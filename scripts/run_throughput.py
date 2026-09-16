#!/usr/bin/env python
"""
Throughput under concurrent load (LEV-17).

Serves `levy.api.app.create_app()` via `uvicorn` -- the same app, same
threadpool dispatch, but built with an explicit `LevyConfig` instead of the
module-level default, using `create_app`'s own `config` parameter (already
public: "Tests pass a mock-provider `config` for offline runs"). No file
outside this script is edited.

The LLM provider is still the mock (no spend, no rate limit enters the
measurement -- the task's own requirement), but its simulated per-call delay
defaults to the REAL measured latency from the actual billed Anthropic run
(LEV-14: `release/latency/llm_calls.json`, `claude-haiku-4-5-20251001`, 600
real calls), not an arbitrary constant. `--llm-latency-seconds` overrides it;
`--llm-latency-percentile {p50,p95}` selects which real figure is the default.

Two arms per level:
  all-hit  -- one prompt, pre-warmed, repeated: every request is a cache hit.
  all-miss -- a globally-unique prompt per request: every request is a miss,
              served by MockLLMClient at the configured (real-measured) delay.

Endpoints are declared `def` (sync), so FastAPI/Starlette dispatches each
request to anyio's worker threadpool (levy/api/app.py:10-14); concurrency is
therefore bounded by that threadpool's size, which this script measures
directly rather than assuming, and records alongside the results.

Percentiles use `levy.latency.benchmark.percentile` (nearest-rank, matching
the LEV-14 convention): a reported p95 always names a real observation.

Output is written under a model-identifier subdirectory of `--out-dir`
(`<out-dir>/<model-id>/throughput.csv` etc.), never directly in `--out-dir` --
a second run against a different model must not silently overwrite the first,
and both must be able to sit side by side for comparison. `<model-id>` is the
resolved model whose latency the run actually reflects: the model named in
`--llm-calls-json`'s provenance under the mock provider (replayed latency), or
`--anthropic-model` under the live provider (real latency). An explicit
`--llm-latency-seconds` override carries no model identity, so that case is
labelled `custom-<seconds>s-latency` rather than guessing.

Usage:
    python scripts/run_throughput.py --out-dir release/throughput
    python scripts/run_throughput.py --out-dir /tmp/throughput --llm-latency-percentile p95
    python scripts/run_throughput.py --out-dir /tmp/throughput --llm-latency-seconds 1.0
    python scripts/run_throughput.py --out-dir release/throughput --llm-provider anthropic --anthropic-model claude-opus-4-8
"""

import argparse
import asyncio
import json
import multiprocessing
import platform
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.latency.benchmark import percentile

ARMS = ("all-hit", "all-miss")
DEFAULT_LLM_CALLS_JSON = Path("release/latency/llm_calls.json")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for throughput.csv and throughput_meta.json (required unless --serve-only)")
    parser.add_argument("--levels", type=str, default="1,4,16,64", help="Comma-separated concurrency ladder (default: 1,4,16,64)")
    parser.add_argument("--requests-per-level", type=int, default=None, help="Requests fired per (level, arm); default: max(level*5, 30)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind the server to (default: an auto-selected free port; required with --serve-only)")
    parser.add_argument("--startup-timeout", type=float, default=60.0, help="Seconds to wait for the server to become ready (default: 60)")
    parser.add_argument("--llm-calls-json", type=Path, default=DEFAULT_LLM_CALLS_JSON, help=f"Source of real measured LLM latency (default: {DEFAULT_LLM_CALLS_JSON})")
    parser.add_argument("--llm-latency-percentile", type=str, default="p50", choices=["p50", "p95"], help="Which real measured percentile to use as the mock's per-call delay (default: p50)")
    parser.add_argument("--llm-latency-seconds", type=float, default=None, help="Override: use this fixed delay instead of reading real data")
    parser.add_argument("--llm-provider", type=str, default="mock", choices=["mock", "anthropic"], help="'mock' (default): no network call, simulated delay per --llm-latency-seconds/--llm-latency-percentile. 'anthropic': real, billed calls to the real API for every all-miss request -- real cost, real rate limits. Requires ANTHROPIC_API_KEY.")
    parser.add_argument("--anthropic-model", type=str, default=None, help="Model to use with --llm-provider anthropic (default: LevyConfig's own default, claude-haiku-4-5-20251001). Also names the output subdirectory, so a second model's run cannot overwrite the first's.")
    parser.add_argument("--serve-only", action="store_true", help="Only start the server (blocking, foreground) on --host/--port; do not generate load or write output. Meant to run inside a resource-isolated container while a separate --client-only process drives it.")
    parser.add_argument("--client-only", action="store_true", help="Do not start a local server; drive load against --base-url instead (an already-running --serve-only instance).")
    parser.add_argument("--base-url", type=str, default=None, help="Server URL to drive load against, with --client-only (e.g. http://127.0.0.1:8000)")
    parser.add_argument("--server-isolation-note", type=str, default=None, help="Free-text provenance of how the server was resource-isolated (e.g. 'docker run --cpus=2'), recorded verbatim in throughput_meta.json; required with --client-only so the isolation claim is stated, not implied")
    return parser


def resolve_llm_latency_seconds(args: argparse.Namespace) -> Tuple[float, dict]:
    """Real measured latency by default; an explicit override is still an
    explicit, stated number, not a silent fallback to an arbitrary constant."""
    if args.llm_latency_seconds is not None:
        return args.llm_latency_seconds, {"source": "explicit --llm-latency-seconds override", "value_ms": args.llm_latency_seconds * 1000.0}

    if not args.llm_calls_json.is_file():
        raise RuntimeError(
            f"{args.llm_calls_json} not found -- pass --llm-latency-seconds explicitly, "
            "or point --llm-calls-json at the real measured LEV-14 output"
        )
    data = json.loads(args.llm_calls_json.read_text(encoding="utf-8"))
    latency_ms = data["latency_ms"][args.llm_latency_percentile]
    provenance = {
        "source": str(args.llm_calls_json),
        "percentile": args.llm_latency_percentile,
        "value_ms": latency_ms,
        "model": data.get("model"),
        "n_calls": data.get("n_calls"),
    }
    return latency_ms / 1000.0, provenance


def resolve_model_id(args: argparse.Namespace, llm_latency_provenance: dict) -> str:
    """The model identifier this run's numbers actually reflect -- the output
    subdirectory name, so two models' results never collide."""
    if args.llm_provider == "anthropic":
        from levy.config import LevyConfig

        return args.anthropic_model or LevyConfig().anthropic_model

    model = llm_latency_provenance.get("model")
    if model:
        return model
    seconds = llm_latency_provenance.get("value_ms", 0.0) / 1000.0
    return f"custom-{seconds:g}s-latency"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _serve(host: str, port: int, llm_latency_seconds: float, llm_provider: str = "mock", anthropic_model: Optional[str] = None) -> None:
    import uvicorn

    from levy.api.app import create_app
    from levy.config import LevyConfig

    if llm_provider == "anthropic":
        kwargs = {"llm_provider": "anthropic", "embedding_provider": "sentence-transformers"}
        if anthropic_model:
            kwargs["anthropic_model"] = anthropic_model
        config = LevyConfig(**kwargs)
    else:
        config = LevyConfig(
            llm_provider="mock",
            mock_llm_latency_seconds=llm_latency_seconds,
            embedding_provider="sentence-transformers",
        )
    app = create_app(config=config)
    uvicorn.run(app, host=host, port=port, log_level="warning")


def start_server(host: str, port: int, llm_latency_seconds: float, llm_provider: str = "mock", anthropic_model: Optional[str] = None) -> multiprocessing.Process:
    proc = multiprocessing.Process(target=_serve, args=(host, port, llm_latency_seconds, llm_provider, anthropic_model), daemon=True)
    proc.start()
    return proc


def wait_ready(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_exc: Optional[Exception] = None
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/admin/cache/stats", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001 - polling until the server accepts connections
            last_exc = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready within {timeout}s (last error: {last_exc})")


def measure_threadpool_worker_count() -> int:
    """The anyio thread limiter FastAPI/Starlette dispatches sync `def` endpoints
    through -- measured directly (it is a fixed default, not derived from
    os.cpu_count()), because it is the bound this whole measurement is against."""
    import anyio

    async def _read() -> int:
        return anyio.to_thread.current_default_thread_limiter().total_tokens

    return asyncio.run(_read())


async def fire_one(client: httpx.AsyncClient, base_url: str, prompt: str) -> Tuple[bool, float, Optional[str]]:
    """Third element is the failure reason (exception class, or `http_<status>`),
    None on success -- a "24% error rate" with no recorded cause is a number
    without a diagnosis, so this is captured rather than discarded."""
    t0 = time.perf_counter()
    reason: Optional[str] = None
    try:
        r = await client.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": prompt}]},
            timeout=30.0,
        )
        ok = r.status_code == 200
        if not ok:
            reason = f"http_{r.status_code}"
    except Exception as exc:  # noqa: BLE001 - a failed request is a measured error, not a script bug
        ok = False
        reason = f"{type(exc).__name__}: {exc}"
    t1 = time.perf_counter()
    return ok, (t1 - t0) * 1000.0, reason


async def run_level(base_url: str, concurrency: int, arm: str, n_requests: int) -> dict:
    limits = httpx.Limits(max_connections=max(200, concurrency * 2), max_keepalive_connections=max(200, concurrency * 2))
    async with httpx.AsyncClient(limits=limits) as client:
        if arm == "all-hit":
            warm_prompt = f"throughput-hit-{concurrency}-{uuid.uuid4()}"
            await fire_one(client, base_url, warm_prompt)  # pre-warm: this one is excluded from timing
            prompts = [warm_prompt] * n_requests
        else:
            prompts = [f"throughput-miss-{concurrency}-{i}-{uuid.uuid4()}" for i in range(n_requests)]

        sem = asyncio.Semaphore(concurrency)
        results: List[Tuple[bool, float, Optional[str]]] = []

        async def worker(prompt: str) -> None:
            async with sem:
                results.append(await fire_one(client, base_url, prompt))

        t_start = time.perf_counter()
        await asyncio.gather(*(worker(p) for p in prompts))
        elapsed = time.perf_counter() - t_start

    oks = [ms for ok, ms, _ in results if ok]
    error_reasons = [reason for ok, _, reason in results if not ok]
    error_count = len(error_reasons)
    n = len(results)
    error_breakdown: dict = {}
    for reason in error_reasons:
        error_breakdown[reason] = error_breakdown.get(reason, 0) + 1
    return {
        "concurrency": concurrency,
        "arm": arm,
        "n_requests": n,
        "elapsed_seconds": elapsed,
        "requests_per_second": (n / elapsed) if elapsed > 0 else None,
        "p50_ms": percentile(oks, 50) if oks else None,
        "p95_ms": percentile(oks, 95) if oks else None,
        "error_count": error_count,
        "error_rate": (error_count / n) if n else None,
        "error_breakdown": error_breakdown,
    }


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    levels = [int(x) for x in args.levels.split(",")]

    if args.llm_provider == "anthropic":
        llm_latency_seconds = None
        llm_latency_provenance = {"note": "real Anthropic API calls -- latency is whatever the live API returns, not simulated"}

    if args.serve_only:
        if args.port is None:
            raise SystemExit("--serve-only requires --port")
        if args.llm_provider != "anthropic":
            llm_latency_seconds, llm_latency_provenance = resolve_llm_latency_seconds(args)
        model_id = resolve_model_id(args, llm_latency_provenance)
        print(f"[run_throughput] serving on {args.host}:{args.port}, llm_provider={args.llm_provider}, model={model_id}, llm_latency_seconds={llm_latency_seconds}", file=sys.stderr)
        _serve(args.host, args.port, llm_latency_seconds, args.llm_provider, args.anthropic_model)  # blocks
        return 0

    if args.client_only:
        if not args.base_url:
            raise SystemExit("--client-only requires --base-url")
        if not args.server_isolation_note:
            raise SystemExit("--client-only requires --server-isolation-note (state how the server was isolated)")
        if not args.out_dir:
            raise SystemExit("--out-dir is required")
        if args.llm_provider != "anthropic":
            llm_latency_seconds, llm_latency_provenance = resolve_llm_latency_seconds(args)
        threadpool_worker_count = measure_threadpool_worker_count()
        base_url = args.base_url
        wait_ready(base_url, args.startup_timeout)
        proc = None
        server_isolation_note = args.server_isolation_note
    else:
        if not args.out_dir:
            raise SystemExit("--out-dir is required")
        if args.llm_provider != "anthropic":
            llm_latency_seconds, llm_latency_provenance = resolve_llm_latency_seconds(args)
        threadpool_worker_count = measure_threadpool_worker_count()
        port = args.port or free_port()
        base_url = f"http://{args.host}:{port}"
        proc = start_server(args.host, port, llm_latency_seconds, args.llm_provider, args.anthropic_model)
        wait_ready(base_url, args.startup_timeout)
        server_isolation_note = "none -- client and server share this process's host and CPU cores"

    model_id = resolve_model_id(args, llm_latency_provenance)

    try:
        rows = []
        for level in levels:
            n_requests = args.requests_per_level or max(level * 5, 30)
            for arm in ARMS:
                print(f"[run_throughput] concurrency={level} arm={arm} n_requests={n_requests}", file=sys.stderr)
                rows.append(asyncio.run(run_level(base_url, level, arm, n_requests)))
    finally:
        if proc is not None:
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=10)

    out_subdir = args.out_dir / model_id
    out_subdir.mkdir(parents=True, exist_ok=True)

    csv_path = out_subdir / "throughput.csv"
    fieldnames = ["concurrency", "arm", "n_requests", "elapsed_seconds", "requests_per_second", "p50_ms", "p95_ms", "error_count", "error_rate"]
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(str(row[f]) for f in fieldnames) + "\n")

    error_breakdowns = {
        f"concurrency={row['concurrency']}|arm={row['arm']}": row["error_breakdown"]
        for row in rows
        if row["error_breakdown"]
    }

    meta = {
        "generated_by": "scripts/run_throughput.py",
        "command": "uvicorn (levy.api.app.create_app(config=...), programmatic, not the CLI default app)",
        "model_id": model_id,
        "llm_provider": args.llm_provider,
        "anthropic_model": model_id if args.llm_provider == "anthropic" else None,
        "mock_llm_latency_seconds_used": llm_latency_seconds,
        "mock_llm_latency_provenance": llm_latency_provenance,
        "embedding_provider": "sentence-transformers",
        "levels": levels,
        "arms": list(ARMS),
        "threadpool_worker_count": threadpool_worker_count,
        "threadpool_note": "anyio.to_thread.current_default_thread_limiter().total_tokens -- the bound FastAPI dispatches sync `def` endpoints through; not derived from os.cpu_count(). Measured on the client process in --client-only mode (anyio's default is a fixed constant, not independently confirmed inside the server's own environment).",
        "server_isolation_note": server_isolation_note,
        "error_breakdowns": error_breakdowns,
        "host": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": __import__("os").cpu_count(),
        },
    }
    meta_path = out_subdir / "throughput_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(f"[run_throughput] wrote {csv_path} and {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
