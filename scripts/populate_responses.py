#!/usr/bin/env python
"""
Populate the real-provider response corpus (LEV-14). **BILLED. NETWORKED.**

This is the second network-touching entry point in the repository, after
`scripts/fetch_corpora.py`, and it is the only one that spends money. It calls
the configured Anthropic model once per unique prompt of a workload and records,
per call, wall-clock end-to-end latency, input/output token counts, the resolved
model identifier and a UTC timestamp. No test invokes it; an AST guard in
`tests/test_corpus_acquisition.py` keeps it that way.

Cost control, in the order it applies:

1. `--dry-run` prints the call count and the estimated cost and sends nothing.
2. Prompts already in the corpus are skipped, so an interrupted run resumes
   without paying twice. The corpus is keyed by `sha256(prompt)` — the same key
   `ExactCache` uses — so one population serves every configuration of the
   workload.
3. `--max-calls` caps this invocation.
4. `AnthropicLLMClient`'s own `_BudgetGuard` halts before sending once the
   accumulated estimate reaches the cap. A halt is a clean stop: everything
   recorded so far is already on disk, one JSON object per line.

Set `--model` and both `--*-price-per-mtok` flags to the model you are actually
running. The defaults in `levy/config.py` describe a different model, and a cost
figure carried over from it would be wrong in both directions.

Examples:
    # What would this cost? Sends nothing.
    python scripts/populate_responses.py --dataset data/ground_truth.full.csv \\
        --workload faq --out-dir results/latency-faq --dry-run

    # The pilot run.
    python scripts/populate_responses.py --dataset data/ground_truth.full.csv \\
        --workload faq --out-dir results/latency-faq \\
        --model claude-haiku-4-5-20251001 \\
        --input-price-per-mtok 1.0 --output-price-per-mtok 5.0
"""

import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.config import LevyConfig
from levy.dataset.io import load_dataset
from levy.latency.corpus import load_corpus, response_key
from levy.latency.population import populate_corpus
from levy.latency.report import price_at, summarise_calls, write_llm_calls
from levy.llm_client import AnthropicLLMClient

RESPONSES_FILENAME = "responses.jsonl"
CALLS_FILENAME = "llm_calls.json"


def build_arg_parser() -> argparse.ArgumentParser:
    defaults = LevyConfig()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path("data/ground_truth.full.csv"), help="Dataset file (.csv or .json); default: data/ground_truth.full.csv")
    parser.add_argument("--workload", type=str, default="faq", help="Workload to populate (default: faq)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for responses.jsonl and llm_calls.json (never a harness result directory)")
    parser.add_argument("--model", type=str, default=defaults.anthropic_model, help=f"Anthropic model id (default: {defaults.anthropic_model})")
    parser.add_argument("--max-tokens", type=int, default=256, help="max_tokens per request (default: 256)")
    parser.add_argument("--input-price-per-mtok", type=float, default=defaults.anthropic_input_price_per_mtok, help="USD per 1M input tokens for the model actually used")
    parser.add_argument("--output-price-per-mtok", type=float, default=defaults.anthropic_output_price_per_mtok, help="USD per 1M output tokens for the model actually used")
    parser.add_argument("--budget-cap-usd", type=float, default=defaults.anthropic_budget_cap_usd, help=f"Hard cap on estimated spend (default: {defaults.anthropic_budget_cap_usd})")
    parser.add_argument("--max-calls", type=int, default=None, help="Stop after this many calls in this invocation (default: no limit beyond the budget cap)")
    parser.add_argument("--reprice-as", type=str, default=None, help="Also record what this run would have cost at another model's prices, e.g. claude-sonnet-5")
    parser.add_argument("--reprice-input-per-mtok", type=float, default=None, help="Input price for --reprice-as")
    parser.add_argument("--reprice-output-per-mtok", type=float, default=None, help="Output price for --reprice-as")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be called and stop. Sends nothing, spends nothing")
    return parser


def unique_prompts(dataset: Path, workload: str) -> List[str]:
    """
    Every distinct query of the workload, in dataset order.

    Both sides of a pair are populated: the replay submits `query_1` and then
    `query_2`, and a pair whose `query_2` misses needs a response of its own.
    """
    pairs = [pair for pair in load_dataset(dataset) if pair.workload == workload]
    seen = set()
    prompts = []
    for pair in pairs:
        for query in (pair.query_1, pair.query_2):
            if query not in seen:
                seen.add(query)
                prompts.append(query)
    return prompts


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    prompts = unique_prompts(args.dataset, args.workload)
    if not prompts:
        print(f"[populate_responses] no {args.workload!r} pairs in {args.dataset}", file=sys.stderr)
        return 1

    responses_path = args.out_dir / RESPONSES_FILENAME
    existing = load_corpus(responses_path)
    pending = [prompt for prompt in prompts if response_key(prompt) not in existing]
    if args.max_calls is not None:
        pending = pending[: args.max_calls]

    print(
        f"[populate_responses] workload={args.workload} unique_prompts={len(prompts)} "
        f"already_recorded={len(prompts) - len([p for p in prompts if response_key(p) not in existing])} "
        f"to_call={len(pending)} model={args.model}"
    )

    if args.dry_run:
        # A rough estimate only: token counts are not known until the calls
        # return. The recorded figure in llm_calls.json is the observation.
        print(
            f"[populate_responses] dry run: {len(pending)} call(s) would be sent at "
            f"${args.input_price_per_mtok}/MTok in, ${args.output_price_per_mtok}/MTok out. "
            "Nothing sent."
        )
        return 0

    if not pending:
        print("[populate_responses] corpus already complete for this workload; nothing to call")

    config = LevyConfig()
    client = AnthropicLLMClient(
        api_key=config.anthropic_api_key,
        model=args.model,
        max_retries=config.anthropic_max_retries,
        budget_cap_usd=args.budget_cap_usd,
        input_price_per_mtok=args.input_price_per_mtok,
        output_price_per_mtok=args.output_price_per_mtok,
    )

    def report(index: int, total: int) -> None:
        if index % 25 == 0 or index == total:
            print(f"[populate_responses] {index}/{total} calls, estimated spend ${client.estimated_cost_usd:.4f}")

    outcome = populate_corpus(
        prompts,
        responses_path,
        client=client,
        max_tokens=args.max_tokens,
        max_calls=args.max_calls,
        on_progress=report,
    )
    if outcome.halted:
        print(f"[populate_responses] {outcome.halted}", file=sys.stderr)
    if outcome.refusals:
        print(f"[populate_responses] {outcome.refusals} refusal(s), not recorded", file=sys.stderr)

    records = load_corpus(responses_path)
    summary = summarise_calls(records.values(), args.input_price_per_mtok, args.output_price_per_mtok)
    extra = {
        "workload": args.workload,
        "dataset_path": str(args.dataset),
        "refusals": outcome.refusals,
        "halted": outcome.halted,
    }
    if args.reprice_as and args.reprice_input_per_mtok is not None and args.reprice_output_per_mtok is not None:
        extra["repriced"] = price_at(
            summary,
            args.reprice_input_per_mtok,
            args.reprice_output_per_mtok,
            args.reprice_as,
        )

    write_llm_calls(summary, args.out_dir / CALLS_FILENAME, extra=extra)
    print(
        f"[populate_responses] corpus holds {len(records)} response(s); "
        f"wrote {args.out_dir / CALLS_FILENAME}"
    )
    return 1 if outcome.halted else 0


if __name__ == "__main__":
    raise SystemExit(main())
