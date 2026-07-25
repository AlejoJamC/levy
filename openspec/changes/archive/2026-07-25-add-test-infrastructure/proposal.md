# Proposal: add-test-infrastructure

**Linear:** LEV-5 | **Maps to:** Deliverable D1 quality gate | **Spec basis:** S&D Report §C D1 "unit tests with pytest (target coverage >80%)"; Proposal QA section.

## Why

D1 promises pytest-based tests with >80% coverage, but pytest, pytest-cov, and coverage are not installed in the `levy` conda env (verified: imports fail); tests run via `unittest` and coverage has never been measured. Without an enforced gate, a change to an untested line goes unnoticed — the project owner has set the enforced bar at **90%** (exceeds the frozen 80% floor, no scope conflict). Separately, the suite's runtime has grown to ~81 s for 120 tests because LEV-4's replay tests pass through `MockLLMClient`'s fixed 0.5 s sleep — a test-infrastructure defect worth fixing in the same change.

## What Changes

- Add `pytest` + `pytest-cov` (and the underlying `coverage`) to `environment.yml` via conda-forge; sync `pyproject.toml` [dev] extras (`pytest-cov` is currently missing there).
- Adopt pytest as the canonical runner. The existing 120 unittest-style tests run natively under pytest and are **not rewritten**; `python -m pytest tests/ -q` becomes the standard command.
- Coverage configuration in `pyproject.toml`: `[tool.coverage.run]` with `source = ["levy"]` and `branch = true`; `[tool.coverage.report]` with `fail_under = 90`. Network-only provider code (OpenAI/Ollama LLM clients, Ollama/SentenceTransformer embedding clients) is excluded via explicit `# pragma: no cover` markers rather than inflating the denominator — coverage reflects tests written with intention.
- Measure coverage once tooling lands and close real gaps in `levy/` with targeted offline tests (mock providers only) until the 90% branch-coverage gate passes.
- Make `MockLLMClient` latency injectable (default 0.5 s unchanged for demos; tests inject 0) to restore a fast suite.
- Docs: README and CLAUDE.md updated with the canonical pytest commands; CLAUDE.md known-gap #7 (pytest declared but not installed) marked resolved.
- Optional (kept last, skippable): minimal GitHub Actions workflow running the offline suite with the coverage gate, supporting the public-release reproducibility story.

## Capabilities

### New Capabilities

- `test-infrastructure`: pytest as the canonical offline test runner with an enforced ≥90% branch-coverage gate on the `levy/` package, intentional exclusions for network-only provider code, and fast deterministic execution (no fixed sleeps in the test path).

### Modified Capabilities

_None. No existing capability's spec-level requirements change; the `MockLLMClient` latency parameter is additive with an unchanged default and no existing spec covers mock latency._

## Impact

- **Environment:** `environment.yml` (+pytest, +pytest-cov via conda-forge), `pyproject.toml` ([dev] extras sync, coverage config; `[tool.pytest.ini_options]` already exists).
- **Code:** `levy/llm_client.py` (injectable mock latency, default unchanged), `# pragma: no cover` markers on network-only client internals, new targeted tests under `tests/` for measured gaps.
- **Docs:** README.md, CLAUDE.md (commands + known-gap #7).
- **CI (optional):** new `.github/workflows/` running the offline suite.
- **Downstream:** every future change (LEV-6/7/8) inherits the gate: coverage below 90% fails the standard test command.
