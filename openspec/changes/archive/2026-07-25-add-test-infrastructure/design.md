# Design: add-test-infrastructure

## Context

Current facts (verified in-session): the `levy` conda env has no pytest/pytest-cov/coverage; 120 unittest-style tests across 8 files pass offline via `python -m unittest discover`; `pyproject.toml` already carries `[tool.pytest.ini_options]` (testpaths, file pattern) and lists `pytest` (only) in [dev] extras; the suite takes ~81 s because `MockLLMClient.generate()` sleeps a fixed 0.5 s and LEV-4's replay/runner tests drive many engine calls through it. The frozen D1 mandates pytest with >80% coverage; the project owner has fixed the enforced bar at 90% (recorded on Linear LEV-5). Everything must keep running fully offline inside the conda env.

## Goals / Non-Goals

**Goals:**
- pytest as the single canonical runner; the 120 existing tests keep passing under it unmodified.
- Enforced coverage gate: `fail_under = 90`, branch coverage, `source = ["levy"]`, so any drop below 90% fails the standard command.
- Honest denominator: network-only provider code excluded with explicit pragmas; every other line of `levy/` counts.
- Fast suite: remove the fixed-sleep tax from the test path (target: back to seconds, not minutes).
- Docs reflect the one true command.

**Non-Goals:**
- No rewrite of existing unittest-style tests into pytest idiom (zero coverage gain, churn on 120 green tests; new tests may use plain pytest style).
- No mutation testing in the gate (optional occasional audit only, per the LEV-5 Linear note).
- No testing of real network providers (OpenAI/Ollama/sentence-transformers loading) — that is exactly what the pragma exclusions declare out of scope.
- No mandatory CI: the GitHub Actions workflow is an optional final task, skippable without affecting acceptance.

## Decisions

1. **Run existing tests as-is under pytest.** pytest collects and runs `unittest.TestCase` natively; `[tool.pytest.ini_options]` already points at `tests/`. *Alternative rejected:* wholesale migration to pytest functions/fixtures — high churn, no behavioral or coverage benefit, and risks breaking green tests.
2. **Install via conda-forge in `environment.yml`** (`pytest`, `pytest-cov`), keeping the repo's rule that everything Python lives in the `levy` env; sync `pyproject.toml` [dev] extras (add `pytest-cov`) so pip-only consumers of the public repo get the same tooling. *Alternative rejected:* pip-install into the env without updating `environment.yml` — irreproducible env, contradicts the reproducibility story.
3. **Coverage config lives in `pyproject.toml`, gate in the report step**: `[tool.coverage.run] source=["levy"], branch=true`; `[tool.coverage.report] fail_under=90, show_missing=true`. Canonical commands: `python -m pytest tests/ -q` (fast run) and `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` (gated run, also used by CI). Branch coverage is deliberate: untested `if` arms are where silent behavior changes hide.
4. **Exclusions are surgical pragmas, not file omits.** `OpenAILLMClient`/`OllamaLLMClient` (httpx calls), `OllamaEmbeddingClient`, and the model-loading internals of `SentenceTransformerClient` get `# pragma: no cover` on their network-touching bodies; their constructors/config plumbing stay counted where testable offline. *Alternative rejected:* `omit = ["levy/llm_client.py", ...]` — omits whole files including logic that IS offline-testable (e.g. `MockLLMClient`, request shaping), hiding real gaps.
5. **`MockLLMClient` gets an injectable `latency_seconds` parameter (default 0.5, unchanged).** Tests construct it with 0 (or the engine test helpers do); demos keep the realistic delay. The LEV-4 `run_meta.json` "synthetic latency" labeling is unaffected — it describes whatever latency the mock produced. *Alternative rejected:* monkeypatching `time.sleep` in tests — implicit, fragile across files.
6. **Gap-closing tests are written against measured misses, not speculation.** First land tooling, run the gated command, read the per-line miss report, then add targeted offline tests (mock providers) module by module until ≥90% branch coverage. No test is written for a line the report doesn't name.
7. **Optional CI = one workflow, one job**: checkout, micromamba/conda env from `environment.yml`, gated pytest command. Kept as the final, skippable task.

## Risks / Trade-offs

- [90% branch coverage may demand tests for hard-to-reach error paths] → Pragmas are allowed for genuinely unreachable/defensive lines, but each one is an explicit, reviewable marker — never a config-level omit of testable logic. If after honest effort the measured ceiling sits below 90, the number is a fact to surface to the owner (who set the bar), not to quietly lower.
- [Adding pytest to the conda env changes the reproducible environment] → `environment.yml` is the single source; the change is recorded there, not ad-hoc.
- [Reducing mock latency to 0 in tests could mask timing-dependent bugs] → Nothing in the sync codebase depends on wall-clock delay; the parameter default (0.5 s) keeps demos realistic, and any future latency-sensitive test can inject a nonzero value explicitly.
- [Coverage gate makes future PRs fail when they add uncovered code] → That is the intended behavior (the owner's stated purpose: notice when someone changes the wrong line); CLAUDE.md will document the gated command so it's never a surprise.

## Migration Plan

Additive tooling + one backward-compatible mock parameter. `unittest discover` keeps working during and after the change (nothing removes it); docs simply stop advertising it as canonical. Rollback = revert the change; no data or API surface involved.

## Open Questions

- None blocking. Whether CI (optional task) uses micromamba vs miniconda setup action is decided at implementation time by whichever runs the env fastest on GitHub runners.
