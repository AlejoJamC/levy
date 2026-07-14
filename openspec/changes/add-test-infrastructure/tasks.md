# Tasks: add-test-infrastructure

## 1. Tooling into the environment

- [ ] 1.1 Add `pytest` and `pytest-cov` to `environment.yml` (conda-forge) and add `pytest-cov` to `pyproject.toml` [dev] extras; install into the existing `levy` env and verify `python -m pytest --version` works inside it.
- [ ] 1.2 Run `python -m pytest tests/ -q`: all 120 existing tests collected and green under pytest, zero test-file rewrites.

## 2. Coverage configuration

- [ ] 2.1 Add `[tool.coverage.run]` (`source = ["levy"]`, `branch = true`) and `[tool.coverage.report]` (`fail_under = 90`, `show_missing = true`) to `pyproject.toml`.
- [ ] 2.2 First measured run: `python -m pytest tests/ -q --cov=levy --cov-branch` — record the actual baseline number and the per-module miss report (facts before fixes).

## 3. Fast suite

- [ ] 3.1 Add injectable latency to `MockLLMClient` (default 0.5 s unchanged); demos untouched.
- [ ] 3.2 Point tests (engine/replay/runner paths) at zero latency; confirm full suite runtime drops from ~81 s to seconds and stays green.

## 4. Honest exclusions and gap closing

- [ ] 4.1 Add `# pragma: no cover` to network-only bodies: `OpenAILLMClient`/`OllamaLLMClient` request paths, `OllamaEmbeddingClient`, `SentenceTransformerClient` model-loading/encode internals. No whole-file omits.
- [ ] 4.2 Close measured gaps with targeted offline tests (mock providers only), module by module per the miss report, until the gated command passes at ≥ 90% branch coverage.
- [ ] 4.3 Gated command green: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` exits zero inside the conda env.

## 5. Docs & sync

- [ ] 5.1 Update README.md and CLAUDE.md: canonical commands (fast + gated), note unittest still works but pytest is canonical, mark CLAUDE.md known-gap #7 resolved.
- [ ] 5.2 `openspec validate --all` passes; keep Linear LEV-5 in sync (reference this change; tick delivered acceptance criteria; record the measured baseline and final coverage numbers).

## 6. Optional CI (skippable)

- [ ] 6.1 Minimal GitHub Actions workflow: build the conda env from `environment.yml`, run the gated pytest command offline. Skippable without affecting acceptance.
