# Tasks: add-release-packaging

## 1. Shared pipeline definition

- [ ] 1.1 Create `scripts/reproduce.sh`: chain `run_experiments.py` → `run_analysis.py` → `check_replication.py` using the verified flags (`--dataset/--out-dir/--embedding-provider`; `--results-dir/--out-dir/--dataset`; `--reference/--dataset`), with overridable dataset and output directory, mock providers by default, `set -euo pipefail`, and non-zero exit on any stage failure.
- [ ] 1.2 Run it end-to-end on the committed fixture; confirm harness outputs + analysis bundle + replication pass.

## 2. Container (D7)

- [ ] 2.1 Add `Dockerfile` on a micromamba/conda-forge base building the env from `environment.yml` (single dependency source; keeps conda-forge `faiss-cpu`), copying the package/scripts/data, with `CMD` invoking `scripts/reproduce.sh` on the fixture with mock providers.
- [ ] 2.2 Add `.dockerignore` (exclude `.git`, `results/`, caches, `__pycache__`, `.env`) to keep the build context small.
- [ ] 2.3 Extend `docker-compose.yml` with a pipeline service run via `docker compose run --rm`; leave the existing `redis:7-alpine` service untouched and keep the default path independent of Redis; Anthropic path opt-in via env var.
- [ ] 2.4 Author-verify locally: build the image, run the one command on a machine with no API key and (as far as practical) no network; record build time and image size for the guide. Note Docker is outside the offline pytest suite.

## 3. Documentation (D5)

- [ ] 3.1 Write `docs/REPRODUCTION.md`: env setup (conda + Docker paths) → dataset build → 30-config run → analysis bundle → replication check, as verbatim commands sourced from `scripts/reproduce.sh`; include the expected outputs, and a "swap in the real dataset" section that changes only `--dataset`.
- [ ] 3.2 Write `docs/ARCHITECTURE.md` (user-facing): component map traced to the frozen spec's three named components (router; embedding manager + Faiss HNSW; Anthropic connector) plus dataset/harness/analysis layers, request flow, ABC+mock provider pattern; link from README and from CLAUDE.md (no duplication).
- [ ] 3.3 De-scaffold `README.md`: rename ticket-numbered headings (`## HTTP API (LEV-7)`, `## Statistical analysis (LEV-8)`, `## Ground-truth dataset tooling (LEV-3)`, `## Experiment harness (LEV-4)`) to capability names, keep content, add links to the two new docs and to the datasheet.
- [ ] 3.4 Document `examples/anthropic_smoke_check.py` as billed/opt-in and confirm no automated path invokes it; verify `simple_replay.py` runs offline and note `ollama_demo.py`'s local-service requirement.
- [ ] 3.5 Flag superseded docs in place (`docs/RESEARCH_OVERVIEW.md` historical). Do NOT touch `docs/Project_Proposal.md` or `docs/Specification_and_Design_Report.md` — frozen; verify they are byte-identical at the end.

## 4. Audit

- [ ] 4.1 Create `scripts/audit_release.sh`: LICENSE present; no tracked `.env`; no secret-shaped strings in the working tree; no secret introduced in git history (`git log -S` over key patterns); no personal-data markers in `data/`; prints pass/fail per check and exits non-zero on any finding.
- [ ] 4.2 Run it; record the clean result in the reproduction guide (or a release-checklist section) so the criterion is re-runnable rather than a one-time claim.

## 5. OpenSpec consistency

- [ ] 5.1 Archive the in-flight changes whose code has shipped (`add-ground-truth-dataset`, `add-experiment-harness`, `add-test-infrastructure`, `add-anthropic-connector`, `add-fastapi-router`, and `add-statistical-analysis` once merged), leaving anything unshipped in flight.
- [ ] 5.2 Sync each archived change's capability into `openspec/specs/` using **main-spec structure** (`## Purpose` + `## Requirements`) — not delta headers (`## ADDED Requirements`), the mistake the LEV-1/LEV-2 archive made and that had to be repaired.
- [ ] 5.3 Run `openspec validate --all` after each archive/sync step and fix immediately; final state green across all changes and specs.
- [ ] 5.4 Update CLAUDE.md's documentation map and architecture section to match the released layout (new docs, scripts, archived changes).

## 6. Final review & sync

- [ ] 6.1 Gated suite green: `python -m pytest tests/ -q --cov=levy --cov-branch --cov-fail-under=90` (new artefacts are shell/docs/config, so the denominator is unchanged; any new Python must be covered).
- [ ] 6.2 Sync Linear LEV-9: reference this change, tick the packaging acceptance criteria, and record that the published-numbers ±5% verification belongs to LEV-11's production run, not here.
