# Design: add-release-packaging

## Context

Verified this session. **Present:** `LICENSE`; `.env` gitignored with only `.env.example` tracked and no secret in git history; `.github/workflows/tests.yml` already running the gated pytest suite via conda; `data/DATASHEET.md` + `data/README.md`; a 460-line README covering every shipped component; `docker-compose.yml` with a single `redis:7-alpine` service. **Absent:** `Dockerfile`/`.dockerignore`, reproduction guide, user-facing architecture doc, OpenSpec consistency (6 in-flight changes / 2 archived; `openspec/specs/` holds `embedding-management` + `vector-store` only). **Verified CLI surface** (the guide and container must match it literally): `run_experiments.py --dataset --out-dir --models --workloads --thresholds --embedding-provider`; `run_analysis.py --results-dir --out-dir --dataset --alpha --kappa-threshold`; `check_replication.py --reference --dataset --embedding-provider --keep-dir --relative-tolerance --absolute-floor --llm-latency-seconds --show-all`. `environment.yml` is the single dependency source: conda-forge for numpy/pandas/statsmodels/matplotlib/faiss-cpu/fastapi/uvicorn/pydantic/redis-py/pytest, pip for sentence-transformers/transformers/anthropic, with a recorded note that the pip faiss wheel segfaults on Apple Silicon.

## Goals / Non-Goals

**Goals:**
- A literal reproduction guide someone else can follow in a clean environment, defaulting to the committed fixture and mock providers.
- One command that runs the whole evaluation pipeline in a container, fully offline (no API key, no model download, no network).
- User-facing architecture documentation traceable to the frozen spec's named components.
- OpenSpec specs and archive consistent with shipped code, with `openspec validate --all` green.
- A recorded, re-runnable licence/secrets/personal-data audit.

**Non-Goals:**
- No published container registry image (local build only), no Kubernetes/deployment manifests.
- No new CI (the gated-suite workflow already exists — extending it is out of scope).
- No dashboard (LEV-10), no changes to any shipped capability's behaviour.
- No production of real research numbers — that is LEV-11's run using these artefacts.
- No edits to the two frozen university documents, ever.

## Decisions

1. **Container base = conda-forge (micromamba/mambaforge), built from `environment.yml`.** `environment.yml` is already the single source of truth for the env, and it deliberately takes `faiss-cpu` from conda-forge because the pip wheel is broken on the author's platform. A conda-based image reproduces the documented environment exactly, so "works in Docker" and "works in the guide" cannot diverge. *Alternative rejected:* `python:3.10-slim` + `pip install`, which would create a second dependency list to keep in sync and reintroduce the faiss wheel risk.
2. **The one-command default path is the offline mock/fixture path.** Container `CMD` runs the pipeline with `--embedding-provider mock` against `data/ground_truth.csv`, so it completes with no network, no `ANTHROPIC_API_KEY`, and no model download — which is the only way "one command in a clean environment" is honestly true. Real embedding providers and the Anthropic backend are opt-in via env vars/flags, documented with their first-run download and billing implications. This also keeps the container aligned with the project's mock-first reproducibility rule.
3. **One shell definition of the pipeline, shared by the guide and the container.** A single `scripts/reproduce.sh` chains `run_experiments.py` → `run_analysis.py` → `check_replication.py` with the verified flags and an overridable dataset/out-dir; the guide shows those same commands and the container's `CMD` invokes the script. *Alternative rejected:* commands duplicated in prose and in the `Dockerfile` — guaranteed to drift the moment a flag changes. Shell keeps it out of the Python coverage denominator while staying one source of truth.
4. **Compose gains an app service; Redis is left exactly as-is.** The existing `redis:7-alpine` service stays so today's documented `docker-compose up -d` keeps working; the new service runs the pipeline via `docker compose run --rm`. Redis is *not* a dependency of the default path (the engine defaults to the memory store), so the pipeline never requires it to be up.
5. **Data-agnosticism is proven by construction in the guide:** the "swap in the real dataset" step changes exactly one argument (`--dataset`) and nothing else — no rewritten steps, no second guide. That is the auditable form of the project rule, and it doubles as the guide's own regression test.
6. **Architecture doc is user-facing and spec-traceable.** `docs/ARCHITECTURE.md` maps the code onto the frozen S&D's three named components (FastAPI router; embedding manager + Faiss HNSW index; Anthropic connector) plus the dataset/harness/analysis layers, with the request flow and the ABC+mock provider pattern. CLAUDE.md remains the agent-facing map and links to it rather than duplicating it.
7. **OpenSpec sync repeats no prior mistake.** Archive only changes whose code has actually shipped, and when syncing each capability into `openspec/specs/`, write main-spec structure (`## Purpose` + `## Requirements`) — the LEV-1/LEV-2 archive previously left delta-style `## ADDED Requirements` headers and no Purpose, which failed `openspec validate --all` and had to be repaired. Validation is re-run after each sync, not once at the end.
8. **Audit is a script, not a claim.** `scripts/audit_release.sh` checks: LICENSE present; no tracked `.env`; no secret-shaped strings in the working tree; no secret ever introduced in git history (`git log -S` over key patterns); dataset files free of personal data markers. It prints a pass/fail summary and exits non-zero on any finding, so the acceptance criterion is re-runnable rather than a one-time eyeball.
9. **README de-scaffolding is a rename, not a rewrite.** Ticket-numbered headings become capability names (`## HTTP API`, `## Statistical analysis`, …); the content stays. Ticket identifiers remain where they belong — CLAUDE.md, OpenSpec changes, and git history — not in user-facing docs of a released artefact.

## Risks / Trade-offs

- [Image size: `environment.yml` pulls sentence-transformers → torch, so the image will be multi-GB even though the default path uses mock embeddings] → Accepted and documented (a research artefact optimised for fidelity, not distribution size); `.dockerignore` keeps build context small, and the guide states the expected build time and size instead of surprising the user.
- [Conda solve time makes the Docker build slow] → Use micromamba and pin via the existing `environment.yml`; document the one-time build cost. Not worth a second, faster-but-divergent dependency path.
- [Docker cannot be verified inside the offline pytest suite] → Explicitly author-verified locally (build + one-command run recorded in the task list); the suite keeps covering the Python it invokes. Stated plainly rather than implied by a green test run.
- [Archiving 6 changes at once could break `openspec validate --all`] → Validate after each archive/sync step; the known failure mode (delta headers in main specs) is called out in the tasks so it is checked, not rediscovered.
- [Guide rot as code evolves] → The guide's commands come from `scripts/reproduce.sh`, so a flag change breaks the script (and its author-run) rather than silently leaving stale prose.

## Migration Plan

Additive files (`Dockerfile`, `.dockerignore`, `docs/REPRODUCTION.md`, `docs/ARCHITECTURE.md`, two shell scripts) plus edits to `docker-compose.yml`, `README.md`, `CLAUDE.md`, and the OpenSpec archive/specs layout. No code behaviour changes, no dependency additions, no data migration. Rollback = delete the new files and revert the doc/compose edits. The frozen documents are untouched throughout.

## Open Questions

- None blocking. Whether the image is ever pushed to a public registry is a submission-time choice for the author and does not affect this change (local build is the deliverable).
