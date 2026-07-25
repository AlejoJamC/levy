# Proposal: add-release-packaging

**Linear:** LEV-9 | **Maps to:** Deliverables D5 (essential) + D7 (desirable) | **Spec basis:** S&D Report §C D5/D7; Proposal success Criterion 3 (reproducibility).

## Why

The engine, dataset tooling, harness, router, and analysis pipeline all ship, but the repo cannot yet be handed to a third party: there is no reproduction guide, no application container (compose runs Redis only), no user-facing architecture document, and the OpenSpec layer has drifted from the shipped code (6 in-flight changes, 2 archived; 2 capability specs for 8 shipped capabilities). This is the last engineering issue before submission, and D5 is essential — the public Apache 2.0 release is judged on whether someone else can run this.

**Boundary (project rule):** this change delivers *packaging only*, and every artefact is data-agnostic — the guide and container run on the committed fixture and work unchanged when the real dataset replaces it. The claim about *published* headline numbers reproducing within ±5% belongs to LEV-11's production run; here the duty is that the guide, the container path, and `scripts/check_replication.py` execute end-to-end. Nothing in this change waits for the corpus.

## What Changes

- **Reproduction guide** (`docs/REPRODUCTION.md`): env setup → dataset build → 30-configuration harness run → analysis bundle → replication check, as literal commands, defaulting to the committed 15-pair fixture and mock providers; identical steps against the real dataset once present.
- **Application container** (D7): `Dockerfile` + `.dockerignore` + a `docker-compose` service so **one command** runs the full evaluation pipeline offline with mocks; the Anthropic path is opt-in via env var and Redis stays optional (the existing `redis:7-alpine` service is kept, not replaced).
- **User-facing architecture documentation** (`docs/ARCHITECTURE.md`): the component map, request flow, and provider abstractions — currently only in `CLAUDE.md`, which is agent-facing.
- **OpenSpec consistency**: archive the completed in-flight changes and sync their capability specs into `openspec/specs/`, so specs describe shipped code.
- **README de-scaffolding**: user-facing headings carry intermediate ticket numbers (`## HTTP API (LEV-7)`, `## Statistical analysis (LEV-8)`, `## Ground-truth dataset tooling (LEV-3)`, `## Experiment harness (LEV-4)`); these become capability-named sections, since the release must not read as work-in-progress.
- **Secrets / personal-data audit**: a recorded, re-runnable check over the working tree *and* git history. Current state is already clean (`LICENSE` present, `.env` gitignored, only `.env.example` tracked, no secret ever committed) — this change turns that from an observation into a repeatable, documented audit.
- **Repo review**: examples verified runnable (with `examples/anthropic_smoke_check.py` documented as billed/opt-in and excluded from every automated path); superseded docs flagged in place — `docs/RESEARCH_OVERVIEW.md` is historical, and the two frozen university documents are never edited.

Not rebuilt (verified present): `LICENSE`, `.env`/`.env.example` hygiene, `.github/workflows/tests.yml` (already runs the gated suite — no CI duplication), `data/DATASHEET.md` + `data/README.md`.

## Capabilities

### New Capabilities

- `release-packaging`: the reproducibility surface of the public artefact — a literal reproduction guide, a one-command containerised evaluation pipeline, user-facing architecture documentation, spec/code consistency, and a recorded licence/secrets/personal-data audit.

### Modified Capabilities

_None. No shipped capability's requirements change; this change documents, containerises, and audits them._

## Impact

- **New files:** `Dockerfile`, `.dockerignore`, `docs/REPRODUCTION.md`, `docs/ARCHITECTURE.md`, an audit script or documented command set.
- **Touched files:** `docker-compose.yml` (add app service, keep Redis), `README.md` (de-scaffold + link the new docs), `CLAUDE.md` (documentation map), `openspec/changes/*` → `openspec/changes/archive/*` and `openspec/specs/*` (sync).
- **Never touched:** `docs/Project_Proposal.md`, `docs/Specification_and_Design_Report.md` (frozen).
- **Dependencies:** no new Python packages; Docker is host tooling, not an import.
- **Verification:** guide and container run by the author locally (Docker is outside the offline pytest suite); the 90% branch-coverage gate stays green — any new Python is covered, the rest is docs/config.
- **Downstream:** LEV-11 uses this packaging to publish the real D2/D3 outputs; LEV-10 (desirable) ships on top of it.
