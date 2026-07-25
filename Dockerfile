# Levy — containerised evaluation pipeline (deliverable D7).
#
# Base: micromamba on conda-forge. The environment is built from
# `environment.yml`, the repository's single dependency specification — the same
# file the documented conda setup uses — so "works in Docker" and "works by
# following the guide" cannot diverge. In particular this keeps `faiss-cpu` from
# conda-forge (the pip wheel segfaults on Apple Silicon arm64).
#
# Default entry point runs the full pipeline offline: the committed fixture
# dataset with mock embeddings, so no API key, no model download, and no network
# access are needed.
#
# Build:
#   docker build -t levy:latest .
#
# Run (one command, offline):
#   docker run --rm -v "$PWD/results:/opt/levy/results" levy:latest
#
# Expect a multi-GB image: `environment.yml` pulls sentence-transformers, hence
# torch. This is a research artefact optimised for environment fidelity, not for
# distribution size.

FROM mambaorg/micromamba:1.5.8

# Create the `levy` environment from the repository's dependency specification.
# Copied alone first so the (slow) solve is cached until environment.yml changes.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba clean --all --yes

# Every subsequent RUN/CMD executes inside the environment.
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt/levy

# The package, the CLIs, the fixture dataset, and the packaging metadata.
# `.dockerignore` keeps the build context free of .git, results/, and caches.
COPY --chown=$MAMBA_USER:$MAMBA_USER levy/ ./levy/
COPY --chown=$MAMBA_USER:$MAMBA_USER scripts/ ./scripts/
COPY --chown=$MAMBA_USER:$MAMBA_USER data/ ./data/
COPY --chown=$MAMBA_USER:$MAMBA_USER tests/ ./tests/
COPY --chown=$MAMBA_USER:$MAMBA_USER examples/ ./examples/
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE ./

ENV PYTHONPATH=/opt/levy \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    LEVY_OUT_DIR=/opt/levy/results/reproduce

# Offline by default: fixture dataset + mock embeddings + mock LLM.
# Opt in to real providers with `-e LEVY_EMBEDDING_PROVIDER=sentence-transformers`
# (downloads model weights: needs network) and, for the HTTP API only,
# `-e ANTHROPIC_API_KEY=...` (billed). The pipeline itself never calls a real LLM.
CMD ["scripts/reproduce.sh"]
