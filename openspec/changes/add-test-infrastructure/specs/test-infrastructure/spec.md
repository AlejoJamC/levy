# test-infrastructure

Capability: pytest as the canonical offline test runner for the `levy` package with an enforced branch-coverage gate, honest exclusions for network-only provider code, and fast deterministic execution.

## ADDED Requirements

### Requirement: pytest is the canonical runner in the conda env
The system SHALL provide pytest and pytest-cov inside the `levy` conda environment (declared in `environment.yml`, mirrored in `pyproject.toml` [dev] extras), and `python -m pytest tests/ -q` SHALL collect and pass the entire existing test suite without modifying the existing unittest-style tests.

#### Scenario: Full suite green under pytest
- **WHEN** `python -m pytest tests/ -q` runs inside the `levy` conda environment
- **THEN** all existing tests are collected and pass, with zero test files rewritten for the migration

#### Scenario: Environment is reproducible
- **WHEN** the conda environment is recreated from `environment.yml`
- **THEN** pytest and pytest-cov are present without any manual pip step

### Requirement: Enforced coverage gate at 90%
The system SHALL measure branch coverage over the `levy/` package (`source = ["levy"]`, `branch = true`) and the gated test command SHALL fail with a non-zero exit when total coverage is below 90%.

#### Scenario: Gate passes at or above the bar
- **WHEN** the gated command runs and measured branch coverage of `levy/` is ≥ 90%
- **THEN** the command exits zero and reports the coverage total

#### Scenario: Gate fails below the bar
- **WHEN** a change drops measured coverage below 90%
- **THEN** the gated command exits non-zero, so the untested change is noticed rather than silently merged

### Requirement: Intentional exclusions only
Code excluded from the coverage denominator SHALL be limited to network-only provider internals (OpenAI/Ollama LLM clients, Ollama/sentence-transformers embedding client network paths), marked with explicit inline `# pragma: no cover` comments. Whole-file omission of modules containing offline-testable logic SHALL NOT be used.

#### Scenario: Exclusions are visible in the diff
- **WHEN** a reviewer inspects why a line is not counted
- **THEN** an inline pragma marks that exact line or block, rather than a configuration entry hiding an entire file

#### Scenario: Offline-testable logic stays counted
- **WHEN** coverage is computed
- **THEN** mock providers, configuration plumbing, and all engine/cache/dataset/experiment logic are included in the denominator

### Requirement: Fully offline suite
The entire test suite, including the gated coverage run, SHALL execute with no network access, using mock providers only.

#### Scenario: No-network execution
- **WHEN** the gated command runs on a machine with no network access
- **THEN** every test passes and coverage is reported

### Requirement: Fast test execution without fixed sleeps
The mock LLM client SHALL accept an injectable latency (default unchanged at 0.5 s for demos), tests SHALL run with zero injected latency, and the full suite SHALL complete in well under a minute.

#### Scenario: Tests are not taxed by mock latency
- **WHEN** the full suite runs with tests injecting zero mock latency
- **THEN** total runtime is a small fraction of the previous ~81 s baseline and no test sleeps a fixed 0.5 s per engine call

#### Scenario: Demo behavior unchanged
- **WHEN** `MockLLMClient` is constructed without a latency argument
- **THEN** it behaves exactly as before (0.5 s simulated delay)

### Requirement: Documented canonical commands
Project documentation SHALL state the canonical fast command (`python -m pytest tests/ -q`) and the gated coverage command, replacing the unittest command as the advertised default while the unittest runner keeps working.

#### Scenario: Docs match reality
- **WHEN** a contributor follows README/CLAUDE.md test instructions verbatim inside the conda env
- **THEN** the commands run successfully and enforce the 90% gate on the gated variant
