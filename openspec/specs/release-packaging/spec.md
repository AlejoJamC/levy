# release-packaging Specification

Capability: the reproducibility surface of the public artefact — a literal reproduction guide, a one-command containerised evaluation pipeline, user-facing architecture documentation, spec/code consistency, and a recorded licence/secrets/personal-data audit.

## Purpose

Package the released system as a reproducible public artefact per the frozen specification's deliverables: a literal, verbatim-executable reproduction guide covering environment setup, dataset build, the 30-configuration harness sweep, the statistical analysis bundle, and the replication check — offline by default against the committed fixture dataset and mock providers, and dataset-agnostic so the real dataset drops in via a single argument change with no other steps or code changes; a one-command containerised path that runs the same pipeline end-to-end with no API key and no network, built from the repository's single dependency specification, alongside the pre-existing Redis service left intact; a single shared definition of the pipeline so the guide and the container entry point cannot silently diverge; user-facing architecture documentation that traces the shipped code back to the frozen specification's named components; release documentation, archived OpenSpec changes, and synced capability specs that describe only what has actually shipped, free of in-flight ticket scaffolding, validated end-to-end; a recorded, re-runnable release audit that checks licensing, tracked and historical secrets, and personal-data exposure and fails loudly on any finding; and honest handling of runnable examples and superseded documents, with the frozen specification documents left untouched.
## Requirements
### Requirement: Literal reproduction guide
The repository SHALL contain a step-by-step reproduction guide covering environment setup, dataset build, the 30-configuration harness run, the analysis bundle, and the replication check, written as commands that can be executed verbatim, defaulting to the committed fixture dataset and mock providers so it requires no API key and no network.

#### Scenario: Guide followed in a clean environment
- **WHEN** a third party follows the guide literally from a fresh checkout
- **THEN** the pipeline completes and the analysis bundle (tables and figures) is produced

#### Scenario: Commands match the shipped CLIs
- **WHEN** the guide's commands are compared against the scripts' actual arguments
- **THEN** every flag shown exists and is spelled as the script defines it

### Requirement: Dataset-agnostic reproduction path
Switching from the committed fixture to the real dataset SHALL require changing only the dataset argument — no different steps, no separate guide, no code change — and the guide SHALL state this explicitly.

#### Scenario: Real dataset swap
- **WHEN** the real dataset replaces the fixture and the documented dataset argument is pointed at it
- **THEN** the same commands run unchanged and produce the same output structure

### Requirement: One-command containerised pipeline
The repository SHALL provide a container definition and a compose service such that a single documented command executes the full evaluation pipeline end-to-end offline with mock providers, requiring no API key, no model download, and no network; the container environment SHALL be built from the repository's single dependency specification. Real providers and the Anthropic backend SHALL be opt-in via environment variables, with their network and billing implications documented.

#### Scenario: Single command, offline
- **WHEN** the documented container command is run on a machine with no network access and no API key
- **THEN** the pipeline completes and writes the analysis bundle

#### Scenario: Environment parity
- **WHEN** the container image is built
- **THEN** its environment derives from the repository's declared dependency specification, not a separately maintained list

#### Scenario: Existing services preserved
- **WHEN** the compose file is extended
- **THEN** the previously documented Redis service still starts and behaves as before, and the default pipeline path does not require it

### Requirement: Pipeline definition has a single source
The command sequence that constitutes the evaluation pipeline SHALL be defined once and shared by the guide and the container entry point, so a flag change cannot leave the documentation stale.

#### Scenario: Flag change surfaces
- **WHEN** a script's argument changes and the shared pipeline definition is not updated
- **THEN** running the pipeline fails visibly rather than silently diverging from the documentation

### Requirement: User-facing architecture documentation
The repository SHALL contain architecture documentation written for a reader of the public artefact, mapping the shipped code onto the frozen specification's named components and describing the request flow and the provider-abstraction pattern; agent-facing internal notes SHALL link to it rather than duplicate it.

#### Scenario: Spec-to-code traceability
- **WHEN** a reviewer reads the architecture document
- **THEN** each component named in the frozen specification is traceable to the module that implements it

### Requirement: Release documentation reflects shipped code
User-facing documentation SHALL describe the released system without intermediate work-in-progress scaffolding — in particular, no issue or ticket identifiers in user-facing section headings — while internal planning artefacts retain their identifiers.

#### Scenario: No ticket scaffolding in user docs
- **WHEN** user-facing documentation is reviewed before release
- **THEN** section headings name capabilities rather than tracker issues, and the content matches the shipped behaviour

### Requirement: Specification and archive consistency
Completed change proposals SHALL be archived and their capability specifications synced into the living spec layer using main-spec structure, and the specification tooling's validation SHALL pass over the whole repository afterwards.

#### Scenario: Validation passes after sync
- **WHEN** completed changes are archived and their capabilities synced
- **THEN** the specification validator reports no failures across all changes and specs

#### Scenario: Only shipped work is archived
- **WHEN** a change's implementation has not shipped
- **THEN** it remains in-flight rather than being archived

### Requirement: Recorded, re-runnable release audit
The repository SHALL provide an executable audit that verifies the licence is present, no secret file is tracked, no secret was introduced anywhere in git history, no personal or sensitive data is present in code or data, and no tracked file contains third-party corpus text; it SHALL exit non-zero on any finding. The corpus-text check SHALL be an enforced gate rather than a documented expectation, and its failure path SHALL be verified.

#### Scenario: Clean repository passes
- **WHEN** the audit runs against the current repository
- **THEN** it reports each check passing and exits zero

#### Scenario: Introduced secret fails the audit
- **WHEN** a secret-shaped value is present in the working tree or history
- **THEN** the audit exits non-zero and identifies the finding

#### Scenario: Committed corpus text fails the audit
- **WHEN** a tracked file contains query text drawn from a third-party corpus
- **THEN** the audit exits non-zero and identifies the offending file

#### Scenario: Synthetic fixtures pass the corpus-text check
- **WHEN** the audit runs with the committed synthetic fixture dataset present
- **THEN** the corpus-text check passes, because synthetic text carries no third-party licence

### Requirement: Examples and superseded documents handled honestly
Runnable examples SHALL be verified as part of release review, any example that incurs real API cost SHALL be documented as opt-in and excluded from automated paths, and superseded or historical documents SHALL be flagged in place — frozen documents SHALL NOT be modified.

#### Scenario: Billed example is fenced off
- **WHEN** the examples are reviewed
- **THEN** the real-API example is documented as billed and opt-in, and no automated path invokes it

#### Scenario: Historical document flagged, frozen documents untouched
- **WHEN** superseded documentation is identified
- **THEN** it carries an in-place status note, and the frozen specification documents remain byte-identical

