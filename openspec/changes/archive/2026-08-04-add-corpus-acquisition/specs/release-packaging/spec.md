## MODIFIED Requirements

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
