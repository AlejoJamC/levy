#!/usr/bin/env bash
#
# Levy — release audit.
#
# The public artefact is released under Apache 2.0, so "no secrets, no personal
# data, licence present" has to be a re-runnable check rather than a one-time
# eyeball. This script is that check.
#
# Checks:
#   1. LICENSE is present and non-empty.
#   2. No .env (or other secret-bearing file) is tracked by git.
#   3. No secret-shaped string appears in the tracked working tree.
#   4. No secret-shaped string was ever introduced in git history.
#   5. No personal-data markers (emails, phone numbers) in data/.
#   6. No tracked file carries query text attributed to a third-party corpus.
#   7. No tracked file contains a string sampled from a populated data/raw/.
#   8. .env is gitignored.
#
# Prints a pass/fail line per check; exits non-zero if any check fails.
#
# Usage:
#   scripts/audit_release.sh

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FAILURES=0
CHECK_NO=0

pass() { printf '  \033[0;32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[0;31mFAIL\033[0m  %s\n' "$1"; FAILURES=$((FAILURES + 1)); }

heading() {
  CHECK_NO=$((CHECK_NO + 1))
  printf '\n[%d] %s\n' "$CHECK_NO" "$1"
}

# Secret-shaped value patterns. These match plausible real credential VALUES,
# not the names of environment variables or documented placeholders such as
# `sk-...` in the README — otherwise the audit would cry wolf on its own docs.
SECRET_PATTERNS=(
  'sk-ant-api[0-9]{2}-[A-Za-z0-9_-]{20,}'      # Anthropic API key
  'sk-[A-Za-z0-9]{32,}'                        # OpenAI-style key
  'sk-proj-[A-Za-z0-9_-]{20,}'                 # OpenAI project key
  'AKIA[0-9A-Z]{16}'                           # AWS access key id
  'gh[pousr]_[A-Za-z0-9]{36}'                  # GitHub token
  'xox[baprs]-[A-Za-z0-9-]{10,}'               # Slack token
  'AIza[0-9A-Za-z_-]{35}'                      # Google API key
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'         # private key material
  'eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.'  # JWT
)

echo "=============================================================="
echo "Levy release audit"
echo "  repository: ${REPO_ROOT}"
echo "=============================================================="

# ---------------------------------------------------------------- 1. licence
heading "Licence present"
if [ -s LICENSE ]; then
  pass "LICENSE present ($(wc -l < LICENSE | tr -d ' ') lines)"
  if grep -qi "Apache License" LICENSE; then
    pass "LICENSE is the Apache License (as declared in pyproject.toml)"
  else
    fail "LICENSE does not look like the declared Apache 2.0 licence"
  fi
else
  fail "LICENSE is missing or empty"
fi

# ------------------------------------------------- 2. no tracked secret files
heading "No secret-bearing file tracked by git"
TRACKED_SECRET_FILES="$(git ls-files \
  | grep -E '(^|/)(\.env|\.env\..*|.*\.pem|.*\.key|.*\.p12|.*\.pfx|credentials|\.netrc|id_rsa.*)$' \
  | grep -v -E '(^|/)\.env\.example$' || true)"
if [ -z "$TRACKED_SECRET_FILES" ]; then
  pass "no .env / key / credential file is tracked (only .env.example)"
else
  fail "secret-bearing file(s) tracked by git:"
  printf '        %s\n' $TRACKED_SECRET_FILES
fi

# --------------------------------------------------- 3. working-tree secrets
heading "No secret-shaped string in the tracked working tree"
TREE_FINDINGS=0
for pattern in "${SECRET_PATTERNS[@]}"; do
  # Search only tracked files: untracked .env is expected to exist locally and
  # is legitimately excluded from the release.
  HITS="$(git grep -n -I -E "$pattern" -- . ':(exclude)scripts/audit_release.sh' 2>/dev/null || true)"
  if [ -n "$HITS" ]; then
    fail "pattern matched in tracked files: ${pattern}"
    printf '        %s\n' "$HITS"
    TREE_FINDINGS=$((TREE_FINDINGS + 1))
  fi
done
if [ "$TREE_FINDINGS" -eq 0 ]; then
  pass "none of ${#SECRET_PATTERNS[@]} secret patterns matched any tracked file"
fi

# -------------------------------------------------------- 4. git history
heading "No secret ever introduced in git history"
HISTORY_FINDINGS=0
for pattern in "${SECRET_PATTERNS[@]}"; do
  # -S with --pickaxe-regex finds commits that changed the NUMBER of matches,
  # i.e. commits that introduced (or removed) a matching string.
  COMMITS="$(git log --all --pickaxe-regex -S"$pattern" --format='%h %ad %s' --date=short 2>/dev/null || true)"
  if [ -n "$COMMITS" ]; then
    fail "history touched a string matching: ${pattern}"
    printf '        %s\n' "$COMMITS"
    HISTORY_FINDINGS=$((HISTORY_FINDINGS + 1))
  fi
done
if [ "$HISTORY_FINDINGS" -eq 0 ]; then
  pass "no commit in any branch introduced a secret-shaped string"
fi

# ----------------------------------------------------- 5. personal data
heading "No personal-data markers in data/"
PERSONAL_FINDINGS=0

# Real email addresses. The author's own contact address in packaging metadata
# is intentional and lives in pyproject.toml, not in the dataset.
EMAIL_HITS="$(git grep -n -I -E '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}' -- 'data/*' 2>/dev/null \
  | grep -v -E 'example\.(com|org|net)|@workload|@levy' || true)"
if [ -n "$EMAIL_HITS" ]; then
  fail "email-shaped string(s) in data/:"
  printf '        %s\n' "$EMAIL_HITS"
  PERSONAL_FINDINGS=$((PERSONAL_FINDINGS + 1))
fi

# Phone-number-shaped strings (international or grouped forms).
PHONE_HITS="$(git grep -n -I -E '(\+[0-9]{1,3}[ -]?)?\(?[0-9]{3}\)?[ -][0-9]{3}[ -][0-9]{4}' -- 'data/*' 2>/dev/null || true)"
if [ -n "$PHONE_HITS" ]; then
  fail "phone-number-shaped string(s) in data/:"
  printf '        %s\n' "$PHONE_HITS"
  PERSONAL_FINDINGS=$((PERSONAL_FINDINGS + 1))
fi

if [ "$PERSONAL_FINDINGS" -eq 0 ]; then
  pass "no email or phone-number markers in tracked data/ files"
fi

# -------------------------------------------- 6. third-party corpus text
# Two of the three source corpora grant no redistribution right (Quora QQP is
# released under Quora's Terms of Service; SODD is CC BY-NC-SA 4.0), so what
# this repository publishes is data/ground_truth.ids.csv — identifiers and
# labels, no query text. That has to be an enforced gate, not a convention:
# one `git add data/ground_truth.full.csv` would publish licensed text.
#
# The rule is about *attribution*, not about text as such: a tracked file may
# carry query text only if every row attributes it to a synthetic source. The
# 15 synthetic fixture pairs in data/ground_truth.{csv,json} therefore pass —
# fabricated text carries no third-party licence.
heading "No third-party corpus text in tracked files"
CORPUS_TEXT_REPORT="$(python3 - <<'PY' 2>&1
import csv, io, json, subprocess, sys
from pathlib import Path

SYNTHETIC = {"synthetic-fixture", "mock"}
TEXT_FIELDS = ("query_1", "query_2")

try:
    registry = json.loads(Path("data/corpora.json").read_text(encoding="utf-8"))
    known = set(registry.get("corpora", {}))
except (OSError, json.JSONDecodeError) as exc:
    print(f"ERROR cannot read data/corpora.json: {exc}")
    sys.exit(1)

tracked = subprocess.run(
    ["git", "ls-files", "data"], capture_output=True, text=True, check=False
).stdout.split()

findings = []
checked = 0
for name in tracked:
    path = Path(name)
    if path.suffix not in {".csv", ".json"} or not path.is_file():
        continue
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    if path.suffix == ".csv":
        reader = csv.DictReader(io.StringIO(raw))
        if not reader.fieldnames or not any(f in reader.fieldnames for f in TEXT_FIELDS):
            continue
        rows = list(reader)
    else:
        try:
            rows = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
            continue
        if not any(f in rows[0] for f in TEXT_FIELDS):
            continue
    checked += 1
    cited = {str(row.get("source_corpus", "")) for row in rows}
    offending = sorted(c for c in cited if c and c not in SYNTHETIC)
    for corpus in offending:
        note = " (a corpus in data/corpora.json)" if corpus in known else ""
        findings.append(
            f"{path}: carries query text attributed to source_corpus={corpus!r}{note}"
        )

for finding in findings:
    print(f"FINDING {finding}")
print(f"CHECKED {checked} tracked file(s) with query-text columns")
sys.exit(1 if findings else 0)
PY
)"
CORPUS_TEXT_STATUS=$?
if [ "$CORPUS_TEXT_STATUS" -eq 0 ]; then
  pass "no tracked file carries query text attributed to a third-party corpus"
  printf '        %s\n' "$(echo "$CORPUS_TEXT_REPORT" | grep '^CHECKED' || true)"
else
  fail "tracked file(s) carry third-party corpus text:"
  printf '        %s\n' "$CORPUS_TEXT_REPORT"
fi

# ---------------------------------- 7. spot-check against acquired corpora
# The attribution check above trusts the source_corpus column. This one does
# not: it takes real strings out of a populated data/raw/ and looks for them
# in tracked files. Skips cleanly when nothing has been acquired, since a
# clean clone has an empty data/raw/ by design.
heading "Tracked files contain no strings sampled from data/raw/"
SPOTCHECK_REPORT="$(python3 - <<'PY' 2>&1
import subprocess, sys
from pathlib import Path

RAW = Path("data/raw")
IGNORED = {".gitkeep", "README.md"}

raw_files = [
    p for p in sorted(RAW.rglob("*"))
    if p.is_file() and p.name not in IGNORED
]
if not raw_files:
    print("SKIP data/raw/ is empty — nothing acquired to spot-check against")
    sys.exit(2)

# Distinctive needles: long, multi-word fields from the text-readable corpora.
# Binary shards (parquet) are skipped; the attribution check covers those.
needles = []
for path in raw_files:
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    for line in content.splitlines()[:5000]:
        for field in line.split("\t"):
            field = field.strip()
            if len(field) >= 40 and " " in field:
                needles.append(field)
needles = needles[:500]
if not needles:
    print("SKIP no text-readable corpus files under data/raw/ to sample from")
    sys.exit(2)

tracked = subprocess.run(
    ["git", "ls-files"], capture_output=True, text=True, check=False
).stdout.split()

findings = []
for name in tracked:
    path = Path(name)
    if not path.is_file():
        continue
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        continue
    hits = [n for n in needles if n in raw]
    if hits:
        findings.append(f"{path}: contains {len(hits)} string(s) from data/raw/, e.g. {hits[0][:60]!r}")

for finding in findings:
    print(f"FINDING {finding}")
print(f"CHECKED {len(needles)} sampled string(s) against {len(tracked)} tracked file(s)")
sys.exit(1 if findings else 0)
PY
)"
SPOTCHECK_STATUS=$?
if [ "$SPOTCHECK_STATUS" -eq 2 ]; then
  pass "spot-check skipped: $(echo "$SPOTCHECK_REPORT" | sed 's/^SKIP //')"
elif [ "$SPOTCHECK_STATUS" -eq 0 ]; then
  pass "no tracked file contains a string sampled from the acquired corpora"
  printf '        %s\n' "$(echo "$SPOTCHECK_REPORT" | grep '^CHECKED' || true)"
else
  fail "tracked file(s) contain text from data/raw/:"
  printf '        %s\n' "$SPOTCHECK_REPORT"
fi

# ------------------------------------------------------- 8. gitignore
heading ".env is gitignored"
if git check-ignore -q .env 2>/dev/null; then
  pass ".env is ignored by git"
elif grep -qE '^\.env$' .gitignore 2>/dev/null; then
  # check-ignore needs the file to exist; fall back to reading the rule.
  pass ".env rule present in .gitignore"
else
  fail ".env is not gitignored — a local key could be committed by accident"
fi

# ------------------------------------------------------------------ summary
echo
echo "=============================================================="
if [ "$FAILURES" -eq 0 ]; then
  echo "RESULT: release audit PASSED (${CHECK_NO} checks)"
  echo "=============================================================="
  exit 0
else
  echo "RESULT: release audit FAILED — ${FAILURES} finding(s)"
  echo "=============================================================="
  exit 1
fi
