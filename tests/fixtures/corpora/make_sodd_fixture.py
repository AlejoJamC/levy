#!/usr/bin/env python
"""
Regenerate the committed SODD parquet fixtures.

`SODDSource` reads gzipped parquet, so its fixture has to be a real parquet
file rather than something a test can write inline — which means a binary blob
in the tree. This script is how that blob is produced, so it is reviewable and
reproducible rather than opaque.

The content is entirely synthetic ("fixture-" prefixed): it carries the real
SODD column structure and label domain, and none of its licensed text.

    python tests/fixtures/corpora/make_sodd_fixture.py
"""

from pathlib import Path

import pandas as pd

OUT_DIR = Path(__file__).resolve().parent / "sodd"

# SODD's real columns. `first_author` / `second_author` are usernames; the
# adapter never reads them, but the fixture carries them so the column-subset
# read is genuinely exercised.
COLUMNS = ["first_post", "second_post", "first_author", "second_author", "label", "page"]

# SODD label classes: 0=duplicates, 1=similar (fulltext), 2=similar (tags),
# 3=different, 4=accepted answer.
_DUPLICATE_TOPICS = [
    ("reverse a fixture-list", "invert a fixture-list in place"),
    ("parse a fixture-date string", "convert a fixture-date string to a date"),
    ("catch a fixture-timeout error", "handle the fixture-timeout exception"),
    ("split a fixture-string on commas", "break a fixture-string at each comma"),
    ("read a fixture-config file", "load settings from a fixture-config file"),
    ("sort a fixture-map by value", "order a fixture-map on its values"),
    ("join two fixture-frames", "merge a pair of fixture-frames"),
    ("mock a fixture-client in tests", "stub out the fixture-client for testing"),
    ("retry a failed fixture-request", "re-issue a fixture-request after failure"),
    ("stream a large fixture-file", "read a large fixture-file in chunks"),
]
_UNRELATED = [
    "configure the fixture-linter",
    "publish a fixture-package",
    "profile fixture-memory usage",
    "rotate the fixture-logfile",
    "pin a fixture-dependency version",
]


def _post(title: str, snippet: str) -> str:
    """A SODD-shaped HTML post: prose plus a code block."""
    return (
        f"<p>How do I {title}?</p>\n"
        f"<pre><code>{snippet}</code></pre>\n"
        f"<p>Any pointer appreciated &mdash; thanks!</p>"
    )


def build_rows():
    rows = []
    for index, (first, second) in enumerate(_DUPLICATE_TOPICS):
        rows.append(
            {
                "first_post": _post(first, f"fixture_call_{index}(a, b)"),
                "second_post": _post(second, f"fixture_call_{index}(x, y)"),
                "first_author": f"fixture-user-{index:02d}",
                "second_author": f"fixture-user-{index + 50:02d}",
                "label": 0,  # duplicates
                "page": "stackoverflow",
            }
        )
    for index, (first, _) in enumerate(_DUPLICATE_TOPICS):
        rows.append(
            {
                "first_post": _post(first, f"fixture_call_{index}(a, b)"),
                "second_post": _post(_UNRELATED[index % len(_UNRELATED)], "fixture_other()"),
                "first_author": f"fixture-user-{index:02d}",
                "second_author": f"fixture-user-{index + 70:02d}",
                "label": 3,  # different
                "page": "stackoverflow",
            }
        )
    # Two hard negatives per "similar" class, excluded from the negative pool
    # unless the adapter's hard-negative option is on.
    for offset, label in ((0, 1), (1, 2)):
        first, second = _DUPLICATE_TOPICS[offset]
        rows.append(
            {
                "first_post": _post(first, f"fixture_similar_{label}(a)"),
                "second_post": _post(second + " but for fixture-batches", f"fixture_similar_{label}(b)"),
                "first_author": f"fixture-user-9{offset}",
                "second_author": f"fixture-user-8{offset}",
                "label": label,
                "page": "stackoverflow",
            }
        )
    # Class 4 (accepted answer) is in the declared domain but is neither a
    # positive nor a negative: the adapter must skip it, not coerce it.
    rows.append(
        {
            "first_post": _post("accept a fixture-answer", "fixture_accept()"),
            "second_post": "<p>The fixture-answer body, not a question.</p>",
            "first_author": "fixture-user-95",
            "second_author": "fixture-user-85",
            "label": 4,
            "page": "stackoverflow",
        }
    )
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = build_rows()

    train = pd.DataFrame(rows, columns=COLUMNS)
    train.to_parquet(OUT_DIR / "SODD_train.parquet.gzip", compression="gzip", index=False)

    # A second shard, so multi-shard reading is exercised.
    dev = pd.DataFrame(
        [
            {
                "first_post": _post("tail a fixture-log", "fixture_tail()"),
                "second_post": _post("follow a fixture-log as it grows", "fixture_follow()"),
                "first_author": "fixture-user-31",
                "second_author": "fixture-user-32",
                "label": 0,
                "page": "stackoverflow",
            },
            {
                "first_post": _post("tail a fixture-log", "fixture_tail()"),
                "second_post": _post("rotate the fixture-logfile", "fixture_rotate()"),
                "first_author": "fixture-user-31",
                "second_author": "fixture-user-33",
                "label": 3,
                "page": "stackoverflow",
            },
        ],
        columns=COLUMNS,
    )
    dev.to_parquet(OUT_DIR / "SODD_dev.parquet.gzip", compression="gzip", index=False)

    print(f"wrote {len(train)} train rows and {len(dev)} dev rows to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
