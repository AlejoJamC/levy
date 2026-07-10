#!/usr/bin/env python
"""
Convert a ground-truth dataset between CSV and JSON (LEV-3 / D2).

Loads `--in` (dispatching on its extension) and writes `--out` (dispatching
on its extension), validating the schema on the way through `levy.dataset.io`.
Useful for regenerating a sibling format after editing one side, or for
verifying the CSV<->JSON round-trip contract that LEV-4 depends on.

Example:
    python scripts/export_dataset.py --in data/ground_truth.json --out /tmp/roundtrip.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from levy.dataset.io import load_dataset, save_csv, save_json


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_path", type=Path, required=True, help="Input dataset (.csv or .json)")
    parser.add_argument("--out", dest="out_path", type=Path, required=True, help="Output dataset (.csv or .json)")
    return parser


def main(argv=None, output_fn=print) -> int:
    args = build_arg_parser().parse_args(argv)
    pairs = load_dataset(args.in_path)

    suffix = args.out_path.suffix.lower()
    if suffix == ".csv":
        save_csv(pairs, args.out_path)
    elif suffix == ".json":
        save_json(pairs, args.out_path)
    else:
        output_fn(f"[export_dataset] unrecognized output extension {suffix!r}; expected .csv or .json")
        return 1

    output_fn(f"[export_dataset] converted {len(pairs)} pairs: {args.in_path} -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
