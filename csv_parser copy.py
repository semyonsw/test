#!/usr/bin/env python3
"""Clean every source CSV in a folder (semicolon-delimited, quoted fields)
into comma-separated columns with no " symbols. Each `NAME.csv` becomes
`NAME_clean.csv` in the same folder."""

import csv
import sys
from pathlib import Path

FOLDER = (
    sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\User\Downloads\Telegram Desktop"
)


def clean_one(src: Path) -> int:
    dst = src.with_name(src.stem + "_clean.csv")
    with open(src, newline="", encoding="utf-8-sig") as f_in:
        reader = csv.reader(f_in, delimiter=";", quotechar='"')
        rows = [[cell.strip().strip('"') for cell in row] for row in reader]
    with open(dst, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(
            f_out, delimiter=",", quoting=csv.QUOTE_NONE, escapechar="\\"
        )
        writer.writerows(rows)
    print(f"  {src.name} -> {dst.name} ({len(rows)} rows)")
    return len(rows)


def main():
    folder = Path(FOLDER)
    # Skip files we already produced, so re-runs don't make _clean_clean.csv
    sources = [p for p in folder.glob("*.csv") if not p.stem.endswith("_clean")]
    if not sources:
        print(f"No source CSVs found in {folder}")
        return
    print(f"Processing {len(sources)} file(s) in {folder}:")
    for src in sorted(sources):
        clean_one(src)


if __name__ == "__main__":
    main()
