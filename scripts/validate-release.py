#!/usr/bin/env python3
"""Validate a release tag against a Cargo manifest and changelog."""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--changelog", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--notes", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with args.manifest.open("rb") as manifest_file:
        package = tomllib.load(manifest_file)["package"]

    version = package["version"]
    expected_tag = f"{args.prefix}{version}"
    if args.tag != expected_tag:
        sys.exit(
            f"tag {args.tag!r} does not match {package['name']} version "
            f"{version!r}; expected {expected_tag!r}"
        )

    changelog = args.changelog.read_text(encoding="utf-8")
    heading = re.compile(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$",
        re.MULTILINE,
    )
    match = heading.search(changelog)
    if match is None:
        sys.exit(
            f"{args.changelog} has no dated release section for version {version}"
        )

    section_end = re.search(
        r"^(?:## \[|\[[^\]]+\]:)", changelog[match.end() :], re.MULTILINE
    )
    end = (
        match.end() + section_end.start()
        if section_end is not None
        else len(changelog)
    )
    notes = changelog[match.start() : end].strip() + "\n"
    if args.notes is not None:
        args.notes.write_text(notes, encoding="utf-8")

    print(f"validated {package['name']} {version} from {args.tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
