"""Materialize a checked participant-level split from the raw session inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data_splits" / "non_miguel_zr_validation.csv"
DEFAULT_RAW_DIR = ROOT.parents[1] / "all_sessions"
DEFAULT_LABEL_DIR = ROOT / "data_all_mixed" / "train_data"
DEFAULT_OUTPUT_DIR = ROOT / "data_non_miguel_zr_validation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))

    required = {"split", "user", "action", "file"}
    if not rows or set(rows[0]) != required:
        raise ValueError(f"{path} must contain exactly {sorted(required)}")

    users: dict[str, set[str]] = defaultdict(set)
    files: set[str] = set()
    for row in rows:
        split, user, file_name = row["split"], row["user"], row["file"]
        if split not in {"train", "valid"}:
            raise ValueError(f"{file_name}: invalid split {split!r}")
        if user == "Miguel":
            raise ValueError("Miguel must not be included in this external-user split")
        if not file_name.endswith(".csv") or Path(file_name).name != file_name:
            raise ValueError(f"Invalid recording name: {file_name!r}")
        if file_name in files:
            raise ValueError(f"Duplicate recording: {file_name}")
        files.add(file_name)
        users[user].add(split)

    overlap = sorted(user for user, splits in users.items() if len(splits) != 1)
    if overlap:
        raise ValueError(f"Participants occur in multiple splits: {overlap}")
    if users.get("ZR") != {"valid"}:
        raise ValueError("All ZR recordings must be in validation")
    return rows


def link(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source, destination.parent))


def materialize(
    rows: list[dict[str, str]], raw_dir: Path, label_dir: Path, output_dir: Path
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to modify existing output directory: {output_dir}")

    for row in rows:
        name = row["file"]
        raw_csv = raw_dir / name
        labeled_csv = label_dir / name
        label = label_dir / Path(name).with_suffix(".txt")
        if not raw_csv.is_file() or not labeled_csv.is_file() or not label.is_file():
            raise FileNotFoundError(f"Missing raw CSV or label sidecar for {name}")
        if sha256(raw_csv) != sha256(labeled_csv):
            raise ValueError(f"Raw and labeled CSV differ for {name}")

    for row in rows:
        target = output_dir / f"{row['split']}_data"
        target.mkdir(parents=True, exist_ok=True)
        name = row["file"]
        link(raw_dir / name, target / name)
        label_name = Path(name).with_suffix(".txt").name
        link(label_dir / label_name, target / label_name)
        detailed_label = label_dir / Path(name).with_suffix(".labels.txt")
        if detailed_label.is_file():
            link(detailed_label, target / detailed_label.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    counts = Counter(row["split"] for row in rows)
    users = {split: sorted({row["user"] for row in rows if row["split"] == split}) for split in counts}
    print(f"train: {counts['train']} recordings from {', '.join(users['train'])}")
    print(f"valid: {counts['valid']} recordings from {', '.join(users['valid'])}")
    if args.dry_run:
        return
    materialize(rows, args.raw_dir, args.label_dir, args.output_dir)
    print(f"Created {args.output_dir}")


if __name__ == "__main__":
    main()
