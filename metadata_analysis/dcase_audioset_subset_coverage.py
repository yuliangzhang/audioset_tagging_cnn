"""Quantify how DCASE2021 clips overlap with Audioset segment lists.

Loads the DCASE strong, weak, unlabelled, and validation metadata,
normalises their filenames to Audioset-style identifiers, and compares
them against the balanced/unbalanced train and eval Audioset segment lists to report
coverage statistics.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

DCASE_BASE = Path("/Users/yuliangzhang/Documents/work_dir/dcase2021_metadata/train")
AUDIOSET_BASE = Path("/Users/yuliangzhang/Documents/work_dir/audioset_metadata")

DCASE_DEFAULTS = {
    "strong": DCASE_BASE / "audioset_strong.tsv",
    "weak": DCASE_BASE / "weak.tsv",
    "unlabel_in_domain": DCASE_BASE / "unlabel_in_domain.tsv",
    "validation": Path("/Users/yuliangzhang/Documents/work_dir/dcase2021_metadata/validation/validation.tsv"),
}

AUDIOSET_DEFAULTS = {
    "balanced_train_segments": AUDIOSET_BASE / "balanced_train_segments.csv",
    "unbalanced_train_segments": AUDIOSET_BASE / "unbalanced_train_segments.csv",
    "eval_segments": AUDIOSET_BASE / "eval_segments.csv",
}


def format_seconds(value) -> str:
    """Return Audioset time stamp with millisecond precision."""
    dec = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return format(dec, ".3f")


def dcase_filename_to_key(filename: str) -> str:
    base = filename[:-4] if filename.endswith(".wav") else filename
    if not base.startswith("Y"):
        raise ValueError(f"Unexpected filename format: {filename}")
    return base


def load_dcase_metadata(path: Path, column: str = "filename") -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=[column])
    unique = df[column].drop_duplicates().reset_index(drop=True)
    result = pd.DataFrame({
        "filename": unique,
        "segment_key": unique.apply(dcase_filename_to_key),
    })
    return result


def load_audioset_segments(path: Path, source_name: str) -> pd.DataFrame:
    cols = ["ytid", "start_seconds", "end_seconds", "positive_labels"]
    df = pd.read_csv(
        path,
        comment="#",
        header=None,
        names=cols,
        skipinitialspace=True,
    )
    if df.empty:
        return pd.DataFrame(columns=["segment_key", "source"])

    df["segment_key"] = (
        "Y"
        + df["ytid"].astype(str)
        + "_"
        + df["start_seconds"].apply(format_seconds)
        + "_"
        + df["end_seconds"].apply(format_seconds)
    )
    df = df.drop_duplicates(subset=["segment_key"])
    df["source"] = source_name
    return df[["segment_key", "source"]]


def combine_audioset_sources(paths: Dict[str, Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for source_name, path in paths.items():
        frames.append(load_audioset_segments(path, source_name))
    combined = pd.concat(frames, ignore_index=True)
    grouped = (
        combined
        .groupby("segment_key", as_index=False)["source"]
        .agg(lambda s: ";".join(sorted(set(s))))
    )
    return grouped


def compute_coverage(dcase: pd.DataFrame, audioset_grouped: pd.DataFrame, sources: Iterable[str]) -> Dict[str, object]:
    merged = dcase.merge(audioset_grouped, on="segment_key", how="left")
    merged["source"] = merged["source"].fillna("not_found")

    merged["source_list"] = merged["source"].apply(lambda value: [] if value == "not_found" else value.split(";"))

    total = len(merged)
    combo_counts = merged["source"].value_counts()
    coverage = {
        "total_files": int(total),
        "found_in_any_audioset": int(merged["source_list"].apply(bool).sum()),
        "source_combo_counts": list(combo_counts.items()),
    }

    for source_name in sources:
        count = merged["source_list"].apply(lambda lst, name=source_name: name in lst).sum()
        coverage[source_name] = int(count)
    return coverage


def present_results(results: Dict[str, Dict[str, object]], sources: Iterable[str]) -> None:
    for dataset_name, stats in results.items():
        total = stats["total_files"]
        print(f"\n=== DCASE {dataset_name} ===")
        print(f"Total unique filenames: {total}")

        found_any = stats["found_in_any_audioset"]
        ratio_any = found_any / total if total else 0.0
        print(f"Matched to any Audioset subset: {found_any} ({ratio_any:.2%})")

        for source_name in sources:
            count = stats[source_name]
            ratio = count / total if total else 0.0
            print(f"  - {source_name}: {count} ({ratio:.2%})")

        combos = stats["source_combo_counts"]
        print("Combination breakdown (top 5):")
        for combo, count in combos[:5]:
            ratio = count / total if total else 0.0
            print(f"  * {combo}: {count} ({ratio:.2%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DCASE datasets against Audioset segment lists")
    parser.add_argument("--dcase-strong", type=Path, default=DCASE_DEFAULTS["strong"])
    parser.add_argument("--dcase-weak", type=Path, default=DCASE_DEFAULTS["weak"])
    parser.add_argument("--dcase-unlabel", type=Path, default=DCASE_DEFAULTS["unlabel_in_domain"])
    parser.add_argument("--dcase-validation", type=Path, default=DCASE_DEFAULTS["validation"])
    parser.add_argument("--audioset-balanced", type=Path, default=AUDIOSET_DEFAULTS["balanced_train_segments"])
    parser.add_argument("--audioset-unbalanced", type=Path, default=AUDIOSET_DEFAULTS["unbalanced_train_segments"])
    parser.add_argument("--audioset-eval", type=Path, default=AUDIOSET_DEFAULTS["eval_segments"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_paths = [
        args.dcase_strong,
        args.dcase_weak,
        args.dcase_unlabel,
        args.dcase_validation,
        args.audioset_balanced,
        args.audioset_unbalanced,
        args.audioset_eval,
    ]
    for path in all_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")

    dcase_datasets = {
        "strong": load_dcase_metadata(args.dcase_strong, column="filename"),
        "weak": load_dcase_metadata(args.dcase_weak, column="filename"),
        "unlabel_in_domain": load_dcase_metadata(args.dcase_unlabel, column="filename"),
        "validation": load_dcase_metadata(args.dcase_validation, column="filename"),
    }

    audioset_sources = combine_audioset_sources({
        "balanced_train_segments": args.audioset_balanced,
        "unbalanced_train_segments": args.audioset_unbalanced,
        "eval_segments": args.audioset_eval,
    })

    source_names: List[str] = [
        "balanced_train_segments",
        "unbalanced_train_segments",
        "eval_segments",
    ]

    coverage_results = {
        name: compute_coverage(df, audioset_sources, source_names)
        for name, df in dcase_datasets.items()
    }

    present_results(coverage_results, source_names)


if __name__ == "__main__":
    main()
