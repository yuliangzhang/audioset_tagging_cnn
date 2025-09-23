"""Compare DCASE2021 strong labels against Audioset strong metadata.

This script loads the DCASE2021 strong-label metadata and the Audioset
strong-eval/train metadata, normalises identifiers, and reports overlap
statistics between the datasets.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# DCASE_DEFAULT = Path("/Users/yuliangzhang/Documents/work_dir/dcase2021_metadata/train/audioset_strong.tsv")
DCASE_DEFAULT = Path("/Users/yuliangzhang/Documents/work_dir/dcase2021_metadata/validation/validation.tsv")
AUDIOSET_TRAIN_DEFAULT = Path("/Users/yuliangzhang/Documents/work_dir/audioset_metadata/audioset_train_strong.tsv")
AUDIOSET_EVAL_DEFAULT = Path("/Users/yuliangzhang/Documents/work_dir/audioset_metadata/audioset_eval_strong.tsv")


def filename_to_segment_id(filename: str) -> str:
    """Convert a DCASE filename to an Audioset-style segment_id."""
    stem = filename[:-4] if filename.endswith(".wav") else filename
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    ytid, start_seconds, _ = parts
    start_ms = int(round(float(start_seconds) * 1000))
    return f"{ytid}_{start_ms}"


def load_dcase_segments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["filename"])
    df = df.drop_duplicates().assign(segment_id=lambda d: d["filename"].apply(filename_to_segment_id))
    return df


def load_audioset_segments(train_path: Path, eval_path: Path) -> pd.DataFrame:
    train = pd.read_csv(train_path, sep="\t", usecols=["segment_id"])
    train["audioset_source"] = "train_strong"

    eval_df = pd.read_csv(eval_path, sep="\t", usecols=["segment_id"])
    eval_df["audioset_source"] = "eval_strong"

    audioset = pd.concat([train, eval_df], ignore_index=True)
    audioset = audioset.drop_duplicates(subset=["segment_id", "audioset_source"])  # drop repeated rows

    # Combine sources that share the same segment id (e.g. present in both splits)
    combined = (
        audioset
        .groupby("segment_id", as_index=False)["audioset_source"]
        .agg(lambda s: ";".join(sorted(set(s))))
    )
    combined['segment_id'] = combined['segment_id'].map(lambda x: 'Y' + x)
    return combined


def build_summary(dcase: pd.DataFrame, audioset: pd.DataFrame) -> Tuple[Dict[str, int], pd.DataFrame]:
    train_count = audioset["audioset_source"].str.contains("train_strong").sum()
    eval_count = audioset["audioset_source"].str.contains("eval_strong").sum()

    merge = dcase.merge(audioset, on="segment_id", how="left", indicator=True)
    overlap = (merge["_merge"] == "both").sum()
    only_dcase = (merge["_merge"] == "left_only").sum()

    summary = {
        "dcase_unique_filenames": int(len(dcase)),
        "audioset_train_unique_segments": int(train_count),
        "audioset_eval_unique_segments": int(eval_count),
        "audioset_combined_unique_segments": int(len(audioset)),
        "dcase_segments_found_in_audioset": int(overlap),
        "dcase_segments_missing_from_audioset": int(only_dcase),
    }
    return summary, merge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DCASE strong files against Audioset strong metadata")
    parser.add_argument("--dcase-path", type=Path, default=DCASE_DEFAULT)
    parser.add_argument("--audioset-train", type=Path, default=AUDIOSET_TRAIN_DEFAULT)
    parser.add_argument("--audioset-eval", type=Path, default=AUDIOSET_EVAL_DEFAULT)
    parser.add_argument("--top-missing", type=int, default=10, help="Number of missing segment ids to display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    for path in (args.dcase_path, args.audioset_train, args.audioset_eval):
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")

    dcase_segments = load_dcase_segments(args.dcase_path)
    audioset_segments = load_audioset_segments(args.audioset_train, args.audioset_eval)

    summary, merged = build_summary(dcase_segments, audioset_segments)

    print("\n=== Dataset Sizes ===")
    print(f"DCASE strong unique filenames: {summary['dcase_unique_filenames']}")
    print(f"Audioset train strong unique segments: {summary['audioset_train_unique_segments']}")
    print(f"Audioset eval strong unique segments: {summary['audioset_eval_unique_segments']}")
    print(f"Audioset combined unique segments: {summary['audioset_combined_unique_segments']}")

    print("\n=== Overlap ===")
    if summary["dcase_unique_filenames"]:
        overlap_ratio = summary['dcase_segments_found_in_audioset'] / summary['dcase_unique_filenames']
    else:
        overlap_ratio = 0.0
    print(
        "DCASE segments present in Audioset: "
        f"{summary['dcase_segments_found_in_audioset']} "
        f"({overlap_ratio:.2%})"
    )
    print(
        "DCASE segments missing from Audioset: "
        f"{summary['dcase_segments_missing_from_audioset']} "
        f"({1 - overlap_ratio:.2%})"
    )

    missing = (
        merged.loc[merged["_merge"] == "left_only", "segment_id"]
        .head(args.top_missing)
        .tolist()
    )
    if missing:
        print("\nSample segment_ids missing from Audioset:")
        for seg in missing:
            print(f"  - {seg}")

    # Provide a quick breakdown of DCASE matches by Audioset split
    matched = merged.loc[merged["_merge"] == "both", "audioset_source"]
    if not matched.empty:
        print("\nAudioset split coverage for matched DCASE segments:")
        counts = (
            matched.str.split(";")
            .explode()
            .value_counts()
            .sort_index()
        )
        for source, count in counts.items():
            print(f"  - {source}: {int(count)}")


if __name__ == "__main__":
    main()
