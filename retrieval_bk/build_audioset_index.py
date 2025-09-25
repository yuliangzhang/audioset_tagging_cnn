from __future__ import annotations

"""
Build an embedding index from AudioSet TFRecord frame-level features.

Pilot scope: only processes files under audioset_features/bal_train.

For each SequenceExample (10s segment), it:
- extracts context fields: video_id, start_time_seconds, end_time_seconds, labels
- stacks per-second 128-dim uint8 embeddings from feature_list.audio_embedding
- converts to float32 in [0, 1] by /255.0, mean-pools across time, L2-normalizes
- saves arrays and metadata for retrieval

Outputs:
- artifacts/audioset_bal_train_index.npz  (embeddings [N,128], segment_ids [N])
- artifacts/audioset_bal_train_meta.csv   (segment_id,video_id,start_ms,end_ms,labels,labels_name,split)
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from .tfrecord_reader import iter_tfrecord_records, parse_sequence_example
# from retrieval.tfrecord_reader import iter_tfrecord_records, parse_sequence_example


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def seq_to_vector(audio_embeddings: List[bytes]) -> np.ndarray:
    if not audio_embeddings:
        raise ValueError("Empty audio_embedding sequence")
    arrs = []
    for b in audio_embeddings:
        x = np.frombuffer(b, dtype=np.uint8)
        if x.size != 128:
            # Unexpected; skip this frame
            continue
        arrs.append(x)
    if not arrs:
        raise ValueError("No valid 128-dim frames in audio_embedding")
    # Dequantize uint8 [0,255] back to approx [-2.0, 2.0] as in VGGish postprocessing
    # See quantization in VGGish: values in [-2, 2] mapped to 8-bit
    mat = np.vstack(arrs).astype(np.float32)
    mat = (mat / 255.0) * 4.0 - 2.0  # [T,128] in approx [-2, 2]
    vec = mat.mean(axis=0)
    vec = l2_normalize(vec)
    return vec.astype(np.float32)


def filename_to_segment_id(ytid: str, start_seconds: float) -> str:
    start_ms = int(round(float(start_seconds) * 1000.0))
    return f"{ytid}_{start_ms}"


def process_tfrecord(path: Path, split: str) -> Tuple[List[np.ndarray], List[str], List[str], List[int], List[int], List[List[int]], List[str]]:
    """Read a TFRecord file and extract vectors + metadata.

    Returns lists of: vectors, segment_ids, video_ids, start_ms, end_ms, labels, splits
    """
    vectors: List[np.ndarray] = []
    segment_ids: List[str] = []
    video_ids: List[str] = []
    start_mss: List[int] = []
    end_mss: List[int] = []
    labels_all: List[List[int]] = []
    splits: List[str] = []

    for rec in iter_tfrecord_records(str(path)):
        try:
            parsed = parse_sequence_example(rec)
        except Exception:
            # Skip malformed records
            continue
        ctx = parsed.get('context', {})
        feat_lists = parsed.get('feature_lists', {})
        if 'audio_embedding' not in feat_lists:
            continue
        video_id = ctx.get('video_id')
        start_time = ctx.get('start_time_seconds')
        end_time = ctx.get('end_time_seconds')
        labels = ctx.get('labels', [])
        if not isinstance(video_id, str) or start_time is None:
            # Need at least video_id and start
            continue
        try:
            vec = seq_to_vector(feat_lists['audio_embedding'])
        except Exception:
            continue
        segid = filename_to_segment_id(video_id, float(start_time))
        vectors.append(vec)
        segment_ids.append(segid)
        video_ids.append(video_id)
        start_mss.append(int(round(float(start_time) * 1000.0)))
        end_mss.append(int(round(float(end_time) * 1000.0)) if end_time is not None else start_mss[-1] + 10000)
        labels_all.append([int(x) for x in labels])
        splits.append(split)

    return vectors, segment_ids, video_ids, start_mss, end_mss, labels_all, splits


def main():
    parser = argparse.ArgumentParser(description="Build AudioSet bal_train index from TFRecords")
    parser.add_argument("--tfrecord-dir", type=Path, default=Path("audioset_features/bal_train"))
    parser.add_argument("--pattern", type=str, default="*.tfrecord")
    parser.add_argument("--out-index", type=Path, default=Path("artifacts/audioset_bal_train_index.npz"))
    parser.add_argument("--out-meta", type=Path, default=Path("artifacts/audioset_bal_train_meta.csv"))
    parser.add_argument("--class-index", type=Path, default=Path("metadata/audioset_metadata/class_labels_indices.csv"))
    # parser.add_argument("--max-files", type=int, default=None, help="Limit number of TFRecord files (pilot)")
    parser.add_argument("--max-files", type=int, default=3, help="Limit number of TFRecord files (pilot)")
    args = parser.parse_args()

    tf_dir: Path = args.tfrecord_dir
    if not tf_dir.is_dir():
        raise FileNotFoundError(f"TFRecord dir not found: {tf_dir}")

    files = sorted(tf_dir.glob(args.pattern))
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No TFRecord files matching {args.pattern} in {tf_dir}")

    all_vecs: List[np.ndarray] = []
    all_segids: List[str] = []
    all_vids: List[str] = []
    all_starts: List[int] = []
    all_ends: List[int] = []
    all_labels: List[List[int]] = []
    all_splits: List[str] = []

    split_name = tf_dir.name
    for i, f in enumerate(files, 1):
        vecs, segids, vids, starts, ends, labels, splits = process_tfrecord(f, split_name)
        if vecs:
            all_vecs.extend(vecs)
            all_segids.extend(segids)
            all_vids.extend(vids)
            all_starts.extend(starts)
            all_ends.extend(ends)
            all_labels.extend(labels)
            all_splits.extend(splits)

    if not all_vecs:
        raise RuntimeError("No vectors extracted from TFRecords")

    os.makedirs(args.out_index.parent, exist_ok=True)
    os.makedirs(args.out_meta.parent, exist_ok=True)

    emb = np.vstack(all_vecs).astype(np.float32)
    seg = np.array(all_segids, dtype=object)
    np.savez_compressed(args.out_index, embeddings=emb, segment_id=seg)

    # Save metadata CSV
    def labels_to_str(lst: List[int]) -> str:
        return "|".join(str(x) for x in lst)

    # Map label indices to display names
    try:
        class_df = pd.read_csv(args.class_index)
        idx_to_name = {int(r['index']): str(r['display_name']) for _, r in class_df.iterrows()}
    except Exception:
        idx_to_name = {}

    def labels_to_names(lst: List[int]) -> str:
        return "|".join(idx_to_name.get(int(x), str(x)) for x in lst)

    meta_df = pd.DataFrame({
        'segment_id': all_segids,
        'video_id': all_vids,
        'start_ms': all_starts,
        'end_ms': all_ends,
        'labels': [labels_to_str(x) for x in all_labels],
        'labels_name': [labels_to_names(x) for x in all_labels],
        'split': all_splits,
    })
    meta_df.to_csv(args.out_meta, index=False)

    print(f"Embeddings: {emb.shape}, segments: {len(all_segids)}")
    print(f"Wrote index: {args.out_index}")
    print(f"Wrote meta : {args.out_meta}")


if __name__ == "__main__":
    main()
