from __future__ import annotations

"""
Shard builder for AudioSet unbalanced_train TFRecords.

Splits the TFRecord file list into N shards and builds one index per shard.
Each index contains mean-pooled, dequantized, L2-normalized 128-D embeddings
and a metadata CSV with labels and label names.

Usage examples:
  python -m retrieval.build_unbal_shards --tfrecord-dir audioset_features/unbal_train \
      --num-shards 40 --shard-index 0

Options also include limiting files for a dry run via --max-files.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .tfrecord_reader import iter_tfrecord_records, parse_sequence_example
from .build_audioset_index import seq_to_vector, filename_to_segment_id


def process_tfrecord(path: Path, split: str) -> Tuple[List[np.ndarray], List[str], List[str], List[int], List[int], List[List[int]], List[str]]:
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
    parser = argparse.ArgumentParser(description="Build shard index from unbalanced_train TFRecords")
    parser.add_argument('--tfrecord-dir', type=Path, default=Path('audioset_features/unbal_train'))
    parser.add_argument('--num-shards', type=int, default=40)
    parser.add_argument('--shard-index', type=int, required=True, help='Which shard to build [0..num_shards-1]')
    parser.add_argument('--pattern', type=str, default='*.tfrecord')
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/unbal_train_shards'))
    parser.add_argument('--class-index', type=Path, default=Path('metadata/audioset_metadata/class_labels_indices.csv'))
    parser.add_argument('--max-files', type=int, default=None)
    args = parser.parse_args()

    tf_dir: Path = args.tfrecord_dir
    if not tf_dir.is_dir():
        raise FileNotFoundError(f"TFRecord dir not found: {tf_dir}")

    files = sorted(tf_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No TFRecord files in {tf_dir}")

    # Round-robin sharding by file index for reproducibility
    target_files = [f for i, f in enumerate(files) if (i % args.num_shards) == args.shard_index]
    if args.max_files is not None:
        target_files = target_files[: args.max_files]
    if not target_files:
        raise RuntimeError(f"No files for shard {args.shard_index} with {args.num_shards} shards")

    out_shard_dir = args.out_dir / f'shard_{args.shard_index:02d}'
    out_shard_dir.mkdir(parents=True, exist_ok=True)
    out_index = out_shard_dir / 'index.npz'
    out_meta = out_shard_dir / 'meta.csv'

    all_vecs: List[np.ndarray] = []
    all_segids: List[str] = []
    all_vids: List[str] = []
    all_starts: List[int] = []
    all_ends: List[int] = []
    all_labels: List[List[int]] = []
    all_splits: List[str] = []

    split_name = 'unbal_train'
    for f in target_files:
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
        raise RuntimeError('No vectors extracted in this shard')

    emb = np.vstack(all_vecs).astype(np.float32)
    seg = np.array(all_segids, dtype=object)
    np.savez_compressed(out_index, embeddings=emb, segment_id=seg)

    # Build label name mapping
    try:
        class_df = pd.read_csv(args.class_index)
        idx_to_name = {int(r['index']): str(r['display_name']) for _, r in class_df.iterrows()}
    except Exception:
        idx_to_name = {}

    def labels_to_str(lst: List[int]) -> str:
        return "|".join(str(x) for x in lst)

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
    meta_df.to_csv(out_meta, index=False)

    print(f"Shard {args.shard_index}/{args.num_shards} -> vecs {emb.shape}, segs {len(all_segids)}")
    print(f"Wrote: {out_index} , {out_meta}")


if __name__ == '__main__':
    main()

