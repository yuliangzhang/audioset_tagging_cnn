from __future__ import annotations

"""
Collect DCASE validation query embeddings from eval, bal_train, and unbal TFRecords.

Produces artifacts/dcase_val_query_vectors.npz with arrays:
- segment_id: list of found query segment_ids
- embeddings: float32 [Q_found, 128], L2-normalized

Note: expected to be < 1168 if some queries are absent (~35 missing typical).
"""

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from .tfrecord_reader import iter_tfrecord_records, parse_sequence_example
from .build_audioset_index import seq_to_vector, filename_to_segment_id, l2_normalize


def dcase_filename_to_segment_id(filename: str) -> str:
    stem = filename[:-4] if filename.endswith('.wav') else filename
    parts = stem.rsplit('_', 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected DCASE filename: {filename}")
    ytid, start_sec, _ = parts
    if ytid.startswith('Y') and len(ytid) > 11:
        ytid = ytid[1:]
    start_ms = int(round(float(start_sec) * 1000))
    return f"{ytid}_{start_ms}"


def load_index_npz(index_npz: Path) -> Tuple[List[str], np.ndarray]:
    data = np.load(index_npz, allow_pickle=True)
    seg = [str(s) for s in data['segment_id']]
    emb = data['embeddings'].astype(np.float32)
    return seg, emb


def collect_from_unbal(missing: Set[str], tfrecord_dir: Path, max_files: int | None = None) -> Tuple[List[str], List[np.ndarray]]:
    found_sids: List[str] = []
    found_vecs: List[np.ndarray] = []
    files = sorted(tfrecord_dir.glob('*.tfrecord'))
    if max_files is not None:
        files = files[: max_files]
    for f in files:
        for rec in iter_tfrecord_records(str(f)):
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
            if not isinstance(video_id, str) or start_time is None:
                continue
            sid = filename_to_segment_id(video_id, float(start_time))
            if sid in missing:
                try:
                    vec = seq_to_vector(feat_lists['audio_embedding'])
                except Exception:
                    continue
                found_sids.append(sid)
                found_vecs.append(vec)
                missing.remove(sid)
                if not missing:
                    return found_sids, found_vecs
    return found_sids, found_vecs


def main():
    parser = argparse.ArgumentParser(description='Collect DCASE validation query vectors from eval/bal_train/unbal')
    parser.add_argument('--dcase-validation', type=Path, default=Path('metadata/dcase2021_metadata/validation/validation.tsv'))
    parser.add_argument('--eval-index', type=Path, default=Path('artifacts/audio_eval_index.npz'))
    parser.add_argument('--bal-index', type=Path, default=Path('artifacts/audio_bal_train_index.npz'))
    parser.add_argument('--unbal-tfrecord-dir', type=Path, default=Path('audioset_features/unbal_train'))
    parser.add_argument('--out', type=Path, default=Path('artifacts/dcase_val_query_vectors.npz'))
    parser.add_argument('--max-files', type=int, default=None, help='Limit unbal TFRecords to scan (pilot)')
    args = parser.parse_args()

    # Load DCASE validation SIDs
    dcase = pd.read_csv(args.dcase_validation, sep='\t')
    dcase_sids = sorted(set(dcase['filename'].apply(dcase_filename_to_segment_id).tolist()))

    # Load eval/bal indexes
    base_seg: List[str] = []
    base_emb: List[np.ndarray] = []
    if args.eval_index.is_file():
        seg, emb = load_index_npz(args.eval_index)
        base_seg.extend(seg)
        base_emb.append(emb)
    if args.bal_index.is_file():
        seg, emb = load_index_npz(args.bal_index)
        base_seg.extend(seg)
        base_emb.append(emb)
    base_emb_mat = np.vstack(base_emb) if base_emb else np.zeros((0, 128), dtype=np.float32)
    base_sid_set = set(base_seg)

    # Found from eval/bal
    found_sid: List[str] = []
    found_vec: List[np.ndarray] = []
    sid_to_idx = {sid: i for i, sid in enumerate(base_seg)}
    for sid in dcase_sids:
        if sid in sid_to_idx:
            found_sid.append(sid)
            found_vec.append(base_emb_mat[sid_to_idx[sid]])

    missing = set(dcase_sids) - set(found_sid)
    if missing:
        add_sid, add_vec = collect_from_unbal(missing, args.unbal_tfrecord_dir, args.max_files)
        found_sid.extend(add_sid)
        found_vec.extend(add_vec)

    if not found_vec:
        raise RuntimeError('No query vectors collected from eval/bal/unbal')

    emb = np.vstack(found_vec).astype(np.float32)
    emb = l2_normalize(emb)
    seg = np.array(found_sid, dtype=object)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, embeddings=emb, segment_id=seg)
    print(f'Collected {len(found_sid)} query vectors (out of {len(dcase_sids)}). Wrote {args.out}')


if __name__ == '__main__':
    main()

