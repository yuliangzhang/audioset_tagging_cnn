from __future__ import annotations

"""
Build 2-second query window embeddings (256-D) from DCASE validation strong
labels using frame-level AudioSet TFRecord features.

Rules:
- Dequantize per-second 128-D uint8 embeddings to ~[-2, 2].
- A 2s window vector is the concatenation of two consecutive seconds (256-D),
  L2-normalized.
- For each validation event (filename, onset, offset, event_label):
  - if offset - onset < 1.0s:
      center c=(onset+offset)/2, s=round(c) clamped into [0, 9]
      produce up to two windows starting at s-1 and s (clamped to [0, 8])
  - else (duration >= 1.0s):
      candidate starts S in [floor(onset)-1, ceil(offset)-1] ∩ [0, 8]
      pick up to two distinct starts from S using a fixed RNG seed

Outputs:
- artifacts2/dcase_val_query_windows.npz with arrays:
  embeddings [Q, 256], query_id, query_segment_id, query_label, onset, offset,
  variant, query_window_start
- artifacts2/dcase_val_query_windows.csv (metadata mirror)

Pilot-friendly options:
- --limit-events N to cap number of validation events processed
- --max-unbal-files to limit unbal scanning when needed
"""

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from retrieval.tfrecord_reader import iter_tfrecord_records, parse_sequence_example


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def dequantize_embed(u8: np.ndarray) -> np.ndarray:
    # uint8 [0, 255] -> approx [-2, 2]
    return (u8.astype(np.float32) / 255.0) * 4.0 - 2.0


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


def filename_to_segment_id(ytid: str, start_seconds: float) -> str:
    start_ms = int(round(float(start_seconds) * 1000.0))
    return f"{ytid}_{start_ms}"


def load_needed_segments(val_tsv: Path, limit_events: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(val_tsv, sep='\t')
    if limit_events is not None:
        df = df.iloc[: int(limit_events)].copy()
    df['segment_id'] = df['filename'].apply(dcase_filename_to_segment_id)
    return df


def collect_segment_frames(
    needed_sids: List[str],
    eval_dir: Path,
    bal_dir: Path,
    unbal_dir: Optional[Path] = None,
    max_unbal_files: Optional[int] = None,
) -> Dict[str, Tuple[np.ndarray, List[int], str]]:
    """Return mapping segment_id -> (frames[T,128] float32, labels[List[int]], split).

    Iterates eval and bal_train TFRecords; optionally scans a limited number of
    unbal TFRecords for any missing segment_ids.
    """
    needed = set(needed_sids)
    out: Dict[str, Tuple[np.ndarray, List[int], str]] = {}

    def scan_dir(tf_dir: Path, split: str, max_files: Optional[int] = None):
        files = sorted(tf_dir.glob('*.tfrecord'))
        if max_files is not None:
            files = files[: int(max_files)]
        for f in files:
            for rec in iter_tfrecord_records(str(f)):
                try:
                    parsed = parse_sequence_example(rec)
                except Exception:
                    continue
                ctx = parsed.get('context', {})
                feats = parsed.get('feature_lists', {})
                if 'audio_embedding' not in feats:
                    continue
                vid = ctx.get('video_id')
                st = ctx.get('start_time_seconds')
                labels = [int(x) for x in ctx.get('labels', [])]
                if not isinstance(vid, str) or st is None:
                    continue
                sid = filename_to_segment_id(vid, float(st))
                if sid not in needed:
                    continue
                frames: List[np.ndarray] = []
                for b in feats['audio_embedding']:
                    arr = np.frombuffer(b, dtype=np.uint8)
                    if arr.size == 128:
                        frames.append(arr)
                if not frames:
                    continue
                mat = dequantize_embed(np.vstack(frames))  # [T,128]
                out[sid] = (mat.astype(np.float32), labels, split)
                if sid in needed:
                    needed.remove(sid)
                if not needed:
                    return

    if eval_dir.is_dir():
        scan_dir(eval_dir, 'eval')
    if bal_dir.is_dir() and needed:
        scan_dir(bal_dir, 'bal_train')
    if unbal_dir is not None and unbal_dir.is_dir() and needed:
        scan_dir(unbal_dir, 'unbal_train', max_files=max_unbal_files)
    return out


def make_window(frames: np.ndarray, start: int) -> Optional[np.ndarray]:
    # frames [T,128], need [start, start+1]
    if start < 0 or start + 1 >= frames.shape[0]:
        return None
    v = np.concatenate([frames[start], frames[start + 1]], axis=0).astype(np.float32)
    return l2_normalize(v)


def build_queries(df: pd.DataFrame, sid_to_frames: Dict[str, Tuple[np.ndarray, List[int], str]], seed: int = 1337) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    rng = np.random.default_rng(seed)
    embeds: List[np.ndarray] = []
    meta: List[Dict[str, object]] = []

    for i, row in df.iterrows():
        sid = row['segment_id']
        try:
            onset = float(row['onset'])
            offset = float(row['offset'])
        except Exception:
            continue
        if math.isnan(onset) or math.isnan(offset):
            continue
        label = str(row['event_label'])
        pack = sid_to_frames.get(sid)
        if pack is None:
            continue
        frames, _, _ = pack
        T = frames.shape[0]
        # ensure we have at least 2 frames
        if T < 2:
            continue
        if offset - onset < 1.0:
            c = 0.5 * (onset + offset)
            s = int(round(c))
            s = max(0, min(9, s))
            starts = []
            if s - 1 >= 0:
                starts.append(s - 1)
            if s <= 8:
                starts.append(s)
            used = 0
            for j, st in enumerate(starts):
                w = make_window(frames, st)
                if w is None:
                    continue
                embeds.append(w)
                meta.append({
                    'query_id': f'{sid}:{label}:{onset:.3f}-{offset:.3f}:short:{j}',
                    'query_segment_id': sid,
                    'query_label': label,
                    'onset': onset,
                    'offset': offset,
                    'variant': 'short_center_prev' if j == 0 else 'short_center_next',
                    'query_window_start': st,
                })
                used += 1
            if used == 0:
                continue
        else:
            lo = int(math.floor(onset) - 1)
            hi = int(math.ceil(offset) - 1)
            cands = [s for s in range(lo, hi + 1) if 0 <= s <= 8]
            if not cands:
                continue
            picks = list(cands)
            rng.shuffle(picks)
            picks = picks[:2]
            for j, st in enumerate(picks):
                w = make_window(frames, st)
                if w is None:
                    continue
                embeds.append(w)
                meta.append({
                    'query_id': f'{sid}:{label}:{onset:.3f}-{offset:.3f}:long:{j}',
                    'query_segment_id': sid,
                    'query_label': label,
                    'onset': onset,
                    'offset': offset,
                    'variant': f'long_random_{j}',
                    'query_window_start': st,
                })
    if not embeds:
        return np.zeros((0, 256), dtype=np.float32), []
    E = np.vstack(embeds).astype(np.float32)
    return E, meta


def main():
    parser = argparse.ArgumentParser(description='Build 2s query windows from DCASE validation')
    parser.add_argument('--dcase-validation', type=Path, default=Path('metadata/dcase2021_metadata/validation/validation.tsv'))
    parser.add_argument('--eval-dir', type=Path, default=Path('audioset_features/eval'))
    parser.add_argument('--bal-dir', type=Path, default=Path('audioset_features/bal_train'))
    parser.add_argument('--unbal-dir', type=Path, default=Path('audioset_features/unbal_train'))
    parser.add_argument('--out-npz', type=Path, default=Path('artifacts2/dcase_val_query_windows.npz'))
    parser.add_argument('--out-csv', type=Path, default=Path('artifacts2/dcase_val_query_windows.csv'))
    parser.add_argument('--limit-events', type=int, default=None)
    parser.add_argument('--max-unbal-files', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()

    df = load_needed_segments(args.dcase_validation, args.limit_events)
    needed = sorted(set(df['segment_id']))

    sid2frames = collect_segment_frames(
        needed, args.eval_dir, args.bal_dir, args.unbal_dir, args.max_unbal_files
    )

    E, meta = build_queries(df, sid2frames, seed=args.seed)
    if E.size == 0:
        print('No query windows built')
        return

    # Save artifacts
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        embeddings=E,
        query_id=np.array([m['query_id'] for m in meta], dtype=object),
        query_segment_id=np.array([m['query_segment_id'] for m in meta], dtype=object),
        query_label=np.array([m['query_label'] for m in meta], dtype=object),
        onset=np.array([m['onset'] for m in meta], dtype=np.float32),
        offset=np.array([m['offset'] for m in meta], dtype=np.float32),
        variant=np.array([m['variant'] for m in meta], dtype=object),
        query_window_start=np.array([m['query_window_start'] for m in meta], dtype=np.int32),
    )

    # CSV mirror
    meta_df = pd.DataFrame(meta)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(args.out_csv, index=False)
    print(f'Built {E.shape[0]} query windows. Saved: {args.out_npz} and {args.out_csv}')


if __name__ == '__main__':
    main()
