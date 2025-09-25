from __future__ import annotations

"""
Stream retrieval of 2s windows (256-D vectors) over eval+bal_train (+optional unbal),
with in-domain label filtering and Top-K per query (allowing multiple windows
from the same audio).

Steps per TFRecord record:
- parse frames (T x 128) uint8, dequantize to ~[-2, 2]
- build all windows (2s, stride 1): W (<=9 x 256), L2-normalized
- collect candidate queries using label-based inverted index
- for each candidate query vector q: compute sims = W @ q; take max, record
  best start_second and sim; push into per-query TopK (K up to 300)

Outputs:
- results2/dcase_val_window_neighbors_topK300_min07.csv
- coverage_overall_windows.csv, coverage_per_class_windows.csv
- windows_top1_similarity_hist.png, windows_top1_similarity_per_class.png, windows_top1_similarity_stats.csv
"""

import argparse
import math
import heapq
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from retrieval.tfrecord_reader import iter_tfrecord_records, parse_sequence_example


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def dequantize_embed(u8: np.ndarray) -> np.ndarray:
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


def default_dcase_to_audioset_mids() -> Dict[str, List[str]]:
    return {
        'Alarm_bell_ringing': ['/m/07pp8cl', '/m/03wwcy', '/m/046dlr', '/m/07pp_mv'],
        'Blender': ['/m/02pjr4'],
        'Cat': ['/m/01yrx'],
        'Dishes': ['/m/04brg2'],
        'Dog': ['/m/0bt9lr'],
        'Electric_shaver_toothbrush': ['/m/02g901', '/m/04fgwm'],
        'Frying': ['/m/0dxrf'],
        'Running_water': ['/m/02jz0l'],
        'Speech': ['/m/09x0r'],
        'Vacuum_cleaner': ['/m/0d31p'],
    }


def load_mid_index_name(class_index_csv: Path) -> Tuple[Dict[str, int], Dict[int, str], Dict[int, str]]:
    df = pd.read_csv(class_index_csv)
    mid_to_idx = {row['mid']: int(row['index']) for _, row in df.iterrows()}
    idx_to_mid = {int(row['index']): row['mid'] for _, row in df.iterrows()}
    idx_to_name = {int(row['index']): row['display_name'] for _, row in df.iterrows()}
    return mid_to_idx, idx_to_mid, idx_to_name


class TopKHeap:
    def __init__(self, k: int):
        self.k = k
        # key by (segment_id, start_second) to allow multiple windows per audio
        self.heap: List[Tuple[float, Tuple[str, str, int, List[int], List[str], str]]] = []
        self.seen: set[Tuple[str, int]] = set()

    def push(self, sim: float, seg_id: str, vid: str, start_second: int, label_idx: List[int], label_names: List[str], split: str):
        key = (seg_id, int(start_second))
        if key in self.seen:
            return
        item = (sim, (seg_id, vid, int(start_second), label_idx, label_names, split))
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
            self.seen.add(key)
        else:
            if sim > self.heap[0][0]:
                heapq.heapreplace(self.heap, item)
                self.seen.add(key)

    def items_sorted(self) -> List[Tuple[float, Tuple[str, str, int, List[int], List[str], str]]]:
        return sorted(self.heap, key=lambda x: -x[0])


def load_queries(npz_path: Path) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    d = np.load(npz_path, allow_pickle=True)
    E = d['embeddings'].astype(np.float32)
    meta = []
    n = E.shape[0]
    for i in range(n):
        meta.append({
            'query_id': str(d['query_id'][i]),
            'query_segment_id': str(d['query_segment_id'][i]),
            'query_label': str(d['query_label'][i]),
            'onset': float(d['onset'][i]),
            'offset': float(d['offset'][i]),
            'query_window_start': int(d['query_window_start'][i]),
        })
    return E, meta


def main():
    parser = argparse.ArgumentParser(description='2s window retrieval over TFRecords')
    parser.add_argument('--queries-npz', type=Path, default=Path('artifacts2/dcase_val_query_windows.npz'))
    parser.add_argument('--class-index', type=Path, default=Path('metadata/audioset_metadata/class_labels_indices.csv'))
    parser.add_argument('--eval-dir', type=Path, default=Path('audioset_features/eval'))
    parser.add_argument('--bal-dir', type=Path, default=Path('audioset_features/bal_train'))
    parser.add_argument('--unbal-dir', type=Path, default=Path('audioset_features/unbal_train'))
    parser.add_argument('--label-filter', type=str, choices=['none', 'mapped'], default='mapped')
    parser.add_argument('--top-k', type=int, default=300)
    parser.add_argument('--min-sim', type=float, default=0.7)
    parser.add_argument('--sweep-thresholds', type=str, default='0.6,0.7,0.8,0.9')
    parser.add_argument('--out', type=Path, default=Path('results2/dcase_val_window_neighbors_topK300_min07.csv'))
    parser.add_argument('--plots-dir', type=Path, default=Path('results2'))
    parser.add_argument('--class-thresholds', type=Path, default=None, help='CSV with columns: class,min_sim to override thresholds per class')
    parser.add_argument('--summary-out', type=Path, default=None, help='Path to write top-1 summary CSV')
    parser.add_argument('--per-class-out-dir', type=Path, default=None, help='Write per-class neighbor CSVs into this directory')
    parser.add_argument('--max-files-eval', type=int, default=None)
    parser.add_argument('--max-files-bal', type=int, default=None)
    parser.add_argument('--max-files-unbal', type=int, default=None)
    parser.add_argument('--limit-queries', type=int, default=None, help='Limit number of queries (None=all)')
    args = parser.parse_args()

    Q, qmeta = load_queries(args.queries_npz)
    if args.limit_queries is not None and Q.shape[0] > int(args.limit_queries):
        Q = Q[: int(args.limit_queries)]
        qmeta = qmeta[: int(args.limit_queries)]
    Q = l2_normalize(Q)
    n_query = Q.shape[0]

    # Build per-query allowed label index sets
    mid_to_idx, idx_to_mid, idx_to_name = load_mid_index_name(args.class_index)
    mapping = default_dcase_to_audioset_mids()
    q_allowed: List[set[int]] = []
    for qm in qmeta:
        lab = str(qm['query_label'])
        allowed = set()
        for mid in mapping.get(lab, []):
            if mid in mid_to_idx:
                allowed.add(int(mid_to_idx[mid]))
        q_allowed.append(allowed)

    # Load class-specific thresholds (optional)
    class_thr: Dict[str, float] = {}
    if args.class_thresholds and Path(args.class_thresholds).is_file():
        df_thr = pd.read_csv(args.class_thresholds)
        for _, r in df_thr.iterrows():
            try:
                cls = str(r['class'])
                thr = float(r['min_sim'])
                class_thr[cls] = thr
            except Exception:
                continue

    # Build label->queries inverted index
    label_to_queries: Dict[int, List[int]] = {}
    for qi, allowed in enumerate(q_allowed):
        for li in allowed:
            label_to_queries.setdefault(li, []).append(qi)

    # Per-query TopK heaps
    heaps = [TopKHeap(args.top_k) for _ in range(n_query)]
    query_max_sim = [float('-inf')] * n_query

    def process_dir(tf_dir: Path, split: str, max_files: Optional[int]):
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
                segid = f"{vid}_{int(round(float(st)*1000.0))}"
                # Exclude self segment for any matching queries
                label_queries: set[int] = set()
                if args.label_filter == 'mapped' and labels:
                    for li in labels:
                        label_queries.update(label_to_queries.get(int(li), []))
                    if not label_queries:
                        continue
                else:
                    # if no filter, all queries are candidates
                    label_queries = set(range(n_query))

                # Build window matrix once per segment
                frames: List[np.ndarray] = []
                for b in feats['audio_embedding']:
                    arr = np.frombuffer(b, dtype=np.uint8)
                    if arr.size == 128:
                        frames.append(arr)
                if len(frames) < 2:
                    continue
                F = dequantize_embed(np.vstack(frames))  # [T,128]
                starts = list(range(0, max(0, F.shape[0]-1)))
                if not starts:
                    continue
                W = np.stack([l2_normalize(np.concatenate([F[s], F[s+1]], axis=0)) for s in starts], axis=0)
                # row_labels (for output)
                label_names = [idx_to_name.get(int(li), str(li)) for li in labels]

                for qi in label_queries:
                    # Exclude self-segment entirely
                    if qmeta[qi]['query_segment_id'] == segid:
                        continue
                    qv = Q[qi]
                    sims = W @ qv
                    if sims.size == 0:
                        continue
                    best_idx = int(np.argmax(sims))
                    best_sim = float(sims[best_idx])
                    # Determine threshold for this query label
                    base_thr = float(args.min_sim) if args.min_sim is not None else 0.0
                    lbl = str(qmeta[qi]['query_label'])
                    eff_thr = max(base_thr, class_thr.get(lbl, -float('inf')))
                    if best_sim < eff_thr:
                        continue
                    query_max_sim[qi] = max(query_max_sim[qi], best_sim)
                    start_second = starts[best_idx]
                    heaps[qi].push(best_sim, segid, vid, start_second, labels, label_names, split)

    if args.eval_dir.is_dir():
        process_dir(args.eval_dir, 'eval', args.max_files_eval)
    if args.bal_dir.is_dir():
        process_dir(args.bal_dir, 'bal_train', args.max_files_bal)
    # For pilot, you can skip unbal by setting max_files_unbal=0 or leaving dir absent
    if args.unbal_dir.is_dir() and (args.max_files_unbal is None or args.max_files_unbal > 0):
        process_dir(args.unbal_dir, 'unbal_train', args.max_files_unbal)

    # Emit results
    rows: List[Dict[str, object]] = []
    for qi in range(n_query):
        qm = qmeta[qi]
        items = heaps[qi].items_sorted()
        for rank, (sim, (c_sid, c_vid, c_start, c_lab_idx, c_lab_names, split)) in enumerate(items, start=1):
            rows.append({
                'query_id': qm['query_id'],
                'query_segment_id': qm['query_segment_id'],
                'query_label': qm['query_label'],
                'candidate_segment_id': c_sid,
                'candidate_video_id': c_vid,
                'candidate_start_second': int(c_start),
                'similarity': float(sim),
                'rank': rank,
                'candidate_label_indices': '|'.join(str(x) for x in c_lab_idx),
                'candidate_label_names': '|'.join(c_lab_names),
                'source_split': split,
            })

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved window neighbors: {args.out} with {len(out_df)} rows (queries={n_query})")

    # Optional per-class CSVs
    if args.per_class_out_dir is not None:
        out_dir = Path(args.per_class_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cls, sub in out_df.groupby('query_label'):
            safe = ''.join(ch if ch.isalnum() or ch in ('_', '-') else '_' for ch in str(cls))
            sub.to_csv(out_dir / f'neighbors_{safe}.csv', index=False)
        print(f"Wrote per-class neighbors to: {out_dir}")

    # Optional top-1 summary per query window
    if args.summary_out is not None:
        if not out_df.empty:
            # Pick top row per query_id by similarity
            idx = out_df.groupby('query_id')['similarity'].idxmax()
            top1 = out_df.loc[idx].sort_values('query_id')
            Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
            top1.to_csv(args.summary_out, index=False)
            print(f"Wrote top-1 summary to: {args.summary_out}")

    # Coverage sweeps
    sweeps = None
    if args.sweep_thresholds:
        try:
            sweeps = [float(x) for x in args.sweep_thresholds.split(',') if x.strip()]
        except Exception:
            sweeps = None
    if sweeps:
        present = [v for v in query_max_sim if not math.isinf(v) and v != float('-inf')]
        # Overall
        overall = []
        denom = len([v for v in query_max_sim if not math.isinf(v) and v != float('-inf')])
        for thr in sweeps:
            covered = sum(1 for v in query_max_sim if (not math.isinf(v) and v != float('-inf') and v >= thr))
            overall.append({'threshold': thr, 'n_queries': denom, 'covered': covered, 'coverage_ratio': covered / denom if denom else 0.0})
        (args.out.parent / 'coverage_overall_windows.csv').write_text(pd.DataFrame(overall).to_csv(index=False))
        # Per-class
        per_cls = []
        # Build query lists per class
        cls_to_qi: Dict[str, List[int]] = {}
        for qi, qm in enumerate(qmeta):
            cls_to_qi.setdefault(str(qm['query_label']), []).append(qi)
        for thr in sweeps:
            for cls, qlist in cls_to_qi.items():
                vals = [query_max_sim[qi] for qi in qlist if not math.isinf(query_max_sim[qi]) and query_max_sim[qi] != float('-inf')]
                covered = sum(1 for v in vals if v >= thr)
                per_cls.append({'threshold': thr, 'class': cls, 'n_queries': len(vals), 'covered': covered, 'coverage_ratio': covered / len(vals) if vals else 0.0})
        (args.out.parent / 'coverage_per_class_windows.csv').write_text(pd.DataFrame(per_cls).to_csv(index=False))
        print(f"Saved coverage CSVs to {args.out.parent}")

    # Plots
    vals = [v for v in query_max_sim if not math.isinf(v) and v != float('-inf')]
    if vals:
        plt.figure(figsize=(6,4))
        plt.hist(vals, bins=50, color='#4C78A8', alpha=0.9)
        plt.title('2s-window Top-1 similarity (pilot)')
        plt.xlabel('cosine similarity')
        plt.ylabel('count')
        plt.tight_layout()
        fig1 = args.plots_dir / 'windows_top1_similarity_hist.png'
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig1, dpi=150)
        print(f"Saved plot: {fig1}")

        # Per-class grid
        classes = sorted(set(str(qm['query_label']) for qm in qmeta))
        cols = 5
        rows = (len(classes) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5), squeeze=False)
        for idx, cls in enumerate(classes):
            r, c = divmod(idx, cols)
            qlist = [qi for qi, qm in enumerate(qmeta) if str(qm['query_label']) == cls]
            v = [query_max_sim[qi] for qi in qlist if not math.isinf(query_max_sim[qi]) and query_max_sim[qi] != float('-inf')]
            ax = axes[r][c]
            if v:
                ax.hist(v, bins=30, color='#F58518', alpha=0.9)
            ax.set_title(cls, fontsize=9)
            ax.set_xlim(0.0, 1.0)
        for idx in range(len(classes), rows*cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis('off')
        plt.tight_layout()
        fig2 = args.plots_dir / 'windows_top1_similarity_per_class.png'
        plt.savefig(fig2, dpi=150)
        print(f"Saved plot: {fig2}")

        stats = {
            'count': len(vals),
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'p90': float(np.quantile(vals, 0.9)),
            'p95': float(np.quantile(vals, 0.95)),
            'p99': float(np.quantile(vals, 0.99)),
            'min': float(np.min(vals)),
            'max': float(np.max(vals)),
        }
        (args.out.parent / 'windows_top1_similarity_stats.csv').write_text(pd.DataFrame([stats]).to_csv(index=False))
        print(f"Saved stats CSV: {args.out.parent / 'windows_top1_similarity_stats.csv'}")


if __name__ == '__main__':
    main()
