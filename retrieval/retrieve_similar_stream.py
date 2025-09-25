from __future__ import annotations

"""
Streamed retrieval across unbalanced shards + base pools.

Loads DCASE validation queries (segment_id -> labels), loads query vectors
(artifacts/dcase_val_query_vectors.npz), initializes per-query Top-K heaps
optionally from an existing CSV (eval+bal train results), and then iterates
over unbal_train shards (artifacts/unbal_train_shards/shard_xx) to update the
heaps using in-domain label filtering. Writes a final merged CSV and optional
coverage sweeps.
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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


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


def build_allowed_label_indices(dcase_labels: List[str], mid_to_idx: Dict[str, int], mapping: Dict[str, List[str]]) -> List[int]:
    allowed = set()
    for lab in dcase_labels:
        for mid in mapping.get(lab, []):
            if mid in mid_to_idx:
                allowed.add(mid_to_idx[mid])
    return sorted(allowed)


class TopKHeap:
    def __init__(self, k: int):
        self.k = k
        self.heap: List[Tuple[float, Tuple[str, str, List[int], List[str], str]]] = []
        self.seen: set[str] = set()

    def push(self, sim: float, seg_id: str, vid: str, label_idx: List[int], label_names: List[str], split: str):
        if seg_id in self.seen:
            return
        item = (sim, (seg_id, vid, label_idx, label_names, split))
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, item)
            self.seen.add(seg_id)
        else:
            if sim > self.heap[0][0]:
                popped = heapq.heapreplace(self.heap, item)
                self.seen.add(seg_id)
            # else discard

    def items_sorted(self) -> List[Tuple[float, Tuple[str, str, List[int], List[str], str]]]:
        return sorted(self.heap, key=lambda x: -x[0])


def load_init_csv(path: Optional[Path]) -> Dict[str, TopKHeap]:
    heaps: Dict[str, TopKHeap] = {}
    if not path or not path.is_file():
        return heaps
    df = pd.read_csv(path)
    for qsid, sub in df.groupby('query_segment_id'):
        h = TopKHeap(k=sub['rank'].max())
        for _, r in sub.iterrows():
            c_sid = str(r['candidate_segment_id'])
            c_vid = str(r['candidate_video_id'])
            sim = float(r['similarity'])
            idxs = [int(x) for x in str(r['candidate_label_indices']).split('|') if x != '' and x != 'nan']
            names = [str(x) for x in str(r['candidate_label_names']).split('|') if x != '' and x != 'nan']
            split = str(r.get('source_split', ''))
            h.push(sim, c_sid, c_vid, idxs, names, split)
        heaps[qsid] = h
    return heaps


def main():
    parser = argparse.ArgumentParser(description='Stream retrieval across unbal shards + base results')
    parser.add_argument('--dcase-validation', type=Path, default=Path('metadata/dcase2021_metadata/validation/validation.tsv'))
    parser.add_argument('--queries-npz', type=Path, default=Path('artifacts/dcase_val_query_vectors.npz'))
    parser.add_argument('--init-csv', type=Path, default=Path('results/dcase_val_neighbors_topK_eval_bal_train_dedup.csv'))
    parser.add_argument('--unbal-shards-dir', type=Path, default=Path('artifacts/unbal_train_shards'))
    parser.add_argument('--class-index', type=Path, default=Path('metadata/audioset_metadata/class_labels_indices.csv'))
    parser.add_argument('--label-filter', type=str, choices=['none', 'mapped'], default='mapped')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--min-sim', type=float, default=None)
    parser.add_argument('--sweep-thresholds', type=str, default=None)
    parser.add_argument('--out', type=Path, default=Path('results/dcase_val_neighbors_topK_eval_bal_train_unbal.csv'))
    args = parser.parse_args()

    # Load DCASE validation labels
    dcase_df = pd.read_csv(args.dcase_validation, sep='\t')
    seg2labels: Dict[str, List[str]] = {}
    for _, row in dcase_df.iterrows():
        sid = dcase_filename_to_segment_id(str(row['filename']))
        lab = str(row['event_label'])
        seg2labels.setdefault(sid, [])
        if lab not in seg2labels[sid]:
            seg2labels[sid].append(lab)

    # Load query vectors
    q_data = np.load(args.queries_npz, allow_pickle=True)
    q_seg = [str(s) for s in q_data['segment_id']]
    q_emb = q_data['embeddings'].astype(np.float32)
    q_emb = l2_normalize(q_emb)

    # Build mapping of queries we have vectors for, and allowed labels per query
    mid_to_idx, idx_to_mid, idx_to_name = load_mid_index_name(args.class_index)
    dcase_to_mids = default_dcase_to_audioset_mids()
    q_allowed: Dict[str, List[int]] = {}
    for sid in q_seg:
        labels = seg2labels.get(sid, [])
        q_allowed[sid] = build_allowed_label_indices(labels, mid_to_idx, dcase_to_mids)

    # Initialize per-query heaps from init CSV if exists
    heaps = load_init_csv(args.init_csv)
    for sid in q_seg:
        heaps.setdefault(sid, TopKHeap(args.top_k))

    # For threshold coverage: seed with existing init CSV maxima if any
    query_max_sim: Dict[str, float] = {}
    for sid in q_seg:
        items = heaps[sid].items_sorted()
        if items:
            # items are sorted desc by sim
            query_max_sim[sid] = float(items[0][0])
        else:
            query_max_sim[sid] = float('-inf')

    # Iterate over shards
    shard_dirs = sorted([p for p in args.unbal_shards_dir.glob('shard_*') if p.is_dir()])
    for sdir in shard_dirs:
        idxp = sdir / 'index.npz'
        metap = sdir / 'meta.csv'
        if not (idxp.is_file() and metap.is_file()):
            continue
        data = np.load(idxp, allow_pickle=True)
        emb = data['embeddings'].astype(np.float32)
        seg_ids = [str(s) for s in data['segment_id']]
        meta_df = pd.read_csv(metap)
        # Pre-parse labels per row
        row_labels: List[List[int]] = []
        row_label_names: List[List[str]] = []
        for _, r in meta_df.iterrows():
            lab = str(r['labels']) if not pd.isna(r['labels']) else ''
            idxs = [int(x) for x in lab.split('|') if x != '']
            row_labels.append(idxs)
            names = str(r.get('labels_name', ''))
            row_label_names.append([x for x in names.split('|') if x != ''])

        # Build inverted index: label_idx -> row indices
        inv: Dict[int, List[int]] = {}
        for i, idxs in enumerate(row_labels):
            for li in idxs:
                inv.setdefault(li, []).append(i)

        # For each query, collect candidate rows using its allowed labels
        sid_to_rowidx = {sid: i for i, sid in enumerate(seg_ids)}
        for qi, qsid in enumerate(q_seg):
            allowed = q_allowed.get(qsid, [])
            # Label filtering
            if args.label_filter == 'mapped' and allowed:
                cand_set = set()
                for li in allowed:
                    for ri in inv.get(li, []):
                        cand_set.add(ri)
                if not cand_set:
                    continue
                cand_idx = np.fromiter(cand_set, dtype=np.int64)
            else:
                cand_idx = np.arange(len(seg_ids))

            # Exclude self segment id
            # If this query segment exists in the shard, remove it from candidates
            self_row = sid_to_rowidx.get(qsid)
            if self_row is not None:
                cand_idx = cand_idx[cand_idx != self_row]
            if cand_idx.size == 0:
                continue

            qv = q_emb[qi]
            sims = emb[cand_idx] @ qv
            if sims.size == 0:
                continue

            # Get top few from this shard (min(top_k, len(cand_idx)))
            top_k = min(heaps[qsid].k, sims.size)
            t_idx = np.argpartition(-sims, kth=top_k - 1)[:top_k]
            order = np.argsort(-sims[t_idx])
            t_sel = cand_idx[t_idx][order]
            t_sims = sims[t_idx][order]

            # Record max similarity seen
            if t_sims.size:
                query_max_sim[qsid] = max(query_max_sim[qsid], float(t_sims[0]))

            # Push into heap
            for ri, sim in zip(t_sel, t_sims):
                c_sid = seg_ids[int(ri)]
                c_vid = str(meta_df.iloc[int(ri)]['video_id'])
                c_lab_idx = row_labels[int(ri)]
                c_lab_names = row_label_names[int(ri)]
                # Apply min similarity threshold if given (for heap insertion)
                if args.min_sim is not None and sim < float(args.min_sim):
                    continue
                heaps[qsid].push(float(sim), c_sid, c_vid, c_lab_idx, c_lab_names, 'unbal_train')

    # Emit merged CSV
    rows: List[Dict[str, object]] = []
    for qsid in q_seg:
        labs = seg2labels.get(qsid, [])
        q_labels_str = '|'.join(sorted(set(labs)))
        items = heaps[qsid].items_sorted()
        for rank, (sim, (c_sid, c_vid, c_lab_idx, c_lab_names, split)) in enumerate(items, start=1):
            rows.append({
                'query_segment_id': qsid,
                'query_labels': q_labels_str,
                'candidate_segment_id': c_sid,
                'candidate_video_id': c_vid,
                'similarity': float(sim),
                'rank': rank,
                'candidate_label_indices': '|'.join(str(x) for x in c_lab_idx),
                'candidate_label_names': '|'.join(c_lab_names),
                'source_split': split if split else 'unbal_train',
            })

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved merged neighbors: {args.out} with {len(out_df)} rows")

    # Coverage sweep
    if args.sweep_thresholds:
        try:
            sweeps = [float(x) for x in args.sweep_thresholds.split(',') if x.strip() != '']
        except Exception:
            sweeps = None
        if sweeps:
            all_q = list(q_seg)
            present_q = [q for q in all_q if q in query_max_sim and not math.isinf(query_max_sim[q])]
            overall = []
            for thr in sweeps:
                covered = sum(1 for q in present_q if query_max_sim.get(q, float('-inf')) >= thr)
                overall.append({'threshold': thr, 'n_queries': len(present_q), 'covered': covered, 'coverage_ratio': covered / len(present_q) if present_q else 0.0})
            (args.out.parent / 'coverage_overall_unbal.csv').write_text(pd.DataFrame(overall).to_csv(index=False))
            print(f"Saved overall coverage sweep: {args.out.parent / 'coverage_overall_unbal.csv'}")

            # Per-class coverage
            per_class_rows = []
            cls_to_queries: Dict[str, List[str]] = {}
            for q in all_q:
                labs = seg2labels.get(q, [])
                for lab in labs:
                    cls_to_queries.setdefault(lab, []).append(q)
            for thr in sweeps:
                for lab, qlist in cls_to_queries.items():
                    q_present = [q for q in qlist if q in query_max_sim and not math.isinf(query_max_sim[q])]
                    covered = sum(1 for q in q_present if query_max_sim.get(q, float('-inf')) >= thr)
                    ratio = covered / len(q_present) if q_present else 0.0
                    per_class_rows.append({'threshold': thr, 'class': lab, 'n_queries': len(q_present), 'covered': covered, 'coverage_ratio': ratio})
            (args.out.parent / 'coverage_per_class_unbal.csv').write_text(pd.DataFrame(per_class_rows).to_csv(index=False))
            print(f"Saved per-class coverage sweep: {args.out.parent / 'coverage_per_class_unbal.csv'}")

    # Similarity distribution plots
    # Top-1 similarity across queries
    top1_vals = [query_max_sim.get(q, float('-inf')) for q in q_seg]
    top1_vals = [v for v in top1_vals if not math.isinf(v)]
    if top1_vals:
        plt.figure(figsize=(6,4))
        plt.hist(top1_vals, bins=50, color='#4C78A8', alpha=0.9)
        plt.title('Top-1 similarity across queries (eval+bal+unbal)')
        plt.xlabel('cosine similarity')
        plt.ylabel('count')
        plt.tight_layout()
        fig_path = args.out.parent / 'unbal_top1_similarity_hist.png'
        plt.savefig(fig_path, dpi=150)
        print(f"Saved plot: {fig_path}")

        # Per-class top-1 hist (multi-panel)
        classes = sorted(set(lab for labs in seg2labels.values() for lab in labs))
        n = len(classes)
        if n:
            cols = 5
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*2.5), squeeze=False)
            for idx, lab in enumerate(classes):
                r, c = divmod(idx, cols)
                qlist = [q for q,labs in seg2labels.items() if lab in labs]
                vals = [query_max_sim.get(q, float('-inf')) for q in qlist]
                vals = [v for v in vals if not math.isinf(v)]
                ax = axes[r][c]
                if vals:
                    ax.hist(vals, bins=30, color='#F58518', alpha=0.9)
                ax.set_title(lab, fontsize=9)
                ax.set_xlim(0.0, 1.0)
            # Turn off empty subplots
            for idx in range(n, rows*cols):
                r, c = divmod(idx, cols)
                axes[r][c].axis('off')
            plt.tight_layout()
            fig2_path = args.out.parent / 'unbal_top1_similarity_per_class.png'
            plt.savefig(fig2_path, dpi=150)
            print(f"Saved plot: {fig2_path}")

        # Save top1 stats CSV
        stats = {
            'count': len(top1_vals),
            'mean': float(np.mean(top1_vals)),
            'median': float(np.median(top1_vals)),
            'p90': float(np.quantile(top1_vals, 0.9)),
            'p95': float(np.quantile(top1_vals, 0.95)),
            'p99': float(np.quantile(top1_vals, 0.99)),
            'min': float(np.min(top1_vals)),
            'max': float(np.max(top1_vals)),
        }
        (args.out.parent / 'unbal_top1_similarity_stats.csv').write_text(pd.DataFrame([stats]).to_csv(index=False))
        print(f"Saved stats: {args.out.parent / 'unbal_top1_similarity_stats.csv'}")


if __name__ == '__main__':
    main()
