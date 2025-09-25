from __future__ import annotations

"""
Retrieve top-K most similar AudioSet segments (from bal_train index) for each
DCASE2021 Task4 validation segment, with optional in-domain label filtering.

Inputs:
- One or more (index, meta) pairs (e.g., bal_train + eval)
- metadata/dcase2021_metadata/validation/validation.tsv
- metadata/audioset_metadata/class_labels_indices.csv

Outputs:
- results/dcase_val_neighbors_topK.csv (configurable)

Config:
- --top-k: number of neighbors per query
- --min-sim: optional similarity threshold (cosine)
- --label-filter: 'none' (no filter) or 'mapped' (in-domain only)
- --mapping-yaml: optional YAML to override DCASE->AudioSet mids mapping
 - --no-dedup: do not deduplicate identical segment_ids across pools
 - --sweep-thresholds: comma-separated thresholds to compute coverage stats
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(x, ord=2, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return x / norm


def filename_to_segment_id(filename: str) -> str:
    stem = filename[:-4] if filename.endswith('.wav') else filename
    parts = stem.rsplit('_', 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected DCASE filename: {filename}")
    ytid, start_sec, _ = parts
    # DCASE filenames often prefix the 11-char YouTube ID with an extra 'Y'.
    if ytid.startswith('Y') and len(ytid) > 11:
        ytid = ytid[1:]
    start_ms = int(round(float(start_sec) * 1000))
    return f"{ytid}_{start_ms}"


def default_dcase_to_audioset_mids() -> Dict[str, List[str]]:
    # Mapping DCASE 10 classes -> AudioSet MIDs (tunable/extendable)
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


def load_index(index_npz: Path, meta_csv: Path) -> Tuple[np.ndarray, List[str], List[List[int]], List[str]]:
    data = np.load(index_npz, allow_pickle=True)
    emb = data['embeddings'].astype(np.float32)
    seg = [str(s) for s in data['segment_id']]
    meta = pd.read_csv(meta_csv)
    # Align meta rows to seg order using a dict
    m = {row['segment_id']: row for _, row in meta.iterrows()}
    labels = []
    vids = []
    for sid in seg:
        row = m.get(sid)
        if row is None:
            labels.append([])
            vids.append('')
        else:
            lab = str(row['labels']) if not (isinstance(row['labels'], float) and math.isnan(row['labels'])) else ''
            labs = [int(x) for x in lab.split('|') if x != '']
            labels.append(labs)
            vids.append(str(row['video_id']))
    emb = l2_normalize(emb)
    return emb, seg, labels, vids


def main():
    parser = argparse.ArgumentParser(description="Retrieve similar AudioSet segments for DCASE validation")
    parser.add_argument('--dcase-validation', type=Path, default=Path('metadata/dcase2021_metadata/validation/validation.tsv'))
    parser.add_argument('--index', type=Path, action='append', help='Path to index .npz (repeatable)')
    parser.add_argument('--meta', type=Path, action='append', help='Path to meta .csv (repeatable)')
    parser.add_argument('--class-index', type=Path, default=Path('metadata/audioset_metadata/class_labels_indices.csv'))
    parser.add_argument('--label-filter', type=str, choices=['none', 'mapped'], default='mapped')
    parser.add_argument('--top-k', type=int, default=100)
    parser.add_argument('--min-sim', type=float, default=None)
    parser.add_argument('--out', type=Path, default=Path('results/dcase_val_neighbors_topK.csv'))
    parser.add_argument('--no-dedup', action='store_true', help='Do not deduplicate identical segment_ids across pools')
    parser.add_argument('--sweep-thresholds', type=str, default=None, help='Comma-separated similarity thresholds to evaluate coverage, e.g., "0.4,0.5,0.6"')
    args = parser.parse_args()

    # Default to eval + bal_train if none provided
    index_list: List[Tuple[Path, Path]] = []
    if not args.index and not args.meta:
        # Try common defaults
        pairs = [
            (Path('artifacts/audio_eval_index.npz'), Path('artifacts/audio_eval_meta.csv')),
            (Path('artifacts/audioset_bal_train_index.npz'), Path('artifacts/audioset_bal_train_meta.csv')),
        ]
        for idxp, mp in pairs:
            if idxp.is_file() and mp.is_file():
                index_list.append((idxp, mp))
    else:
        if not args.index or not args.meta or len(args.index) != len(args.meta):
            raise ValueError('Provide equal counts of --index and --meta, or none to use defaults')
        index_list = list(zip(args.index, args.meta))

    if not index_list:
        raise FileNotFoundError('No usable (index, meta) pairs found')

    # Load and concatenate pools
    emb_list = []
    sid_list: List[str] = []
    lab_list: List[List[int]] = []
    vid_list: List[str] = []
    for idxp, mp in index_list:
        e, s, l, v = load_index(idxp, mp)
        emb_list.append(e)
        sid_list.extend(s)
        lab_list.extend(l)
        vid_list.extend(v)
    emb = np.vstack(emb_list)
    seg_ids = sid_list
    seg_labels = lab_list
    seg_vids = vid_list

    # Deduplicate identical segment_ids across pools (keep first occurrence)
    if not args.no_dedup:
        seen = {}
        keep_idx = []
        for i, sid in enumerate(seg_ids):
            if sid not in seen:
                seen[sid] = i
                keep_idx.append(i)
        keep_idx = np.array(keep_idx, dtype=np.int64)
        emb = emb[keep_idx]
        seg_ids = [seg_ids[i] for i in keep_idx]
        seg_labels = [seg_labels[i] for i in keep_idx]
        seg_vids = [seg_vids[i] for i in keep_idx]
    emb = l2_normalize(emb)
    sid_to_idx = {sid: i for i, sid in enumerate(seg_ids)}

    # Load class index and build mapping
    mid_to_idx, idx_to_mid, idx_to_name = load_mid_index_name(args.class_index)
    dcase_to_mids = default_dcase_to_audioset_mids()

    # Load DCASE validation and group labels per segment
    dcase = pd.read_csv(args.dcase_validation, sep='\t')
    seg2labels: Dict[str, List[str]] = {}
    for _, row in dcase.iterrows():
        fn = str(row['filename'])
        lab = str(row['event_label'])
        sid = filename_to_segment_id(fn)
        seg2labels.setdefault(sid, [])
        if lab not in seg2labels[sid]:
            seg2labels[sid].append(lab)

    # Prepare output rows and stats holders
    out_rows: List[Dict[str, object]] = []
    missing_queries: List[str] = []
    query_max_sim: Dict[str, float] = {}

    for q_sid, q_dcase_labels in seg2labels.items():
        idx = sid_to_idx.get(q_sid)
        if idx is None:
            missing_queries.append(q_sid)
            continue
        q = emb[idx]
        sims = emb @ q  # cosine similarities (L2-normalized)
        candidate_indices = np.arange(len(seg_ids))

        if args.label_filter == 'mapped':
            allowed_idx = set(build_allowed_label_indices(q_dcase_labels, mid_to_idx, dcase_to_mids))
            if allowed_idx:
                mask = [any((lab in allowed_idx) for lab in seg_labels[i]) for i in candidate_indices]
                candidate_indices = candidate_indices[np.array(mask, dtype=bool)]

        # Exclude the query segment itself from its neighbors (by index and by segment_id)
        # (User prefers no special handling of same video_id; keep other segments from same video.)
        candidate_indices = candidate_indices[candidate_indices != idx]
        if candidate_indices.size:
            sid_arr = np.array([seg_ids[i] for i in candidate_indices])
            candidate_indices = candidate_indices[sid_arr != q_sid]

        if candidate_indices.size == 0:
            query_max_sim[q_sid] = float('nan')
            continue

        cand_sims = sims[candidate_indices]
        # Partial top-k selection
        top_k = min(int(args.top_k), candidate_indices.size)
        top_idx = np.argpartition(-cand_sims, kth=top_k - 1)[:top_k]
        order = np.argsort(-cand_sims[top_idx])
        top_sel = candidate_indices[top_idx][order]
        top_sims = cand_sims[top_idx][order]

        # Record max similarity before thresholding for coverage stats
        if top_sims.size:
            query_max_sim[q_sid] = float(top_sims[0])
        else:
            query_max_sim[q_sid] = float('nan')

        # Apply min-sim threshold if provided
        if args.min_sim is not None:
            keep = top_sims >= float(args.min_sim)
            top_sel = top_sel[keep]
            top_sims = top_sims[keep]

        # Emit rows
        q_labels_str = '|'.join(sorted(set(q_dcase_labels)))
        for rank, (cand_i, sim) in enumerate(zip(top_sel, top_sims), start=1):
            c_sid = seg_ids[cand_i]
            c_vid = seg_vids[cand_i]
            c_label_idx = seg_labels[cand_i]
            c_label_names = [idx_to_name.get(li, str(li)) for li in c_label_idx]
            out_rows.append({
                'query_segment_id': q_sid,
                'query_labels': q_labels_str,
                'candidate_segment_id': c_sid,
                'candidate_video_id': c_vid,
                'similarity': float(sim),
                'rank': rank,
                'candidate_label_indices': '|'.join(str(x) for x in c_label_idx),
                'candidate_label_names': '|'.join(c_label_names),
                'source_split': 'bal_train',
            })

    out_df = pd.DataFrame(out_rows)
    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Saved neighbors: {out_path} with {len(out_df)} rows")
    if missing_queries:
        print(f"Missing queries in index: {len(missing_queries)} (e.g., {missing_queries[:5]})")

    # Coverage stats and optional threshold sweep
    sweep = None
    if args.sweep_thresholds:
        try:
            sweep = [float(x) for x in args.sweep_thresholds.split(',') if x.strip() != '']
        except Exception:
            sweep = None
    if sweep:
        # Overall coverage
        all_queries = list(seg2labels.keys())
        # Exclude queries missing from index
        present_queries = [q for q in all_queries if q in query_max_sim]
        overall_rows = []
        for thr in sweep:
            covered = sum(1 for q in present_queries if (not np.isnan(query_max_sim.get(q, float('nan'))) and query_max_sim[q] >= thr))
            overall_rows.append({'threshold': thr, 'n_queries': len(present_queries), 'covered': covered, 'coverage_ratio': covered / len(present_queries) if present_queries else 0.0})
        overall_df = pd.DataFrame(overall_rows)
        cov_overall_path = out_path.parent / 'coverage_overall.csv'
        overall_df.to_csv(cov_overall_path, index=False)
        print(f"Saved overall coverage sweep: {cov_overall_path}")

        # Per-class coverage
        per_class_rows = []
        # Build query set per class
        cls_to_queries: Dict[str, List[str]] = {}
        for q, labs in seg2labels.items():
            for lab in labs:
                cls_to_queries.setdefault(lab, []).append(q)
        for thr in sweep:
            for lab, qlist in cls_to_queries.items():
                q_present = [q for q in qlist if q in query_max_sim]
                covered = sum(1 for q in q_present if (not np.isnan(query_max_sim.get(q, float('nan'))) and query_max_sim[q] >= thr))
                per_class_rows.append({'threshold': thr, 'class': lab, 'n_queries': len(q_present), 'covered': covered, 'coverage_ratio': covered / len(q_present) if q_present else 0.0})
        per_class_df = pd.DataFrame(per_class_rows)
        cov_class_path = out_path.parent / 'coverage_per_class.csv'
        per_class_df.to_csv(cov_class_path, index=False)
        print(f"Saved per-class coverage sweep: {cov_class_path}")


if __name__ == '__main__':
    main()
