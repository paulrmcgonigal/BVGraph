#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BVGraph2D.py

Produce a symmetrical, mirror-preserving layout for networks with paired (chiral)
and unpaired (achiral) nodes. The layout enforces:
 - mirror symmetry: chiral node pairs are placed as left/right copies (same |x|, same y)
 - achiral nodes are arranged on the central vertical axis (x == 0)
 - achiral nodes that are directly connected are placed as nearest neighbours on the axis
 - iterative local optimizations that prioritize node spacing and edge compactness, while still reducing edge crossings
 - post-processing to avoid node/edge overlaps and to enforce minimum lateral clearance.

Main features:
 - Uses modularity-based community layout for coarse rep placement
 - Enforces pairwise mirroring for enantiomeric pairs
 - Arranges achiral components as vertical blocks on the central axis with uniform spacing
 - Local move evaluations combine edge-length, spacing, and crossing terms for efficiency
 - Chiral relaxation adds edge-length regularisation and local repulsion to reduce cramped or stretched layouts
 - Exports a GEXF file including viz:position attributes for viewing in Gephi

Dependencies:
  - Python 3.8+
  - networkx
  - numpy
  - pandas

See also the corresponding 3D layout utilities (BVGraph3D.py).

Author: Paul McGonigal
"""
from __future__ import annotations
import argparse
import math
import random
import time
from collections import defaultdict
from typing import Tuple, List, Dict, Set
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

# ---------------------------
# Utilities
# ---------------------------
def _clean_id(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\u200b", "").replace("\ufeff", "").strip()
    return s

def _normalize_barcode(code: str, width: int = 10) -> str:
    code = _clean_id(code)
    return code.zfill(width) if len(code) < width else code

def _split_arms(code10: str) -> Tuple[str, str, str, str]:
    a, b, c, center = code10[:3], code10[3:6], code10[6:9], code10[9]
    return a, b, c, center

def _rotations(triple: Tuple[str, str, str]) -> List[Tuple[str, str, str]]:
    a, b, c = triple
    return [(a, b, c), (b, c, a), (c, a, b)]

def _canonical_rotation(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    return min(_rotations(triple))

def _reflect_order(triple: Tuple[str, str, str]) -> Tuple[str, str, str]:
    a, b, c = triple
    return (a, c, b)

def _make_code(arms: Tuple[str, str, str], center: str) -> str:
    return "".join(arms) + center

# ---------------------------
# Node annotation
# ---------------------------
def annotate_nodes(nodes_df: pd.DataFrame, id_col="id") -> pd.DataFrame:
    df = nodes_df.copy()
    df[id_col] = df[id_col].astype(str).apply(_clean_id)
    rows = []
    original_ids = set(df[id_col].tolist())
    for _, r in df.iterrows():
        nid = _clean_id(r[id_col])
        try:
            code10 = _normalize_barcode(nid, 10)
            a, b, c, center = _split_arms(code10)
            can = _canonical_rotation((a, b, c))
            chiral = len(set(can)) == 3
            partner_can = _canonical_rotation(_reflect_order(can)) if chiral else can
            partner_norm = _make_code(partner_can, center)
            partner_unpadded = partner_norm.lstrip("0") or "0"
            norm_code = _make_code(can, center)
        except Exception:
            chiral = False
            partner_norm = ""
            partner_unpadded = ""
            norm_code = ""
        partner_id = partner_norm if partner_norm in original_ids else (
            partner_unpadded if partner_unpadded in original_ids else "")
        rows.append({
            id_col: nid,
            "Barcode_Normalized": norm_code,
            "Chirality": "chiral" if chiral else "achiral",
            "Enantiomer_Suggested": partner_unpadded,
            "Enantiomer_Id": _clean_id(partner_id),
        })
    out = df.merge(pd.DataFrame(rows), on=id_col, how="left")
    return out

# ---------------------------
# Graph IO & attach annotations
# ---------------------------
def read_graph(nodes_csv, edges_csv, id_col="id", energy_col="Energy"):
    nodes = pd.read_csv(nodes_csv, dtype={id_col: str})
    edges = pd.read_csv(edges_csv, dtype=str)
    nodes[id_col] = nodes[id_col].astype(str).apply(_clean_id)
    # normalize edge column names case-insensitively
    cols = {c.lower(): c for c in edges.columns}
    src_col = cols.get('source', cols.get('s', None))
    tgt_col = cols.get('target', cols.get('t', None))
    ts_col = cols.get('ts_energy', cols.get('ts', None))
    if src_col is None or tgt_col is None:
        raise ValueError('Edges CSV must contain Source and Target columns (case-insensitive).')
    edges['Source'] = edges[src_col].astype(str).apply(_clean_id)
    edges['Target'] = edges[tgt_col].astype(str).apply(_clean_id)

    G = nx.Graph()
    # allow energy column case-insensitively
    node_cols = {c.lower(): c for c in nodes.columns}
    energy_col_actual = node_cols.get(energy_col.lower(), energy_col) if isinstance(energy_col, str) else energy_col
    for _, r in nodes.iterrows():
        nid = _clean_id(r[id_col])
        energy = float(r.get(energy_col_actual, 0.0))
        G.add_node(nid, **{energy_col: energy})
        # placeholders for annotation keys
        for k in ["Chirality", "Enantiomer_Id", "Barcode_Normalized", "Enantiomer_Suggested"]:
            if k in nodes.columns:
                G.nodes[nid][k] = _clean_id(r.get(k, ""))
            else:
                G.nodes[nid][k] = ""
    energies = nx.get_node_attributes(G, energy_col)
    if energies:
        emin = min(energies.values())
        for n in G.nodes:
            G.nodes[n]["DeltaE"] = float(G.nodes[n][energy_col]) - float(emin)
    for _, r in edges.iterrows():
        s, t = r["Source"], r["Target"]
        if s not in G or t not in G:
            continue
        if G.has_edge(s, t):
            continue
        G.add_edge(s, t)
    return G, nodes, edges

def attach_annotation_from_df(G, annot_df, id_col="id"):
    annot = annot_df.set_index(id_col)
    for n in G.nodes():
        if G.nodes[n].get("Chirality", "") == "" and n in annot.index:
            G.nodes[n]["Chirality"] = str(annot.loc[n].get("Chirality", "")).strip().lower()
        if G.nodes[n].get("Enantiomer_Id", "") == "" and n in annot.index:
            G.nodes[n]["Enantiomer_Id"] = _clean_id(annot.loc[n].get("Enantiomer_Id", ""))
        for k in ["Barcode_Normalized", "Enantiomer_Suggested"]:
            if k in annot.columns and n in annot.index:
                G.nodes[n][k] = _clean_id(annot.loc[n].get(k, ""))
    return G

# ---------------------------
# Pair detection & rep graph
# ---------------------------
def build_pairs(G):
    pairs = {}
    node2pair = {}
    achiral_or_unmatched = set()
    issues = []
    for n in G.nodes():
        chir = G.nodes[n].get("Chirality", "")
        if chir == "achiral":
            achiral_or_unmatched.add(n)
            continue
        p = _clean_id(G.nodes[n].get("Enantiomer_Id", ""))
        if (not p) or (p not in G):
            achiral_or_unmatched.add(n)
            issues.append({"id": n, "partner": p, "reason": "empty_or_missing_partner"})
            continue
        pm = _clean_id(G.nodes[p].get("Enantiomer_Id", "")) if p in G else ""
        if pm != n:
            achiral_or_unmatched.add(n)
            issues.append({"id": n, "partner": p, "reason": "not_mutual_mapping"})
            continue
        a, b = sorted([n, p])
        if a not in pairs:
            pairs[a] = (a, b)
        node2pair[n] = a
        node2pair[p] = a
    return pairs, node2pair, achiral_or_unmatched, issues

def pair_graph_for_bisection(G, pairs, node2pair, weight_mode="inverse_ts"):
    H = nx.Graph()
    for pid in pairs.keys():
        H.add_node(pid)
    accum = defaultdict(float)
    for u, v in G.edges():
        if u not in node2pair or v not in node2pair:
            continue
        pi, pj = node2pair[u], node2pair[v]
        if pi == pj:
            continue
        key = tuple(sorted((pi, pj)))
        accum[key] += 1.0
    for (pi, pj), w in accum.items():
        H.add_edge(pi, pj, weight=w)
    return H

def balanced_bisection_pairs(H, seed=42):
    nodes = list(H.nodes())
    if len(nodes) <= 1:
        return set(nodes), set()
    rng = random.Random(seed)
    rng.shuffle(nodes)
    half = len(nodes) // 2
    A = set(nodes[:half])
    B = set(nodes[half:])
    try:
        A2, B2 = nx.algorithms.community.kernighan_lin_bisection(H, partition=(A, B), weight='weight')
        return set(A2), set(B2)
    except Exception:
        return A, B

def rep_graph_from_partition(G, pairs, A_side):
    rep_of = {}
    reps = set()
    for pid, (A, B) in pairs.items():
        rep = A if pid in A_side else B
        reps.add(rep)
        rep_of[A] = rep
        rep_of[B] = rep
    Grep = nx.Graph()
    for n in reps:
        Grep.add_node(n, **G.nodes[n])
    for u, v in G.edges():
        ru, rv = rep_of.get(u, u), rep_of.get(v, v)
        if ru in reps and rv in reps and ru != rv:
            if Grep.has_edge(ru, rv):
                continue
            Grep.add_edge(ru, rv)
    return rep_of, reps, Grep
# ---------------------------
# Representative layout (scale-aware)
# ---------------------------
def compute_layout_scale(G, min_sep=80.0, layout_scale=None, layout_scale_factor=8.0):
    if layout_scale is not None and float(layout_scale) > 0.0:
        return float(layout_scale)
    n = max(1, G.number_of_nodes())
    return max(3.0 * float(min_sep), float(layout_scale_factor) * math.sqrt(float(n)))

def layout_reps_scale_aware(Grep, layout_scale=900.0, sweeps=4, seed=42):
    nodes = list(Grep.nodes())
    if not nodes:
        return {}
    if Grep.number_of_edges() == 0:
        radius = max(1.0, float(layout_scale))
        k = len(nodes)
        return {n: (radius * math.cos(2 * math.pi * i / max(1, k)), radius * math.sin(2 * math.pi * i / max(1, k))) for i, n in enumerate(nodes)}

    k_val = max(1e-6, float(layout_scale) / max(1.0, math.sqrt(max(1, len(nodes)))))
    pos = nx.spring_layout(Grep, seed=seed, k=k_val, iterations=max(25, 25 * max(1, sweeps)), scale=float(layout_scale), center=(0.0, 0.0), dim=2)
    return {n: (float(x), float(y)) for n, (x, y) in pos.items()}
# ---------------------------
# Mirror enforcement and initial positions
# ---------------------------
def enforce_full_mirror(G, pairs, A_side, reps_pos, snap_achiral=True, axis_lateral_gap=None):
    pos = {}
    sides = {}
    for pid, (A, B) in pairs.items():
        rep = A if pid in A_side else B
        x, y = reps_pos.get(rep, (0.0, 0.0))
        if rep == A:
            pos[A] = (abs(x), y)
            pos[B] = (-abs(x), y)
            sides[A] = "right"; sides[B] = "left"
        else:
            pos[B] = (abs(x), y)
            pos[A] = (-abs(x), y)
            sides[B] = "right"; sides[A] = "left"
    for n in G.nodes():
        if G.nodes[n].get("Chirality", "") == "achiral" or not G.nodes[n].get("Enantiomer_Id", ""):
            ys = [pos[v][1] for v in G[n] if v in pos]
            y = float(np.median(ys)) if ys else 0.0
            pos[n] = (0.0 if snap_achiral else 0.0, y)
            sides[n] = "axis"
    for n in G.nodes():
        if n not in pos:
            pos[n] = (0.0, 0.0)
            sides[n] = "axis"
    if axis_lateral_gap is not None and axis_lateral_gap > 0.0:
        for n in list(pos.keys()):
            if sides.get(n, "") == "axis":
                continue
            x, y = pos[n]
            if abs(x) < axis_lateral_gap:
                sign = 1.0 if x >= 0 else -1.0
                if abs(x) < 1e-12:
                    sign = 1.0 if sides.get(n, "") == "right" else -1.0
                pos[n] = (sign * axis_lateral_gap, y)
    return pos, sides

# ---------------------------
# Axis and chiral alignment helpers
# ---------------------------

def match_axis_chiral_stats(pos: Dict[str, Tuple[float, float]], sides: Dict[str, str]) -> Dict[str, Tuple[float, float]]:
    """
    Attempt to align the central (axis) group and chiral hemispheres so that
    their means and medians are close. Perfect equality of both statistics
    is not always achievable by pure translations; we therefore:
      - compute mean and median for each group,
      - compute a single target for mean alignment (the mean of the two means),
      - translate groups so their means match the target (centroid alignment),
      - then additionally apply a small balanced correction that reduces the median
        difference (minimizes the squared error of mean & median differences).
    This aggressively keeps relative ordering while reducing vertical drift
    between axis and hemispheres.
    """
    newpos = dict(pos)
    axis_nodes = [n for n, s in sides.items() if s == 'axis' and n in pos]
    chiral_nodes = [n for n, s in sides.items() if s != 'axis' and n in pos]

    if not axis_nodes or not chiral_nodes:
        return newpos

    axis_ys = [pos[n][1] for n in axis_nodes]
    chiral_ys = [pos[n][1] for n in chiral_nodes]

    mean_axis = float(np.mean(axis_ys))
    mean_chiral = float(np.mean(chiral_ys))
    med_axis = float(np.median(axis_ys))
    med_chiral = float(np.median(chiral_ys))

    # First, set both group means to the common mean target (simple centroid alignment)
    mean_target = 0.5 * (mean_axis + mean_chiral)
    delta_axis = mean_target - mean_axis
    delta_chiral = mean_target - mean_chiral

    for n in axis_nodes:
        x, y = newpos[n]
        newpos[n] = (x, y + delta_axis)
    for n in chiral_nodes:
        x, y = newpos[n]
        newpos[n] = (x, y + delta_chiral)

    # Recompute medians after mean alignment
    axis_ys2 = [newpos[n][1] for n in axis_nodes]
    chiral_ys2 = [newpos[n][1] for n in chiral_nodes]
    med_axis2 = float(np.median(axis_ys2))
    med_chiral2 = float(np.median(chiral_ys2))

    # If medians still differ, apply a balanced correction: move both groups half the median difference
    med_diff = med_axis2 - med_chiral2
    if abs(med_diff) > 1e-9:
        # move axis by -med_diff/2 and chiral by +med_diff/2 so they meet halfway
        corr_axis = -0.5 * med_diff
        corr_chiral = 0.5 * med_diff
        for n in axis_nodes:
            x, y = newpos[n]
            newpos[n] = (x, y + corr_axis)
        for n in chiral_nodes:
            x, y = newpos[n]
            newpos[n] = (x, y + corr_chiral)

    return newpos

    """
    Translate either the axis group or the chiral group (whichever one is non-empty)
    so that mean(y_axis_nodes) == mean(y_chiral_nodes). Preserve relative ordering.
    """
    newpos = dict(pos)
    axis_nodes = [n for n, s in sides.items() if s == 'axis' and n in pos]
    chiral_nodes = [n for n, s in sides.items() if s != 'axis' and n in pos]

    if not axis_nodes or not chiral_nodes:
        return newpos

    axis_ys = [pos[n][1] for n in axis_nodes]
    chiral_ys = [pos[n][1] for n in chiral_nodes]
    mean_axis = float(np.mean(axis_ys))
    mean_chiral = float(np.mean(chiral_ys))

    # If already equal (within tiny tol) do nothing
    if abs(mean_axis - mean_chiral) < 1e-9:
        return newpos

    # Decide which group to move: move the smaller-span group for minimal disruption
    span_axis = max(axis_ys) - min(axis_ys) if len(axis_ys) > 1 else 0.0
    span_chiral = max(chiral_ys) - min(chiral_ys) if len(chiral_ys) > 1 else 0.0

    # If both spans zero, move axis
    if span_chiral < span_axis:
        # move chiral group to match axis
        delta = mean_axis - mean_chiral
        for n in chiral_nodes:
            x, y = newpos[n]
            newpos[n] = (x, y + delta)
    else:
        # move axis group to match chiral mean
        delta = mean_chiral - mean_axis
        for n in axis_nodes:
            x, y = newpos[n]
            newpos[n] = (x, y + delta)

    return newpos

# ---------------------------
# Equalisation with target ratio
# ---------------------------
def equalize_hemi_span(pos: Dict[str, Tuple[float, float]],
                       sides: Dict[str, str],
                       tolerance: float = 0.10,
                       min_nodes: int = 2,
                       hemi_max_scale: float = 1.5,
                       target_ratio: float = 1.0) -> Dict[str, Tuple[float, float]]:
    newpos = dict(pos)
    axis_nodes = [n for n, s in sides.items() if s == 'axis' and n in pos]
    chiral_nodes = [n for n, s in sides.items() if s != 'axis' and n in pos]

    axis_ok = len(axis_nodes) >= 2
    chiral_ok = len(chiral_nodes) >= min_nodes

    if not axis_ok and not chiral_ok:
        return newpos

    axis_ys = [pos[n][1] for n in axis_nodes] if axis_ok else []
    chiral_ys = [pos[n][1] for n in chiral_nodes] if chiral_ok else []

    axis_span = (max(axis_ys) - min(axis_ys)) if axis_ok else 0.0
    ch_span = (max(chiral_ys) - min(chiral_ys)) if chiral_ok else 0.0

    if axis_span < 1e-12 and ch_span < 1e-12:
        return newpos

    if ch_span < 1e-12:
        return newpos

    desired_axis_span = ch_span * float(target_ratio)

    if axis_span < 1e-12:
        raw_scale_axis = desired_axis_span / max(1e-12, axis_span) if axis_span > 0 else (desired_axis_span / (ch_span + 1e-12))
    else:
        raw_scale_axis = desired_axis_span / axis_span

    if axis_span >= desired_axis_span * (1.0 - tolerance) and axis_span <= desired_axis_span * (1.0 + tolerance):
        return newpos

    if axis_span < desired_axis_span:
        scale_target_group = 'axis'
        current_span = axis_span
        target_span = desired_axis_span
    else:
        scale_target_group = 'chiral'
        current_span = ch_span
        target_span = axis_span / max(1e-12, float(target_ratio))

    if current_span < 1e-12:
        return newpos

    raw_scale = target_span / current_span
    max_s = max(1.0, float(hemi_max_scale))
    clamped_scale = max(1.0 / max_s, min(raw_scale, max_s))

    if scale_target_group == 'axis' and axis_ok:
        y_med = float(np.median(axis_ys))
        for n in axis_nodes:
            x, y = pos[n]
            y_new = y_med + (y - y_med) * clamped_scale
            newpos[n] = (x, float(y_new))
    elif scale_target_group == 'chiral' and chiral_ok:
        y_med = float(np.median(chiral_ys))
        for n in chiral_nodes:
            x, y = pos[n]
            y_new = y_med + (y - y_med) * clamped_scale
            newpos[n] = (x, float(y_new))

    return newpos

# ---------------------------
# Achiral axis arrangement & adjacency
# ---------------------------
def arrange_achiral_on_axis(G, pos, sides, pairs, min_sep=60.0, vertical_gap=None):
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    comp_orderings = {}
    if not axis_nodes:
        return pos, comp_orderings
    subG = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(subG))
    if vertical_gap is None:
        vertical_gap = 1.6 * min_sep
    occupied_ranges = []
    comp_counter = 0
    for comp in comps:
        comp_sub = subG.subgraph(comp).copy()
        adj = {n: set(comp_sub[n]) for n in comp_sub.nodes()}
        order_blocks = []
        used = set()
        leaves = [n for n in comp_sub.nodes() if len(adj[n]) <= 1]
        for leaf in leaves:
            if leaf in used:
                continue
            path = [leaf]
            used.add(leaf)
            cur = leaf
            prev = None
            while True:
                nbrs = [x for x in adj[cur] if x != prev and x not in used]
                if not nbrs:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                used.add(nxt)
                prev, cur = cur, nxt
            order_blocks.append(path)
        remaining = [n for n in comp_sub.nodes() if n not in used]
        while remaining:
            start = remaining[0]
            path = [start]
            used.add(start)
            cur = start
            prev = None
            while True:
                nbrs = [x for x in adj[cur] if x != prev]
                next_candidates = [x for x in nbrs if x not in used]
                if next_candidates:
                    nxt = next_candidates[0]
                    path.append(nxt)
                    used.add(nxt)
                    prev, cur = cur, nxt
                    continue
                break
            order_blocks.append(path)
            remaining = [n for n in comp_sub.nodes() if n not in used]
        ordering = []
        for blk in order_blocks:
            ordering.extend(blk)
        if not ordering:
            continue
        m = len(ordering)
        height = (m - 1) * vertical_gap
        neigh_ys = []
        for n in ordering:
            for nbr in G[n]:
                if nbr not in comp:
                    neigh_ys.append(pos.get(nbr, (0.0, 0.0))[1])
        center_y = float(np.median(neigh_ys)) if neigh_ys else (0.0 if not occupied_ranges else (max(r[1] for r in occupied_ranges) + vertical_gap + height / 2.0))
        start_y = center_y + height / 2.0
        for i, n in enumerate(ordering):
            y = start_y - i * vertical_gap
            pos[n] = (0.0, y)
        occupied_ranges.append((center_y - height / 2.0, center_y + height / 2.0))
        comp_orderings[comp_counter] = ordering
        comp_counter += 1
    return pos, comp_orderings

def infer_axis_component_orderings(G, pos, sides):
    """Infer axis component orderings from existing coordinates without moving nodes."""
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    if not axis_nodes:
        return {}
    subG = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(subG))
    comp_orderings = {}
    for cid, comp in enumerate(comps):
        ordering = sorted(list(comp), key=lambda n: (pos.get(n, (0.0, 0.0))[1], str(n)), reverse=True)
        comp_orderings[cid] = ordering
    return comp_orderings

def enforce_achiral_direct_adjacency(G, pos, sides, comp_orderings, achiral_gap):
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    if not axis_nodes:
        return pos, {}

    subG = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(subG))
    new_comp_orderings = {}
    comp_id = 0

    for comp in comps:
        comp_sub = subG.subgraph(comp).copy()
        degs = dict(comp_sub.degree())
        if len(comp_sub) == 0:
            ordering = []
        else:
            leaves = [n for n, d in degs.items() if d == 1]
            visited = set()
            ordering = []
            if leaves:
                start = leaves[0]
            else:
                start = next(iter(comp_sub.nodes()))
            cur = start
            prev = None
            while True:
                ordering.append(cur)
                visited.add(cur)
                nbrs = [x for x in comp_sub[cur] if x != prev]
                next_candidates = [x for x in nbrs if x not in visited]
                if next_candidates:
                    nxt = next_candidates[0]
                    prev, cur = cur, nxt
                    continue
                remaining = [n for n in comp_sub.nodes() if n not in visited]
                if remaining:
                    cur = remaining[0]
                    prev = None
                    continue
                break

        new_comp_orderings[comp_id] = ordering
        comp_id += 1

    occupied_ranges = []
    for cid, ordering in new_comp_orderings.items():
        m = len(ordering)
        if m == 0:
            continue
        neigh_ys = []
        for n in ordering:
            for nbr in G[n]:
                if nbr not in ordering:
                    neigh_ys.append(pos.get(nbr, (0.0, 0.0))[1])
        center_y = float(np.median(neigh_ys)) if neigh_ys else 0.0
        height = (m - 1) * achiral_gap
        half = height / 2.0
        start_y = center_y + half
        for _ in range(100):
            low = start_y - height
            high = start_y
            collision = False
            for (ol, oh) in occupied_ranges:
                if not (high + 1e-6 < ol or low - 1e-6 > oh):
                    start_y = oh - 1e-6
                    collision = True
                    break
            if not collision:
                break
        for idx, n in enumerate(ordering):
            y = start_y - idx * achiral_gap
            pos[n] = (0.0, y)
        occupied_ranges.append((start_y - height, start_y))

    return pos, new_comp_orderings

def adjust_axis_components_min_gap(G, pos, sides, comp_orderings, achiral_gap, min_axis_gap, max_iters=6):
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    if not axis_nodes:
        return pos, comp_orderings

    subG = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(subG))
    comp_list = []
    for cid, comp in enumerate(comps):
        ordering = None
        for k, ordlist in comp_orderings.items():
            if set(ordlist) == set(comp):
                ordering = list(ordlist)
                break
        if ordering is None:
            ordering = sorted(list(comp), key=lambda n: pos[n][1])
        comp_list.append(ordering)

    blocks = []
    for ordering in comp_list:
        m = len(ordering)
        height = (m - 1) * achiral_gap if m > 0 else 0.0
        center = np.median([pos[n][1] for n in ordering]) if ordering else 0.0
        blocks.append({"ordering": ordering, "height": height, "center": center})

    blocks.sort(key=lambda b: -b["center"])

    y_cursor = None
    new_pos = dict(pos)
    for b in blocks:
        ordering = b["ordering"]
        height = b["height"]
        if y_cursor is None:
            center = b["center"]
            start_y = center + height / 2.0
        else:
            start_y = y_cursor - min_axis_gap
        for idx, n in enumerate(ordering):
            new_pos[n] = (0.0, start_y - idx * achiral_gap)
        bottom = start_y - (len(ordering) - 1) * achiral_gap if ordering else start_y
        y_cursor = bottom - min_axis_gap

    return new_pos, comp_orderings

def relax_axis_node_clearance(
    G,
    pos,
    sides,
    focus_nodes=None,
    clearance=25.0,
    max_shift=200.0,
    step=5.0,
):
    """
    Nudge selected axis nodes up/down the axis to reduce edge overlap/underpass.
    Keeps x == 0 for all axis nodes.
    """
    newpos = dict(pos)
    axis_nodes = [n for n in G.nodes() if sides.get(n, "") == "axis" and n in newpos]

    if focus_nodes is None:
        focus_nodes = axis_nodes
    else:
        focus_nodes = [n for n in focus_nodes if n in axis_nodes]

    if not focus_nodes:
        return newpos

    edges = list(G.edges())
    n_steps = max(1, int(max_shift / step))

    for n in focus_nodes:
        x0, y0 = newpos[n]
        best_y = y0
        best_score = None

        # Try offsets around the current y position.
        for k in range(n_steps + 1):
            offsets = [0.0] if k == 0 else [k * step, -k * step]
            for dy in offsets:
                y = y0 + dy
                score = 0.0

                for u, v in edges:
                    if n in (u, v):
                        continue
                    d = _point_segment_distance((0.0, y), newpos[u], newpos[v])
                    if d < clearance:
                        score += (clearance - d) ** 2

                # Keep movement modest unless it really helps.
                score += 1e-3 * (y - y0) ** 2

                if best_score is None or score < best_score:
                    best_score = score
                    best_y = y

        newpos[n] = (0.0, best_y)

    return newpos

# ---------------------------

def spread_axis_nodes_min_sep(pos, sides, min_sep, recenter=True):
    """Enforce a minimum vertical separation between all axis nodes."""
    newpos = dict(pos)
    axis_nodes = [n for n, s in sides.items() if s == 'axis' and n in newpos]
    if len(axis_nodes) < 2:
        for n in axis_nodes:
            newpos[n] = (0.0, float(newpos[n][1]))
        return newpos

    ordered = sorted(axis_nodes, key=lambda n: (float(newpos[n][1]), str(n)))
    y_vals = [float(newpos[n][1]) for n in ordered]
    target = list(y_vals)

    for i in range(1, len(target)):
        if target[i] - target[i - 1] < min_sep:
            target[i] = target[i - 1] + min_sep

    if recenter:
        old_med = float(np.median(y_vals))
        new_med = float(np.median(target))
        delta = old_med - new_med
        if abs(delta) > 1e-9:
            target = [y + delta for y in target]

    for n, y in zip(ordered, target):
        newpos[n] = (0.0, float(y))

    return newpos
# Overlap resolution & strict pass
# ---------------------------
def _point_segment_distance(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx = bx - ax; dy = by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    proj = (ax + t * dx, ay + t * dy)
    return math.hypot(px - proj[0], py - proj[1])

def adjust_positions_with_constraints(G, pos, pairs, sides, min_sep=40.0, edge_clearance=12.0, max_iter=200, lr=0.25, fixed_achiral_y=None, axis_lateral_gap=None):
    nodes = list(G.nodes())
    pcoords = {n: [float(pos.get(n, (0.0, 0.0))[0]), float(pos.get(n, (0.0, 0.0))[1])] for n in nodes}
    axis_set = {n for n in nodes if sides.get(n, '') == 'axis'}

    def enforce_mirror():
        for pid, (A, B) in pairs.items():
            ax = abs(pcoords[A][0])
            bx = abs(pcoords[B][0])
            avg = max(1e-6, 0.5 * (ax + bx))
            sideA = sides.get(A, ''); sideB = sides.get(B, '')
            if sideA == 'right' or sideB == 'left':
                pcoords[A][0] = abs(avg)
                pcoords[B][0] = -abs(avg)
            else:
                pcoords[A][0] = -abs(avg)
                pcoords[B][0] = abs(avg)

    def enforce_achiral_axis():
        for n in axis_set:
            pcoords[n][0] = 0.0
            if fixed_achiral_y and n in fixed_achiral_y:
                pcoords[n][1] = float(fixed_achiral_y[n])

    for it in range(max_iter):
        moved = 0
        for i in range(len(nodes)):
            ni = nodes[i]
            for j in range(i + 1, len(nodes)):
                nj = nodes[j]
                pi = pcoords[ni]; pj = pcoords[nj]
                d = math.hypot(pi[0] - pj[0], pi[1] - pj[1])
                if d < 1e-6:
                    ang = random.random() * 2 * math.pi
                    dx = math.cos(ang) * 1e-1
                    dy = math.sin(ang) * 1e-1
                    if ni not in axis_set:
                        pcoords[ni][0] += dx; pcoords[ni][1] += dy
                    if nj not in axis_set:
                        pcoords[nj][0] -= dx; pcoords[nj][1] -= dy
                    moved += 1
                    continue
                if d < min_sep:
                    overlap = (min_sep - d)
                    ux = (pi[0] - pj[0]) / d
                    uy = (pi[1] - pj[1]) / d
                    shift = 0.5 * overlap * lr
                    if ni in axis_set and nj not in axis_set:
                        pcoords[nj][0] -= ux * (overlap * lr)
                        pcoords[nj][1] -= uy * (overlap * lr)
                    elif nj in axis_set and ni not in axis_set:
                        pcoords[ni][0] += ux * (overlap * lr)
                        pcoords[ni][1] += uy * (overlap * lr)
                    else:
                        pcoords[ni][0] += ux * shift; pcoords[ni][1] += uy * shift
                        pcoords[nj][0] -= ux * shift; pcoords[nj][1] -= uy * shift
                    moved += 1
        for n in nodes:
            for u, v in G.edges():
                if n == u or n == v:
                    continue
                a = tuple(pcoords[u]); b = tuple(pcoords[v]); p = tuple(pcoords[n])
                dist = _point_segment_distance(p, a, b)
                if dist < edge_clearance:
                    ax, ay = a; bx, by = b; px, py = p
                    dx = bx - ax; dy = by - ay
                    if dx == 0 and dy == 0:
                        continue
                    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
                    t = max(0.0, min(1.0, t))
                    projx = ax + t * dx; projy = ay + t * dy
                    vx = px - projx; vy = py - projy
                    norm = math.hypot(vx, vy)
                    if norm < 1e-6:
                        vx, vy = -dy, dx
                        norm = math.hypot(vx, vy)
                    vx /= norm; vy /= norm
                    shift = (edge_clearance - dist) * lr
                    if n not in axis_set:
                        pcoords[n][0] += vx * shift
                        pcoords[n][1] += vy * shift
                    moved += 1
        enforce_achiral_axis()
        enforce_mirror()
        if axis_lateral_gap is not None and axis_lateral_gap > 0.0:
            for n in pcoords:
                if n in axis_set:
                    continue
                xval = pcoords[n][0]
                if abs(xval) < axis_lateral_gap:
                    sign = 1.0 if xval >= 0 else -1.0
                    if abs(xval) < 1e-12:
                        sign = 1.0
                    pcoords[n][0] = sign * axis_lateral_gap
        if moved == 0:
            break

    def enforce_strict_min_sep(pcoords_local, min_sep_local, axis_set_local, pairs_local, sides_local, max_rounds=50):
        nodes_local = list(pcoords_local.keys())
        for round_i in range(max_rounds):
            any_moved = False
            for i in range(len(nodes_local)):
                ni = nodes_local[i]
                for j in range(i + 1, len(nodes_local)):
                    nj = nodes_local[j]
                    pi = pcoords_local[ni]; pj = pcoords_local[nj]
                    dx = pi[0] - pj[0]; dy = pi[1] - pj[1]
                    d = math.hypot(dx, dy)
                    if d < 1e-12:
                        ang = random.random() * 2 * math.pi
                        dx = math.cos(ang) * 1e-3
                        dy = math.sin(ang) * 1e-3
                        d = math.hypot(dx, dy)
                    if d < min_sep_local:
                        need = (min_sep_local - d)
                        ux = dx / d; uy = dy / d
                        ni_axis = (ni in axis_set_local)
                        nj_axis = (nj in axis_set_local)
                        if ni_axis and not nj_axis:
                            pcoords_local[nj][0] -= ux * need
                            pcoords_local[nj][1] -= uy * need
                        elif nj_axis and not ni_axis:
                            pcoords_local[ni][0] += ux * need
                            pcoords_local[ni][1] += uy * need
                        else:
                            pcoords_local[ni][0] += 0.5 * ux * need
                            pcoords_local[ni][1] += 0.5 * uy * need
                            pcoords_local[nj][0] -= 0.5 * ux * need
                            pcoords_local[nj][1] -= 0.5 * uy * need
                        any_moved = True
            for n in axis_set_local:
                pcoords_local[n][0] = 0.0
            for pid, (A, B) in pairs_local.items():
                yA = pcoords_local[A][1]; yB = pcoords_local[B][1]
                yavg = 0.5 * (yA + yB)
                xA = pcoords_local[A][0]; xB = pcoords_local[B][0]
                mag = 0.5 * (abs(xA) + abs(xB))
                if abs(xA) < 1e-12 and abs(xB) < 1e-12:
                    mag = max(mag, min_sep_local * 0.6)
                signA = 1.0 if xA >= 0 else -1.0
                pcoords_local[A][0], pcoords_local[A][1] = signA * mag, yavg
                pcoords_local[B][0], pcoords_local[B][1] = -signA * mag, yavg
            if not any_moved:
                break

    enforce_strict_min_sep(pcoords, min_sep, axis_set, pairs, sides)
    newpos = {n: (float(pcoords[n][0]), float(pcoords[n][1])) for n in nodes}
    return newpos

# ---------------------------
# Segment bbox + intersection (fast)
# ---------------------------
def seg_bbox(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float, float, float]:
    x1, y1 = a; x2, y2 = b
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

def bbox_overlap(a_minx, a_miny, a_maxx, a_maxy, b_minx, b_miny, b_maxx, b_maxy) -> bool:
    return not (a_maxx < b_minx or b_maxx < a_minx or a_maxy < b_miny or b_maxy < a_miny)

def _segments_intersect(a1, a2, b1, b2):
    aminx, aminy, amaxx, amaxy = seg_bbox(a1, a2)
    bminx, bminy, bmaxx, bmaxy = seg_bbox(b1, b2)
    if not bbox_overlap(aminx, aminy, amaxx, amaxy, bminx, bminy, bmaxx, bmaxy):
        return False
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])
    try:
        return (ccw(a1, b1, b2) != ccw(a2, b1, b2)) and (ccw(a1, a2, b1) != ccw(a1, a2, b2))
    except Exception:
        return False

# ---------------------------
# Crossing counters & delta evaluator
# ---------------------------
def count_edge_crossings(G, pos, force_exact: bool = False):
    policy = CROSSING_POLICY
    if policy is not None and not force_exact and policy.use_sampled_by_default():
        return policy.sampled_score(G, pos, edges_all=list(G.edges()))

    edges = list(G.edges())
    coords = pos
    cnt = 0
    for i in range(len(edges)):
        u, v = edges[i]
        a1 = coords[u]; a2 = coords[v]
        for j in range(i + 1, len(edges)):
            x, y = edges[j]
            if u in (x, y) or v in (x, y):
                continue
            b1 = coords[x]; b2 = coords[y]
            if _segments_intersect(a1, a2, b1, b2):
                cnt += 1
    return cnt

def build_edge_cache(G, pos):
    edges = [tuple(sorted((u, v))) for u, v in G.edges()]
    seen = set(); unique = []
    for e in edges:
        if e not in seen:
            seen.add(e); unique.append(e)
    edge_bboxes = {e: seg_bbox(pos[e[0]], pos[e[1]]) for e in unique}
    return unique, edge_bboxes


def positions_exactly_equal(pos_a: Dict[str, Tuple[float, float]], pos_b: Dict[str, Tuple[float, float]]) -> bool:
    """Return True only if every node has exactly the same coordinates."""
    if pos_a.keys() != pos_b.keys():
        return False
    for n in pos_a:
        if pos_a[n] != pos_b[n]:
            return False
    return True

def edge_cross_count_against_list(edge, edge_list, pos, edge_bboxes):
    a1 = pos[edge[0]]; a2 = pos[edge[1]]
    aminx, aminy, amaxx, amaxy = seg_bbox(a1, a2)
    cnt = 0
    for oth in edge_list:
        if set(edge) & set(oth):
            continue
        b1 = pos[oth[0]]; b2 = pos[oth[1]]
        bminx, bminy, bmaxx, bmaxy = edge_bboxes[oth]
        if not bbox_overlap(aminx, aminy, amaxx, amaxy, bminx, bminy, bmaxx, bmaxy):
            continue
        if _segments_intersect(a1, a2, b1, b2):
            cnt += 1
    return cnt

def delta_crossings_for_move(G, pos, moved_nodes: Set[str], candidate_pos: Dict[str, Tuple[float, float]], edges_all, edge_bboxes):
    E_changed_set = set()
    for n in moved_nodes:
        for nbr in G[n]:
            e = tuple(sorted((n, nbr)))
            E_changed_set.add(e)
    if not E_changed_set:
        return 0
    E_changed = list(E_changed_set)
    E_other = [e for e in edges_all if e not in E_changed_set]

    old_count = 0
    for e in E_changed:
        old_count += edge_cross_count_against_list(e, E_other, pos, edge_bboxes)
    for i in range(len(E_changed)):
        for j in range(i + 1, len(E_changed)):
            a = E_changed[i]; b = E_changed[j]
            if _segments_intersect(pos[a[0]], pos[a[1]], pos[b[0]], pos[b[1]]):
                old_count += 1

    newpos = dict(pos)
    for n, p in candidate_pos.items():
        newpos[n] = p

    new_edge_bboxes = {}
    for e in E_changed:
        new_edge_bboxes[e] = seg_bbox(newpos[e[0]], newpos[e[1]])

    new_count = 0
    for e in E_changed:
        a1 = newpos[e[0]]; a2 = newpos[e[1]]
        aminx, aminy, amaxx, amaxy = new_edge_bboxes[e]
        for oth in E_other:
            b1 = newpos[oth[0]]; b2 = newpos[oth[1]]
            bminx, bminy, bmaxx, bmaxy = edge_bboxes[oth]
            if not bbox_overlap(aminx, aminy, amaxx, amaxy, bminx, bminy, bmaxx, bmaxy):
                continue
            if _segments_intersect(a1, a2, b1, b2):
                new_count += 1
    for i in range(len(E_changed)):
        for j in range(i + 1, len(E_changed)):
            a = E_changed[i]; b = E_changed[j]
            if _segments_intersect(newpos[a[0]], newpos[a[1]], newpos[b[0]], newpos[b[1]]):
                new_count += 1

    return new_count - old_count

# ---------------------------
# Fast sampled crossing estimates
# ---------------------------
def _sample_crossing_pair(edges_all, rng, changed_edges=None):
    """Sample a pair of edges.

    If changed_edges is supplied, sample from pairs involving at least one changed edge.
    Otherwise sample from all unordered edge pairs.
    """
    if changed_edges:
        changed_edges = list(changed_edges)
        cset = set(changed_edges)
        others = [e for e in edges_all if e not in cset]
        n_c = len(changed_edges)
        n_o = len(others)
        cc_total = n_c * (n_c - 1) // 2
        co_total = n_c * n_o
        total = cc_total + co_total
        if total <= 0:
            return None
        k = rng.randrange(total)
        if k < cc_total:
            i = rng.randrange(n_c)
            j = rng.randrange(n_c - 1)
            if j >= i:
                j += 1
            return changed_edges[i], changed_edges[j]
        i = rng.randrange(n_c)
        j = rng.randrange(n_o)
        return changed_edges[i], others[j]
    else:
        n = len(edges_all)
        if n < 2:
            return None
        i = rng.randrange(n)
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        return edges_all[i], edges_all[j]

def estimate_crossings_sampled(G, pos, edges_all=None, sample_size=4000, seed=42):
    """Estimate the total number of crossings by Monte Carlo sampling of edge pairs."""
    if edges_all is None:
        edges_all = [tuple(sorted((u, v))) for u, v in G.edges()]
    if len(edges_all) < 2:
        return 0
    total_pairs = len(edges_all) * (len(edges_all) - 1) // 2
    sample_size = max(1, min(int(sample_size), total_pairs))
    rng = random.Random(seed)
    hits = 0
    trials = 0
    while trials < sample_size:
        pair = _sample_crossing_pair(edges_all, rng)
        if pair is None:
            break
        e1, e2 = pair
        if set(e1) & set(e2):
            continue
        if _segments_intersect(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]]):
            hits += 1
        trials += 1
    if trials == 0:
        return 0
    return int(round((hits / float(trials)) * total_pairs))

def estimate_delta_crossings_for_move_sampled(G, pos, moved_nodes: Set[str], candidate_pos: Dict[str, Tuple[float, float]], edges_all, edge_bboxes, sample_size=4000, seed=42):
    """Approximate crossing delta for a move by sampling affected edge pairs."""
    E_changed_set = set()
    for n in moved_nodes:
        for nbr in G[n]:
            E_changed_set.add(tuple(sorted((n, nbr))))
    if not E_changed_set:
        return 0
    E_changed = list(E_changed_set)
    E_other = [e for e in edges_all if e not in E_changed_set]
    old_total = len(E_changed) * len(E_other) + (len(E_changed) * (len(E_changed) - 1) // 2)
    if old_total <= 0:
        return 0
    sample_size = max(1, min(int(sample_size), old_total))
    rng = random.Random(seed)

    newpos = dict(pos)
    newpos.update(candidate_pos)

    old_hits = 0
    new_hits = 0
    trials = 0
    while trials < sample_size:
        pair = _sample_crossing_pair(edges_all, rng, changed_edges=E_changed)
        if pair is None:
            break
        e1, e2 = pair
        if set(e1) & set(e2):
            continue
        old_cross = _segments_intersect(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]])
        new_cross = _segments_intersect(newpos[e1[0]], newpos[e1[1]], newpos[e2[0]], newpos[e2[1]])
        old_hits += 1 if old_cross else 0
        new_hits += 1 if new_cross else 0
        trials += 1
    if trials == 0:
        return 0
    scale = old_total / float(trials)
    return int(round((new_hits - old_hits) * scale))

def get_crossing_delta_fn(fast_mode: bool, sample_size: int, seed: int):
    if not fast_mode:
        return None
    def _delta(G, pos, moved_nodes, candidate_pos, edges_all, edge_bboxes, _seed_offset=0):
        return estimate_delta_crossings_for_move_sampled(
            G, pos, moved_nodes, candidate_pos, edges_all, edge_bboxes,
            sample_size=sample_size, seed=seed + _seed_offset)
    return _delta

def get_crossing_score_fn(fast_mode: bool, sample_size: int, seed: int):
    if not fast_mode:
        return None
    def _score(G, pos, edges_all, _seed_offset=0):
        return estimate_crossings_sampled(G, pos, edges_all=edges_all, sample_size=sample_size, seed=seed + _seed_offset)
    return _score

# Global crossing policy used to switch between exact and sampled counting.
CROSSING_POLICY = None

class CrossingPolicy:
    def __init__(self, mode: str = "exact", sample_size: int = 4000, seed: int = 42, exact_every: int = 1):
        self.mode = mode
        self.sample_size = int(sample_size)
        self.seed = int(seed)
        self.exact_every = max(1, int(exact_every))

    def use_sampled_by_default(self) -> bool:
        return self.mode in {"estimate", "cycle_end", "step_interval"}

    def sampled_score(self, G, pos, edges_all=None, seed_offset: int = 0):
        return estimate_crossings_sampled(
            G,
            pos,
            edges_all=edges_all,
            sample_size=self.sample_size,
            seed=self.seed + seed_offset,
        )


def set_crossing_policy(mode: str, sample_size: int, seed: int, exact_every: int = 1):
    global CROSSING_POLICY
    CROSSING_POLICY = CrossingPolicy(mode=mode, sample_size=sample_size, seed=seed, exact_every=exact_every)
    return CROSSING_POLICY

# ---------------------------
# Optimization passes
# ---------------------------
def optimize_chiral_pair_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                               max_iters=500, seed=42, tries_per_iter=None, verbose=True,
                               fast_mode=False, sample_size=4000):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    pair_side_map = {}
    for pid, (A, B) in pairs.items():
        a_side = sides.get(A, ''); b_side = sides.get(B, '')
        if a_side == 'left' and b_side == 'right':
            left, right = A, B
        elif b_side == 'left' and a_side == 'right':
            left, right = B, A
        else:
            if pos.get(A, (0.0, 0.0))[0] <= pos.get(B, (0.0, 0.0))[0]:
                left, right = A, B
            else:
                left, right = B, A
        pair_side_map[pid] = (left, right)

    candidates = [pid for pid in pair_ids if not (sides.get(pairs[pid][0], '') == 'axis' and sides.get(pairs[pid][1], '') == 'axis')]
    if not candidates:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(candidates)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    best_cross = before_cross
    improved_any = False
    it = 0
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [chiral-y-swaps] candidates={len(candidates)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    while it < max_iters:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            a, b = rng.sample(candidates, 2)
            if a == b:
                continue
            aL, aR = pair_side_map[a]; bL, bR = pair_side_map[b]
            candidate_pos = {
                aL: (best_pos[aL][0], best_pos[bL][1]),
                aR: (best_pos[aR][0], best_pos[bL][1]),
                bL: (best_pos[bL][0], best_pos[aL][1]),
                bR: (best_pos[bR][0], best_pos[aL][1]),
            }
            moved = {aL, aR, bL, bR}
            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
            if delta < 0:
                for n, p in candidate_pos.items():
                    best_pos[n] = p
                current_cross += delta
                best_cross = current_cross
                improved = True
                improved_any = True
                if verbose:
                    print(f"    [chiral-y-swaps] iter={it} applied swap {a}<->{b} delta={delta} new_cross={current_cross}")
                break
        if not improved:
            if verbose:
                print(f"    [chiral-y-swaps] iter={it} no improvement, stopping")
            break
    return best_pos, improved_any, before_cross, best_cross

def optimize_pair_of_pairs_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                                 max_iters=200, seed=42, attempts_per_iter=100, verbose=True,
                                 fast_mode=False, sample_size=4000):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    pair_side_map = {}
    for pid, (A, B) in pairs.items():
        a_side = sides.get(A, ''); b_side = sides.get(B, '')
        if a_side == 'left' and b_side == 'right':
            left, right = A, B
        elif b_side == 'left' and a_side == 'right':
            left, right = B, A
        else:
            if pos.get(A, (0.0, 0.0))[0] <= pos.get(B, (0.0, 0.0))[0]:
                left, right = A, B
            else:
                left, right = B, A
        pair_side_map[pid] = (left, right)

    candidates = [pid for pid in pair_ids if not (sides.get(pairs[pid][0], '') == 'axis' and sides.get(pairs[pid][1], '') == 'axis')]
    if len(candidates) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    best_cross = before_cross
    improved_any = False
    it = 0
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [pair-of-pairs] candidates={len(candidates)} attempts_per_iter={attempts_per_iter} max_iters={max_iters} before_cross={before_cross}")

    while it < max_iters:
        it += 1
        changed = False
        for _ in range(attempts_per_iter):
            a, b = rng.sample(candidates, 2)
            if a == b:
                continue
            aL, aR = pair_side_map[a]; bL, bR = pair_side_map[b]
            candidate_pos = {
                aL: best_pos[bL],
                aR: best_pos[bR],
                bL: best_pos[aL],
                bR: best_pos[aR],
            }
            moved = {aL, aR, bL, bR}
            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
            if delta < 0:
                for n, p in candidate_pos.items():
                    best_pos[n] = p
                current_cross += delta
                best_cross = current_cross
                improved_any = True
                changed = True
                if verbose:
                    print(f"    [pair-of-pairs] iter={it} applied swap {a} <-> {b} delta={delta} new_cross={current_cross}")
                break
        if not changed:
            if verbose:
                print(f"    [pair-of-pairs] iter={it} no change, stopping")
            break
    return best_pos, improved_any, before_cross, best_cross

def optimize_chiral_pair_flips(G, pos, pairs, sides, edges_all, edge_bboxes,
                               max_iters=500, seed=42, tries_per_iter=None, verbose=True,
                               fast_mode=False, sample_size=4000):
    """
    Try flipping the two members of a chiral pair across the mirror plane.
    This swaps the positions of the enantiomeric partners within the same pair.
    """
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 1:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(pair_ids)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    best_cross = before_cross
    improved_any = False
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [pair-flips] candidates={len(pair_ids)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    it = 0
    while it < max_iters:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            pid = rng.choice(pair_ids)
            A, B = pairs[pid]

            # Flip the pair: swap the two node positions
            candidate_pos = {
                A: best_pos[B],
                B: best_pos[A],
            }
            moved = {A, B}

            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)

            if delta < 0:
                best_pos[A] = candidate_pos[A]
                best_pos[B] = candidate_pos[B]
                current_cross += delta
                best_cross = current_cross
                improved = True
                improved_any = True
                if verbose:
                    print(f"    [pair-flips] iter={it} flipped {pid} delta={delta} new_cross={current_cross}")
                break

        if not improved:
            if verbose:
                print(f"    [pair-flips] iter={it} no improvement, stopping")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, improved_any, before_cross, after_cross

def optimize_enantiomer_x_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                                max_iters=100, seed=42, tries_per_iter=None, verbose=True,
                                fast_mode=False, sample_size=4000):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    candidates = []
    for pid in pair_ids:
        A, B = pairs[pid]
        if A not in pos or B not in pos:
            continue
        if sides.get(A, "") == "axis" and sides.get(B, "") == "axis":
            continue
        candidates.append(pid)

    if len(candidates) < 2:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(candidates)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    improved_any = False
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [enantiomer-x-swaps] start candidates={len(candidates)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    it = 0
    while it < max_iters:
        it += 1
        made_progress = False
        for _ in range(tries_per_iter):
            a_pid, b_pid = rng.sample(candidates, 2)
            if a_pid == b_pid:
                continue
            aA, aB = pairs[a_pid]
            bA, bB = pairs[b_pid]

            def left_member(pair):
                n1, n2 = pair
                if best_pos[n1][0] < best_pos[n2][0]:
                    return n1, n2
                else:
                    return n2, n1

            a_left, a_right = left_member((aA, aB))
            b_left, b_right = left_member((bA, bB))

            ax_mag = abs(best_pos[a_left][0])
            bx_mag = abs(best_pos[b_left][0])

            if abs(ax_mag - bx_mag) < 1e-12:
                continue

            sign_a_left = 1.0 if best_pos[a_left][0] >= 0 else -1.0
            sign_b_left = 1.0 if best_pos[b_left][0] >= 0 else -1.0

            cand = {
                a_left: (sign_a_left * bx_mag, best_pos[a_left][1]),
                a_right: (-sign_a_left * bx_mag, best_pos[a_right][1]),
                b_left: (sign_b_left * ax_mag, best_pos[b_left][1]),
                b_right: (-sign_b_left * ax_mag, best_pos[b_right][1]),
            }

            moved_nodes = set(cand.keys())
            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved_nodes, cand, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved_nodes, cand, edges_all, edge_bboxes)
            if delta < 0:
                for n, p in cand.items():
                    best_pos[n] = p
                current_cross += delta
                made_progress = True
                improved_any = True
                if verbose:
                    print(f"    [enantiomer-x-swaps] iter={it} accepted swap {a_pid}<->{b_pid} delta={delta} crossings={current_cross}")
                break
        if not made_progress:
            if verbose:
                print(f"    [enantiomer-x-swaps] iter={it} no improvement; stopping early")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, improved_any, before_cross, after_cross

def optimize_achiral_swaps(G, pos, pairs, sides, comp_orderings, achiral_gap,
                           edges_all, edge_bboxes,
                           max_iters=1000, seed=42,
                           cross_weight=1000.0, gap_weight=1.0, nonadj_weight=50.0, verbose=True,
                           tries_per_iter_override=None, fast_mode=False, sample_size=4000):
    start_time = time.time()
    node_to_slot = {}
    base_y_by_comp = {}
    for comp_id, ordering in comp_orderings.items():
        if not ordering:
            continue
        ys = [pos[n][1] for n in ordering]
        center_y = float(np.median(ys))
        base_y = center_y - ((len(ordering) - 1) * achiral_gap / 2.0)
        base_y_by_comp[comp_id] = base_y
        for idx, n in enumerate(ordering):
            node_to_slot[n] = (comp_id, idx)
            pos[n] = (0.0, base_y + idx * achiral_gap)

    candidates = list(node_to_slot.keys())
    if len(candidates) < 2:
        return pos, count_edge_crossings(G, pos)

    def evaluate_spacing_and_nonadj(pos_local, node_to_slot_local):
        axis_nodes = [n for n in pos_local.keys() if sides.get(n, '') == 'axis']
        ys = sorted((pos_local[n][1] for n in axis_nodes))
        if len(ys) <= 1:
            gap_var = 0.0
        else:
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            gap_var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        nonadj = 0
        for u, v in G.edges():
            if sides.get(u, '') != 'axis' or sides.get(v, '') != 'axis':
                continue
            if u not in node_to_slot_local or v not in node_to_slot_local:
                continue
            cu, iu = node_to_slot_local[u]
            cv, iv = node_to_slot_local[v]
            if cu != cv:
                continue
            if abs(iu - iv) != 1:
                nonadj += 1
        return gap_var, nonadj

    def combined_score_approx(crossings, pos_local, node_to_slot_local):
        gap_var, nonadj = evaluate_spacing_and_nonadj(pos_local, node_to_slot_local)
        return cross_weight * crossings + gap_weight * gap_var + nonadj_weight * nonadj

    best_pos = dict(pos)
    best_node_to_slot = dict(node_to_slot)
    if fast_mode:
        current_cross = estimate_crossings_sampled(G, best_pos, edges_all=edges_all, sample_size=sample_size, seed=seed)
    else:
        current_cross = count_edge_crossings(G, best_pos)
    best_score = combined_score_approx(current_cross, best_pos, best_node_to_slot)
    rng = random.Random(seed)
    it = 0
    improved = True
    tries_per_iter = tries_per_iter_override if tries_per_iter_override is not None else min(200, max(20, len(candidates)))
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [achiral-swaps] starting score={best_score:.3f} crossings={current_cross} candidates={len(candidates)} tries_per_iter={tries_per_iter}")

    while it < max_iters and improved:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            a, b = rng.sample(candidates, 2)
            ca, ia = best_node_to_slot[a]; cb, ib = best_node_to_slot[b]
            candidate_pos = {
                a: (0.0, base_y_by_comp[cb] + ib * achiral_gap),
                b: (0.0, base_y_by_comp[ca] + ia * achiral_gap),
            }
            moved = {a, b}
            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
            new_cross = current_cross + delta
            new_pos_temp = dict(best_pos)
            new_pos_temp[a] = candidate_pos[a]; new_pos_temp[b] = candidate_pos[b]
            new_score = combined_score_approx(new_cross, new_pos_temp, {**best_node_to_slot, a: (cb, ib), b: (ca, ia)})
            if new_score < best_score:
                best_score = new_score
                best_pos = new_pos_temp
                best_node_to_slot = dict(best_node_to_slot)
                best_node_to_slot[a] = (cb, ib)
                best_node_to_slot[b] = (ca, ia)
                current_cross = new_cross
                improved = True
                if verbose:
                    print(f"    [achiral-swaps] iter={it} accepted swap {a}<->{b} delta_cross={delta} crossings={current_cross} score={best_score:.3f}")
                break
        if verbose and not improved:
            elapsed = time.time() - start_time
            print(f"    [achiral-swaps] iter={it} no improvement (elapsed={elapsed:.1f}s) current_cross={current_cross}")
    for n, (comp_id, idx) in best_node_to_slot.items():
        base = base_y_by_comp.get(comp_id, None)
        if base is not None:
            best_pos[n] = (0.0, base + idx * achiral_gap)
    return best_pos, current_cross

def _median_edge_length_from_pos(pos, edges):
    lengths = []
    for u, v in edges:
        if u in pos and v in pos:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            lengths.append(math.hypot(float(x1) - float(x2), float(y1) - float(y2)))
    return float(np.median(lengths)) if lengths else 0.0


def _build_spatial_hash(pos, cell_size):
    cell_size = max(float(cell_size), 1e-9)
    inv_cell = 1.0 / cell_size
    grid = defaultdict(list)
    for n, (x, y) in pos.items():
        key = (int(math.floor(float(x) * inv_cell)), int(math.floor(float(y) * inv_cell)))
        grid[key].append(n)
    return grid, inv_cell


def _local_edge_length_penalty(best_pos, candidate_pos, moved_nodes, edges_all, target_length):
    if target_length <= 0.0:
        return 0.0
    moved_set = set(moved_nodes)
    seen = set()
    penalty = 0.0
    for u, v in edges_all:
        if u not in moved_set and v not in moved_set:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        pu = candidate_pos.get(u, best_pos.get(u))
        pv = candidate_pos.get(v, best_pos.get(v))
        if pu is None or pv is None:
            continue
        d = math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
        if d > target_length:
            excess = (d - target_length) / target_length
            penalty += excess * excess
    return penalty


def _local_repulsion_penalty(best_pos, candidate_pos, moved_nodes, grid, inv_cell, repulse_dist):
    if repulse_dist <= 0.0:
        return 0.0
    repulse_dist = float(repulse_dist)
    moved_set = set(moved_nodes)
    checked = set()
    penalty = 0.0
    for n in moved_set:
        p = candidate_pos.get(n, best_pos.get(n))
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        ix = int(math.floor(x * inv_cell))
        iy = int(math.floor(y * inv_cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((ix + dx, iy + dy), []):
                    if other == n or other in moved_set:
                        continue
                    key = (n, other) if n <= other else (other, n)
                    if key in checked:
                        continue
                    checked.add(key)
                    op = best_pos.get(other)
                    if op is None:
                        continue
                    d = math.hypot(x - float(op[0]), y - float(op[1]))
                    if d < repulse_dist:
                        deficit = (repulse_dist - d) / repulse_dist
                        penalty += deficit * deficit
    return penalty


def optimize_chiral_coordinate_relaxation(
    G, pos, pairs, sides, edges_all, edge_bboxes,
    max_iters=150, seed=42, tries_per_iter=None,
    x_rate=0.35, y_rate=0.35,
    spacing_weight=0.02, edge_length_weight=0.05, repulsion_weight=0.02,
    min_sep=40.0, edge_target_factor=1.0, min_x=5.0,
    verbose=True, fast_mode=False, sample_size=4000
):
    """
    Mirror-preserving local relaxation for chiral pairs.
    Moves both x and y of a pair together, keeps the two partners mirrored,
    and uses neighbouring pair centres as soft targets.

    The score combines crossing count with a soft neighbour-centre pull,
    a mild penalty for overly long incident edges, and a weak local repulsion
    term to discourage cramped layouts before the hard constraint pass.
    """
    rng = random.Random(seed)
    pair_of = {}
    chiral_pair_ids = []

    for pid, (A, B) in pairs.items():
        pair_of[A] = pid
        pair_of[B] = pid
        if sides.get(A, "") != "axis" or sides.get(B, "") != "axis":
            chiral_pair_ids.append(pid)

    if len(chiral_pair_ids) < 1:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(chiral_pair_ids)))

    def pair_right_left(pid, pmap):
        A, B = pairs[pid]
        a_side = sides.get(A, "")
        b_side = sides.get(B, "")
        if a_side == "right" or b_side == "left":
            return A, B
        if b_side == "right" or a_side == "left":
            return B, A
        return (A, B) if pmap[A][0] >= pmap[B][0] else (B, A)

    def pair_center(pid, pmap):
        A, B = pairs[pid]
        mag = 0.5 * (abs(pmap[A][0]) + abs(pmap[B][0]))
        cy = 0.5 * (pmap[A][1] + pmap[B][1])
        return mag, cy

    def neighbour_targets(pid, pmap):
        neigh = set()
        for n in pairs[pid]:
            for nbr in G[n]:
                q = pair_of.get(nbr)
                if q is not None and q != pid:
                    neigh.add(q)
        if not neigh:
            return None
        mags = []
        ys = []
        for q in neigh:
            A, B = pairs[q]
            mags.append(0.5 * (abs(pmap[A][0]) + abs(pmap[B][0])))
            ys.append(0.5 * (pmap[A][1] + pmap[B][1]))
        return float(np.median(mags)), float(np.median(ys))

    current_pos = dict(pos)
    edge_target_length = _median_edge_length_from_pos(current_pos, edges_all)
    if edge_target_length <= 1e-9:
        edge_target_length = max(1.0, 0.75 * float(min_sep))
    edge_target_length *= float(edge_target_factor)
    repulse_dist = max(1e-9, float(min_sep))

    if fast_mode:
        current_cross = estimate_crossings_sampled(
            G, current_pos, edges_all=edges_all,
            sample_size=sample_size, seed=seed
        )
    else:
        current_cross = count_edge_crossings(G, current_pos)

    best_pos = dict(current_pos)
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(
            f"  [chiral-relax] start pairs={len(chiral_pair_ids)} tries_per_iter={tries_per_iter} "
            f"max_iters={max_iters} crossings={current_cross} edge_target={edge_target_length:.3f} "
            f"edge_w={edge_length_weight:.3f} repulse_w={repulsion_weight:.3f}"
        )

    for it in range(1, max_iters + 1):
        improved = False
        spatial_grid, inv_cell = _build_spatial_hash(best_pos, repulse_dist)
        for _ in range(tries_per_iter):
            pid = rng.choice(chiral_pair_ids)
            target = neighbour_targets(pid, best_pos)
            if target is None:
                continue

            target_mag, target_y = target
            cur_mag, cur_y = pair_center(pid, best_pos)

            new_mag = max(min_x, cur_mag + x_rate * (target_mag - cur_mag))
            new_y = cur_y + y_rate * (target_y - cur_y)

            right, left = pair_right_left(pid, best_pos)
            candidate_pos = {
                right: (abs(new_mag), new_y),
                left: (-abs(new_mag), new_y),
            }

            moved = {right, left}
            current_edge_pen = _local_edge_length_penalty(best_pos, {}, moved, edges_all, edge_target_length)
            current_repulse_pen = _local_repulsion_penalty(best_pos, {}, moved, spatial_grid, inv_cell, repulse_dist)
            current_spacing = abs(cur_mag - target_mag) + abs(cur_y - target_y)
            current_score = (
                float(current_cross)
                + spacing_weight * current_spacing
                + edge_length_weight * current_edge_pen
                + repulsion_weight * current_repulse_pen
            )

            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes,
                                 _seed_offset=it)
                new_cross = current_cross + delta
            else:
                new_cross = count_edge_crossings(G, {**best_pos, **candidate_pos})

            new_edge_pen = _local_edge_length_penalty(best_pos, candidate_pos, moved, edges_all, edge_target_length)
            new_repulse_pen = _local_repulsion_penalty(best_pos, candidate_pos, moved, spatial_grid, inv_cell, repulse_dist)
            new_spacing = abs(new_mag - target_mag) + abs(new_y - target_y)
            new_score = (
                float(new_cross)
                + spacing_weight * new_spacing
                + edge_length_weight * new_edge_pen
                + repulsion_weight * new_repulse_pen
            )

            if new_score < current_score:
                best_pos = dict(best_pos)
                best_pos.update(candidate_pos)
                current_cross = new_cross
                improved = True
                if verbose:
                    print(
                        f"    [chiral-relax] iter={it} moved {pid} "
                        f"crossings={current_cross} score={new_score:.3f} "
                        f"edge_pen={new_edge_pen:.3f} repulse_pen={new_repulse_pen:.3f}"
                    )
                break

        if not improved:
            if verbose:
                print(f"    [chiral-relax] iter={it} no improvement, stopping")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, (after_cross < count_edge_crossings(G, pos)), count_edge_crossings(G, pos), after_cross

def max_node_shift(pos_a, pos_b):
    common = [n for n in pos_a.keys() if n in pos_b]
    if not common:
        return 0.0
    return max(
        math.hypot(float(pos_a[n][0]) - float(pos_b[n][0]), float(pos_a[n][1]) - float(pos_b[n][1]))
        for n in common
    )

def pair_block_relocation_refinement(G, pos, sides, comp_orderings, achiral_gap,
                                     edges_all, edge_bboxes,
                                     max_trials=200, seed=42, verbose=True,
                                     fast_mode=False, sample_size=4000):
    rng = random.Random(seed)
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    axis_sub = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(axis_sub))
    node2comp = {}
    comp_orders = {}
    for cid, comp in enumerate(comps):
        comp_set = set(comp)
        ordering = None
        for k, ordlist in comp_orderings.items():
            if set(ordlist) == comp_set:
                ordering = list(ordlist)
                break
        if ordering is None:
            ordering = sorted(list(comp), key=lambda n: pos[n][1])
        comp_orders[cid] = ordering
        for n in ordering:
            node2comp[n] = cid

    def build_pos_from_comp_orders(base_pos, comp_orders_local, gap):
        newp = dict(base_pos)
        comp_info = []
        for cid, ordering in comp_orders_local.items():
            med = np.median([base_pos[n][1] for n in ordering]) if ordering else 0.0
            comp_info.append((cid, ordering, med))
        comp_info.sort(key=lambda x: -x[2])
        y_cursor = None
        for cid, ordering, _ in comp_info:
            m = len(ordering)
            height = (m - 1) * gap if m > 0 else 0.0
            if y_cursor is None:
                center = np.median([base_pos[n][1] for n in ordering]) if m > 0 else 0.0
                start_y = center + height / 2.0
            else:
                start_y = y_cursor - gap
            for i, n in enumerate(ordering):
                newp[n] = (0.0, start_y - i * gap)
            bottom = start_y - (m - 1) * gap
            y_cursor = bottom - gap
        return newp

    baseline_pos = dict(pos)
    if fast_mode:
        baseline_cross = estimate_crossings_sampled(G, baseline_pos, edges_all=edges_all, sample_size=sample_size, seed=seed)
    else:
        baseline_cross = count_edge_crossings(G, baseline_pos)
    best_pos = dict(baseline_pos)
    best_cross = baseline_cross
    trials = 0
    improved_any = False
    score_fn = get_crossing_score_fn(fast_mode, sample_size, seed)

    candidate_pairs = []
    for u, v in G.edges():
        if sides.get(u, '') != 'axis' or sides.get(v, '') != 'axis':
            continue
        if node2comp.get(u, None) is None or node2comp.get(v, None) is None:
            continue
        if node2comp[u] != node2comp[v]:
            continue
        cid = node2comp[u]
        ordlist = comp_orders[cid]
        try:
            iu = ordlist.index(u); iv = ordlist.index(v)
        except ValueError:
            continue
        if abs(iu - iv) == 1:
            i0 = min(iu, iv)
            a = ordlist[i0]; b = ordlist[i0 + 1]
            candidate_pairs.append((cid, i0, (a, b)))
    seen_pairs = set()
    uniq_candidates = []
    for cid, i0, pair in candidate_pairs:
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        uniq_candidates.append((cid, i0, pair))
    rng.shuffle(uniq_candidates)

    if verbose:
        print(f"  [pair-block-relocation] candidates={len(uniq_candidates)} max_trials={max_trials}")

    for cid, i0, pair in uniq_candidates:
        if trials >= max_trials:
            break
        ordering = list(comp_orders[cid])
        a, b = pair
        try:
            idx_a = ordering.index(a); idx_b = ordering.index(b)
        except ValueError:
            continue
        if abs(idx_a - idx_b) != 1:
            continue
        if idx_b < idx_a:
            a, b = b, a
            idx_a, idx_b = idx_b, idx_a
        base_ordering = ordering[:idx_a] + ordering[idx_b + 1:]
        insertion_positions = list(range(0, len(base_ordering) + 1))
        rng.shuffle(insertion_positions)
        original_insert = idx_a
        for ins in insertion_positions:
            trials += 1
            if trials > max_trials:
                break
            if ins == original_insert:
                continue
            new_order = base_ordering[:ins] + [a, b] + base_ordering[ins:]
            comp_orders_candidate = dict(comp_orders)
            comp_orders_candidate[cid] = new_order
            cand_pos = build_pos_from_comp_orders(baseline_pos, comp_orders_candidate, achiral_gap)
            if score_fn is not None:
                cand_cross = score_fn(G, cand_pos, edges_all, _seed_offset=trials)
            else:
                cand_cross = count_edge_crossings(G, cand_pos)
            if cand_cross < best_cross:
                if verbose:
                    print(f"    [pair-block] improved: {best_cross} -> {cand_cross} (moved pair {pair} to insertion {ins})")
                best_cross = cand_cross
                best_pos = dict(cand_pos)
                comp_orders = dict(comp_orders_candidate)
                baseline_pos = dict(cand_pos)
                improved_any = True
                break
    return best_pos, improved_any, baseline_cross, best_cross

# ---------------------------
# Final mirror & GEXF writer
# ---------------------------
def enforce_strict_mirror(pos, pairs, sides, tol=1e-8):
    pos2 = dict(pos)
    changed = False
    for pid, (A, B) in pairs.items():
        yA = pos2.get(A, (0.0, 0.0))[1]
        yB = pos2.get(B, (0.0, 0.0))[1]
        yavg = 0.5 * (yA + yB)
        xA = pos2.get(A, (0.0, 0.0))[0]
        xB = pos2.get(B, (0.0, 0.0))[0]
        mag = 0.5 * (abs(xA) + abs(xB))
        signA = 1.0 if xA >= 0 else -1.0
        if abs(xA) < tol and abs(xB) < tol:
            signA = 1.0
        posA_new = (signA * mag, yavg)
        posB_new = (-signA * mag, yavg)
        if (abs(pos2[A][0] - posA_new[0]) > tol) or (abs(pos2[A][1] - posA_new[1]) > tol):
            pos2[A] = posA_new
            changed = True
        if (abs(pos2[B][0] - posB_new[0]) > tol) or (abs(pos2[B][1] - posB_new[1]) > tol):
            pos2[B] = posB_new
            changed = True
    for n, s in sides.items():
        if s == 'axis':
            x, y = pos2.get(n, (0.0, 0.0))
            if abs(x) > tol:
                pos2[n] = (0.0, y)
                changed = True
    return pos2, changed

def write_gexf_with_viz(G, pos, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">\n')
        f.write('  <graph mode="static" defaultedgetype="undirected">\n')
        f.write('    <attributes class="node">\n')
        f.write('      <attribute id="0" title="Energy" type="float"/>\n')
        f.write('      <attribute id="1" title="DeltaE" type="float"/>\n')
        f.write('      <attribute id="2" title="Chirality" type="string"/>\n')
        f.write('      <attribute id="3" title="Enantiomer_Id" type="string"/>\n')
        f.write('    </attributes>\n')
        f.write('    <attributes class="edge">\n')
        f.write('      <attribute id="10" title="TS_Energy" type="float"/>\n')
        f.write('    </attributes>\n')
        f.write('    <nodes>\n')
        for n, data in G.nodes(data=True):
            x, y = pos.get(n, (0.0, 0.0))
            f.write(f'      <node id="{n}" label="{n}">\n')
            f.write('        <attvalues>\n')
            f.write(f'          <attvalue for="0" value="{float(data.get("Energy", 0.0))}"/>\n')
            f.write(f'          <attvalue for="1" value="{float(data.get("DeltaE", 0.0))}"/>\n')
            f.write(f'          <attvalue for="2" value="{data.get("Chirality", "")}"/>\n')
            f.write(f'          <attvalue for="3" value="{data.get("Enantiomer_Id", "")}"/>\n')
            f.write('        </attvalues>\n')
            f.write(f'        <viz:position x="{x}" y="{y}" z="0"/>\n')
            f.write('      </node>\n')
        f.write('    </nodes>\n')
        f.write('    <edges>\n')
        i = 0
        for u, v, data in G.edges(data=True):
            f.write(f'      <edge id="{i}" source="{u}" target="{v}">\n')
            f.write(f'        <attvalues><attvalue for="10" value="{float(data.get("TS_Energy", 0.0))}"/></attvalues>\n')
            f.write('      </edge>\n')
            i += 1
        f.write('    </edges>\n')
        f.write('  </graph>\n')
        f.write('</gexf>\n')

def load_positions_from_gexf(gexf_path, nodes=None):
    """Load node positions from a Gephi/GEXF file written by write_gexf_with_viz()."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(gexf_path)
    root = tree.getroot()
    ns = {
        "g": "http://www.gexf.net/1.2draft",
        "viz": "http://www.gexf.net/1.2draft/viz",
    }
    pos = {}
    for node in root.findall('.//g:node', ns):
        node_id = node.get('id')
        pos_el = node.find('viz:position', ns)
        if pos_el is None or node_id is None:
            continue
        try:
            x = float(pos_el.get('x', '0.0'))
            y = float(pos_el.get('y', '0.0'))
        except (TypeError, ValueError):
            continue
        pos[node_id] = (x, y)

    if nodes is not None:
        # Ensure every graph node has a coordinate entry.
        for n in nodes:
            pos.setdefault(n, (0.0, 0.0))
    return pos

# ---------------------------
# Main orchestration
# ---------------------------

# ---------------------------
# Cytoscape XGMML writer and CSV exports
# ---------------------------
import xml.sax.saxutils as saxutils
def write_xgmml_with_viz(G, pos, out_path, graph_label="BVGraph"):
    """
    Write an XGMML file suitable for Cytoscape import that includes node attributes
    and <graphics> position elements. Coordinates are written as 'x' and 'y'
    attributes inside each node's <graphics> subelement.
    """
    def attr_xml(attr_name, attr_value, attr_type="string"):
        val = saxutils.escape(str(attr_value))
        return f'    <att name="{saxutils.escape(attr_name)}" value="{val}" type="{attr_type}"/>\n'

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\\n')
        f.write(f'<graph label="{saxutils.escape(graph_label)}" directed="0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.cs.rpi.edu/XGMML">\\n')
        # nodes
        for n, data in G.nodes(data=True):
            x, y = pos.get(n, (0.0, 0.0))
            nid = saxutils.escape(str(n))
            f.write(f'  <node id="{nid}" label="{nid}">\\n')
            # primary attributes (Energy, DeltaE, Chirality, Enantiomer_Id, Barcode_Normalized)
            f.write(attr_xml("Energy", float(data.get("Energy", 0.0)), "real"))
            f.write(attr_xml("DeltaE", float(data.get("DeltaE", 0.0)), "real"))
            f.write(attr_xml("Chirality", data.get("Chirality", ""), "string"))
            f.write(attr_xml("Enantiomer_Id", data.get("Enantiomer_Id", ""), "string"))
            f.write(attr_xml("Barcode_Normalized", data.get("Barcode_Normalized", ""), "string"))
            # add graphics element with coordinates
            f.write(f'    <graphics x="{float(x)}" y="{float(y)}"/>\\n')
            f.write('  </node>\\n')
        # edges
        edge_id = 0
        for u, v, data in G.edges(data=True):
            uid = saxutils.escape(str(u))
            vid = saxutils.escape(str(v))
            f.write(f'  <edge id="e{edge_id}" label="e{edge_id}" source="{uid}" target="{vid}">\\n')
            f.write(attr_xml("TS_Energy", float(data.get("TS_Energy", 0.0)), "real"))
            f.write('  </edge>\\n')
            edge_id += 1
        f.write('</graph>\\n')

def write_nodes_coords_csv(G, pos, out_path, id_col="id"):
    """
    Writes a CSV with columns: id, x, y, Barcode_Normalized, Chirality, Energy, DeltaE, Enantiomer_Id
    """
    import pandas as _pd
    rows = []
    for n, data in G.nodes(data=True):
        x, y = pos.get(n, (0.0, 0.0))
        rows.append({
            id_col: n,
            "x": float(x),
            "y": float(y),
            "Barcode_Normalized": data.get("Barcode_Normalized", ""),
            "Chirality": data.get("Chirality", ""),
            "Energy": float(data.get("Energy", 0.0)),
            "DeltaE": float(data.get("DeltaE", 0.0)),
            "Enantiomer_Id": data.get("Enantiomer_Id", "")
        })
    df = _pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def write_edges_csv(G, out_path):
    """
    Writes a CSV with columns: Source, Target, TS_Energy
    """
    import pandas as _pd
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({"Source": u, "Target": v, "TS_Energy": float(data.get("TS_Energy", 0.0))})
    df = _pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def main():
    ap = argparse.ArgumentParser(description="Create mirror-symmetric layouts with achiral axis and ratio-based span equalisation.")
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--edges", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--annot-out", default=None)
    ap.add_argument("--dump-sides", default=None)
    ap.add_argument("--sweeps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--snap-achiral", type=int, default=1)
    ap.add_argument("--min-sep", type=float, default=60.0)
    ap.add_argument("--edge-clearance", type=float, default=12.0)
    ap.add_argument("--overlap-iter", type=int, default=200)
    ap.add_argument("--achiral-gap", type=float, default=None, help="If not set, will be set equal to --min-sep.")
    ap.add_argument("--swap-iters", type=int, default=1000)
    ap.add_argument("--axis-min-gap", type=float, default=None)
    ap.add_argument("--axis-adjust-iters", type=int, default=6)
    ap.add_argument("--pair-relocation-trials", type=int, default=200)
    ap.add_argument("--axis-lateral-gap", type=float, default=None)
    ap.add_argument("--chiral-swap-iters", type=int, default=500)
    ap.add_argument("--pairpair-iters", type=int, default=100)
    ap.add_argument("--chiral-relax", action="store_true",
                    help="Relax chiral pair coordinates while preserving mirror symmetry.")
    ap.add_argument("--chiral-relax-iters", type=int, default=150)
    ap.add_argument("--chiral-relax-x-rate", type=float, default=0.35)
    ap.add_argument("--chiral-relax-y-rate", type=float, default=0.35)
    ap.add_argument("--chiral-relax-spacing-weight", type=float, default=0.02)
    ap.add_argument("--chiral-relax-edge-length-weight", type=float, default=0.05,
                    help="Weight for the edge-length penalty in chiral coordinate relaxation.")
    ap.add_argument("--chiral-relax-repulsion-weight", type=float, default=0.05,
                    help="Weight for the local repulsion penalty in chiral coordinate relaxation.")
    ap.add_argument("--chiral-relax-edge-target-factor", type=float, default=1.0,
                    help="Multiplier applied to the median incident edge length used as the relaxation target.")
    ap.add_argument("--chiral-relax-min-x", type=float, default=5.0)
    ap.add_argument("--cycle-shift-tol", type=float, default=0.25,
                    help="Stop refinement early when the maximum node displacement over a cycle falls below this tolerance.")
    ap.add_argument("--axis-node-relax", action="store_true",
                    help="Nudge selected axis nodes away from nearby edges before final export.")
    ap.add_argument("--axis-node-relax-id", default=None,
                    help="Comma-separated axis node IDs to relax. If omitted, all axis nodes are considered.")
    ap.add_argument("--axis-node-relax-clearance", type=float, default=25.0,
                    help="Target edge clearance for axis-node relaxation.")
    ap.add_argument("--axis-node-relax-shift", type=float, default=200.0,
                    help="Maximum vertical shift allowed during axis-node relaxation.")
    ap.add_argument("--axis-node-relax-step", type=float, default=5.0,
                    help="Vertical step size used when searching for a better axis-node position.")
    ap.add_argument("--refine-cycles", type=int, default=1000)
    ap.add_argument("--achiral-adjacency-weight", type=float, default=50.0)
    ap.add_argument("--achiral-gap-weight", type=float, default=1.0)
    ap.add_argument("--hemi-span-tol", type=float, default=0.10, help="Tolerance for hemisphere/axis span equilisation.")
    ap.add_argument("--hemi-max-scale", type=float, default=1.5, help="Maximum per-call scale factor for equalize_hemi_span.")
    ap.add_argument("--hemi-min-nodes", type=int, default=2, help="Minimum # chiral nodes required to attempt hemi equalisation.")
    ap.add_argument("--hemi-target-ratio", type=float, default=1.0, help="Desired axis_span / chiral_span ratio (default 1.0).")
    ap.add_argument("--enantiomer-swap-iters", type=int, default=100, help="Attempts per-cycle for enantiomer X-swaps (default 100).")
    ap.add_argument(
        "--crossing-mode",
        choices=["exact", "estimate", "cycle_end", "step_interval"],
        default=None,
        help=(
            "Crossing-count strategy: exact for every evaluation; estimate for all evaluations; "
            "cycle_end for sampled steps plus exact counts at the end of each cycle; "
            "step_interval for sampled steps plus exact counts every N optimisation steps."
        ),
    )
    ap.add_argument(
        "--crossing-exact-every",
        type=int,
        default=5,
        help="When crossing-mode is step_interval, force an exact crossing count every N optimisation steps.",
    )
    ap.add_argument("--fast-mode", action="store_true", help="Backward-compatible alias for --crossing-mode estimate.")
    ap.add_argument("--fast-sample-size", type=int, default=4000, help="Random edge-pair samples used per fast estimate.")
    ap.add_argument("--checkpoint-gephi", default=None, help="Optional temporary GEXF checkpoint file to overwrite during refinement.")
    ap.add_argument("--checkpoint-every", type=int, default=5, help="Write checkpoint every N completed refinement cycles.")
    ap.add_argument("--resume-gephi", default=None, help="Resume starting coordinates from a temporary GEXF checkpoint file.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-xgmml", default=None, help="Optional output XGMML file path (Cytoscape).")
    ap.add_argument("--out-nodes-csv", default=None, help="Optional output CSV with node coords & barcodes.")
    ap.add_argument("--out-edges-csv", default=None, help="Optional output CSV with edge data.")
    args = ap.parse_args()

    verbose = args.verbose
    if args.crossing_mode is None:
        args.crossing_mode = "estimate" if args.fast_mode else "exact"
    elif args.fast_mode and args.crossing_mode != "estimate":
        print("[WARN] --fast-mode was supplied, but --crossing-mode takes precedence.", flush=True)

    if args.axis_lateral_gap is None:
        args.axis_lateral_gap = max(1.2 * args.min_sep, 1.0)

    # The optimisation steps use sampled delta checks whenever exact counting is not required everywhere.
    args.fast_mode = args.crossing_mode != "exact"
    set_crossing_policy(args.crossing_mode, args.fast_sample_size, args.seed, exact_every=args.crossing_exact_every)

    print("[START] Reading graph", flush=True)
    G, nodes_raw, edges_df = read_graph(args.nodes, args.edges)
    print(f"[INFO] Nodes={G.number_of_nodes()} Edges={G.number_of_edges()}", flush=True)

    print("[START] Annotating nodes", flush=True)
    annotated = annotate_nodes(nodes_raw, id_col="id")
    G = attach_annotation_from_df(G, annotated, id_col="id")

    print("[START] Building pairs", flush=True)
    pairs, node2pair, ach_axis, issues = build_pairs(G)
    print(f"[INFO] Pairs found: {len(pairs)}  achiral_or_unmatched={len(ach_axis)}", flush=True)
    if issues and verbose:
        from collections import Counter
        cnt = Counter([x["reason"] for x in issues])
        print("[PAIR DIAG]", dict(cnt), flush=True)
        for ex in issues[:10]:
            print(f"   id={ex['id']} partner={ex['partner']} reason={ex['reason']}", flush=True)

    if len(pairs) <= 1:
        A_side = set(pairs.keys()); B_side = set()
    else:
        H = pair_graph_for_bisection(G, pairs, node2pair)
        if H.number_of_edges() == 0:
            pids = list(H.nodes()); half = len(pids) // 2
            A_side, B_side = set(pids[:half]), set(pids[half:])
        else:
            A_side, B_side = balanced_bisection_pairs(H, seed=args.seed)

    layout_scale = compute_layout_scale(G, min_sep=args.min_sep)
    print("[START] Preparing initial layout or resuming checkpoint", flush=True)
    if args.resume_gephi:
        print(f"[START] Resuming coordinates from {args.resume_gephi}", flush=True)
        best_pos = load_positions_from_gexf(args.resume_gephi, nodes=G.nodes())
        print(f"[INFO] Loaded checkpoint coordinates for {len(best_pos)} nodes", flush=True)

        sides = {}
        for n, data in G.nodes(data=True):
            if data.get("Chirality", "") == "achiral" or not data.get("Enantiomer_Id", ""):
                sides[n] = "axis"

        for pid, (A, B) in pairs.items():
            if pid in A_side:
                sides[A] = "right"
                sides[B] = "left"
            else:
                sides[A] = "left"
                sides[B] = "right"

        pos = dict(best_pos)
        comp_orderings = infer_axis_component_orderings(G, pos, sides)
        achiral_gap = args.achiral_gap if args.achiral_gap is not None else max(args.min_sep, 0.9 * layout_scale / max(1.0, math.sqrt(max(1, G.number_of_nodes()))))
    else:
        print("[START] Laying out representative graph with size-aware scale", flush=True)
        rep_of, reps, Grep = rep_graph_from_partition(G, pairs, A_side)
        layout_scale = compute_layout_scale(Grep, min_sep=args.min_sep)
        reps_pos = layout_reps_scale_aware(Grep, layout_scale=layout_scale, sweeps=args.sweeps, seed=args.seed)

        if args.axis_lateral_gap is not None and args.axis_lateral_gap > 0.0:
            for k, (rx, ry) in list(reps_pos.items()):
                if abs(rx) < args.axis_lateral_gap:
                    sign = 1.0 if rx >= 0 else -1.0
                    if abs(rx) < 1e-12:
                        sign = 1.0
                    reps_pos[k] = (sign * args.axis_lateral_gap, ry)

        print("[START] Enforcing full mirror", flush=True)
        pos, sides = enforce_full_mirror(G, pairs, A_side, reps_pos, snap_achiral=bool(args.snap_achiral), axis_lateral_gap=args.axis_lateral_gap)
        pos = match_axis_chiral_stats(pos, sides)
        pos = equalize_hemi_span(pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio)
        pos = match_axis_chiral_stats(pos, sides)

        print("[START] Arranging achiral nodes on axis", flush=True)
        pos, comp_orderings = arrange_achiral_on_axis(G, pos, sides, pairs, min_sep=args.min_sep, vertical_gap=args.achiral_gap)
        achiral_gap = args.achiral_gap if args.achiral_gap is not None else max(args.min_sep, 0.9 * layout_scale / max(1.0, math.sqrt(max(1, G.number_of_nodes()))))
        pos, comp_orderings = enforce_achiral_direct_adjacency(G, pos, sides, comp_orderings, achiral_gap)
        pos = match_axis_chiral_stats(pos, sides)
        axis_min_gap = args.axis_min_gap if args.axis_min_gap is not None else max(args.min_sep, 0.9 * achiral_gap)
        pos, comp_orderings = adjust_axis_components_min_gap(G, pos, sides, comp_orderings, achiral_gap, axis_min_gap, max_iters=args.axis_adjust_iters)
        pos = match_axis_chiral_stats(pos, sides)
        pos = equalize_hemi_span(pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio)
        pos = match_axis_chiral_stats(pos, sides)

        best_pos = dict(pos)

    print("[INFO] Building edge cache for delta crossing computations", flush=True)
    edges_all, edge_bboxes = build_edge_cache(G, pos)

    best_pos = dict(pos)
    current_cross = count_edge_crossings(G, best_pos)
    current_layout_score = evaluate_layout_score(G, best_pos, edges_all, sides=sides, min_sep=args.min_sep)
    print(f"[REFINE] starting crossings{'~' if args.crossing_mode != 'exact' else ''}={current_cross}{' (sampled)' if args.crossing_mode != 'exact' else ''}", flush=True)

    step_counter = 0
    exact_every = max(1, int(args.crossing_exact_every))

    def maybe_force_exact(final_cycle: bool = False):
        nonlocal current_cross, best_pos
        if args.crossing_mode == "exact":
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "cycle_end" and final_cycle:
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "step_interval" and step_counter > 0 and (step_counter % exact_every == 0):
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)

    def after_step():
        nonlocal step_counter, current_cross
        step_counter += 1
        maybe_force_exact()

    for cycle in range(args.refine_cycles):
        print(f"[REFINE] cycle {cycle+1}/{args.refine_cycles} starting (current crossings={current_cross})", flush=True)
        cycle_start_pos = dict(best_pos)
        cycle_checkpoint_pos = None
        if args.checkpoint_gephi and Path(args.checkpoint_gephi).exists():
            try:
                cycle_checkpoint_pos = load_positions_from_gexf(args.checkpoint_gephi, nodes=G.nodes())
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Could not load checkpoint for convergence check: {exc}", flush=True)
        improved_cycle = False

        print(f"[REFINE][cycle {cycle+1}] Step: chiral Y-swaps (max_iters={args.chiral_swap_iters})", flush=True)
        pos_chiral, ch_improved, ch_before, ch_after = optimize_chiral_pair_swaps(
            G, best_pos, pairs, sides, edges_all, edge_bboxes,
            max_iters=args.chiral_swap_iters, seed=args.seed + cycle, tries_per_iter=None, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if ch_improved:
            pos_chiral = match_axis_chiral_stats(pos_chiral, sides)
            new_cross = count_edge_crossings(G, pos_chiral)
            new_layout_score = evaluate_layout_score(G, pos_chiral, edges_all, sides=sides, min_sep=args.min_sep)
            if new_layout_score < current_layout_score:
                best_pos = pos_chiral
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] chiral Y-swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] chiral Y-swaps no improvement", flush=True)
        after_step()

        print(f"[REFINE][cycle {cycle+1}] Step: pair-of-pairs swaps (attempts_per_iter={args.pairpair_iters})", flush=True)
        pos_pairpair, pp_improved, pp_before, pp_after = optimize_pair_of_pairs_swaps(
            G, best_pos, pairs, sides, edges_all, edge_bboxes,
            max_iters=args.pairpair_iters, seed=args.seed + 1000 + cycle, attempts_per_iter=args.pairpair_iters, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if pp_improved:
            pos_pairpair = match_axis_chiral_stats(pos_pairpair, sides)
            new_cross = count_edge_crossings(G, pos_pairpair)
            new_layout_score = evaluate_layout_score(G, pos_pairpair, edges_all, sides=sides, min_sep=args.min_sep)
            if new_layout_score < current_layout_score:
                best_pos = pos_pairpair
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] pair-of-pairs improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] pair-of-pairs no improvement", flush=True)
        after_step()

        print(f"[REFINE][cycle {cycle+1}] Step: enantiomer X-swaps (iters={args.enantiomer_swap_iters})", flush=True)
        pos_enant, enant_improved, en_before, en_after = optimize_enantiomer_x_swaps(
            G, best_pos, pairs, sides, edges_all, edge_bboxes,
            max_iters=args.enantiomer_swap_iters, seed=args.seed + 5000 + cycle, tries_per_iter=None, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if enant_improved:
            pos_enant = match_axis_chiral_stats(pos_enant, sides)
            new_cross = count_edge_crossings(G, pos_enant)
            new_layout_score = evaluate_layout_score(G, pos_enant, edges_all, sides=sides, min_sep=args.min_sep)
            if new_layout_score < current_layout_score:
                best_pos = pos_enant
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] enantiomer X-swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] enantiomer X-swaps no improvement", flush=True)
        after_step()

        print(f"[REFINE][cycle {cycle+1}] Step: achiral-slot swaps (max_iters={args.swap_iters})", flush=True)
        pos_achiral, ach_cross = optimize_achiral_swaps(
            G, best_pos, pairs, sides, comp_orderings, achiral_gap,
            edges_all, edge_bboxes,
            max_iters=args.swap_iters, seed=args.seed + 2000 + cycle,
            cross_weight=0.20, gap_weight=args.achiral_gap_weight, nonadj_weight=args.achiral_adjacency_weight, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        pos_achiral = match_axis_chiral_stats(pos_achiral, sides)
        ach_score = evaluate_layout_score(G, pos_achiral, edges_all, sides=sides, min_sep=args.min_sep)
        if ach_score < current_layout_score:
            best_pos = pos_achiral
            current_cross = ach_cross
            current_layout_score = ach_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] achiral-slot swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print(f"  [RESULT] achiral-slot swaps no improvement (score={ach_score:.3f}, crossings={ach_cross})", flush=True)
        after_step()

        print(f"[REFINE][cycle {cycle+1}] Step: pair-block relocation (max_trials={args.pair_relocation_trials})", flush=True)
        pos_reloc, relocated, pre_reloc_cross, post_reloc_cross = pair_block_relocation_refinement(
            G, best_pos, sides, comp_orderings, achiral_gap,
            edges_all, edge_bboxes,
            max_trials=args.pair_relocation_trials, seed=args.seed + 3000 + cycle, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if relocated:
            pos_reloc = match_axis_chiral_stats(pos_reloc, sides)
        pos_reloc_score = evaluate_layout_score(G, pos_reloc, edges_all, sides=sides, min_sep=args.min_sep)
        if relocated and pos_reloc_score < current_layout_score:
            best_pos = pos_reloc
            current_cross = post_reloc_cross
            current_layout_score = pos_reloc_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] pair-block relocation improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] pair-block relocation no improvement", flush=True)
        after_step()

        axis_min_gap = args.axis_min_gap if args.axis_min_gap is not None else (0.9 * args.min_sep)
        print(f"[REFINE][cycle {cycle+1}] Step: axis block adjust (min_axis_gap={axis_min_gap})", flush=True)
        best_pos, comp_orderings = adjust_axis_components_min_gap(G, best_pos, sides, comp_orderings, achiral_gap, axis_min_gap, max_iters=1)
        best_pos = spread_axis_nodes_min_sep(best_pos, sides, args.min_sep)
        best_pos = match_axis_chiral_stats(best_pos, sides)
        post_adjust_cross = count_edge_crossings(G, best_pos)
        post_adjust_score = evaluate_layout_score(G, best_pos, edges_all, sides=sides, min_sep=args.min_sep)
        if post_adjust_score < current_layout_score:
            current_cross = post_adjust_cross
            current_layout_score = post_adjust_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] axis adjust improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print(f"  [RESULT] axis adjust no improvement (score={post_adjust_score:.3f}, crossings={post_adjust_cross})", flush=True)
        after_step()

        if args.chiral_relax:
            pos_relax, relax_improved, relax_before, relax_after = optimize_chiral_coordinate_relaxation(
                G, best_pos, pairs, sides, edges_all, edge_bboxes,
                max_iters=args.chiral_relax_iters,
                seed=args.seed + 7000 + cycle,
                tries_per_iter=None,
                x_rate=args.chiral_relax_x_rate,
                y_rate=args.chiral_relax_y_rate,
                spacing_weight=args.chiral_relax_spacing_weight,
                edge_length_weight=args.chiral_relax_edge_length_weight,
                repulsion_weight=args.chiral_relax_repulsion_weight,
                min_sep=args.min_sep,
                edge_target_factor=args.chiral_relax_edge_target_factor,
                min_x=args.chiral_relax_min_x,
                verbose=verbose,
                fast_mode=args.fast_mode,
                sample_size=args.fast_sample_size,
            )
            pos_relax = match_axis_chiral_stats(pos_relax, sides)
            relax_score = evaluate_layout_score(G, pos_relax, edges_all, sides=sides, min_sep=args.min_sep)
            if relax_score < current_layout_score:
                best_pos = pos_relax
                current_cross = relax_after
                current_layout_score = relax_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] chiral relaxation improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
            elif verbose:
                print(f"  [RESULT] chiral relaxation no improvement (score={relax_score:.3f})", flush=True)
            after_step()

            print(f"[REFINE][cycle {cycle+1}] Step: pair flips", flush=True)
            pos_flip, flip_improved, flip_before, flip_after = optimize_chiral_pair_flips(
                G, best_pos, pairs, sides, edges_all, edge_bboxes,
                max_iters=50, seed=args.seed + 9000 + cycle, tries_per_iter=None, verbose=verbose,
                fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
            pos_flip = match_axis_chiral_stats(pos_flip, sides)
            flip_score = evaluate_layout_score(G, pos_flip, edges_all, sides=sides, min_sep=args.min_sep)
            if flip_score < current_layout_score:
                best_pos = pos_flip
                current_cross = flip_after
                current_layout_score = flip_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] pair flips improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
            elif verbose:
                print(f"  [RESULT] pair flips no improvement (score={flip_score:.3f})", flush=True)
            after_step()

        best_pos = equalize_hemi_span(best_pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio)
        best_pos = match_axis_chiral_stats(best_pos, sides)
        current_layout_score = evaluate_layout_score(G, best_pos, edges_all, sides=sides, min_sep=args.min_sep)

        edges_all, edge_bboxes = build_edge_cache(G, best_pos)
        current_cross = count_edge_crossings(G, best_pos)
        if args.crossing_mode == "cycle_end":
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "step_interval" and step_counter > 0 and (step_counter % exact_every == 0):
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)

        cycle_shift = max_node_shift(cycle_start_pos, best_pos)
        print(f"[REFINE] cycle {cycle+1} complete: current_crossings={current_cross} improved_cycle={improved_cycle} max_node_shift={cycle_shift:.6f}", flush=True)

        if args.checkpoint_gephi and ((cycle + 1) % max(1, args.checkpoint_every) == 0):
            write_gexf_with_viz(G, best_pos, args.checkpoint_gephi)
            print(f"[CHECKPOINT] wrote checkpoint -> {args.checkpoint_gephi}", flush=True)

        if cycle_checkpoint_pos is not None and positions_exactly_equal(best_pos, cycle_checkpoint_pos):
            print(f"[REFINE] coordinates unchanged from checkpoint at end of cycle {cycle+1}; stopping early.", flush=True)
            break

        if cycle_shift <= args.cycle_shift_tol:
            print(f"[REFINE] maximum node shift {cycle_shift:.6f} <= tol {args.cycle_shift_tol:.6f}; stopping early.", flush=True)
            break

        if not improved_cycle:
            print(f"[REFINE] no improvement in cycle {cycle+1}; stopping early.", flush=True)
            break

    best_pos = equalize_hemi_span(best_pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio)
    best_pos = match_axis_chiral_stats(best_pos, sides)
    current_layout_score = evaluate_layout_score(G, best_pos, edges_all, sides=sides, min_sep=args.min_sep)

    print("[FINAL] Running overlap resolution while keeping axis nodes on x=0 but allowing y-separation", flush=True)
    pos_after_relax = adjust_positions_with_constraints(G, best_pos, pairs, sides, min_sep=args.min_sep, edge_clearance=args.edge_clearance, max_iter=args.overlap_iter, lr=0.25, fixed_achiral_y=None, axis_lateral_gap=args.axis_lateral_gap)
    pos_after_relax = match_axis_chiral_stats(pos_after_relax, sides)
    pos_after_relax = spread_axis_nodes_min_sep(pos_after_relax, sides, args.min_sep)

    if args.axis_node_relax:
        if args.axis_node_relax_id:
            focus_nodes = [x.strip() for x in args.axis_node_relax_id.split(",") if x.strip()]
        else:
            focus_nodes = None
        pos_after_relax = relax_axis_node_clearance(
            G,
            pos_after_relax,
            sides,
            focus_nodes=focus_nodes,
            clearance=args.axis_node_relax_clearance,
            max_shift=args.axis_node_relax_shift,
            step=args.axis_node_relax_step,
        )
        pos_after_relax = spread_axis_nodes_min_sep(pos_after_relax, sides, args.min_sep)

    print("[FINAL] Enforcing strict mirror symmetry", flush=True)
    pos_final, mirror_changed = enforce_strict_mirror(pos_after_relax, pairs, sides, tol=1e-8)
    if mirror_changed:
        print("[MIRROR] Adjusted positions to enforce strict mirror symmetry (left-right x sign, identical y).", flush=True)
    final_cross = count_edge_crossings(G, pos_final, force_exact=(args.crossing_mode != "estimate"))
    print(f"[FINAL] crossings after all refinements = {final_cross}", flush=True)

    if args.checkpoint_gephi:
        write_gexf_with_viz(G, pos_final, args.checkpoint_gephi)
        print(f"[CHECKPOINT] wrote final checkpoint -> {args.checkpoint_gephi}", flush=True)

    write_gexf_with_viz(G, pos_final, args.out)
    if args.out_xgmml:
        write_xgmml_with_viz(G, pos_final, args.out_xgmml, graph_label="BVGraph")
        print(f"[OK] Wrote XGMML -> {args.out_xgmml}", flush=True)

    if args.out_nodes_csv:
        write_nodes_coords_csv(G, pos_final, args.out_nodes_csv)
        print(f"[OK] Wrote node coords CSV -> {args.out_nodes_csv}", flush=True)

    if args.out_edges_csv:
        write_edges_csv(G, args.out_edges_csv)
        print(f"[OK] Wrote edges CSV -> {args.out_edges_csv}", flush=True)

    print(f"[OK] Wrote {args.out}  Nodes={G.number_of_nodes()}  Edges={G.number_of_edges()}  Pairs={len(pairs)}", flush=True)
    print(f"[DIAG] final crossings={final_cross}", flush=True)



LAYOUT_CROSSING_WEIGHT = 0.20
LAYOUT_EDGE_LENGTH_WEIGHT = 1.00
LAYOUT_SPACING_WEIGHT = 8.00
LAYOUT_SOFT_SEP_FACTOR = 1.35
LAYOUT_AXIS_GAP_WEIGHT = 0.05


def _local_edge_length_score(best_pos, candidate_pos, moved_nodes, edges_all):
    moved_set = set(moved_nodes)
    seen = set()
    score = 0.0
    for u, v in edges_all:
        if u not in moved_set and v not in moved_set:
            continue
        key = (u, v) if u <= v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        pu = candidate_pos.get(u, best_pos.get(u))
        pv = candidate_pos.get(v, best_pos.get(v))
        if pu is None or pv is None:
            continue
        score += math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
    return score


def _local_spacing_penalty(best_pos, candidate_pos, moved_nodes, grid, inv_cell, min_sep, soft_sep_factor=LAYOUT_SOFT_SEP_FACTOR):
    if min_sep <= 0.0:
        return 0.0
    min_sep = float(min_sep)
    soft_sep = max(min_sep, float(min_sep) * float(soft_sep_factor))
    moved_set = set(moved_nodes)
    checked = set()
    penalty = 0.0
    for n in moved_set:
        p = candidate_pos.get(n, best_pos.get(n))
        if p is None:
            continue
        x, y = float(p[0]), float(p[1])
        ix = int(math.floor(x * inv_cell))
        iy = int(math.floor(y * inv_cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((ix + dx, iy + dy), []):
                    if other == n:
                        continue
                    key = (n, other) if n <= other else (other, n)
                    if key in checked:
                        continue
                    checked.add(key)
                    op = candidate_pos.get(other, best_pos.get(other))
                    if op is None:
                        continue
                    d = math.hypot(x - float(op[0]), y - float(op[1]))
                    if d < min_sep:
                        deficit = (min_sep - d) / min_sep
                        penalty += 4.0 * deficit * deficit
                    elif d < soft_sep:
                        deficit = (soft_sep - d) / max(soft_sep - min_sep, 1e-9)
                        penalty += 0.5 * deficit * deficit
    return penalty


def _axis_gap_penalty(pos, sides):
    axis_nodes = [n for n in pos.keys() if sides.get(n, '') == 'axis']
    ys = sorted((pos[n][1] for n in axis_nodes))
    if len(ys) <= 1:
        return 0.0
    gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 1e-12:
        return 0.0
    return sum(((g - mean_gap) / mean_gap) ** 2 for g in gaps) / len(gaps)


def _global_edge_length_score(pos, edges_all):
    total = 0.0
    for u, v in edges_all:
        pu = pos.get(u)
        pv = pos.get(v)
        if pu is None or pv is None:
            continue
        total += math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
    return total


def _global_spacing_penalty(pos, min_sep, soft_sep_factor=LAYOUT_SOFT_SEP_FACTOR):
    if min_sep <= 0.0 or len(pos) < 2:
        return 0.0
    grid, inv_cell = _build_spatial_hash(pos, min_sep)
    min_sep = float(min_sep)
    soft_sep = max(min_sep, float(min_sep) * float(soft_sep_factor))
    checked = set()
    penalty = 0.0
    for n, (x, y) in pos.items():
        x = float(x); y = float(y)
        ix = int(math.floor(x * inv_cell))
        iy = int(math.floor(y * inv_cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other in grid.get((ix + dx, iy + dy), []):
                    if other == n:
                        continue
                    key = (n, other) if n <= other else (other, n)
                    if key in checked:
                        continue
                    checked.add(key)
                    op = pos.get(other)
                    if op is None:
                        continue
                    d = math.hypot(x - float(op[0]), y - float(op[1]))
                    if d < min_sep:
                        deficit = (min_sep - d) / min_sep
                        penalty += 4.0 * deficit * deficit
                    elif d < soft_sep:
                        deficit = (soft_sep - d) / max(soft_sep - min_sep, 1e-9)
                        penalty += 0.5 * deficit * deficit
    return penalty


def evaluate_layout_score(G, pos, edges_all, sides=None, min_sep=40.0,
                          crossing_weight=LAYOUT_CROSSING_WEIGHT,
                          edge_length_weight=LAYOUT_EDGE_LENGTH_WEIGHT,
                          spacing_weight=LAYOUT_SPACING_WEIGHT,
                          axis_gap_weight=LAYOUT_AXIS_GAP_WEIGHT):
    if sides is None:
        sides = {}
    crossings = count_edge_crossings(G, pos)
    edge_score = _global_edge_length_score(pos, edges_all) / max(float(min_sep), 1e-9)
    spacing_score = _global_spacing_penalty(pos, min_sep)
    axis_score = _axis_gap_penalty(pos, sides) if sides else 0.0
    return (
        crossing_weight * float(crossings)
        + edge_length_weight * float(edge_score)
        + spacing_weight * float(spacing_score)
        + axis_gap_weight * float(axis_score)
    )


def optimize_chiral_coordinate_relaxation(
    G, pos, pairs, sides, edges_all, edge_bboxes,
    max_iters=150, seed=42, tries_per_iter=None,
    x_rate=0.35, y_rate=0.35,
    spacing_weight=8.0, edge_length_weight=1.0, repulsion_weight=10.0,
    min_sep=40.0, edge_target_factor=1.0, min_x=5.0,
    verbose=True, fast_mode=False, sample_size=4000
):
    """
    Mirror-preserving local relaxation for chiral pairs.
    Moves both x and y of a pair together, keeps the two partners mirrored,
    and uses neighbouring pair centres as soft targets.

    The score prioritises compact, well-spaced layouts and treats crossings as
    a secondary penalty.
    """
    rng = random.Random(seed)
    pair_of = {}
    chiral_pair_ids = []

    for pid, (A, B) in pairs.items():
        pair_of[A] = pid
        pair_of[B] = pid
        if sides.get(A, "") != "axis" or sides.get(B, "") != "axis":
            chiral_pair_ids.append(pid)

    if len(chiral_pair_ids) < 1:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(chiral_pair_ids)))

    def pair_right_left(pid, pmap):
        A, B = pairs[pid]
        a_side = sides.get(A, "")
        b_side = sides.get(B, "")
        if a_side == "right" or b_side == "left":
            return A, B
        if b_side == "right" or a_side == "left":
            return B, A
        return (A, B) if pmap[A][0] >= pmap[B][0] else (B, A)

    def pair_center(pid, pmap):
        A, B = pairs[pid]
        mag = 0.5 * (abs(pmap[A][0]) + abs(pmap[B][0]))
        cy = 0.5 * (pmap[A][1] + pmap[B][1])
        return mag, cy

    def neighbour_targets(pid, pmap):
        neigh = set()
        for n in pairs[pid]:
            for nbr in G[n]:
                q = pair_of.get(nbr)
                if q is not None and q != pid:
                    neigh.add(q)
        if not neigh:
            return None
        mags = []
        ys = []
        for q in neigh:
            A, B = pairs[q]
            mags.append(0.5 * (abs(pmap[A][0]) + abs(pmap[B][0])))
            ys.append(0.5 * (pmap[A][1] + pmap[B][1]))
        return float(np.median(mags)), float(np.median(ys))

    current_pos = dict(pos)
    edge_target_length = _median_edge_length_from_pos(current_pos, edges_all)
    if edge_target_length <= 1e-9:
        edge_target_length = max(1.0, 0.75 * float(min_sep))
    edge_target_length *= float(edge_target_factor)
    repulse_dist = max(1e-9, float(min_sep))

    if fast_mode:
        current_cross = estimate_crossings_sampled(
            G, current_pos, edges_all=edges_all,
            sample_size=sample_size, seed=seed
        )
    else:
        current_cross = count_edge_crossings(G, current_pos)

    best_pos = dict(current_pos)
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(
            f"  [chiral-relax] start pairs={len(chiral_pair_ids)} tries_per_iter={tries_per_iter} "
            f"max_iters={max_iters} crossings={current_cross} edge_target={edge_target_length:.3f} "
            f"edge_w={edge_length_weight:.3f} repulse_w={repulsion_weight:.3f}"
        )

    for it in range(1, max_iters + 1):
        improved = False
        spatial_grid, inv_cell = _build_spatial_hash(best_pos, repulse_dist)
        for _ in range(tries_per_iter):
            pid = rng.choice(chiral_pair_ids)
            target = neighbour_targets(pid, best_pos)
            if target is None:
                continue

            target_mag, target_y = target
            cur_mag, cur_y = pair_center(pid, best_pos)

            new_mag = max(min_x, cur_mag + x_rate * (target_mag - cur_mag))
            new_y = cur_y + y_rate * (target_y - cur_y)

            right, left = pair_right_left(pid, best_pos)
            candidate_pos = {
                right: (abs(new_mag), new_y),
                left: (-abs(new_mag), new_y),
            }

            moved = {right, left}
            current_edge_pen = _local_edge_length_score(best_pos, {}, moved, edges_all) / max(float(min_sep), 1e-9)
            current_repulse_pen = _local_spacing_penalty(best_pos, {}, moved, spatial_grid, inv_cell, repulse_dist)
            current_spacing = abs(cur_mag - target_mag) + abs(cur_y - target_y)
            current_score = (
                float(current_cross) * LAYOUT_CROSSING_WEIGHT
                + edge_length_weight * current_edge_pen
                + spacing_weight * current_spacing
                + repulsion_weight * current_repulse_pen
            )

            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes,
                                 _seed_offset=it)
                new_cross = current_cross + delta
            else:
                new_cross = count_edge_crossings(G, {**best_pos, **candidate_pos})

            new_edge_pen = _local_edge_length_score(best_pos, candidate_pos, moved, edges_all) / max(float(min_sep), 1e-9)
            new_repulse_pen = _local_spacing_penalty(best_pos, candidate_pos, moved, spatial_grid, inv_cell, repulse_dist)
            new_spacing = abs(new_mag - target_mag) + abs(new_y - target_y)
            new_score = (
                float(new_cross) * LAYOUT_CROSSING_WEIGHT
                + edge_length_weight * new_edge_pen
                + spacing_weight * new_spacing
                + repulsion_weight * new_repulse_pen
            )

            if new_score < current_score:
                best_pos = dict(best_pos)
                best_pos.update(candidate_pos)
                current_cross = new_cross
                improved = True
                if verbose:
                    print(
                        f"    [chiral-relax] iter={it} moved {pid} "
                        f"crossings={current_cross} score={new_score:.3f} "
                        f"edge_pen={new_edge_pen:.3f} repulse_pen={new_repulse_pen:.3f}"
                    )
                break

        if not improved:
            if verbose:
                print(f"    [chiral-relax] iter={it} no improvement, stopping")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, (after_cross < count_edge_crossings(G, pos)), count_edge_crossings(G, pos), after_cross


def _pair_local_score(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, current_cross, min_sep,
                      delta_fn, fast_mode, sample_size, seed, crossing_weight=LAYOUT_CROSSING_WEIGHT,
                      edge_weight=LAYOUT_EDGE_LENGTH_WEIGHT, spacing_weight=LAYOUT_SPACING_WEIGHT):
    if delta_fn is not None:
        delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=seed)
    else:
        delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
    new_cross = current_cross + delta
    spatial_grid, inv_cell = _build_spatial_hash(best_pos, min_sep)
    current_edge = _local_edge_length_score(best_pos, {}, moved, edges_all) / max(float(min_sep), 1e-9)
    new_edge = _local_edge_length_score(best_pos, candidate_pos, moved, edges_all) / max(float(min_sep), 1e-9)
    current_spacing = _local_spacing_penalty(best_pos, {}, moved, spatial_grid, inv_cell, min_sep)
    new_spacing = _local_spacing_penalty(best_pos, candidate_pos, moved, spatial_grid, inv_cell, min_sep)
    current_score = crossing_weight * float(current_cross) + edge_weight * current_edge + spacing_weight * current_spacing
    new_score = crossing_weight * float(new_cross) + edge_weight * new_edge + spacing_weight * new_spacing
    return delta, new_cross, current_score, new_score


def optimize_chiral_pair_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                               max_iters=500, seed=42, tries_per_iter=None, verbose=True,
                               fast_mode=False, sample_size=4000, min_sep=40.0):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    pair_side_map = {}
    for pid, (A, B) in pairs.items():
        a_side = sides.get(A, ''); b_side = sides.get(B, '')
        if a_side == 'left' and b_side == 'right':
            left, right = A, B
        elif b_side == 'left' and a_side == 'right':
            left, right = B, A
        else:
            if pos.get(A, (0.0, 0.0))[0] <= pos.get(B, (0.0, 0.0))[0]:
                left, right = A, B
            else:
                left, right = B, A
        pair_side_map[pid] = (left, right)

    candidates = [pid for pid in pair_ids if not (sides.get(pairs[pid][0], '') == 'axis' and sides.get(pairs[pid][1], '') == 'axis')]
    if not candidates:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(candidates)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    improved_any = False
    it = 0
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [chiral-y-swaps] candidates={len(candidates)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    while it < max_iters:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            a, b = rng.sample(candidates, 2)
            if a == b:
                continue
            aL, aR = pair_side_map[a]; bL, bR = pair_side_map[b]
            candidate_pos = {
                aL: (best_pos[aL][0], best_pos[bL][1]),
                aR: (best_pos[aR][0], best_pos[bL][1]),
                bL: (best_pos[bL][0], best_pos[aL][1]),
                bR: (best_pos[bR][0], best_pos[aL][1]),
            }
            moved = {aL, aR, bL, bR}
            delta, new_cross, current_score, new_score = _pair_local_score(
                G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, current_cross, min_sep,
                delta_fn, fast_mode, sample_size, seed=it)
            if new_score < current_score:
                for n, p in candidate_pos.items():
                    best_pos[n] = p
                current_cross = new_cross
                improved = True
                improved_any = True
                if verbose:
                    print(f"    [chiral-y-swaps] iter={it} applied swap {a}<->{b} delta={delta} new_cross={current_cross} score={new_score:.3f}")
                break
        if not improved:
            if verbose:
                print(f"    [chiral-y-swaps] iter={it} no improvement, stopping")
            break
    return best_pos, improved_any, before_cross, current_cross


def optimize_pair_of_pairs_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                                 max_iters=200, seed=42, attempts_per_iter=100, verbose=True,
                                 fast_mode=False, sample_size=4000, min_sep=40.0):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    pair_side_map = {}
    for pid, (A, B) in pairs.items():
        a_side = sides.get(A, ''); b_side = sides.get(B, '')
        if a_side == 'left' and b_side == 'right':
            left, right = A, B
        elif b_side == 'left' and a_side == 'right':
            left, right = B, A
        else:
            if pos.get(A, (0.0, 0.0))[0] <= pos.get(B, (0.0, 0.0))[0]:
                left, right = A, B
            else:
                left, right = B, A
        pair_side_map[pid] = (left, right)

    candidates = [pid for pid in pair_ids if not (sides.get(pairs[pid][0], '') == 'axis' and sides.get(pairs[pid][1], '') == 'axis')]
    if len(candidates) < 2:
        return dict(pos), False, count_edge_crossings(G, pos), count_edge_crossings(G, pos)

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    improved_any = False
    it = 0
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [pair-of-pairs] candidates={len(candidates)} attempts_per_iter={attempts_per_iter} max_iters={max_iters} before_cross={before_cross}")

    while it < max_iters:
        it += 1
        changed = False
        for _ in range(attempts_per_iter):
            a, b = rng.sample(candidates, 2)
            if a == b:
                continue
            aL, aR = pair_side_map[a]; bL, bR = pair_side_map[b]
            candidate_pos = {
                aL: best_pos[bL],
                aR: best_pos[bR],
                bL: best_pos[aL],
                bR: best_pos[aR],
            }
            moved = {aL, aR, bL, bR}
            delta, new_cross, current_score, new_score = _pair_local_score(
                G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, current_cross, min_sep,
                delta_fn, fast_mode, sample_size, seed=it)
            if new_score < current_score:
                for n, p in candidate_pos.items():
                    best_pos[n] = p
                current_cross = new_cross
                improved_any = True
                changed = True
                if verbose:
                    print(f"    [pair-of-pairs] iter={it} applied swap {a} <-> {b} delta={delta} new_cross={current_cross} score={new_score:.3f}")
                break
        if not changed:
            if verbose:
                print(f"    [pair-of-pairs] iter={it} no change, stopping")
            break
    return best_pos, improved_any, before_cross, current_cross


def optimize_chiral_pair_flips(G, pos, pairs, sides, edges_all, edge_bboxes,
                               max_iters=500, seed=42, tries_per_iter=None, verbose=True,
                               fast_mode=False, sample_size=4000, min_sep=40.0):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    if len(pair_ids) < 1:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(pair_ids)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    improved_any = False
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [pair-flips] candidates={len(pair_ids)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    it = 0
    while it < max_iters:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            pid = rng.choice(pair_ids)
            A, B = pairs[pid]
            candidate_pos = {A: best_pos[B], B: best_pos[A]}
            moved = {A, B}
            delta, new_cross, current_score, new_score = _pair_local_score(
                G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, current_cross, min_sep,
                delta_fn, fast_mode, sample_size, seed=it)
            if new_score < current_score:
                best_pos[A] = candidate_pos[A]
                best_pos[B] = candidate_pos[B]
                current_cross = new_cross
                improved = True
                improved_any = True
                if verbose:
                    print(f"    [pair-flips] iter={it} flipped {pid} delta={delta} new_cross={current_cross} score={new_score:.3f}")
                break
        if not improved:
            if verbose:
                print(f"    [pair-flips] iter={it} no improvement, stopping")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, improved_any, before_cross, after_cross


def optimize_enantiomer_x_swaps(G, pos, pairs, sides, edges_all, edge_bboxes,
                                max_iters=100, seed=42, tries_per_iter=None, verbose=True,
                                fast_mode=False, sample_size=4000, min_sep=40.0):
    rng = random.Random(seed)
    pair_ids = list(pairs.keys())
    candidates = []
    for pid in pair_ids:
        A, B = pairs[pid]
        if A not in pos or B not in pos:
            continue
        if sides.get(A, "") == "axis" and sides.get(B, "") == "axis":
            continue
        candidates.append(pid)

    if len(candidates) < 2:
        before = count_edge_crossings(G, pos)
        return dict(pos), False, before, before

    if tries_per_iter is None:
        tries_per_iter = min(200, max(20, len(candidates)))

    best_pos = dict(pos)
    before_cross = count_edge_crossings(G, best_pos)
    current_cross = before_cross
    improved_any = False
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [enantiomer-x-swaps] start candidates={len(candidates)} tries_per_iter={tries_per_iter} max_iters={max_iters} before_cross={before_cross}")

    it = 0
    while it < max_iters:
        it += 1
        made_progress = False
        for _ in range(tries_per_iter):
            a_pid, b_pid = rng.sample(candidates, 2)
            if a_pid == b_pid:
                continue
            aA, aB = pairs[a_pid]
            bA, bB = pairs[b_pid]

            def left_member(pair):
                n1, n2 = pair
                return (n1, n2) if best_pos[n1][0] < best_pos[n2][0] else (n2, n1)

            a_left, a_right = left_member((aA, aB))
            b_left, b_right = left_member((bA, bB))

            ax_mag = abs(best_pos[a_left][0])
            bx_mag = abs(best_pos[b_left][0])
            if abs(ax_mag - bx_mag) < 1e-12:
                continue

            sign_a_left = 1.0 if best_pos[a_left][0] >= 0 else -1.0
            sign_b_left = 1.0 if best_pos[b_left][0] >= 0 else -1.0

            cand = {
                a_left: (sign_a_left * bx_mag, best_pos[a_left][1]),
                a_right: (-sign_a_left * bx_mag, best_pos[a_right][1]),
                b_left: (sign_b_left * ax_mag, best_pos[b_left][1]),
                b_right: (-sign_b_left * ax_mag, best_pos[b_right][1]),
            }

            moved_nodes = set(cand.keys())
            delta, new_cross, current_score, new_score = _pair_local_score(
                G, best_pos, moved_nodes, cand, edges_all, edge_bboxes, current_cross, min_sep,
                delta_fn, fast_mode, sample_size, seed=it)
            if new_score < current_score:
                for n, p in cand.items():
                    best_pos[n] = p
                current_cross = new_cross
                made_progress = True
                improved_any = True
                if verbose:
                    print(f"    [enantiomer-x-swaps] iter={it} accepted swap {a_pid}<->{b_pid} delta={delta} crossings={current_cross} score={new_score:.3f}")
                break
        if not made_progress:
            if verbose:
                print(f"    [enantiomer-x-swaps] iter={it} no improvement; stopping early")
            break

    after_cross = count_edge_crossings(G, best_pos)
    return best_pos, improved_any, before_cross, after_cross


def optimize_achiral_swaps(G, pos, pairs, sides, comp_orderings, achiral_gap,
                           edges_all, edge_bboxes,
                           max_iters=1000, seed=42,
                           cross_weight=0.20, gap_weight=1.0, nonadj_weight=50.0, verbose=True,
                           tries_per_iter_override=None, fast_mode=False, sample_size=4000, min_sep=40.0):
    start_time = time.time()
    node_to_slot = {}
    base_y_by_comp = {}
    for comp_id, ordering in comp_orderings.items():
        if not ordering:
            continue
        ys = [pos[n][1] for n in ordering]
        center_y = float(np.median(ys))
        base_y = center_y - ((len(ordering) - 1) * achiral_gap / 2.0)
        base_y_by_comp[comp_id] = base_y
        for idx, n in enumerate(ordering):
            node_to_slot[n] = (comp_id, idx)
            pos[n] = (0.0, base_y + idx * achiral_gap)

    candidates = list(node_to_slot.keys())
    if len(candidates) < 2:
        return pos, count_edge_crossings(G, pos)

    def evaluate_spacing_and_nonadj(pos_local, node_to_slot_local):
        axis_nodes = [n for n in pos_local.keys() if sides.get(n, '') == 'axis']
        ys = sorted((pos_local[n][1] for n in axis_nodes))
        if len(ys) <= 1:
            gap_var = 0.0
        else:
            gaps = [ys[i + 1] - ys[i] for i in range(len(ys) - 1)]
            mean_gap = sum(gaps) / len(gaps)
            gap_var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
        nonadj = 0
        for u, v in G.edges():
            if sides.get(u, '') != 'axis' or sides.get(v, '') != 'axis':
                continue
            if u not in node_to_slot_local or v not in node_to_slot_local:
                continue
            cu, iu = node_to_slot_local[u]
            cv, iv = node_to_slot_local[v]
            if cu != cv:
                continue
            if abs(iu - iv) != 1:
                nonadj += 1
        return gap_var, nonadj

    def combined_score_approx(crossings, pos_local, node_to_slot_local):
        gap_var, nonadj = evaluate_spacing_and_nonadj(pos_local, node_to_slot_local)
        return (
            cross_weight * crossings
            + LAYOUT_EDGE_LENGTH_WEIGHT * (_global_edge_length_score(pos_local, edges_all) / max(float(min_sep), 1e-9))
            + LAYOUT_SPACING_WEIGHT * _global_spacing_penalty(pos_local, min_sep)
            + gap_weight * gap_var
            + nonadj_weight * nonadj
        )

    best_pos = dict(pos)
    best_node_to_slot = dict(node_to_slot)
    if fast_mode:
        current_cross = estimate_crossings_sampled(G, best_pos, edges_all=edges_all, sample_size=sample_size, seed=seed)
    else:
        current_cross = count_edge_crossings(G, best_pos)
    best_score = combined_score_approx(current_cross, best_pos, best_node_to_slot)
    rng = random.Random(seed)
    it = 0
    improved = True
    tries_per_iter = tries_per_iter_override if tries_per_iter_override is not None else min(200, max(20, len(candidates)))
    delta_fn = get_crossing_delta_fn(fast_mode, sample_size, seed)

    if verbose:
        print(f"  [achiral-swaps] starting score={best_score:.3f} crossings={current_cross} candidates={len(candidates)} tries_per_iter={tries_per_iter}")

    while it < max_iters and improved:
        it += 1
        improved = False
        for _ in range(tries_per_iter):
            a, b = rng.sample(candidates, 2)
            ca, ia = best_node_to_slot[a]; cb, ib = best_node_to_slot[b]
            candidate_pos = {
                a: (0.0, base_y_by_comp[cb] + ib * achiral_gap),
                b: (0.0, base_y_by_comp[ca] + ia * achiral_gap),
            }
            moved = {a, b}
            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=it)
            else:
                delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
            new_cross = current_cross + delta
            new_pos_temp = dict(best_pos)
            new_pos_temp[a] = candidate_pos[a]; new_pos_temp[b] = candidate_pos[b]
            new_score = combined_score_approx(new_cross, new_pos_temp, {**best_node_to_slot, a: (cb, ib), b: (ca, ia)})
            if new_score < best_score:
                best_score = new_score
                best_pos = new_pos_temp
                best_node_to_slot = dict(best_node_to_slot)
                best_node_to_slot[a] = (cb, ib)
                best_node_to_slot[b] = (ca, ia)
                current_cross = new_cross
                improved = True
                if verbose:
                    print(f"    [achiral-swaps] iter={it} accepted swap {a}<->{b} delta_cross={delta} crossings={current_cross} score={best_score:.3f}")
                break
        if verbose and not improved:
            elapsed = time.time() - start_time
            print(f"    [achiral-swaps] iter={it} no improvement (elapsed={elapsed:.1f}s) current_cross={current_cross}")
    for n, (comp_id, idx) in best_node_to_slot.items():
        base = base_y_by_comp.get(comp_id, None)
        if base is not None:
            best_pos[n] = (0.0, base + idx * achiral_gap)
    return best_pos, current_cross


def pair_block_relocation_refinement(G, pos, sides, comp_orderings, achiral_gap,
                                     edges_all, edge_bboxes,
                                     max_trials=200, seed=42, verbose=True,
                                     fast_mode=False, sample_size=4000, min_sep=40.0):
    rng = random.Random(seed)
    axis_nodes = [n for n in G.nodes() if sides.get(n, '') == 'axis']
    axis_sub = G.subgraph(axis_nodes).copy()
    comps = list(nx.connected_components(axis_sub))
    node2comp = {}
    comp_orders = {}
    for cid, comp in enumerate(comps):
        comp_set = set(comp)
        ordering = None
        for k, ordlist in comp_orderings.items():
            if set(ordlist) == comp_set:
                ordering = list(ordlist)
                break
        if ordering is None:
            ordering = sorted(list(comp), key=lambda n: pos[n][1])
        comp_orders[cid] = ordering
        for n in ordering:
            node2comp[n] = cid

    def build_pos_from_comp_orders(base_pos, comp_orders_local, gap):
        newp = dict(base_pos)
        comp_info = []
        for cid, ordering in comp_orders_local.items():
            med = np.median([base_pos[n][1] for n in ordering]) if ordering else 0.0
            comp_info.append((cid, ordering, med))
        comp_info.sort(key=lambda x: -x[2])
        y_cursor = None
        for cid, ordering, _ in comp_info:
            m = len(ordering)
            height = (m - 1) * gap if m > 0 else 0.0
            if y_cursor is None:
                center = np.median([base_pos[n][1] for n in ordering]) if m > 0 else 0.0
                start_y = center + height / 2.0
            else:
                start_y = y_cursor - gap
            for i, n in enumerate(ordering):
                newp[n] = (0.0, start_y - i * gap)
            bottom = start_y - (m - 1) * gap
            y_cursor = bottom - gap
        return newp

    baseline_pos = dict(pos)
    if fast_mode:
        baseline_cross = estimate_crossings_sampled(G, baseline_pos, edges_all=edges_all, sample_size=sample_size, seed=seed)
    else:
        baseline_cross = count_edge_crossings(G, baseline_pos)
    best_pos = dict(baseline_pos)
    best_score = evaluate_layout_score(G, baseline_pos, edges_all, sides=sides, min_sep=min_sep)
    trials = 0
    improved_any = False

    candidate_pairs = []
    for u, v in G.edges():
        if sides.get(u, '') != 'axis' or sides.get(v, '') != 'axis':
            continue
        if node2comp.get(u, None) is None or node2comp.get(v, None) is None:
            continue
        if node2comp[u] != node2comp[v]:
            continue
        cid = node2comp[u]
        ordlist = comp_orders[cid]
        try:
            iu = ordlist.index(u); iv = ordlist.index(v)
        except ValueError:
            continue
        if abs(iu - iv) == 1:
            i0 = min(iu, iv)
            a = ordlist[i0]; b = ordlist[i0 + 1]
            candidate_pairs.append((cid, i0, (a, b)))
    seen_pairs = set()
    uniq_candidates = []
    for cid, i0, pair in candidate_pairs:
        key = tuple(sorted(pair))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        uniq_candidates.append((cid, i0, pair))
    rng.shuffle(uniq_candidates)

    if verbose:
        print(f"  [pair-block-relocation] candidates={len(uniq_candidates)} max_trials={max_trials}")

    for cid, i0, pair in uniq_candidates:
        if trials >= max_trials:
            break
        ordering = list(comp_orders[cid])
        a, b = pair
        try:
            idx_a = ordering.index(a); idx_b = ordering.index(b)
        except ValueError:
            continue
        if abs(idx_a - idx_b) != 1:
            continue
        if idx_b < idx_a:
            a, b = b, a
            idx_a, idx_b = idx_b, idx_a
        base_ordering = ordering[:idx_a] + ordering[idx_b + 1:]
        insertion_positions = list(range(0, len(base_ordering) + 1))
        rng.shuffle(insertion_positions)
        original_insert = idx_a
        for ins in insertion_positions:
            trials += 1
            if trials > max_trials:
                break
            if ins == original_insert:
                continue
            new_order = base_ordering[:ins] + [a, b] + base_ordering[ins:]
            comp_orders_candidate = dict(comp_orders)
            comp_orders_candidate[cid] = new_order
            cand_pos = build_pos_from_comp_orders(baseline_pos, comp_orders_candidate, achiral_gap)
            cand_score = evaluate_layout_score(G, cand_pos, edges_all, sides=sides, min_sep=min_sep)
            if cand_score < best_score:
                if verbose:
                    print(f"    [pair-block] improved: {best_score:.3f} -> {cand_score:.3f} (moved pair {pair} to insertion {ins})")
                best_score = cand_score
                best_pos = dict(cand_pos)
                comp_orders = dict(comp_orders_candidate)
                baseline_pos = dict(cand_pos)
                improved_any = True
                break
    return best_pos, improved_any, baseline_cross, count_edge_crossings(G, best_pos)

if __name__ == "__main__":
    main()
