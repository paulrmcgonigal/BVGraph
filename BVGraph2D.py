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
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from typing import Tuple, List, Dict, Set
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

R_KJ_MOL_K = 0.00831446261815324
LAYOUT_EDGE_OBJECTIVE_SCOPE = "all"

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
def _find_column(df, requested, aliases=()):
    """Return the actual column name using case-insensitive, whitespace-normalised matching."""
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in (requested, *aliases):
        if candidate is None:
            continue
        actual = lookup.get(str(candidate).strip().lower())
        if actual is not None:
            return actual
    return None


def _optional_finite_float(value, label, row_number):
    """Parse a numeric CSV value, returning None for an empty cell."""
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} at CSV row {row_number} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} at CSV row {row_number} must be finite: {value!r}")
    return parsed


def read_graph(
    nodes_csv,
    edges_csv,
    id_col="id",
    energy_col="Energy",
    edge_weighting="none",
    node_energy_column="Relative Energy (kJ/mol)",
    ts_energy_column="Relative TS Energy (kJ/mol)",
    temperature_k=298.15,
    spring_weight_floor_ratio=0.05,
    transient_flux_file=None,
    transient_flux_metric="absolute-net",
    transient_flux_column=None,
):
    # index_col=False keeps legacy CSVs with trailing empty fields from being
    # interpreted as a MultiIndex by newer pandas releases.  Reading every
    # column as text also preserves leading zeroes in barcode metadata.
    nodes = pd.read_csv(nodes_csv, dtype=str, index_col=False)
    edges = pd.read_csv(edges_csv, dtype=str, index_col=False)
    nodes[id_col] = nodes[id_col].astype(str).apply(_clean_id)
    weighting_mode = str(edge_weighting).strip().lower()
    if weighting_mode not in {"none", "equilibrium-exchange", "transient-flux"}:
        raise ValueError(
            "edge_weighting must be 'none', 'equilibrium-exchange', or 'transient-flux'."
        )
    weighting_enabled = weighting_mode != "none"
    if temperature_k <= 0.0 or not math.isfinite(float(temperature_k)):
        raise ValueError("temperature_k must be a positive finite number.")
    floor_ratio = float(spring_weight_floor_ratio)
    if not math.isfinite(floor_ratio) or not (0.0 < floor_ratio <= 1.0):
        raise ValueError("spring_weight_floor_ratio must be greater than 0 and no greater than 1.")
    transient_flux_metric = str(transient_flux_metric).strip().lower()
    if transient_flux_metric not in {"absolute-net", "gross"}:
        raise ValueError("transient_flux_metric must be 'absolute-net' or 'gross'.")
    if weighting_mode == "transient-flux" and not transient_flux_file:
        raise ValueError("--transient-flux-file is required for transient-flux weighting.")

    # normalize edge column names case-insensitively
    src_col = _find_column(edges, "source", aliases=("s",))
    tgt_col = _find_column(edges, "target", aliases=("t",))
    ts_col = _find_column(edges, ts_energy_column, aliases=("TS_Energy", "ts_energy", "ts"))
    if src_col is None or tgt_col is None:
        raise ValueError('Edges CSV must contain Source and Target columns (case-insensitive).')
    if weighting_mode == "equilibrium-exchange" and ts_col is None:
        raise ValueError(
            f"Edges CSV is missing transition-state energy column {ts_energy_column!r}. "
            "Use --ts-energy-column to select a different heading."
        )
    edges['Source'] = edges[src_col].astype(str).apply(_clean_id)
    edges['Target'] = edges[tgt_col].astype(str).apply(_clean_id)

    G = nx.Graph()
    # allow energy column case-insensitively
    energy_col_actual = _find_column(nodes, energy_col)
    relative_energy_col = _find_column(nodes, node_energy_column)
    seen_node_ids = set()
    for row_index, r in nodes.iterrows():
        csv_row = int(row_index) + 2
        nid = _clean_id(r[id_col])
        if not nid:
            raise ValueError(f"Node id at CSV row {csv_row} is empty.")
        if nid in seen_node_ids:
            raise ValueError(f"Duplicate node id {nid!r} at CSV row {csv_row}.")
        seen_node_ids.add(nid)
        relative_energy = _optional_finite_float(
            r.get(relative_energy_col) if relative_energy_col is not None else None,
            node_energy_column,
            csv_row,
        )
        legacy_energy = _optional_finite_float(
            r.get(energy_col_actual) if energy_col_actual is not None else None,
            energy_col,
            csv_row,
        )
        energy = relative_energy if relative_energy is not None else (legacy_energy if legacy_energy is not None else 0.0)
        node_attrs = {energy_col: float(energy)}
        if relative_energy is not None:
            node_attrs["Relative_Energy"] = float(relative_energy)
            node_attrs["Energy_Data_Status"] = "complete"
        else:
            node_attrs["Energy_Data_Status"] = "missing_node_energy"
        G.add_node(nid, **node_attrs)
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
    grouped_edges = defaultdict(list)
    skipped_unknown = 0
    for row_index, r in edges.iterrows():
        csv_row = int(row_index) + 2
        s, t = r["Source"], r["Target"]
        if s not in G or t not in G:
            if weighting_enabled:
                missing = [n for n in (s, t) if n not in G]
                raise ValueError(
                    f"Edge at CSV row {csv_row} references unknown node(s): {', '.join(repr(n) for n in missing)}."
                )
            skipped_unknown += 1
            continue
        ts_energy = _optional_finite_float(
            r.get(ts_col) if ts_col is not None else None,
            ts_energy_column,
            csv_row,
        )
        key = (s, t) if s <= t else (t, s)
        grouped_edges[key].append({"row": csv_row, "ts_energy": ts_energy})

    pending_attrs = {}
    for (u, v), records in grouped_edges.items():
        finite_ts = [record["ts_energy"] for record in records if record["ts_energy"] is not None]
        selected_ts = min(finite_ts) if finite_ts else None
        attrs = {
            "Layout_Spring_Weight": 0.0 if u == v else 1.0,
            "Layout_Weighting_Mode": weighting_mode,
            "Layout_Weight_Data_Available": False,
            "Weight_Data_Status": "unweighted",
            "Input_Edge_Rows": int(len(records)),
        }
        if selected_ts is not None:
            attrs["Relative_TS_Energy"] = float(selected_ts)
            attrs["TS_Energy"] = float(selected_ts)  # legacy export name
        if u == v:
            attrs["Weight_Data_Status"] = "self_loop"
        elif weighting_mode == "equilibrium-exchange" and selected_ts is not None:
            attrs["Layout_Weight_Data_Available"] = True
            attrs["Weight_Data_Status"] = "complete"
            attrs["Layout_Weight_Input_Value"] = float(selected_ts)
            attrs["Layout_Weight_Input_Metric"] = "relative_ts_free_energy_kj_mol"
        elif weighting_mode == "equilibrium-exchange":
            attrs["Weight_Data_Status"] = "missing_ts_energy"
        elif weighting_mode == "none" and selected_ts is not None:
            attrs["Weight_Data_Status"] = "available_unweighted"
        pending_attrs[(u, v)] = attrs

    if weighting_mode == "transient-flux":
        flux = pd.read_csv(transient_flux_file, dtype=str, index_col=False)
        flux_src = _find_column(flux, "source", aliases=("s",))
        flux_tgt = _find_column(flux, "target", aliases=("t",))
        default_flux_column = (
            "Integrated Absolute Net Flux" if transient_flux_metric == "absolute-net"
            else "Integrated Gross Flux"
        )
        selected_flux_col = _find_column(flux, transient_flux_column or default_flux_column)
        absolute_col = _find_column(flux, "Integrated Absolute Net Flux")
        gross_col = _find_column(flux, "Integrated Gross Flux")
        signed_col = _find_column(flux, "Integrated Signed Net Flux")
        status_col = _find_column(flux, "Flux Data Status")
        if flux_src is None or flux_tgt is None or selected_flux_col is None:
            raise ValueError(
                "Transient-flux CSV must contain source, target, and the selected flux column "
                f"{transient_flux_column or default_flux_column!r}."
            )
        seen_flux_edges = set()
        for row_index, row in flux.iterrows():
            csv_row = int(row_index) + 2
            s, t = _clean_id(row.get(flux_src)), _clean_id(row.get(flux_tgt))
            key = (s, t) if s <= t else (t, s)
            if s not in G or t not in G:
                raise ValueError(
                    f"Transient-flux CSV row {csv_row} references unknown node(s): {s!r}, {t!r}."
                )
            if key not in pending_attrs:
                raise ValueError(
                    f"Transient-flux CSV row {csv_row} references non-topology edge {s!r}-{t!r}."
                )
            if key in seen_flux_edges:
                raise ValueError(
                    f"Duplicate undirected transient-flux edge {key[0]!r}-{key[1]!r} at CSV row {csv_row}."
                )
            seen_flux_edges.add(key)
            selected_flux = _optional_finite_float(row.get(selected_flux_col), selected_flux_col, csv_row)
            if selected_flux is not None and selected_flux < 0.0:
                raise ValueError(f"{selected_flux_col} at CSV row {csv_row} must be non-negative.")
            status = _clean_id(row.get(status_col)) if status_col is not None else "complete"
            status_lower = status.lower()
            unavailable = any(token in status_lower for token in ("unavailable", "missing", "omitted"))
            attrs = pending_attrs[key]
            for column, name in (
                (absolute_col, "Integrated_Absolute_Net_Flux"),
                (gross_col, "Integrated_Gross_Flux"),
                (signed_col, "Integrated_Signed_Net_Flux"),
            ):
                value = _optional_finite_float(row.get(column), column, csv_row) if column is not None else None
                if (
                    value is not None
                    and name in {"Integrated_Absolute_Net_Flux", "Integrated_Gross_Flux"}
                    and value < 0.0
                ):
                    raise ValueError(f"{column} at CSV row {csv_row} must be non-negative.")
                if value is not None:
                    attrs[name] = float(value)
            if key[0] == key[1]:
                attrs["Weight_Data_Status"] = "self_loop"
            elif selected_flux is None or unavailable:
                attrs["Weight_Data_Status"] = status or "unavailable_flux"
            else:
                attrs["Layout_Weight_Data_Available"] = True
                attrs["Weight_Data_Status"] = status or "complete"
                attrs["Layout_Weight_Input_Value"] = float(selected_flux)
                attrs["Layout_Weight_Input_Metric"] = (
                    "integrated_absolute_net_flux" if transient_flux_metric == "absolute-net"
                    else "integrated_gross_flux"
                )
        for key, attrs in pending_attrs.items():
            if key[0] != key[1] and key not in seen_flux_edges:
                attrs["Weight_Data_Status"] = "missing_flux_row"

    if weighting_enabled:
        raw_weights = {}
        available_values = [
            float(attrs["Layout_Weight_Input_Value"])
            for (u, v), attrs in pending_attrs.items()
            if u != v and attrs.get("Layout_Weight_Data_Available")
        ]
        reference_min = min(available_values) if available_values else None
        reference_max = max(available_values) if available_values else None
        for (u, v), attrs in pending_attrs.items():
            if u == v:
                continue
            value = attrs.get("Layout_Weight_Input_Value")
            if value is None:
                raw_weights[(u, v)] = floor_ratio
            elif weighting_mode == "equilibrium-exchange":
                exponent = -(float(value) - float(reference_min)) / (R_KJ_MOL_K * float(temperature_k))
                raw_weights[(u, v)] = floor_ratio + (1.0 - floor_ratio) * math.exp(exponent)
            elif reference_max is None or reference_max <= 0.0:
                raw_weights[(u, v)] = floor_ratio
            else:
                raw_weights[(u, v)] = floor_ratio + (1.0 - floor_ratio) * float(value) / float(reference_max)
        raw_mean = sum(raw_weights.values()) / len(raw_weights) if raw_weights else 1.0
        if raw_mean <= 0.0:
            raw_mean = 1.0
        for key, raw_weight in raw_weights.items():
            pending_attrs[key]["Layout_Spring_Weight"] = float(raw_weight / raw_mean)

    for (u, v), attrs in pending_attrs.items():
        G.add_edge(u, v, **attrs)

    if weighting_enabled:
        complete = sum(1 for u, v, d in G.edges(data=True) if u != v and d.get("Layout_Weight_Data_Available"))
        unavailable = sum(1 for u, v, d in G.edges(data=True) if u != v and not d.get("Layout_Weight_Data_Available"))
        self_loops = nx.number_of_selfloops(G)
        duplicate_rows = sum(max(0, len(records) - 1) for records in grouped_edges.values())
        print(
            f"[WEIGHTING] mode={weighting_mode} metric={transient_flux_metric if weighting_mode == 'transient-flux' else 'relative_ts_free_energy'} "
            f"complete_edges={complete} unavailable_edges={unavailable} "
            f"self_loops={self_loops} duplicate_rows={duplicate_rows} rejected_rows=0",
            flush=True,
        )
        if not available_values:
            print(
                "[WARN] No available non-self-loop weighting values were found; all non-self-loop spring weights normalise to 1.",
                flush=True,
            )
    elif skipped_unknown:
        print(f"[WARN] Skipped {skipped_unknown} edge row(s) that reference unknown nodes.", flush=True)
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

def pair_graph_for_bisection(G, pairs, node2pair):
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
    rep_edge_weights = defaultdict(list)
    for u, v, data in G.edges(data=True):
        ru, rv = rep_of.get(u, u), rep_of.get(v, v)
        if ru in reps and rv in reps and ru != rv:
            key = (ru, rv) if ru <= rv else (rv, ru)
            rep_edge_weights[key].append(float(data.get("Layout_Spring_Weight", 1.0)))
    for (ru, rv), weights in rep_edge_weights.items():
        # Averaging preserves the unweighted layout when all underlying weights are 1.
        Grep.add_edge(ru, rv, Layout_Spring_Weight=float(sum(weights) / len(weights)))
    return rep_of, reps, Grep
# ---------------------------
# Representative layout (scale-aware)
# ---------------------------
def compute_layout_scale(G, min_sep=80.0, layout_scale=None, layout_scale_factor=0.75):
    if layout_scale is not None and float(layout_scale) > 0.0:
        return float(layout_scale)
    n = max(1, G.number_of_nodes())
    # A feasible 2D canvas must grow with both node count and required spacing.
    return max(3.0 * float(min_sep), float(layout_scale_factor) * float(min_sep) * math.sqrt(float(n)))

def layout_reps_scale_aware(Grep, layout_scale=900.0, sweeps=4, seed=42):
    nodes = list(Grep.nodes())
    if not nodes:
        return {}
    if Grep.number_of_edges() == 0:
        radius = max(1.0, float(layout_scale))
        k = len(nodes)
        return {n: (radius * math.cos(2 * math.pi * i / max(1, k)), radius * math.sin(2 * math.pi * i / max(1, k))) for i, n in enumerate(nodes)}

    k_val = max(1e-6, float(layout_scale) / max(1.0, math.sqrt(max(1, len(nodes)))))
    pos = nx.spring_layout(
        Grep,
        seed=seed,
        k=k_val,
        iterations=max(25, 25 * max(1, sweeps)),
        scale=float(layout_scale),
        center=(0.0, 0.0),
        dim=2,
        weight="Layout_Spring_Weight",
    )
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
                       target_ratio: float = 1.0,
                       adjustment: str = "expand-smaller") -> Dict[str, Tuple[float, float]]:
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

    if adjustment not in {"expand-smaller", "compress-larger"}:
        raise ValueError("hemisphere span adjustment must be 'expand-smaller' or 'compress-larger'.")

    if axis_span < desired_axis_span:
        if adjustment == "compress-larger":
            scale_target_group = 'chiral'
            current_span = ch_span
            target_span = axis_span / max(1e-12, float(target_ratio))
        else:
            scale_target_group = 'axis'
            current_span = axis_span
            target_span = desired_axis_span
    else:
        if adjustment == "compress-larger":
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


def axis_chiral_span_metrics(pos: Dict[str, Tuple[float, float]], sides: Dict[str, str]) -> Dict[str, float]:
    axis_y = [float(pos[n][1]) for n, side in sides.items() if side == "axis" and n in pos]
    chiral_y = [float(pos[n][1]) for n, side in sides.items() if side != "axis" and n in pos]
    axis_span = max(axis_y) - min(axis_y) if len(axis_y) > 1 else 0.0
    chiral_span = max(chiral_y) - min(chiral_y) if len(chiral_y) > 1 else 0.0
    return {
        "axis_y_span": float(axis_span),
        "chiral_y_span": float(chiral_span),
        "axis_to_chiral_span_ratio": float(axis_span / chiral_span) if chiral_span > 1e-12 else float("inf"),
    }


def expand_chiral_span_to_axis(pos: Dict[str, Tuple[float, float]], sides: Dict[str, str], target_ratio: float = 1.0):
    """Expand, but never compress, the chiral y-span to match the axis target."""
    metrics = axis_chiral_span_metrics(pos, sides)
    target_chiral_span = metrics["axis_y_span"] / max(float(target_ratio), 1e-12)
    if metrics["chiral_y_span"] <= 1e-12 or metrics["chiral_y_span"] >= target_chiral_span:
        return dict(pos), metrics
    scale = target_chiral_span / metrics["chiral_y_span"]
    chiral_y = [float(pos[n][1]) for n, side in sides.items() if side != "axis" and n in pos]
    centre = float(np.median(chiral_y))
    expanded = dict(pos)
    for node, side in sides.items():
        if side == "axis" or node not in expanded:
            continue
        x, y = expanded[node]
        expanded[node] = (x, centre + (float(y) - centre) * scale)
    return expanded, axis_chiral_span_metrics(expanded, sides)

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

def adjust_positions_with_constraints(G, pos, pairs, sides, min_sep=40.0, edge_clearance=12.0, max_iter=200, lr=0.25, fixed_achiral_y=None, axis_lateral_gap=None, strict_min_sep_rounds=0):
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

    if strict_min_sep_rounds:
        enforce_strict_min_sep(pcoords, min_sep, axis_set, pairs, sides, max_rounds=strict_min_sep_rounds)
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


def transactional_cycle_is_better(start_valid, start_score, end_valid, end_score, tolerance=1e-12):
    """Decide whether to retain a cycle once full separation is established."""
    if not start_valid:
        return True
    return bool(end_valid) and float(end_score) < float(start_score) - float(tolerance)


def spearman_rank_correlation(pairs):
    """Compute Spearman correlation with average tie ranks and no SciPy dependency."""
    if len(pairs) < 2:
        return None
    x = pd.Series([float(pair[0]) for pair in pairs], dtype="float64").rank(method="average").to_numpy()
    y = pd.Series([float(pair[1]) for pair in pairs], dtype="float64").rank(method="average").to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    if float(np.std(x)) <= 1e-15 or float(np.std(y)) <= 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])

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

    old_hits = 0
    new_hits = 0
    trials = 0
    n_changed = len(E_changed)
    n_other = len(E_other)
    changed_changed_total = n_changed * (n_changed - 1) // 2
    changed_other_total = n_changed * n_other
    sampling_total = changed_changed_total + changed_other_total

    def point(node):
        return candidate_pos.get(node, pos[node])

    while trials < sample_size:
        if sampling_total <= 0:
            break
        draw = rng.randrange(sampling_total)
        if draw < changed_changed_total:
            i = rng.randrange(n_changed)
            j = rng.randrange(n_changed - 1)
            if j >= i:
                j += 1
            e1, e2 = E_changed[i], E_changed[j]
        else:
            e1 = E_changed[rng.randrange(n_changed)]
            e2 = E_other[rng.randrange(n_other)]
        if set(e1) & set(e2):
            continue
        old_cross = _segments_intersect(pos[e1[0]], pos[e1[1]], pos[e2[0]], pos[e2[1]])
        new_cross = _segments_intersect(point(e1[0]), point(e1[1]), point(e2[0]), point(e2[1]))
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

    def proposal_direction(pid, pmap, grid, inv_cell):
        """Return a mirror-coordinate direction from weighted pull and soft repulsion."""
        force_x, force_y = 0.0, 0.0
        soft_sep = float(min_sep) * float(LAYOUT_SOFT_SEP_FACTOR)
        own_nodes = set(pairs[pid])
        for node in own_nodes:
            x, y = pmap[node]
            sign = 1.0 if x >= 0.0 else -1.0
            for nbr, edge_data in G[node].items():
                if nbr in own_nodes:
                    continue
                if not edge_in_length_objective(G, node, nbr):
                    continue
                nx_, ny_ = pmap[nbr]
                weight = float(edge_data.get("Layout_Spring_Weight", 1.0))
                force_x += sign * weight * (nx_ - x)
                force_y += weight * (ny_ - y)
            ix, iy = int(math.floor(x * inv_cell)), int(math.floor(y * inv_cell))
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for other in grid.get((ix + dx_cell, iy + dy_cell), []):
                        if other in own_nodes:
                            continue
                        ox, oy = pmap[other]
                        dx, dy = x - ox, y - oy
                        distance = math.hypot(dx, dy)
                        if distance >= soft_sep:
                            continue
                        if distance < 1e-12:
                            dx, dy, distance = 1.0, 0.0, 1.0
                        repulsion = (soft_sep - distance) / soft_sep
                        force_x += sign * repulsion * (dx / distance) * float(min_sep)
                        force_y += repulsion * (dy / distance) * float(min_sep)
        magnitude = math.hypot(force_x, force_y)
        return None if magnitude < 1e-12 else (force_x / magnitude, force_y / magnitude)

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
            direction = proposal_direction(pid, best_pos, spatial_grid, inv_cell)
            if direction is None:
                continue
            cur_mag, cur_y = pair_center(pid, best_pos)
            step = 0.5 * float(min_sep)
            new_mag = max(min_x, cur_mag + x_rate * step * direction[0])
            new_y = cur_y + y_rate * step * direction[1]

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


def separation_diagnostics(pos, min_sep):
    """Return spatially indexed hard-separation diagnostics.

    ``total_squared_deficit`` is continuous and is used to decide whether a
    coordinate-changing operation has made an infeasible layout worse.  The
    count and minimum distance remain useful human-readable diagnostics.
    """
    target = float(min_sep)
    if target <= 0.0 or len(pos) < 2:
        return {
            "threshold": target,
            "remaining_violations": 0,
            "minimum_distance": float("inf"),
            "maximum_fractional_deficit": 0.0,
            "total_squared_deficit": 0.0,
        }
    grid, inv_cell = _build_spatial_hash(pos, target)
    seen = set()
    remaining = 0
    min_distance = float("inf")
    max_deficit = 0.0
    total_deficit = 0.0
    for node, (x, y) in pos.items():
        x, y = float(x), float(y)
        ix, iy = int(math.floor(x * inv_cell)), int(math.floor(y * inv_cell))
        for dx_cell in (-1, 0, 1):
            for dy_cell in (-1, 0, 1):
                for other in grid.get((ix + dx_cell, iy + dy_cell), []):
                    if other == node:
                        continue
                    key = (node, other) if node <= other else (other, node)
                    if key in seen:
                        continue
                    seen.add(key)
                    ox, oy = pos[other]
                    distance = math.hypot(x - float(ox), y - float(oy))
                    min_distance = min(min_distance, distance)
                    if distance >= target:
                        continue
                    remaining += 1
                    deficit = (target - distance) / target
                    max_deficit = max(max_deficit, deficit)
                    total_deficit += deficit * deficit
    return {
        "threshold": target,
        "remaining_violations": remaining,
        "minimum_distance": min_distance,
        "maximum_fractional_deficit": max_deficit,
        "total_squared_deficit": total_deficit,
    }


def separation_not_worse(before, after, tolerance=1e-10):
    """Accept equal/improved infeasibility, and preserve feasibility once reached."""
    if before["remaining_violations"] == 0:
        return after["remaining_violations"] == 0
    scale = max(1.0, float(before["total_squared_deficit"]))
    return float(after["total_squared_deficit"]) <= float(before["total_squared_deficit"]) + tolerance * scale


def enforce_minimum_separation_mirror_preserving(
    pos, pairs, sides, min_sep=60.0, max_iter=200, damping=0.65, stall_limit=12
):
    """Project close nodes apart with damped sequential mirror-safe updates.

    A spatial hash finds only nearby pairs.  Corrections are applied
    sequentially (Gauss-Seidel style), so later collisions see earlier moves
    rather than all nodes responding to stale coordinates.  A pass is retained
    only when the continuous separation deficit decreases.
    """
    target = float(min_sep)
    # A tiny overshoot avoids floating-point/stall-limit residues just below
    # the requested hard distance without visibly changing the layout scale.
    projection_target = target + max(1e-8, abs(target) * 1e-4)
    work, _ = enforce_strict_mirror(pos, pairs, sides)
    initial_diag = separation_diagnostics(work, target)
    if target <= 0.0 or len(work) < 2 or max_iter <= 0 or initial_diag["remaining_violations"] == 0:
        return work, {"iterations": 0, "converged": initial_diag["remaining_violations"] == 0, **initial_diag}

    # When a substantial fraction of the graph is crowded, a small uniform
    # expansion reduces collective jamming before local corrections.  The cap
    # limits any one projection event to three percent; crossings and mirror
    # symmetry are unchanged by this transformation.
    pre_scale = 1.0
    crowded_cutoff = max(100, int(0.10 * len(work)))
    if initial_diag["remaining_violations"] >= crowded_cutoff and initial_diag["minimum_distance"] > 0.0:
        desired_scale = math.sqrt(projection_target / float(initial_diag["minimum_distance"]))
        pre_scale = min(1.03, max(1.0, desired_scale))
        if pre_scale > 1.000001:
            center_y = float(np.median([float(point[1]) for point in work.values()]))
            work = {
                node: (
                    float(point[0]) * pre_scale,
                    center_y + (float(point[1]) - center_y) * pre_scale,
                )
                for node, point in work.items()
            }
            work, _ = enforce_strict_mirror(work, pairs, sides)
            initial_diag = separation_diagnostics(work, target)

    partner = {}
    for a, b in pairs.values():
        partner[a] = b
        partner[b] = a

    def entity(node):
        if sides.get(node) == "axis" or node not in partner:
            return ("axis", node)
        return ("pair", min(node, partner[node]))

    def stable_direction(a, b, iteration):
        value = sum((index + 1) * ord(ch) for index, ch in enumerate(f"{a}|{b}|{iteration}")) % 360
        angle = math.radians(float(value))
        return math.cos(angle), math.sin(angle)

    def move_entity_through_node(node, dx, dy, step_cap):
        length = math.hypot(dx, dy)
        if length > step_cap:
            scale = step_cap / length
            dx, dy = dx * scale, dy * scale
        key = entity(node)
        if key[0] == "axis":
            work[node] = (0.0, float(work[node][1]) + dy)
            return
        other = partner[node]
        right, left = (node, other) if work[node][0] >= 0.0 else (other, node)
        physical_sign = 1.0 if work[node][0] >= 0.0 else -1.0
        magnitude = max(0.5 * projection_target, abs(float(work[right][0])) + physical_sign * dx)
        common_y = 0.5 * (float(work[right][1]) + float(work[left][1])) + dy
        work[right] = (magnitude, common_y)
        work[left] = (-magnitude, common_y)

    best_work = dict(work)
    best_diag = initial_diag
    current_damping = min(1.0, max(0.05, float(damping)))
    stalled = 0
    iterations_used = 0
    for iteration in range(1, int(max_iter) + 1):
        iterations_used = iteration
        grid, inv_cell = _build_spatial_hash(work, target)
        seen = set()
        violations = []
        for node, (x, y) in work.items():
            x, y = float(x), float(y)
            ix, iy = int(math.floor(x * inv_cell)), int(math.floor(y * inv_cell))
            for dx_cell in (-1, 0, 1):
                for dy_cell in (-1, 0, 1):
                    for other in grid.get((ix + dx_cell, iy + dy_cell), []):
                        if other == node:
                            continue
                        key = (node, other) if node <= other else (other, node)
                        if key in seen:
                            continue
                        seen.add(key)
                        ox, oy = work[other]
                        distance = math.hypot(x - float(ox), y - float(oy))
                        if distance < target:
                            violations.append((distance, node, other))
        if not violations:
            final_diag = separation_diagnostics(work, target)
            return work, {"iterations": iteration - 1, "converged": True, "pre_scale": pre_scale, **final_diag}

        # Resolve the deepest overlaps first; reverse ties on alternate passes
        # so a stable identifier ordering cannot trap the same crowded region.
        violations.sort(key=lambda item: (item[0], item[1], item[2]), reverse=(iteration % 2 == 0))
        snapshot = dict(work)
        step_cap = max(1e-9, 0.25 * target)
        for _, node, other in violations:
            x, y = work[node]
            ox, oy = work[other]
            dx, dy = float(x) - float(ox), float(y) - float(oy)
            distance = math.hypot(dx, dy)
            if distance >= target:
                continue
            if entity(node) == entity(other):
                if entity(node)[0] == "pair":
                    a, b = node, other
                    right, left = (a, b) if work[a][0] >= 0.0 else (b, a)
                    common_y = 0.5 * (float(work[right][1]) + float(work[left][1]))
                    work[right] = (max(abs(float(work[right][0])), 0.5 * projection_target), common_y)
                    work[left] = (-max(abs(float(work[right][0])), 0.5 * projection_target), common_y)
                continue
            if distance < 1e-12:
                ux, uy = stable_direction(node, other, iteration)
            else:
                ux, uy = dx / distance, dy / distance
            correction = current_damping * (projection_target - distance)
            move_entity_through_node(node, 0.5 * correction * ux, 0.5 * correction * uy, step_cap)
            move_entity_through_node(other, -0.5 * correction * ux, -0.5 * correction * uy, step_cap)

        work, _ = enforce_strict_mirror(work, pairs, sides)
        candidate_diag = separation_diagnostics(work, target)
        if candidate_diag["remaining_violations"] == 0:
            return work, {"iterations": iteration, "converged": True, "pre_scale": pre_scale, **candidate_diag}
        previous = float(best_diag["total_squared_deficit"])
        candidate = float(candidate_diag["total_squared_deficit"])
        if candidate + 1e-12 < previous:
            best_work = dict(work)
            best_diag = candidate_diag
            relative_gain = (previous - candidate) / max(previous, 1e-12)
            stalled = stalled + 1 if relative_gain < 1e-5 else 0
            current_damping = min(0.85, current_damping * 1.02)
        else:
            work = dict(best_work)
            current_damping = max(0.05, current_damping * 0.5)
            stalled += 1
        if stalled >= int(stall_limit):
            break

    # Dense layouts can approach the boundary asymptotically.  Once every
    # residual is within one percent of the threshold, a sub-percent uniform
    # expansion about the median y coordinate guarantees feasibility while
    # preserving mirror symmetry and the complete crossing topology.
    min_distance = float(best_diag["minimum_distance"])
    if math.isfinite(min_distance) and min_distance > 0.0:
        fallback_scale = projection_target / min_distance
        if fallback_scale <= 1.01:
            center_y = float(np.median([float(point[1]) for point in best_work.values()]))
            expanded = {
                node: (
                    float(point[0]) * fallback_scale,
                    center_y + (float(point[1]) - center_y) * fallback_scale,
                )
                for node, point in best_work.items()
            }
            expanded, _ = enforce_strict_mirror(expanded, pairs, sides)
            expanded_diag = separation_diagnostics(expanded, target)
            if expanded_diag["remaining_violations"] == 0:
                return expanded, {
                    "iterations": iterations_used,
                    "converged": True,
                    "pre_scale": pre_scale,
                    "fallback_scale": fallback_scale,
                    **expanded_diag,
                }
    return best_work, {"iterations": iterations_used, "converged": False, "pre_scale": pre_scale, **best_diag}


def project_candidate_for_separation(
    base_pos,
    candidate_pos,
    pairs,
    sides,
    active_separation,
    max_iter=100,
    stall_limit=12,
):
    """Repair a candidate before objective scoring when it worsens separation.

    Optimisation stages are allowed to pass temporarily through an infeasible
    layout.  If a complete stage worsens the active separation constraint, its
    result is projected back toward feasibility while preserving mirror
    symmetry.  The caller must score the returned coordinates, not the raw
    candidate, so a move is retained only when its repaired form improves the
    full objective.
    """
    target = float(active_separation)
    before = separation_diagnostics(base_pos, target)
    candidate_diag = separation_diagnostics(candidate_pos, target)
    if target <= 0.0 or separation_not_worse(before, candidate_diag):
        return dict(candidate_pos), candidate_diag, False, True

    projected, projected_diag = enforce_minimum_separation_mirror_preserving(
        candidate_pos,
        pairs,
        sides,
        min_sep=target,
        max_iter=max_iter,
        stall_limit=stall_limit,
    )
    return projected, projected_diag, True, separation_not_worse(before, projected_diag)

def write_gexf_with_viz(G, pos, out_path):
    with open(filesystem_path(out_path), "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<gexf xmlns="http://www.gexf.net/1.2draft" xmlns:viz="http://www.gexf.net/1.2draft/viz" version="1.2">\n')
        f.write('  <graph mode="static" defaultedgetype="undirected">\n')
        f.write('    <attributes class="node">\n')
        f.write('      <attribute id="0" title="Energy" type="float"/>\n')
        f.write('      <attribute id="1" title="DeltaE" type="float"/>\n')
        f.write('      <attribute id="2" title="Chirality" type="string"/>\n')
        f.write('      <attribute id="3" title="Enantiomer_Id" type="string"/>\n')
        f.write('      <attribute id="4" title="Relative_Energy" type="float"/>\n')
        f.write('      <attribute id="5" title="Central_Layout_Target" type="boolean"/>\n')
        f.write('      <attribute id="6" title="Hemisphere_Optimization" type="string"/>\n')
        f.write('    </attributes>\n')
        f.write('    <attributes class="edge">\n')
        f.write('      <attribute id="10" title="TS_Energy" type="float"/>\n')
        f.write('      <attribute id="11" title="Relative_TS_Energy" type="float"/>\n')
        f.write('      <attribute id="12" title="Weight_Data_Status" type="string"/>\n')
        f.write('      <attribute id="13" title="Layout_Weighting_Mode" type="string"/>\n')
        f.write('      <attribute id="14" title="Layout_Spring_Weight" type="float"/>\n')
        f.write('      <attribute id="15" title="Input_Edge_Rows" type="integer"/>\n')
        f.write('      <attribute id="16" title="Layout_Weight_Input_Metric" type="string"/>\n')
        f.write('      <attribute id="17" title="Layout_Weight_Input_Value" type="float"/>\n')
        f.write('      <attribute id="18" title="Layout_Weight_Data_Available" type="boolean"/>\n')
        f.write('      <attribute id="19" title="Integrated_Absolute_Net_Flux" type="float"/>\n')
        f.write('      <attribute id="20" title="Integrated_Gross_Flux" type="float"/>\n')
        f.write('      <attribute id="21" title="Integrated_Signed_Net_Flux" type="float"/>\n')
        f.write('      <attribute id="22" title="Cross_Axis_At_Final" type="boolean"/>\n')
        f.write('      <attribute id="23" title="Hemisphere_Optimization" type="string"/>\n')
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
            if "Relative_Energy" in data:
                f.write(f'          <attvalue for="4" value="{float(data["Relative_Energy"])}"/>\n')
            f.write(f'          <attvalue for="5" value="{str(bool(data.get("Central_Layout_Target", False))).lower()}"/>\n')
            f.write(f'          <attvalue for="6" value="{data.get("Hemisphere_Optimization", "local")}"/>\n')
            f.write('        </attvalues>\n')
            f.write(f'        <viz:position x="{x}" y="{y}" z="0"/>\n')
            f.write('      </node>\n')
        f.write('    </nodes>\n')
        f.write('    <edges>\n')
        i = 0
        for u, v, data in G.edges(data=True):
            f.write(f'      <edge id="{i}" source="{u}" target="{v}">\n')
            f.write('        <attvalues>\n')
            if "TS_Energy" in data:
                f.write(f'          <attvalue for="10" value="{float(data["TS_Energy"])}"/>\n')
            if "Relative_TS_Energy" in data:
                f.write(f'          <attvalue for="11" value="{float(data["Relative_TS_Energy"])}"/>\n')
            f.write(f'          <attvalue for="12" value="{data.get("Weight_Data_Status", "unweighted")}"/>\n')
            f.write(f'          <attvalue for="13" value="{data.get("Layout_Weighting_Mode", "none")}"/>\n')
            f.write(f'          <attvalue for="14" value="{float(data.get("Layout_Spring_Weight", 1.0))}"/>\n')
            f.write(f'          <attvalue for="15" value="{int(data.get("Input_Edge_Rows", 1))}"/>\n')
            if data.get("Layout_Weight_Input_Metric"):
                f.write(f'          <attvalue for="16" value="{data["Layout_Weight_Input_Metric"]}"/>\n')
            if "Layout_Weight_Input_Value" in data:
                f.write(f'          <attvalue for="17" value="{float(data["Layout_Weight_Input_Value"])}"/>\n')
            f.write(f'          <attvalue for="18" value="{str(bool(data.get("Layout_Weight_Data_Available", False))).lower()}"/>\n')
            for attr_id, attr_name in (("19", "Integrated_Absolute_Net_Flux"), ("20", "Integrated_Gross_Flux"), ("21", "Integrated_Signed_Net_Flux")):
                if attr_name in data:
                    f.write(f'          <attvalue for="{attr_id}" value="{float(data[attr_name])}"/>\n')
            f.write(f'          <attvalue for="22" value="{str(bool(data.get("Cross_Axis_At_Final", False))).lower()}"/>\n')
            f.write(f'          <attvalue for="23" value="{data.get("Hemisphere_Optimization", "local")}"/>\n')
            f.write('        </attvalues>\n')
            f.write('      </edge>\n')
            i += 1
        f.write('    </edges>\n')
        f.write('  </graph>\n')
        f.write('</gexf>\n')

def load_positions_from_gexf(gexf_path, nodes=None):
    """Load node positions from a Gephi/GEXF file written by write_gexf_with_viz()."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(filesystem_path(gexf_path))
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


def filesystem_path(path):
    """Use Windows extended-length syntax while keeping ordinary paths elsewhere."""
    path = Path(path)
    if os.name != "nt":
        return path
    absolute = os.path.abspath(str(path))
    if absolute.startswith("\\\\?\\"):
        return Path(absolute)
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def replace_atomic_with_retries(
    temporary, destination, retry_attempts=8, retry_delay=0.25, recreate_temporary=None
):
    """Replace atomically, retrying locks and preserving a recovery file if needed."""
    temporary = Path(temporary)
    destination = Path(destination)
    attempts = max(1, int(retry_attempts))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            filesystem_path(temporary).replace(filesystem_path(destination))
            return destination
        except (PermissionError, FileNotFoundError) as exc:
            last_error = exc
            if isinstance(exc, FileNotFoundError) and recreate_temporary is not None:
                recreate_temporary(temporary)
            if attempt < attempts:
                if attempt == 1:
                    print(
                        f"[CHECKPOINT] destination is locked; retrying atomic replace -> {destination}",
                        flush=True,
                    )
                time.sleep(max(0.0, float(retry_delay)) * attempt)

    timestamp = time.strftime("%Y%m%dT%H%M%S")
    recovery = destination.with_name(
        f"{destination.stem}.recovery-{timestamp}-{time.time_ns() % 1_000_000_000:09d}{destination.suffix}"
    )
    if not filesystem_path(temporary).exists() and recreate_temporary is not None:
        recreate_temporary(temporary)
    filesystem_path(temporary).replace(filesystem_path(recovery))
    print(
        f"[CHECKPOINT] WARNING: could not replace locked destination after {attempts} attempts; "
        f"preserved newest data at {recovery}. Last error: {last_error}",
        flush=True,
    )
    return recovery


def unique_atomic_temporary_path(destination):
    """Return a short, same-directory, per-write temporary checkpoint path."""
    path = Path(destination)
    return path.with_name(f".bv-{os.getpid()}-{time.time_ns()}")


def write_gexf_atomic(G, pos, out_path):
    """Write a GEXF checkpoint atomically and return the path actually written."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the unique basename short so deeply nested Windows output paths do
    # not cross the classic 260-character path boundary.
    temporary = unique_atomic_temporary_path(path)
    def recreate(temp_path):
        write_gexf_with_viz(G, pos, temp_path)
    recreate(temporary)
    return replace_atomic_with_retries(
        temporary, path, recreate_temporary=recreate
    )


def write_json_atomic(payload, out_path):
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = unique_atomic_temporary_path(path)
    def recreate(temp_path):
        filesystem_path(temp_path).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    recreate(temporary)
    return replace_atomic_with_retries(
        temporary, path, recreate_temporary=recreate
    )


def parse_separation_levels(text, min_sep):
    """Parse increasing fractions or absolute distances ending at min_sep."""
    if text is None or not str(text).strip():
        values = [0.33, 0.50, 0.67, 0.83, 0.90, 0.95, 1.00]
    else:
        try:
            values = [float(part.strip()) for part in str(text).split(",") if part.strip()]
        except ValueError as exc:
            raise ValueError("--separation-levels must be a comma-separated list of positive numbers.") from exc
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--separation-levels must contain positive numbers.")
    target = float(min_sep)
    distances = [value * target if value <= 1.0 else value for value in values]
    distances.append(target)
    return sorted(set(min(target, max(1e-9, float(value))) for value in distances))

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
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<graph label="{saxutils.escape(graph_label)}" directed="0" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns="http://www.cs.rpi.edu/XGMML">\n')
        # nodes
        for n, data in G.nodes(data=True):
            x, y = pos.get(n, (0.0, 0.0))
            nid = saxutils.escape(str(n))
            f.write(f'  <node id="{nid}" label="{nid}">\n')
            # primary attributes (Energy, DeltaE, Chirality, Enantiomer_Id, Barcode_Normalized)
            f.write(attr_xml("Energy", float(data.get("Energy", 0.0)), "real"))
            f.write(attr_xml("DeltaE", float(data.get("DeltaE", 0.0)), "real"))
            f.write(attr_xml("Chirality", data.get("Chirality", ""), "string"))
            f.write(attr_xml("Enantiomer_Id", data.get("Enantiomer_Id", ""), "string"))
            f.write(attr_xml("Barcode_Normalized", data.get("Barcode_Normalized", ""), "string"))
            if "Relative_Energy" in data:
                f.write(attr_xml("Relative_Energy", float(data["Relative_Energy"]), "real"))
            f.write(attr_xml("Central_Layout_Target", bool(data.get("Central_Layout_Target", False)), "boolean"))
            f.write(attr_xml("Hemisphere_Optimization", data.get("Hemisphere_Optimization", "local"), "string"))
            # add graphics element with coordinates
            f.write(f'    <graphics x="{float(x)}" y="{float(y)}"/>\n')
            f.write('  </node>\n')
        # edges
        edge_id = 0
        for u, v, data in G.edges(data=True):
            uid = saxutils.escape(str(u))
            vid = saxutils.escape(str(v))
            f.write(f'  <edge id="e{edge_id}" label="e{edge_id}" source="{uid}" target="{vid}">\n')
            if "TS_Energy" in data:
                f.write(attr_xml("TS_Energy", float(data["TS_Energy"]), "real"))
            if "Relative_TS_Energy" in data:
                f.write(attr_xml("Relative_TS_Energy", float(data["Relative_TS_Energy"]), "real"))
            f.write(attr_xml("Weight_Data_Status", data.get("Weight_Data_Status", "unweighted"), "string"))
            f.write(attr_xml("Layout_Weighting_Mode", data.get("Layout_Weighting_Mode", "none"), "string"))
            f.write(attr_xml("Layout_Weight_Input_Metric", data.get("Layout_Weight_Input_Metric", ""), "string"))
            if "Layout_Weight_Input_Value" in data:
                f.write(attr_xml("Layout_Weight_Input_Value", float(data["Layout_Weight_Input_Value"]), "real"))
            f.write(attr_xml("Layout_Weight_Data_Available", bool(data.get("Layout_Weight_Data_Available", False)), "boolean"))
            for attr_name in ("Integrated_Absolute_Net_Flux", "Integrated_Gross_Flux", "Integrated_Signed_Net_Flux"):
                if attr_name in data:
                    f.write(attr_xml(attr_name, float(data[attr_name]), "real"))
            f.write(attr_xml("Layout_Spring_Weight", float(data.get("Layout_Spring_Weight", 1.0)), "real"))
            f.write(attr_xml("Input_Edge_Rows", int(data.get("Input_Edge_Rows", 1)), "integer"))
            f.write(attr_xml("Cross_Axis_At_Final", bool(data.get("Cross_Axis_At_Final", False)), "boolean"))
            f.write(attr_xml("Hemisphere_Optimization", data.get("Hemisphere_Optimization", "local"), "string"))
            f.write('  </edge>\n')
            edge_id += 1
        f.write('</graph>\n')

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
            "Relative Energy (kJ/mol)": data.get("Relative_Energy", ""),
            "Energy Data Status": data.get("Energy_Data_Status", ""),
            "Enantiomer_Id": data.get("Enantiomer_Id", ""),
            "Central Layout Target": bool(data.get("Central_Layout_Target", False)),
            "Hemisphere Optimization": data.get("Hemisphere_Optimization", "local"),
        })
    df = _pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

def write_edges_csv(G, out_path):
    """
    Writes a CSV with layout-energy metadata for each simple undirected edge.
    """
    import pandas as _pd
    rows = []
    for u, v, data in G.edges(data=True):
        rows.append({
            "Source": u,
            "Target": v,
            "TS_Energy": data.get("TS_Energy", ""),
            "Relative TS Energy (kJ/mol)": data.get("Relative_TS_Energy", ""),
            "Weight Data Status": data.get("Weight_Data_Status", "unweighted"),
            "Layout Weighting Mode": data.get("Layout_Weighting_Mode", "none"),
            "Layout Weight Input Metric": data.get("Layout_Weight_Input_Metric", ""),
            "Layout Weight Input Value": data.get("Layout_Weight_Input_Value", ""),
            "Layout Weight Data Available": bool(data.get("Layout_Weight_Data_Available", False)),
            "Integrated Absolute Net Flux": data.get("Integrated_Absolute_Net_Flux", ""),
            "Integrated Gross Flux": data.get("Integrated_Gross_Flux", ""),
            "Integrated Signed Net Flux": data.get("Integrated_Signed_Net_Flux", ""),
            "Layout Spring Weight": float(data.get("Layout_Spring_Weight", 1.0)),
            "Input Edge Rows": int(data.get("Input_Edge_Rows", 1)),
            "Cross Axis At Final": bool(data.get("Cross_Axis_At_Final", False)),
            "Hemisphere Optimization": data.get("Hemisphere_Optimization", "local"),
        })
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
    ap.add_argument("--overlap-iter", type=int, default=0,
                    help="Optional all-pairs node/edge clearance iterations; leave at 0 for scalable layouts.")
    ap.add_argument("--final-separation-iters", type=int, default=200,
                    help="Mirror-preserving spatial-grid iterations used to enforce --min-sep after final mirroring.")
    ap.add_argument(
        "--separation-mode", choices=["legacy_final", "progressive"], default="legacy_final",
        help="Use legacy final-only enforcement or progressive mirror-preserving separation during refinement.",
    )
    ap.add_argument(
        "--separation-levels", default="0.33,0.50,0.67,0.83,0.90,0.95,1.00",
        help="Comma-separated fractions of --min-sep (values <=1) or absolute distances for progressive enforcement.",
    )
    ap.add_argument("--separation-project-every", type=int, default=5,
                    help="Run a progressive global projection every N completed cycles while the active level has violations.")
    ap.add_argument("--separation-project-iters", type=int, default=100,
                    help="Maximum damped spatial projection passes at each progressive repair event.")
    ap.add_argument("--separation-stall-passes", type=int, default=12,
                    help="Stop a projection after this many passes without material separation-deficit improvement.")
    ap.add_argument("--layout-scale-factor", type=float, default=0.75,
                    help="Initial layout radius factor multiplied by --min-sep and sqrt(node count).")
    ap.add_argument("--objective-mode", choices=["legacy", "initial_normalized"], default="legacy",
                    help="Use original raw score coefficients or normalize crossings and weighted edge length to the starting layout.")
    ap.add_argument("--objective-reference-crossings", type=float, default=None,
                    help="Optional fixed crossing reference for initial_normalized mode (use when continuing a prior run).")
    ap.add_argument("--objective-reference-weighted-length", type=float, default=None,
                    help="Optional fixed weighted-edge-length reference for initial_normalized mode (use when continuing a prior run).")
    ap.add_argument("--objective-reference-spacing-penalty", type=float, default=None,
                    help="Optional fixed soft-spacing reference for initial_normalized mode (use when continuing a prior run).")
    ap.add_argument("--objective-reference-center-penalty", type=float, default=None,
                    help="Optional fixed central-isomer reference penalty for initial_normalized resume runs.")
    ap.add_argument("--objective-crossing-weight", type=float, default=0.20,
                    help="Crossing importance; in initial_normalized mode this is the requested normalized coefficient.")
    ap.add_argument("--objective-edge-length-weight", type=float, default=1.00,
                    help="Energy-weighted edge-length importance; normalized to the starting value when requested.")
    ap.add_argument(
        "--edge-objective-scope",
        choices=["all", "weighted-data"],
        default="all",
        help="Edges included in edge-length objective terms; weighted-data excludes records without valid mode-specific weighting data.",
    )
    ap.add_argument("--objective-spacing-weight", type=float, default=8.00,
                    help="Soft spacing importance; normalized to the starting penalty in initial_normalized mode. Hard minimum separation remains non-negotiable.")
    ap.add_argument("--edge-length-power", type=float, default=1.0,
                    help="Power applied to normalized edge lengths in the layout objective; 1.0 preserves the standard linear score.")
    ap.add_argument("--cross-axis-edge-weight", type=float, default=0.0,
                    help="Optional normalized penalty for weighted chiral-chiral edges whose endpoints occupy opposite hemispheres.")
    ap.add_argument("--soft-separation-factor", type=float, default=1.35,
                    help="Soft preferred separation as a multiple of --min-sep (must be at least 1).")
    ap.add_argument("--out-metrics-json", default=None,
                    help="Optional JSON file recording objective references and final layout metrics.")
    ap.add_argument("--achiral-gap", type=float, default=None, help="If not set, will be set equal to --min-sep.")
    ap.add_argument("--swap-iters", type=int, default=1000)
    ap.add_argument("--axis-min-gap", type=float, default=None)
    ap.add_argument("--axis-adjust-iters", type=int, default=6)
    ap.add_argument("--pair-relocation-trials", type=int, default=200)
    ap.add_argument("--axis-lateral-gap", type=float, default=None)
    ap.add_argument("--chiral-swap-iters", type=int, default=500)
    ap.add_argument("--pairpair-iters", type=int, default=100)
    ap.add_argument("--pair-flip-iters", type=int, default=50,
                    help="Maximum pair-flip attempts per refinement cycle.")
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
    ap.add_argument("--convergence-patience", type=int, default=1,
                    help="Require this many consecutive stalled cycles before stopping early (default 1).")
    ap.add_argument(
        "--transactional-valid-cycles",
        action="store_true",
        help="Once full minimum separation is reached, retain a cycle only if its repaired final layout remains valid and improves the cycle-start objective.",
    )
    ap.add_argument(
        "--stage-diagnostics",
        action="store_true",
        help="Log retained-coordinate movement, objective and separation diagnostics after every move class.",
    )
    ap.add_argument("--run-all-refine-cycles", action="store_true",
                    help="Run every requested refinement cycle even if a cycle has no accepted move or falls below --cycle-shift-tol.")
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
    ap.add_argument(
        "--max-runtime-hours",
        type=float,
        default=0.0,
        help="Optional wall-clock limit; 0 disables it. A clean stop occurs after a completed cycle and checkpoint.",
    )
    ap.add_argument("--achiral-adjacency-weight", type=float, default=50.0)
    ap.add_argument("--achiral-gap-weight", type=float, default=1.0)
    ap.add_argument("--hemi-span-tol", type=float, default=0.10, help="Tolerance for hemisphere/axis span equilisation.")
    ap.add_argument("--hemi-max-scale", type=float, default=1.5, help="Maximum per-call scale factor for equalize_hemi_span.")
    ap.add_argument("--hemi-min-nodes", type=int, default=2, help="Minimum # chiral nodes required to attempt hemi equalisation.")
    ap.add_argument("--hemi-target-ratio", type=float, default=1.0, help="Desired axis_span / chiral_span ratio (default 1.0).")
    ap.add_argument(
        "--hemi-span-adjustment",
        choices=["expand-smaller", "compress-larger"],
        default="expand-smaller",
        help=(
            "How objective-gated span proposals approach --hemi-target-ratio. "
            "The default expands the shorter group; compress-larger instead contracts the taller group."
        ),
    )
    ap.add_argument(
        "--final-span-expansion",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable or disable the final unconditional expansion of the chiral y-span to the axis target. "
            "Use off to retain the objective-optimised span at export."
        ),
    )
    ap.add_argument("--enantiomer-swap-iters", type=int, default=100, help="Attempts per-cycle for enantiomer X-swaps (default 100).")
    ap.add_argument(
        "--hemisphere-optimization",
        choices=["local", "global", "adaptive"],
        default="local",
        help=(
            "Enantiomer-side optimisation scale: local retains the standard pairwise moves; "
            "global adds a full orientation pass; adaptive also tries connected block flips and block relaxation."
        ),
    )
    ap.add_argument("--hemisphere-pass-every", type=int, default=5,
                    help="Run adaptive global orientation every N completed cycles.")
    ap.add_argument("--hemisphere-stall-trigger", type=int, default=3,
                    help="Run an adaptive orientation pass after this many consecutive non-improving cycles.")
    ap.add_argument("--hemisphere-block-max-pairs", type=int, default=128,
                    help="Maximum number of enantiomeric pairs in an adaptive connected block candidate.")
    ap.add_argument("--hemisphere-block-seeds", type=int, default=16,
                    help="Maximum costly pair-pair connections used to seed adaptive block candidates per pass.")
    ap.add_argument("--hemisphere-block-relax-iters", type=int, default=5,
                    help="Damped coherent relaxation iterations proposed after an accepted block flip.")
    ap.add_argument("--hemisphere-block-relax-halo", type=int, default=1,
                    help="Number of neighbouring pair layers included with reduced mobility during block relaxation.")
    ap.add_argument("--center-isomer", "--centre-isomer", dest="center_isomer", default=None,
                    help="Optional node barcode whose node, or complete enantiomeric pair, is softly prioritised at the vertical midpoint.")
    ap.add_argument("--center-weight", "--centre-weight", dest="center_weight", type=float, default=None,
                    help="Normalized centrality coefficient; defaults to 0.10 when --center-isomer is supplied and is otherwise inactive.")
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
    ap.add_argument("--checkpoint-state-json", default=None,
                    help="Atomic JSON sidecar for cycle, objective, separation-stage and best-layout restart metadata.")
    ap.add_argument("--resume-state-json", default=None,
                    help="Optional checkpoint JSON whose cycle and progressive-separation state should be resumed.")
    ap.add_argument("--best-checkpoint-gephi", default=None,
                    help="Optional atomic GEXF retaining the best state at the most advanced separation level.")
    ap.add_argument("--cycle-offset", type=int, default=0,
                    help="Completed global cycles before this invocation; used for numbering and non-repeating random seeds.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out-xgmml", default=None, help="Optional output XGMML file path (Cytoscape).")
    ap.add_argument("--out-nodes-csv", default=None, help="Optional output CSV with node coords & barcodes.")
    ap.add_argument("--out-edges-csv", default=None, help="Optional output CSV with edge data.")
    ap.add_argument(
        "--edge-weighting",
        choices=["none", "equilibrium-exchange", "transient-flux"],
        default="none",
        help="Select uniform, absolute-TS equilibrium-exchange, or externally supplied transient-flux attraction.",
    )
    ap.add_argument(
        "--node-energy-column",
        default="Relative Energy (kJ/mol)",
        help="Nodes CSV column containing energies relative to the lowest-energy ground-state node.",
    )
    ap.add_argument(
        "--ts-energy-column",
        default="Relative TS Energy (kJ/mol)",
        help="Edges CSV column containing TS energies relative to the lowest-energy ground-state node.",
    )
    ap.add_argument("--temperature-k", type=float, default=298.15, help="Temperature in kelvin for equilibrium-exchange weighting.")
    ap.add_argument(
        "--spring-weight-floor-ratio",
        type=float,
        default=0.05,
        help="Minimum raw attraction relative to the strongest weighted edge (0 < value <= 1).",
    )
    ap.add_argument(
        "--transient-flux-file",
        default=None,
        help="Edge CSV containing integrated transient-flux metrics; required in transient-flux mode.",
    )
    ap.add_argument(
        "--transient-flux-metric",
        choices=["absolute-net", "gross"],
        default="absolute-net",
        help="Transient metric used for layout attraction (default: absolute-net).",
    )
    ap.add_argument(
        "--transient-flux-column",
        default=None,
        help="Optional override for the selected transient-flux CSV value column.",
    )
    args = ap.parse_args()

    global LAYOUT_EDGE_OBJECTIVE_SCOPE, LAYOUT_EDGE_LENGTH_POWER, LAYOUT_CENTER_TARGETS
    LAYOUT_EDGE_OBJECTIVE_SCOPE = args.edge_objective_scope
    verbose = args.verbose
    args.center_isomer = _clean_id(args.center_isomer) if args.center_isomer is not None else None
    if args.center_weight is None:
        args.center_weight = 0.10 if args.center_isomer else 0.0
    if args.center_weight < 0.0 or not math.isfinite(float(args.center_weight)):
        raise ValueError("--center-weight must be zero or a positive finite number.")
    if not args.center_isomer and args.center_weight > 0.0:
        raise ValueError("--center-weight requires --center-isomer.")
    if args.edge_length_power <= 0.0 or not math.isfinite(float(args.edge_length_power)):
        raise ValueError("--edge-length-power must be a positive finite number.")
    LAYOUT_EDGE_LENGTH_POWER = float(args.edge_length_power)
    if args.cross_axis_edge_weight < 0.0 or not math.isfinite(float(args.cross_axis_edge_weight)):
        raise ValueError("--cross-axis-edge-weight must be zero or a positive finite number.")
    if args.hemisphere_pass_every <= 0 or args.hemisphere_stall_trigger <= 0:
        raise ValueError("Hemisphere pass cadence and stall trigger must be positive integers.")
    if args.hemisphere_block_max_pairs < 2 or args.hemisphere_block_seeds <= 0:
        raise ValueError("Adaptive hemisphere block limits must be positive and include at least two pairs.")
    if args.hemisphere_block_relax_iters < 0 or args.hemisphere_block_relax_halo < 0:
        raise ValueError("Hemisphere block relaxation settings must be non-negative integers.")
    if args.cycle_offset < 0:
        raise ValueError("--cycle-offset must be non-negative.")
    if args.max_runtime_hours < 0.0 or not math.isfinite(float(args.max_runtime_hours)):
        raise ValueError("--max-runtime-hours must be zero or a positive finite number.")
    if args.separation_project_every <= 0 or args.separation_project_iters <= 0:
        raise ValueError("Progressive separation cadence and iteration counts must be positive.")
    separation_levels = parse_separation_levels(args.separation_levels, args.min_sep)
    resume_state = {}
    if args.resume_state_json:
        resume_state = json.loads(Path(args.resume_state_json).read_text(encoding="utf-8"))
        saved_parameters = resume_state.get("parameters", {})
        resume_compatibility = {
            "hemisphere_optimization": args.hemisphere_optimization,
            "hemi_span_adjustment": args.hemi_span_adjustment,
            "center_isomer": args.center_isomer,
            "center_weight": float(args.center_weight),
            "edge_weighting": args.edge_weighting,
            "edge_length_power": float(args.edge_length_power),
            "cross_axis_edge_weight": float(args.cross_axis_edge_weight),
            "edge_objective_scope": args.edge_objective_scope,
            "hemisphere_pass_every": int(args.hemisphere_pass_every),
            "hemisphere_stall_trigger": int(args.hemisphere_stall_trigger),
            "hemisphere_block_max_pairs": int(args.hemisphere_block_max_pairs),
            "hemisphere_block_seeds": int(args.hemisphere_block_seeds),
            "hemisphere_block_relax_iters": int(args.hemisphere_block_relax_iters),
            "hemisphere_block_relax_halo": int(args.hemisphere_block_relax_halo),
        }
        for field, current_value in resume_compatibility.items():
            if field not in saved_parameters:
                continue
            saved_value = saved_parameters[field]
            if isinstance(current_value, float):
                compatible = math.isclose(
                    float(saved_value), current_value, rel_tol=0.0, abs_tol=1e-12
                )
            else:
                compatible = saved_value == current_value
            if not compatible:
                raise ValueError(
                    f"Checkpoint is incompatible: {field} was {saved_value!r}, "
                    f"but the current command requests {current_value!r}."
                )
        if args.cycle_offset == 0:
            args.cycle_offset = int(resume_state.get("completed_global_cycle", 0))
        objective_state = resume_state.get("objective", {})
        if args.objective_reference_crossings is None:
            args.objective_reference_crossings = objective_state.get("initial_crossings")
        if args.objective_reference_weighted_length is None:
            args.objective_reference_weighted_length = objective_state.get("initial_weighted_length")
        if args.objective_reference_spacing_penalty is None:
            args.objective_reference_spacing_penalty = objective_state.get("initial_spacing_penalty")
        if args.objective_reference_center_penalty is None:
            args.objective_reference_center_penalty = objective_state.get("initial_center_penalty")
    if args.checkpoint_gephi and args.checkpoint_state_json is None:
        args.checkpoint_state_json = str(Path(args.checkpoint_gephi).with_name(Path(args.checkpoint_gephi).name + ".state.json"))
    if args.checkpoint_gephi and args.best_checkpoint_gephi is None:
        checkpoint_path = Path(args.checkpoint_gephi)
        args.best_checkpoint_gephi = str(checkpoint_path.with_name(checkpoint_path.stem + "_best" + checkpoint_path.suffix))
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
    G, nodes_raw, edges_df = read_graph(
        args.nodes,
        args.edges,
        edge_weighting=args.edge_weighting,
        node_energy_column=args.node_energy_column,
        ts_energy_column=args.ts_energy_column,
        temperature_k=args.temperature_k,
        spring_weight_floor_ratio=args.spring_weight_floor_ratio,
        transient_flux_file=args.transient_flux_file,
        transient_flux_metric=args.transient_flux_metric,
        transient_flux_column=args.transient_flux_column,
    )
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

    center_targets = resolve_center_targets(G, pairs, args.center_isomer)
    LAYOUT_CENTER_TARGETS = tuple(center_targets)
    for node in G.nodes():
        G.nodes[node]["Central_Layout_Target"] = bool(node in center_targets)
    if center_targets:
        print(
            f"[CENTER] requested={args.center_isomer} targets={list(center_targets)} weight={args.center_weight}",
            flush=True,
        )

    if len(pairs) <= 1:
        A_side = set(pairs.keys()); B_side = set()
    else:
        H = pair_graph_for_bisection(G, pairs, node2pair)
        if H.number_of_edges() == 0:
            pids = list(H.nodes()); half = len(pids) // 2
            A_side, B_side = set(pids[:half]), set(pids[half:])
        else:
            A_side, B_side = balanced_bisection_pairs(H, seed=args.seed)

    layout_scale = compute_layout_scale(G, min_sep=args.min_sep, layout_scale_factor=args.layout_scale_factor)
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
        layout_scale = compute_layout_scale(Grep, min_sep=args.min_sep, layout_scale_factor=args.layout_scale_factor)
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
        pos = equalize_hemi_span(pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio, adjustment=args.hemi_span_adjustment)
        pos = match_axis_chiral_stats(pos, sides)

        print("[START] Arranging achiral nodes on axis", flush=True)
        pos, comp_orderings = arrange_achiral_on_axis(G, pos, sides, pairs, min_sep=args.min_sep, vertical_gap=args.achiral_gap)
        achiral_gap = args.achiral_gap if args.achiral_gap is not None else max(args.min_sep, 0.9 * layout_scale / max(1.0, math.sqrt(max(1, G.number_of_nodes()))))
        pos, comp_orderings = enforce_achiral_direct_adjacency(G, pos, sides, comp_orderings, achiral_gap)
        pos = match_axis_chiral_stats(pos, sides)
        axis_min_gap = args.axis_min_gap if args.axis_min_gap is not None else max(args.min_sep, 0.9 * achiral_gap)
        pos, comp_orderings = adjust_axis_components_min_gap(G, pos, sides, comp_orderings, achiral_gap, axis_min_gap, max_iters=args.axis_adjust_iters)
        pos = match_axis_chiral_stats(pos, sides)
        pos = equalize_hemi_span(pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes, hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio, adjustment=args.hemi_span_adjustment)
        pos = match_axis_chiral_stats(pos, sides)

        best_pos = dict(pos)

        if center_targets:
            pos = place_center_targets_at_midpoint(pos, center_targets)
            pos, _ = enforce_strict_mirror(pos, pairs, sides)
            best_pos = dict(pos)
            print(f"[CENTER] placed requested target at the initial vertical midpoint", flush=True)

    print("[INFO] Building edge cache for delta crossing computations", flush=True)
    edges_all, edge_bboxes = build_edge_cache(G, pos)

    best_pos = dict(pos)
    current_cross = count_edge_crossings(G, best_pos)
    resume_start_crossings = current_cross
    resume_start_weighted_length = _global_edge_length_score(G, best_pos, edges_all) / _edge_length_scale(args.min_sep)
    resume_start_spacing_penalty = _global_spacing_penalty(best_pos, args.min_sep)
    initial_crossings = args.objective_reference_crossings if args.objective_reference_crossings is not None else resume_start_crossings
    initial_weighted_length = args.objective_reference_weighted_length if args.objective_reference_weighted_length is not None else resume_start_weighted_length
    initial_spacing_penalty = args.objective_reference_spacing_penalty if args.objective_reference_spacing_penalty is not None else resume_start_spacing_penalty
    resume_start_center_penalty = centrality_metrics(best_pos, center_targets)["penalty"]
    initial_center_penalty = args.objective_reference_center_penalty if args.objective_reference_center_penalty is not None else resume_start_center_penalty
    configure_layout_objective(
        args.objective_mode,
        args.objective_crossing_weight,
        args.objective_edge_length_weight,
        args.objective_spacing_weight,
        args.soft_separation_factor,
        initial_crossings=initial_crossings,
        initial_weighted_length=initial_weighted_length,
        initial_spacing_penalty=initial_spacing_penalty,
        edge_length_power=args.edge_length_power,
        cross_axis_weight=args.cross_axis_edge_weight,
        center_weight=args.center_weight,
        center_targets=center_targets,
        initial_center_penalty=initial_center_penalty,
    )
    current_layout_score = evaluate_layout_score(G, best_pos, edges_all, sides=sides, min_sep=args.min_sep)
    print(f"[OBJECTIVE] {OBJECTIVE_CONFIGURATION}", flush=True)
    print(f"[REFINE] starting crossings{'~' if args.crossing_mode != 'exact' else ''}={current_cross}{' (sampled)' if args.crossing_mode != 'exact' else ''}", flush=True)

    hemisphere_totals = {
        "mode": args.hemisphere_optimization,
        "global_passes_attempted": 0,
        "global_passes_accepted": 0,
        "global_pair_flips_proposed": 0,
        "block_passes_attempted": 0,
        "block_flips_accepted": 0,
        "block_pairs_accepted": 0,
        "block_relaxations_accepted": 0,
    }
    for key, value in resume_state.get("hemisphere_statistics", {}).items():
        if key in hemisphere_totals and key != "mode":
            hemisphere_totals[key] = int(value)
    orientation_uniform = args.edge_weighting == "none"
    if args.hemisphere_optimization in {"global", "adaptive"}:
        hemisphere_totals["global_passes_attempted"] += 1
        before_proxy = orientation_proxy_score(G, best_pos, pairs, uniform=orientation_uniform)
        oriented_pos, oriented_sides, orientation_diag = propose_global_orientation(
            G, best_pos, sides, pairs, uniform=orientation_uniform
        )
        hemisphere_totals["global_pair_flips_proposed"] += orientation_diag["pair_flips"]
        after_proxy = orientation_proxy_score(G, oriented_pos, pairs, uniform=orientation_uniform)
        oriented_score = evaluate_layout_score(
            G, oriented_pos, edges_all, sides=oriented_sides, min_sep=args.min_sep
        )
        orientation_accept = oriented_score < current_layout_score - 1e-12
        if args.objective_edge_length_weight == 0.0:
            orientation_accept = oriented_score <= current_layout_score + 1e-12 and after_proxy < before_proxy - 1e-9
        if orientation_accept:
            best_pos, sides = oriented_pos, oriented_sides
            current_layout_score = oriented_score
            current_cross = count_edge_crossings(G, best_pos)
            edges_all, edge_bboxes = build_edge_cache(G, best_pos)
            hemisphere_totals["global_passes_accepted"] += 1
            print(
                f"[HEMISPHERE] accepted initial global orientation: "
                f"pair_flips={orientation_diag['pair_flips']} proxy={before_proxy:.6f}->{after_proxy:.6f} "
                f"objective={current_layout_score:.12f}",
                flush=True,
            )
        else:
            print(
                f"[HEMISPHERE] rejected initial global orientation: "
                f"pair_flips={orientation_diag['pair_flips']} proxy={before_proxy:.6f}->{after_proxy:.6f} "
                f"candidate_objective={oriented_score:.12f}",
                flush=True,
            )

    separation_stage_index = 0
    if args.separation_mode == "progressive":
        saved_index = resume_state.get("separation", {}).get("stage_index")
        if saved_index is not None:
            separation_stage_index = min(max(0, int(saved_index)), len(separation_levels) - 1)
        else:
            while separation_stage_index < len(separation_levels) - 1:
                diag = separation_diagnostics(best_pos, separation_levels[separation_stage_index])
                if diag["remaining_violations"]:
                    break
                separation_stage_index += 1
        active_separation = separation_levels[separation_stage_index]
        active_separation_diag = separation_diagnostics(best_pos, active_separation)
        print(
            f"[SEPARATION] progressive levels={separation_levels} active={active_separation:.6f} "
            f"violations={active_separation_diag['remaining_violations']} "
            f"deficit={active_separation_diag['total_squared_deficit']:.9f}",
            flush=True,
        )
    else:
        active_separation = 0.0
        active_separation_diag = separation_diagnostics(best_pos, 0.0)

    def prepare_candidate_for_scoring(base_pos, candidate_pos, label):
        if args.separation_mode != "progressive":
            return candidate_pos, True
        repaired, repaired_diag, was_projected, allowed = project_candidate_for_separation(
            base_pos,
            candidate_pos,
            pairs,
            sides,
            active_separation,
            max_iter=args.separation_project_iters,
            stall_limit=args.separation_stall_passes,
        )
        if was_projected:
            status = "repaired" if allowed else "rejected"
            print(
                f"  [SEPARATION] {label} candidate {status} before scoring: "
                f"threshold={active_separation:.6f} "
                f"violations={repaired_diag['remaining_violations']} "
                f"deficit={repaired_diag['total_squared_deficit']:.9f} "
                f"iterations={repaired_diag['iterations']}",
                flush=True,
            )
        return repaired, allowed

    best_checkpoint_rank = tuple(resume_state.get("best_rank", [-1, float("-inf"), float("-inf")]))
    saved_best_valid_objective = resume_state.get("best_valid_objective")
    best_valid_objective = (
        float(saved_best_valid_objective) if saved_best_valid_objective is not None else float("inf")
    )
    best_valid_cycle = resume_state.get("best_valid_cycle")
    initial_full_diag = separation_diagnostics(best_pos, args.min_sep)
    if initial_full_diag["remaining_violations"] == 0 and not math.isfinite(best_valid_objective):
        best_valid_objective = float(current_layout_score)
        best_valid_cycle = int(args.cycle_offset)

    def checkpoint_state(completed_global_cycle, separation_diag, best_rank):
        return {
            "version": 2,
            "completed_global_cycle": int(completed_global_cycle),
            "seed": int(args.seed),
            "objective": dict(OBJECTIVE_CONFIGURATION),
            "separation": {
                "mode": args.separation_mode,
                "levels": list(separation_levels),
                "stage_index": int(separation_stage_index),
                "active_threshold": float(active_separation),
                "diagnostics": dict(separation_diag),
            },
            "best_rank": list(best_rank),
            "best_valid_objective": None if not math.isfinite(best_valid_objective) else float(best_valid_objective),
            "best_valid_cycle": best_valid_cycle,
            "parameters": {
                "min_sep": float(args.min_sep),
                "crossing_mode": args.crossing_mode,
                "fast_sample_size": int(args.fast_sample_size),
                "separation_project_every": int(args.separation_project_every),
                "separation_project_iters": int(args.separation_project_iters),
                "transactional_valid_cycles": bool(args.transactional_valid_cycles),
                "edge_objective_scope": args.edge_objective_scope,
                "edge_weighting": args.edge_weighting,
                "edge_length_power": float(args.edge_length_power),
                "cross_axis_edge_weight": float(args.cross_axis_edge_weight),
                "hemisphere_optimization": args.hemisphere_optimization,
                "hemi_span_adjustment": args.hemi_span_adjustment,
                "final_span_expansion": args.final_span_expansion,
                "hemisphere_pass_every": int(args.hemisphere_pass_every),
                "hemisphere_stall_trigger": int(args.hemisphere_stall_trigger),
                "hemisphere_block_max_pairs": int(args.hemisphere_block_max_pairs),
                "hemisphere_block_seeds": int(args.hemisphere_block_seeds),
                "hemisphere_block_relax_iters": int(args.hemisphere_block_relax_iters),
                "hemisphere_block_relax_halo": int(args.hemisphere_block_relax_halo),
                "center_isomer": args.center_isomer,
                "center_weight": float(args.center_weight),
            },
            "hemisphere_statistics": dict(hemisphere_totals),
        }

    step_counter = 0
    cycle_metrics = []
    consecutive_stalled_cycles = 0
    exact_every = max(1, int(args.crossing_exact_every))
    refinement_started_monotonic = time.monotonic()
    runtime_limit_reached = False

    def maybe_force_exact(final_cycle: bool = False):
        nonlocal current_cross, best_pos
        if args.crossing_mode == "exact":
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "cycle_end" and final_cycle:
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "step_interval" and step_counter > 0 and (step_counter % exact_every == 0):
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)

    def report_stage_diagnostics(label, retained_before):
        if args.stage_diagnostics and label is not None and retained_before is not None:
            retained_diag = separation_diagnostics(best_pos, active_separation)
            retained_score = evaluate_layout_score(
                G, best_pos, edges_all, sides=sides, min_sep=args.min_sep
            )
            print(
                f"[STAGE DIAGNOSTICS] {label}: "
                f"retained_max_shift={max_node_shift(retained_before, best_pos):.6f} "
                f"objective={retained_score:.12f} "
                f"threshold={active_separation:.6f} "
                f"violations={retained_diag['remaining_violations']} "
                f"deficit={retained_diag['total_squared_deficit']:.9f}",
                flush=True,
            )

    def after_step(label=None, retained_before=None):
        nonlocal step_counter, current_cross
        step_counter += 1
        maybe_force_exact()
        report_stage_diagnostics(label, retained_before)

    for cycle in range(args.refine_cycles):
        global_cycle = args.cycle_offset + cycle + 1
        final_requested_cycle = args.cycle_offset + args.refine_cycles
        print(f"[REFINE] cycle {global_cycle}/{final_requested_cycle} starting (current crossings={current_cross})", flush=True)
        cycle_start_pos = dict(best_pos)
        cycle_start_sides = dict(sides)
        cycle_start_score = float(current_layout_score)
        cycle_start_cross = current_cross
        cycle_start_comp_orderings = copy.deepcopy(comp_orderings)
        cycle_start_stage_index = int(separation_stage_index)
        cycle_start_active_separation = float(active_separation)
        cycle_start_full_diag = separation_diagnostics(cycle_start_pos, args.min_sep)
        cycle_start_full_valid = cycle_start_full_diag["remaining_violations"] == 0
        cycle_checkpoint_pos = None
        if args.checkpoint_gephi and Path(args.checkpoint_gephi).exists():
            try:
                cycle_checkpoint_pos = load_positions_from_gexf(args.checkpoint_gephi, nodes=G.nodes())
            except Exception as exc:
                if verbose:
                    print(f"[WARN] Could not load checkpoint for convergence check: {exc}", flush=True)
        improved_cycle = False
        cycle_hemisphere = {
            "global_attempted": 0,
            "global_accepted": 0,
            "global_pair_flips_proposed": 0,
            "block_candidates": 0,
            "block_accepted": 0,
            "block_pairs_accepted": 0,
            "block_relaxation_accepted": 0,
            "transaction_rolled_back": False,
        }

        print(f"[REFINE][cycle {global_cycle}] Step: chiral Y-swaps (max_iters={args.chiral_swap_iters})", flush=True)
        stage_start_pos = dict(best_pos)
        pos_chiral, ch_improved, ch_before, ch_after = optimize_chiral_pair_swaps(
            G, dict(best_pos), pairs, sides, edges_all, edge_bboxes,
            max_iters=args.chiral_swap_iters, seed=args.seed + global_cycle - 1, tries_per_iter=None, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if ch_improved:
            pos_chiral = match_axis_chiral_stats(pos_chiral, sides)
            raw_layout_score = evaluate_layout_score(G, pos_chiral, edges_all, sides=sides, min_sep=args.min_sep)
            separation_ok = False
            if raw_layout_score < current_layout_score:
                pos_chiral, separation_ok = prepare_candidate_for_scoring(best_pos, pos_chiral, "chiral Y-swaps")
            new_cross = count_edge_crossings(G, pos_chiral)
            new_layout_score = evaluate_layout_score(G, pos_chiral, edges_all, sides=sides, min_sep=args.min_sep)
            if separation_ok and new_layout_score < current_layout_score:
                best_pos = pos_chiral
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] chiral Y-swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] chiral Y-swaps no improvement", flush=True)
        after_step("chiral Y-swaps", stage_start_pos)

        print(f"[REFINE][cycle {global_cycle}] Step: pair-of-pairs swaps (attempts_per_iter={args.pairpair_iters})", flush=True)
        stage_start_pos = dict(best_pos)
        pos_pairpair, pp_improved, pp_before, pp_after = optimize_pair_of_pairs_swaps(
            G, dict(best_pos), pairs, sides, edges_all, edge_bboxes,
            max_iters=args.pairpair_iters, seed=args.seed + 1000 + global_cycle - 1, attempts_per_iter=args.pairpair_iters, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if pp_improved:
            pos_pairpair = match_axis_chiral_stats(pos_pairpair, sides)
            raw_layout_score = evaluate_layout_score(G, pos_pairpair, edges_all, sides=sides, min_sep=args.min_sep)
            separation_ok = False
            if raw_layout_score < current_layout_score:
                pos_pairpair, separation_ok = prepare_candidate_for_scoring(best_pos, pos_pairpair, "pair-of-pairs")
            new_cross = count_edge_crossings(G, pos_pairpair)
            new_layout_score = evaluate_layout_score(G, pos_pairpair, edges_all, sides=sides, min_sep=args.min_sep)
            if separation_ok and new_layout_score < current_layout_score:
                best_pos = pos_pairpair
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] pair-of-pairs improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] pair-of-pairs no improvement", flush=True)
        after_step("pair-of-pairs swaps", stage_start_pos)

        print(f"[REFINE][cycle {global_cycle}] Step: enantiomer X-swaps (iters={args.enantiomer_swap_iters})", flush=True)
        stage_start_pos = dict(best_pos)
        pos_enant, enant_improved, en_before, en_after = optimize_enantiomer_x_swaps(
            G, dict(best_pos), pairs, sides, edges_all, edge_bboxes,
            max_iters=args.enantiomer_swap_iters, seed=args.seed + 5000 + global_cycle - 1, tries_per_iter=None, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if enant_improved:
            pos_enant = match_axis_chiral_stats(pos_enant, sides)
            raw_layout_score = evaluate_layout_score(G, pos_enant, edges_all, sides=sides, min_sep=args.min_sep)
            separation_ok = False
            if raw_layout_score < current_layout_score:
                pos_enant, separation_ok = prepare_candidate_for_scoring(best_pos, pos_enant, "enantiomer X-swaps")
            new_cross = count_edge_crossings(G, pos_enant)
            new_layout_score = evaluate_layout_score(G, pos_enant, edges_all, sides=sides, min_sep=args.min_sep)
            if separation_ok and new_layout_score < current_layout_score:
                best_pos = pos_enant
                current_cross = new_cross
                current_layout_score = new_layout_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] enantiomer X-swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] enantiomer X-swaps no improvement", flush=True)
        after_step("enantiomer X-swaps", stage_start_pos)

        print(f"[REFINE][cycle {global_cycle}] Step: achiral-slot swaps (max_iters={args.swap_iters})", flush=True)
        stage_start_pos = dict(best_pos)
        pos_achiral, ach_cross = optimize_achiral_swaps(
            G, dict(best_pos), pairs, sides, copy.deepcopy(comp_orderings), achiral_gap,
            edges_all, edge_bboxes,
            max_iters=args.swap_iters, seed=args.seed + 2000 + global_cycle - 1,
            cross_weight=LAYOUT_CROSSING_WEIGHT, gap_weight=args.achiral_gap_weight, nonadj_weight=args.achiral_adjacency_weight, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        pos_achiral = match_axis_chiral_stats(pos_achiral, sides)
        ach_score = evaluate_layout_score(G, pos_achiral, edges_all, sides=sides, min_sep=args.min_sep)
        separation_ok = False
        if ach_score < current_layout_score:
            pos_achiral, separation_ok = prepare_candidate_for_scoring(best_pos, pos_achiral, "achiral-slot swaps")
            ach_score = evaluate_layout_score(G, pos_achiral, edges_all, sides=sides, min_sep=args.min_sep)
            ach_cross = count_edge_crossings(G, pos_achiral)
        if separation_ok and ach_score < current_layout_score:
            best_pos = pos_achiral
            current_cross = ach_cross
            current_layout_score = ach_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] achiral-slot swaps improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print(f"  [RESULT] achiral-slot swaps no improvement (score={ach_score:.3f}, crossings={ach_cross})", flush=True)
        after_step("achiral-slot swaps", stage_start_pos)

        print(f"[REFINE][cycle {global_cycle}] Step: pair-block relocation (max_trials={args.pair_relocation_trials})", flush=True)
        stage_start_pos = dict(best_pos)
        pos_reloc, relocated, pre_reloc_cross, post_reloc_cross = pair_block_relocation_refinement(
            G, dict(best_pos), sides, copy.deepcopy(comp_orderings), achiral_gap,
            edges_all, edge_bboxes,
            max_trials=args.pair_relocation_trials, seed=args.seed + 3000 + global_cycle - 1, verbose=verbose,
            fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
        if relocated:
            pos_reloc = match_axis_chiral_stats(pos_reloc, sides)
        pos_reloc_score = evaluate_layout_score(G, pos_reloc, edges_all, sides=sides, min_sep=args.min_sep)
        separation_ok = False
        if relocated and pos_reloc_score < current_layout_score:
            pos_reloc, separation_ok = prepare_candidate_for_scoring(best_pos, pos_reloc, "pair-block relocation")
            pos_reloc_score = evaluate_layout_score(G, pos_reloc, edges_all, sides=sides, min_sep=args.min_sep)
            post_reloc_cross = count_edge_crossings(G, pos_reloc)
        if relocated and separation_ok and pos_reloc_score < current_layout_score:
            best_pos = pos_reloc
            current_cross = post_reloc_cross
            current_layout_score = pos_reloc_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] pair-block relocation improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print("  [RESULT] pair-block relocation no improvement", flush=True)
        after_step("pair-block relocation", stage_start_pos)

        axis_min_gap = args.axis_min_gap if args.axis_min_gap is not None else (0.9 * args.min_sep)
        print(f"[REFINE][cycle {global_cycle}] Step: axis block adjust (min_axis_gap={axis_min_gap})", flush=True)
        stage_start_pos = dict(best_pos)
        axis_candidate, comp_orderings_candidate = adjust_axis_components_min_gap(
            G, dict(best_pos), sides, copy.deepcopy(comp_orderings), achiral_gap, axis_min_gap, max_iters=1
        )
        axis_candidate = spread_axis_nodes_min_sep(axis_candidate, sides, args.min_sep)
        axis_candidate = match_axis_chiral_stats(axis_candidate, sides)
        post_adjust_score = evaluate_layout_score(G, axis_candidate, edges_all, sides=sides, min_sep=args.min_sep)
        separation_ok = False
        if post_adjust_score < current_layout_score:
            axis_candidate, separation_ok = prepare_candidate_for_scoring(best_pos, axis_candidate, "axis adjust")
            post_adjust_score = evaluate_layout_score(G, axis_candidate, edges_all, sides=sides, min_sep=args.min_sep)
        post_adjust_cross = count_edge_crossings(G, axis_candidate)
        if separation_ok and post_adjust_score < current_layout_score:
            best_pos = axis_candidate
            comp_orderings = comp_orderings_candidate
            current_cross = post_adjust_cross
            current_layout_score = post_adjust_score
            improved_cycle = True
            if verbose:
                print(f"  [RESULT] axis adjust improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
        else:
            if verbose:
                print(f"  [RESULT] axis adjust no improvement (score={post_adjust_score:.3f}, crossings={post_adjust_cross})", flush=True)
        after_step("axis block adjust", stage_start_pos)

        if args.chiral_relax:
            stage_start_pos = dict(best_pos)
            pos_relax, relax_improved, relax_before, relax_after = optimize_chiral_coordinate_relaxation(
                G, dict(best_pos), pairs, sides, edges_all, edge_bboxes,
                max_iters=args.chiral_relax_iters,
                seed=args.seed + 7000 + global_cycle - 1,
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
            separation_ok = False
            if relax_score < current_layout_score:
                pos_relax, separation_ok = prepare_candidate_for_scoring(best_pos, pos_relax, "chiral relaxation")
                relax_score = evaluate_layout_score(G, pos_relax, edges_all, sides=sides, min_sep=args.min_sep)
                relax_after = count_edge_crossings(G, pos_relax)
            if separation_ok and relax_score < current_layout_score:
                best_pos = pos_relax
                current_cross = relax_after
                current_layout_score = relax_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] chiral relaxation improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
            elif verbose:
                print(f"  [RESULT] chiral relaxation no improvement (score={relax_score:.3f})", flush=True)
            after_step("chiral relaxation", stage_start_pos)

            print(f"[REFINE][cycle {global_cycle}] Step: pair flips", flush=True)
            stage_start_pos = dict(best_pos)
            pos_flip, flip_improved, flip_before, flip_after = optimize_chiral_pair_flips(
                G, dict(best_pos), pairs, sides, edges_all, edge_bboxes,
                max_iters=args.pair_flip_iters, seed=args.seed + 9000 + global_cycle - 1, tries_per_iter=None, verbose=verbose,
                fast_mode=args.fast_mode, sample_size=args.fast_sample_size, min_sep=args.min_sep)
            pos_flip = match_axis_chiral_stats(pos_flip, sides)
            flip_score = evaluate_layout_score(G, pos_flip, edges_all, sides=sides, min_sep=args.min_sep)
            separation_ok = False
            if flip_score < current_layout_score:
                pos_flip, separation_ok = prepare_candidate_for_scoring(best_pos, pos_flip, "pair flips")
                flip_score = evaluate_layout_score(G, pos_flip, edges_all, sides=sides, min_sep=args.min_sep)
                flip_after = count_edge_crossings(G, pos_flip)
            if separation_ok and flip_score < current_layout_score:
                best_pos = pos_flip
                if args.hemisphere_optimization != "local":
                    sides = synchronise_sides_from_positions(best_pos, sides)
                current_cross = flip_after
                current_layout_score = flip_score
                improved_cycle = True
                if verbose:
                    print(f"  [RESULT] pair flips improved layout -> score={current_layout_score:.3f}, crossings={current_cross}", flush=True)
            elif verbose:
                print(f"  [RESULT] pair flips no improvement (score={flip_score:.3f})", flush=True)
            after_step("pair flips", stage_start_pos)

        if args.hemisphere_optimization == "adaptive":
            stage_start_pos = dict(best_pos)
            run_global_pass = (
                global_cycle % args.hemisphere_pass_every == 0
                or consecutive_stalled_cycles >= args.hemisphere_stall_trigger
            )
            if run_global_pass:
                hemisphere_totals["global_passes_attempted"] += 1
                cycle_hemisphere["global_attempted"] = 1
                before_proxy = orientation_proxy_score(G, best_pos, pairs, uniform=orientation_uniform)
                oriented_pos, oriented_sides, orientation_diag = propose_global_orientation(
                    G, best_pos, sides, pairs, uniform=orientation_uniform
                )
                cycle_hemisphere["global_pair_flips_proposed"] = orientation_diag["pair_flips"]
                hemisphere_totals["global_pair_flips_proposed"] += orientation_diag["pair_flips"]
                after_proxy = orientation_proxy_score(G, oriented_pos, pairs, uniform=orientation_uniform)
                oriented_score = evaluate_layout_score(
                    G, oriented_pos, edges_all, sides=oriented_sides, min_sep=args.min_sep
                )
                accept_orientation = oriented_score < current_layout_score - 1e-12
                if args.objective_edge_length_weight == 0.0:
                    accept_orientation = (
                        oriented_score <= current_layout_score + 1e-12
                        and after_proxy < before_proxy - 1e-9
                    )
                if accept_orientation:
                    best_pos, sides = oriented_pos, oriented_sides
                    current_layout_score = oriented_score
                    current_cross = count_edge_crossings(G, best_pos)
                    edges_all, edge_bboxes = build_edge_cache(G, best_pos)
                    improved_cycle = True
                    cycle_hemisphere["global_accepted"] = 1
                    hemisphere_totals["global_passes_accepted"] += 1
                    print(
                        f"[HEMISPHERE][cycle {global_cycle}] accepted global orientation: "
                        f"pair_flips={orientation_diag['pair_flips']} "
                        f"proxy={before_proxy:.6f}->{after_proxy:.6f} objective={current_layout_score:.12f}",
                        flush=True,
                    )
                elif verbose:
                    print(
                        f"[HEMISPHERE][cycle {global_cycle}] global orientation not retained: "
                        f"pair_flips={orientation_diag['pair_flips']} candidate_objective={oriented_score:.12f}",
                        flush=True,
                    )

            hemisphere_totals["block_passes_attempted"] += 1
            block_proposals = adaptive_block_candidates(
                G,
                best_pos,
                sides,
                pairs,
                uniform=orientation_uniform,
                max_pairs=args.hemisphere_block_max_pairs,
                seed_limit=args.hemisphere_block_seeds,
            )
            cycle_hemisphere["block_candidates"] = len(block_proposals)
            current_proxy = orientation_proxy_score(G, best_pos, pairs, uniform=orientation_uniform)
            selected = None
            for proxy_delta, block_ids, block_pos, block_sides in block_proposals[:8]:
                block_score = evaluate_layout_score(
                    G, block_pos, edges_all, sides=block_sides, min_sep=args.min_sep
                )
                acceptable = block_score < current_layout_score - 1e-12
                if args.objective_edge_length_weight == 0.0:
                    acceptable = block_score <= current_layout_score + 1e-12 and proxy_delta < -1e-9
                if not acceptable:
                    continue
                rank = (block_score, proxy_delta, len(block_ids), block_ids)
                if selected is None or rank < selected[0]:
                    selected = (rank, block_ids, block_pos, block_sides, block_score)
            if selected is not None:
                _, block_ids, block_pos, block_sides, block_score = selected
                best_pos, sides = block_pos, block_sides
                current_layout_score = block_score
                current_cross = count_edge_crossings(G, best_pos)
                edges_all, edge_bboxes = build_edge_cache(G, best_pos)
                improved_cycle = True
                cycle_hemisphere["block_accepted"] = 1
                cycle_hemisphere["block_pairs_accepted"] = len(block_ids)
                hemisphere_totals["block_flips_accepted"] += 1
                hemisphere_totals["block_pairs_accepted"] += len(block_ids)
                print(
                    f"[HEMISPHERE][cycle {global_cycle}] accepted connected block flip: "
                    f"pairs={len(block_ids)} proxy={current_proxy:.6f}->"
                    f"{orientation_proxy_score(G, best_pos, pairs, uniform=orientation_uniform):.6f} "
                    f"objective={current_layout_score:.12f}",
                    flush=True,
                )

                if args.hemisphere_block_relax_iters > 0:
                    relaxed_pos, relaxed_sides, relax_diag = coherent_block_relaxation(
                        G,
                        best_pos,
                        sides,
                        pairs,
                        block_ids,
                        iterations=args.hemisphere_block_relax_iters,
                        halo_layers=args.hemisphere_block_relax_halo,
                        min_sep=args.min_sep,
                        min_x=max(args.chiral_relax_min_x, args.axis_lateral_gap),
                        uniform=orientation_uniform,
                    )
                    separation_ok = True
                    if args.separation_mode == "progressive":
                        relaxed_pos, _, was_projected, separation_ok = project_candidate_for_separation(
                            best_pos,
                            relaxed_pos,
                            pairs,
                            relaxed_sides,
                            active_separation,
                            max_iter=args.separation_project_iters,
                            stall_limit=args.separation_stall_passes,
                        )
                        if was_projected:
                            relaxed_sides = synchronise_sides_from_positions(relaxed_pos, relaxed_sides)
                    relaxed_score = evaluate_layout_score(
                        G, relaxed_pos, edges_all, sides=relaxed_sides, min_sep=args.min_sep
                    )
                    if separation_ok and relaxed_score < current_layout_score - 1e-12:
                        best_pos, sides = relaxed_pos, relaxed_sides
                        current_layout_score = relaxed_score
                        current_cross = count_edge_crossings(G, best_pos)
                        edges_all, edge_bboxes = build_edge_cache(G, best_pos)
                        cycle_hemisphere["block_relaxation_accepted"] = 1
                        hemisphere_totals["block_relaxations_accepted"] += 1
                        print(
                            f"[HEMISPHERE][cycle {global_cycle}] accepted coherent block relaxation: "
                            f"active_pairs={relax_diag['active_pairs']} objective={current_layout_score:.12f}",
                            flush=True,
                        )
            after_step("adaptive hemisphere optimisation", stage_start_pos)

        stage_start_pos = dict(best_pos)
        span_candidate = equalize_hemi_span(
            dict(best_pos), sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes,
            hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio,
            adjustment=args.hemi_span_adjustment
        )
        span_candidate = match_axis_chiral_stats(span_candidate, sides)
        span_score = evaluate_layout_score(G, span_candidate, edges_all, sides=sides, min_sep=args.min_sep)
        separation_ok = False
        if span_score < current_layout_score:
            span_candidate, separation_ok = prepare_candidate_for_scoring(best_pos, span_candidate, "hemi-span adjustment")
            span_score = evaluate_layout_score(G, span_candidate, edges_all, sides=sides, min_sep=args.min_sep)
        if separation_ok and span_score < current_layout_score:
            best_pos = span_candidate
            current_cross = count_edge_crossings(G, best_pos)
            current_layout_score = span_score
            improved_cycle = True
        report_stage_diagnostics("hemi-span adjustment", stage_start_pos)

        if args.separation_mode == "progressive":
            active_separation_diag = separation_diagnostics(best_pos, active_separation)
            should_project = (
                active_separation_diag["remaining_violations"] > 0
                and (cycle == 0 or global_cycle % args.separation_project_every == 0)
            )
            if should_project:
                before_projection = active_separation_diag
                projected_pos, projected_diag = enforce_minimum_separation_mirror_preserving(
                    best_pos,
                    pairs,
                    sides,
                    min_sep=active_separation,
                    max_iter=args.separation_project_iters,
                    stall_limit=args.separation_stall_passes,
                )
                if separation_not_worse(before_projection, projected_diag) and (
                    projected_diag["total_squared_deficit"] + 1e-12
                    < before_projection["total_squared_deficit"]
                    or projected_diag["remaining_violations"] == 0
                ):
                    best_pos = projected_pos
                    active_separation_diag = projected_diag
                    current_layout_score = evaluate_layout_score(
                        G, best_pos, edges_all, sides=sides, min_sep=args.min_sep
                    )
                    improved_cycle = True
                    print(
                        f"[SEPARATION][cycle {global_cycle}] projected threshold={active_separation:.6f} "
                        f"violations={before_projection['remaining_violations']}->{projected_diag['remaining_violations']} "
                        f"deficit={before_projection['total_squared_deficit']:.9f}->"
                        f"{projected_diag['total_squared_deficit']:.9f} iterations={projected_diag['iterations']}",
                        flush=True,
                    )
                else:
                    print(
                        f"[SEPARATION][cycle {global_cycle}] projection made no retained improvement at "
                        f"threshold={active_separation:.6f}",
                        flush=True,
                    )

            active_separation_diag = separation_diagnostics(best_pos, active_separation)
            while active_separation_diag["remaining_violations"] == 0 and separation_stage_index < len(separation_levels) - 1:
                separation_stage_index += 1
                active_separation = separation_levels[separation_stage_index]
                active_separation_diag = separation_diagnostics(best_pos, active_separation)
                print(
                    f"[SEPARATION][cycle {global_cycle}] advanced to threshold={active_separation:.6f} "
                    f"violations={active_separation_diag['remaining_violations']} "
                    f"deficit={active_separation_diag['total_squared_deficit']:.9f}",
                    flush=True,
                )

        if args.transactional_valid_cycles and cycle_start_full_valid:
            end_full_diag = separation_diagnostics(best_pos, args.min_sep)
            if end_full_diag["remaining_violations"] > 0:
                repaired_pos, repaired_diag = enforce_minimum_separation_mirror_preserving(
                    best_pos,
                    pairs,
                    sides,
                    min_sep=args.min_sep,
                    max_iter=args.final_separation_iters,
                    stall_limit=args.separation_stall_passes,
                )
                if repaired_diag["remaining_violations"] == 0:
                    best_pos = repaired_pos
                    end_full_diag = repaired_diag
                    current_layout_score = evaluate_layout_score(
                        G, best_pos, edges_all, sides=sides, min_sep=args.min_sep
                    )
                    active_separation_diag = separation_diagnostics(best_pos, active_separation)
                    print(
                        f"[TRANSACTION][cycle {global_cycle}] repaired final candidate to full separation "
                        f"in {repaired_diag['iterations']} iterations before cycle scoring.",
                        flush=True,
                    )
                else:
                    end_full_diag = repaired_diag

            current_layout_score = evaluate_layout_score(
                G, best_pos, edges_all, sides=sides, min_sep=args.min_sep
            )
            end_full_valid = end_full_diag["remaining_violations"] == 0
            if not transactional_cycle_is_better(
                cycle_start_full_valid,
                cycle_start_score,
                end_full_valid,
                current_layout_score,
            ):
                rejected_score = current_layout_score
                rejected_violations = end_full_diag["remaining_violations"]
                best_pos = cycle_start_pos
                sides = cycle_start_sides
                current_layout_score = cycle_start_score
                current_cross = cycle_start_cross
                comp_orderings = cycle_start_comp_orderings
                separation_stage_index = cycle_start_stage_index
                active_separation = cycle_start_active_separation
                active_separation_diag = separation_diagnostics(best_pos, active_separation)
                improved_cycle = False
                cycle_hemisphere["transaction_rolled_back"] = True
                print(
                    f"[TRANSACTION][cycle {global_cycle}] rolled back final cycle candidate: "
                    f"start_score={cycle_start_score:.12f} end_score={rejected_score:.12f} "
                    f"end_full_separation_violations={rejected_violations}.",
                    flush=True,
                )
            else:
                print(
                    f"[TRANSACTION][cycle {global_cycle}] retained valid final cycle: "
                    f"score={cycle_start_score:.12f}->{current_layout_score:.12f}.",
                    flush=True,
                )

        edges_all, edge_bboxes = build_edge_cache(G, best_pos)
        current_cross = count_edge_crossings(G, best_pos)
        if args.crossing_mode == "cycle_end":
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)
        elif args.crossing_mode == "step_interval" and step_counter > 0 and (step_counter % exact_every == 0):
            current_cross = count_edge_crossings(G, best_pos, force_exact=True)

        cycle_shift = max_node_shift(cycle_start_pos, best_pos)
        print(f"[REFINE] cycle {global_cycle} complete: current_crossings={current_cross} improved_cycle={improved_cycle} max_node_shift={cycle_shift:.6f}", flush=True)
        span_metrics = axis_chiral_span_metrics(best_pos, sides)
        center_metrics = centrality_metrics(best_pos, center_targets)
        cycle_record = {
            "cycle": global_cycle,
            "local_cycle": cycle + 1,
            "crossings": current_cross,
            "crossings_fraction_of_initial": float(current_cross) / max(float(initial_crossings), 1.0),
            "weighted_edge_length_score": _global_edge_length_score(G, best_pos, edges_all) / _edge_length_scale(args.min_sep),
            "weighted_edge_length_fraction_of_initial": (_global_edge_length_score(G, best_pos, edges_all) / _edge_length_scale(args.min_sep)) / max(float(initial_weighted_length), 1e-12),
            "soft_spacing_penalty": _global_spacing_penalty(best_pos, args.min_sep),
            "objective_score": current_layout_score,
            "max_node_shift": cycle_shift,
            "separation_threshold": active_separation,
            "separation_violations": active_separation_diag["remaining_violations"],
            "separation_minimum_distance": active_separation_diag["minimum_distance"],
            "separation_total_squared_deficit": active_separation_diag["total_squared_deficit"],
            "weighted_opposite_side_edge_fraction": _global_cross_axis_penalty(
                G, best_pos, edges_all, sides
            ),
            "unweighted_opposite_side_edge_fraction": _global_cross_axis_penalty(
                G, best_pos, edges_all, sides, uniform=True
            ),
            "center_absolute_offset": center_metrics["absolute_offset"],
            "center_fractional_offset": center_metrics["fractional_offset"],
            "center_penalty": center_metrics["penalty"],
            "hemisphere": dict(cycle_hemisphere),
            **span_metrics,
        }
        cycle_metrics.append(cycle_record)
        print(f"[CYCLE METRICS] {cycle_record}", flush=True)

        cycle_full_valid = separation_diagnostics(best_pos, args.min_sep)["remaining_violations"] == 0
        best_valid_improved = False
        if cycle_full_valid and current_layout_score < best_valid_objective - 1e-12:
            best_valid_objective = float(current_layout_score)
            best_valid_cycle = int(global_cycle)
            best_valid_improved = True
            print(
                f"[CONVERGENCE] new best valid objective={best_valid_objective:.12f} "
                f"at cycle {best_valid_cycle}.",
                flush=True,
            )

        candidate_rank = (
            int(separation_stage_index),
            -float(active_separation_diag["total_squared_deficit"]),
            -float(current_layout_score),
        )
        if args.best_checkpoint_gephi and candidate_rank > best_checkpoint_rank:
            best_written = write_gexf_atomic(G, best_pos, args.best_checkpoint_gephi)
            if best_written == Path(args.best_checkpoint_gephi):
                best_checkpoint_rank = candidate_rank
                print(f"[CHECKPOINT] wrote best checkpoint -> {best_written}", flush=True)
            else:
                print(
                    "[CHECKPOINT] best destination remained locked; recovery copy was written and "
                    "the best rank was not advanced so the primary path will be retried next cycle.",
                    flush=True,
                )

        if args.checkpoint_gephi and ((cycle + 1) % max(1, args.checkpoint_every) == 0):
            checkpoint_written = write_gexf_atomic(G, best_pos, args.checkpoint_gephi)
            if checkpoint_written == Path(args.checkpoint_gephi):
                print(f"[CHECKPOINT] wrote checkpoint -> {checkpoint_written}", flush=True)
            else:
                print(f"[CHECKPOINT] wrote recovery checkpoint -> {checkpoint_written}", flush=True)
            if args.checkpoint_state_json and checkpoint_written == Path(args.checkpoint_gephi):
                state_written = write_json_atomic(
                    checkpoint_state(global_cycle, active_separation_diag, best_checkpoint_rank),
                    args.checkpoint_state_json,
                )
                print(f"[CHECKPOINT] wrote state -> {state_written}", flush=True)
            elif args.checkpoint_state_json:
                print(
                    "[CHECKPOINT] state JSON was not advanced because the primary GEXF remained locked; "
                    "the recovery GEXF is self-contained.",
                    flush=True,
                )

        if args.max_runtime_hours > 0.0:
            elapsed_hours = (time.monotonic() - refinement_started_monotonic) / 3600.0
            if elapsed_hours >= args.max_runtime_hours:
                runtime_limit_reached = True
                print(
                    f"[REFINE] runtime limit reached after completed cycle {global_cycle}: "
                    f"elapsed={elapsed_hours:.3f} h limit={args.max_runtime_hours:.3f} h.",
                    flush=True,
                )
                break

        if not args.run_all_refine_cycles and args.transactional_valid_cycles:
            if cycle_full_valid:
                if best_valid_improved:
                    consecutive_stalled_cycles = 0
                else:
                    consecutive_stalled_cycles += 1
                    patience = max(1, int(args.convergence_patience))
                    print(
                        f"[CONVERGENCE] no new best valid objective for "
                        f"{consecutive_stalled_cycles}/{patience} consecutive valid cycles.",
                        flush=True,
                    )
                    if consecutive_stalled_cycles >= patience:
                        print(
                            f"[REFINE] best-valid convergence patience reached at cycle {global_cycle}; "
                            f"stopping early (best cycle {best_valid_cycle}).",
                            flush=True,
                        )
                        break
            else:
                consecutive_stalled_cycles = 0
        elif not args.run_all_refine_cycles:
            stall_reasons = []
            if cycle_checkpoint_pos is not None and positions_exactly_equal(best_pos, cycle_checkpoint_pos):
                stall_reasons.append("coordinates unchanged from checkpoint")
            if cycle_shift <= args.cycle_shift_tol:
                stall_reasons.append(
                    f"maximum node shift {cycle_shift:.6f} <= tolerance {args.cycle_shift_tol:.6f}"
                )
            if not improved_cycle:
                stall_reasons.append("no accepted improvement")
            if stall_reasons:
                consecutive_stalled_cycles += 1
                patience = max(1, int(args.convergence_patience))
                print(
                    f"[REFINE] stalled cycle {consecutive_stalled_cycles}/{patience}: "
                    + "; ".join(stall_reasons),
                    flush=True,
                )
                if consecutive_stalled_cycles >= patience:
                    print(
                        f"[REFINE] convergence patience reached at cycle {global_cycle}; stopping early.",
                        flush=True,
                    )
                    break
            else:
                consecutive_stalled_cycles = 0

    final_balance_candidate = equalize_hemi_span(
        best_pos, sides, tolerance=args.hemi_span_tol, min_nodes=args.hemi_min_nodes,
        hemi_max_scale=args.hemi_max_scale, target_ratio=args.hemi_target_ratio,
        adjustment=args.hemi_span_adjustment
    )
    final_balance_candidate = match_axis_chiral_stats(final_balance_candidate, sides)
    final_balance_score = evaluate_layout_score(
        G, final_balance_candidate, edges_all, sides=sides, min_sep=args.min_sep
    )
    final_balance_ok = False
    if final_balance_score < current_layout_score:
        final_balance_candidate, final_balance_ok = prepare_candidate_for_scoring(
            best_pos, final_balance_candidate, "final hemi-span adjustment"
        )
        final_balance_score = evaluate_layout_score(
            G, final_balance_candidate, edges_all, sides=sides, min_sep=args.min_sep
        )
    if final_balance_ok and final_balance_score < current_layout_score:
        best_pos = final_balance_candidate
        current_layout_score = final_balance_score

    print("[FINAL] Running overlap resolution while keeping axis nodes on x=0 but allowing y-separation", flush=True)
    pos_after_relax = adjust_positions_with_constraints(G, best_pos, pairs, sides, min_sep=args.min_sep, edge_clearance=args.edge_clearance, max_iter=args.overlap_iter, lr=0.25, fixed_achiral_y=None, axis_lateral_gap=args.axis_lateral_gap, strict_min_sep_rounds=0)
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
    span_before_final_balance = axis_chiral_span_metrics(pos_final, sides)
    if args.final_span_expansion == "on":
        pos_final, span_after_final_balance = expand_chiral_span_to_axis(pos_final, sides, target_ratio=args.hemi_target_ratio)
        if span_after_final_balance != span_before_final_balance:
            print(f"[FINAL SPAN] expanded chiral y-span: before={span_before_final_balance} after={span_after_final_balance}", flush=True)
    else:
        span_after_final_balance = dict(span_before_final_balance)
        print(f"[FINAL SPAN] unconditional expansion disabled; retaining objective-optimised spans: {span_after_final_balance}", flush=True)
    print("[FINAL] Enforcing mirror-preserving minimum node separation", flush=True)
    pos_final, separation_diag = enforce_minimum_separation_mirror_preserving(
        pos_final, pairs, sides, min_sep=args.min_sep, max_iter=args.final_separation_iters
    )
    print(
        "[SEPARATION] iterations={iterations} remaining_violations={remaining_violations} "
        "minimum_distance={minimum_distance:.6f}".format(**separation_diag),
        flush=True,
    )
    completed_global_cycle = args.cycle_offset + len(cycle_metrics)
    if args.checkpoint_gephi:
        write_gexf_atomic(G, pos_final, args.checkpoint_gephi)
        if args.checkpoint_state_json:
            write_json_atomic(
                checkpoint_state(completed_global_cycle, separation_diag, best_checkpoint_rank),
                args.checkpoint_state_json,
            )
        print(f"[CHECKPOINT] wrote pre-final validated coordinates -> {args.checkpoint_gephi}", flush=True)
    if separation_diag["remaining_violations"]:
        raise RuntimeError(
            "Final mirror-preserving separation did not converge: "
            f"{separation_diag['remaining_violations']} pairs remain below {args.min_sep}. "
            "The resumable pre-final checkpoint has been preserved."
        )
    final_cross = count_edge_crossings(G, pos_final, force_exact=(args.crossing_mode != "estimate"))
    print(f"[FINAL] crossings after all refinements = {final_cross}", flush=True)

    final_weighted_length = _global_edge_length_score(G, pos_final, edges_all) / _edge_length_scale(args.min_sep)
    final_center_metrics = centrality_metrics(pos_final, center_targets)
    final_cross_axis_fraction = _global_cross_axis_penalty(
        G, pos_final, edges_all, sides
    )
    final_cross_axis_fraction_unweighted = _global_cross_axis_penalty(
        G, pos_final, edges_all, sides, uniform=True
    )
    for node in G.nodes():
        G.nodes[node]["Central_Layout_Target"] = bool(node in center_targets)
        G.nodes[node]["Hemisphere_Optimization"] = args.hemisphere_optimization
    for u, v, data in G.edges(data=True):
        data["Hemisphere_Optimization"] = args.hemisphere_optimization
        data["Cross_Axis_At_Final"] = bool(
            sides.get(u) != "axis" and sides.get(v) != "axis"
            and float(pos_final[u][0]) * float(pos_final[v][0]) < 0.0
        )
    metric_lengths = []
    for u, v, data in G.edges(data=True):
        if "Layout_Weight_Input_Value" not in data or not data.get("Layout_Weight_Data_Available"):
            continue
        pu, pv = pos_final[u], pos_final[v]
        metric_lengths.append((float(data["Layout_Weight_Input_Value"]), math.hypot(pu[0] - pv[0], pu[1] - pv[1])))
    metric_length_spearman = None
    if len(metric_lengths) >= 2:
        metric_length_spearman = spearman_rank_correlation(metric_lengths)
    run_metrics = {
        "objective": OBJECTIVE_CONFIGURATION,
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "resume_start_crossings": resume_start_crossings,
        "resume_start_weighted_edge_length_score": resume_start_weighted_length,
        "resume_start_soft_spacing_penalty": resume_start_spacing_penalty,
        "initial_crossings": initial_crossings,
        "final_crossings": final_cross,
        "initial_weighted_edge_length_score": OBJECTIVE_CONFIGURATION.get("initial_weighted_length"),
        "final_weighted_edge_length_score": final_weighted_length,
        "layout_weight_input_metric": next((d.get("Layout_Weight_Input_Metric") for _, _, d in G.edges(data=True) if d.get("Layout_Weight_Input_Metric")), None),
        "weight_input_vs_edge_length_spearman": metric_length_spearman,
        "runtime_limit_reached": runtime_limit_reached,
        "hemisphere_optimization": args.hemisphere_optimization,
        "hemi_span_adjustment": args.hemi_span_adjustment,
        "final_span_expansion": args.final_span_expansion,
        "hemisphere_statistics": hemisphere_totals,
        "center_isomer": args.center_isomer,
        "center_targets": list(center_targets),
        "final_centrality": final_center_metrics,
        "final_cross_axis_edge_fraction": final_cross_axis_fraction,
        "final_unweighted_cross_axis_edge_fraction": final_cross_axis_fraction_unweighted,
        "final_span_before_balance": span_before_final_balance,
        "final_span_after_balance": axis_chiral_span_metrics(pos_final, sides),
        "separation": separation_diag,
        "cycles": cycle_metrics,
    }
    print(f"[METRICS] {run_metrics}", flush=True)
    if args.out_metrics_json:
        Path(args.out_metrics_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_metrics_json).write_text(json.dumps(run_metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.checkpoint_gephi:
        write_gexf_atomic(G, pos_final, args.checkpoint_gephi)
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
LAYOUT_EDGE_LENGTH_POWER = 1.0
LAYOUT_CROSS_AXIS_WEIGHT = 0.0
LAYOUT_CENTER_WEIGHT = 0.0
LAYOUT_CENTER_TARGETS = tuple()
LAYOUT_CENTER_REFERENCE = 0.01
OBJECTIVE_CONFIGURATION = {"mode": "legacy"}


def configure_layout_objective(mode, crossing_weight, edge_length_weight, spacing_weight, soft_sep_factor,
                               initial_crossings=None, initial_weighted_length=None, initial_spacing_penalty=None,
                               edge_length_power=1.0, cross_axis_weight=0.0,
                               center_weight=0.0, center_targets=(), initial_center_penalty=None):
    """Configure comparable objective terms for a single layout run."""
    global LAYOUT_CROSSING_WEIGHT, LAYOUT_EDGE_LENGTH_WEIGHT, LAYOUT_SPACING_WEIGHT, LAYOUT_SOFT_SEP_FACTOR
    global LAYOUT_EDGE_LENGTH_POWER, LAYOUT_CROSS_AXIS_WEIGHT, LAYOUT_CENTER_WEIGHT
    global LAYOUT_CENTER_TARGETS, LAYOUT_CENTER_REFERENCE, OBJECTIVE_CONFIGURATION
    mode = str(mode).strip().lower()
    if mode not in {"legacy", "initial_normalized"}:
        raise ValueError("objective mode must be 'legacy' or 'initial_normalized'.")
    if soft_sep_factor < 1.0:
        raise ValueError("soft separation factor must be at least 1.")
    if mode == "initial_normalized":
        if initial_crossings is None or initial_weighted_length is None or initial_spacing_penalty is None:
            raise ValueError("Initial objective references are required for normalized mode.")
        LAYOUT_CROSSING_WEIGHT = float(crossing_weight) / max(float(initial_crossings), 1.0)
        LAYOUT_EDGE_LENGTH_WEIGHT = float(edge_length_weight) / max(float(initial_weighted_length), 1e-12)
        LAYOUT_SPACING_WEIGHT = float(spacing_weight) / max(float(initial_spacing_penalty), 1e-12)
    else:
        LAYOUT_CROSSING_WEIGHT = float(crossing_weight)
        LAYOUT_EDGE_LENGTH_WEIGHT = float(edge_length_weight)
        LAYOUT_SPACING_WEIGHT = float(spacing_weight)
    LAYOUT_SOFT_SEP_FACTOR = float(soft_sep_factor)
    LAYOUT_EDGE_LENGTH_POWER = float(edge_length_power)
    LAYOUT_CROSS_AXIS_WEIGHT = float(cross_axis_weight)
    LAYOUT_CENTER_TARGETS = tuple(center_targets or ())
    LAYOUT_CENTER_REFERENCE = max(float(initial_center_penalty or 0.0), 0.01)
    if mode == "initial_normalized":
        LAYOUT_CENTER_WEIGHT = float(center_weight) / LAYOUT_CENTER_REFERENCE
    else:
        LAYOUT_CENTER_WEIGHT = float(center_weight)
    OBJECTIVE_CONFIGURATION = {
        "mode": mode,
        "requested_crossing_weight": float(crossing_weight),
        "requested_edge_length_weight": float(edge_length_weight),
        "spacing_weight": float(spacing_weight),
        "center_weight": float(center_weight),
        "cross_axis_edge_weight": float(cross_axis_weight),
        "soft_sep_factor": float(soft_sep_factor),
        "edge_length_power": float(edge_length_power),
        "initial_crossings": None if initial_crossings is None else float(initial_crossings),
        "initial_weighted_length": None if initial_weighted_length is None else float(initial_weighted_length),
        "initial_spacing_penalty": None if initial_spacing_penalty is None else float(initial_spacing_penalty),
        "initial_center_penalty": None if initial_center_penalty is None else float(initial_center_penalty),
        "effective_crossing_weight": LAYOUT_CROSSING_WEIGHT,
        "effective_edge_length_weight": LAYOUT_EDGE_LENGTH_WEIGHT,
        "effective_spacing_weight": LAYOUT_SPACING_WEIGHT,
        "effective_center_weight": LAYOUT_CENTER_WEIGHT,
        "center_targets": list(LAYOUT_CENTER_TARGETS),
        "edge_objective_scope": LAYOUT_EDGE_OBJECTIVE_SCOPE,
    }


def edge_in_length_objective(G, u, v):
    """Return whether an edge contributes to scientific length objectives."""
    if LAYOUT_EDGE_OBJECTIVE_SCOPE == "all":
        return True
    return bool(G.edges[u, v].get("Layout_Weight_Data_Available", False))


def _local_edge_length_score(G, best_pos, candidate_pos, moved_nodes, edges_all):
    moved_set = set(moved_nodes)
    seen = set()
    score = 0.0
    for node in moved_set:
        for other in G[node]:
            u, v = (node, other) if node <= other else (other, node)
            key = (u, v)
            if key in seen:
                continue
            seen.add(key)
            if not edge_in_length_objective(G, u, v):
                continue
            pu = candidate_pos.get(u, best_pos.get(u))
            pv = candidate_pos.get(v, best_pos.get(v))
            if pu is None or pv is None:
                continue
            weight = float(G.edges[u, v].get("Layout_Spring_Weight", 1.0))
            distance = math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
            score += weight * (distance ** LAYOUT_EDGE_LENGTH_POWER)
    return score


def _edge_length_scale(min_sep):
    return max(float(min_sep), 1e-9) ** LAYOUT_EDGE_LENGTH_POWER


def _local_spacing_penalty(best_pos, candidate_pos, moved_nodes, grid, inv_cell, min_sep, soft_sep_factor=None):
    if min_sep <= 0.0:
        return 0.0
    min_sep = float(min_sep)
    if soft_sep_factor is None:
        soft_sep_factor = LAYOUT_SOFT_SEP_FACTOR
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


def _global_edge_length_score(G, pos, edges_all):
    total = 0.0
    for u, v in edges_all:
        if not edge_in_length_objective(G, u, v):
            continue
        pu = pos.get(u)
        pv = pos.get(v)
        if pu is None or pv is None:
            continue
        weight = float(G.edges[u, v].get("Layout_Spring_Weight", 1.0))
        distance = math.hypot(float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1]))
        total += weight * (distance ** LAYOUT_EDGE_LENGTH_POWER)
    return total


def centrality_metrics(pos, targets=()):
    present = [node for node in targets if node in pos]
    if not pos or not present:
        return {
            "target_y": None,
            "layout_y_min": None,
            "layout_y_max": None,
            "layout_y_midpoint": None,
            "absolute_offset": 0.0,
            "fractional_offset": 0.0,
            "penalty": 0.0,
        }
    ys = [float(point[1]) for point in pos.values()]
    ymin, ymax = min(ys), max(ys)
    midpoint = 0.5 * (ymin + ymax)
    target_y = sum(float(pos[node][1]) for node in present) / len(present)
    span = ymax - ymin
    offset = abs(target_y - midpoint)
    fractional = 0.0 if span <= 1e-12 else 2.0 * offset / span
    return {
        "target_y": target_y,
        "layout_y_min": ymin,
        "layout_y_max": ymax,
        "layout_y_midpoint": midpoint,
        "absolute_offset": offset,
        "fractional_offset": fractional,
        "penalty": fractional * fractional,
    }


def _global_cross_axis_penalty(G, pos, edges_all, sides, uniform=False):
    weighted_opposite = 0.0
    weighted_total = 0.0
    for u, v in edges_all:
        if u == v or sides.get(u) == "axis" or sides.get(v) == "axis":
            continue
        if not uniform and not edge_in_length_objective(G, u, v):
            continue
        weight = 1.0 if uniform else float(
            G.edges[u, v].get("Layout_Spring_Weight", 1.0)
        )
        weighted_total += weight
        if float(pos[u][0]) * float(pos[v][0]) < 0.0:
            weighted_opposite += weight
    return 0.0 if weighted_total <= 0.0 else weighted_opposite / weighted_total


def _global_spacing_penalty(pos, min_sep, soft_sep_factor=None):
    if min_sep <= 0.0 or len(pos) < 2:
        return 0.0
    grid, inv_cell = _build_spatial_hash(pos, min_sep)
    min_sep = float(min_sep)
    if soft_sep_factor is None:
        soft_sep_factor = LAYOUT_SOFT_SEP_FACTOR
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
                          crossing_weight=None, edge_length_weight=None,
                          spacing_weight=None, axis_gap_weight=None):
    if sides is None:
        sides = {}
    if crossing_weight is None:
        crossing_weight = LAYOUT_CROSSING_WEIGHT
    if edge_length_weight is None:
        edge_length_weight = LAYOUT_EDGE_LENGTH_WEIGHT
    if spacing_weight is None:
        spacing_weight = LAYOUT_SPACING_WEIGHT
    if axis_gap_weight is None:
        axis_gap_weight = LAYOUT_AXIS_GAP_WEIGHT
    crossings = count_edge_crossings(G, pos)
    edge_score = _global_edge_length_score(G, pos, edges_all) / _edge_length_scale(min_sep)
    spacing_score = _global_spacing_penalty(pos, min_sep)
    axis_score = _axis_gap_penalty(pos, sides) if sides else 0.0
    cross_axis_score = _global_cross_axis_penalty(G, pos, edges_all, sides) if sides else 0.0
    center_score = centrality_metrics(pos, LAYOUT_CENTER_TARGETS)["penalty"]
    return (
        crossing_weight * float(crossings)
        + edge_length_weight * float(edge_score)
        + spacing_weight * float(spacing_score)
        + axis_gap_weight * float(axis_score)
        + LAYOUT_CROSS_AXIS_WEIGHT * float(cross_axis_score)
        + LAYOUT_CENTER_WEIGHT * float(center_score)
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
            current_edge_pen = _local_edge_length_score(G, best_pos, {}, moved, edges_all) / _edge_length_scale(min_sep)
            current_repulse_pen = _local_spacing_penalty(best_pos, {}, moved, spatial_grid, inv_cell, repulse_dist)
            current_score = (
                float(current_cross) * LAYOUT_CROSSING_WEIGHT
                + LAYOUT_EDGE_LENGTH_WEIGHT * current_edge_pen
                + LAYOUT_SPACING_WEIGHT * current_repulse_pen
            )

            if delta_fn is not None:
                delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes,
                                 _seed_offset=it)
                new_cross = current_cross + delta
            else:
                new_cross = count_edge_crossings(G, {**best_pos, **candidate_pos})

            new_edge_pen = _local_edge_length_score(G, best_pos, candidate_pos, moved, edges_all) / _edge_length_scale(min_sep)
            new_repulse_pen = _local_spacing_penalty(best_pos, candidate_pos, moved, spatial_grid, inv_cell, repulse_dist)
            new_score = (
                float(new_cross) * LAYOUT_CROSSING_WEIGHT
                + LAYOUT_EDGE_LENGTH_WEIGHT * new_edge_pen
                + LAYOUT_SPACING_WEIGHT * new_repulse_pen
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
                      delta_fn, fast_mode, sample_size, seed, crossing_weight=None,
                      edge_weight=None, spacing_weight=None):
    if crossing_weight is None:
        crossing_weight = LAYOUT_CROSSING_WEIGHT
    if edge_weight is None:
        edge_weight = LAYOUT_EDGE_LENGTH_WEIGHT
    if spacing_weight is None:
        spacing_weight = LAYOUT_SPACING_WEIGHT
    if delta_fn is not None:
        delta = delta_fn(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes, _seed_offset=seed)
    else:
        delta = delta_crossings_for_move(G, best_pos, moved, candidate_pos, edges_all, edge_bboxes)
    new_cross = current_cross + delta
    spatial_grid, inv_cell = _build_spatial_hash(best_pos, min_sep)
    current_edge = _local_edge_length_score(G, best_pos, {}, moved, edges_all) / _edge_length_scale(min_sep)
    new_edge = _local_edge_length_score(G, best_pos, candidate_pos, moved, edges_all) / _edge_length_scale(min_sep)
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
            + LAYOUT_EDGE_LENGTH_WEIGHT * (_global_edge_length_score(G, pos_local, edges_all) / _edge_length_scale(min_sep))
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


# ---------------------------
# Optional hemisphere and central-target optimisation
# ---------------------------
def resolve_center_targets(G, pairs, requested):
    """Resolve a requested barcode to one achiral node or a complete chiral pair."""
    if requested is None or _clean_id(requested) == "":
        return tuple()
    node = _clean_id(requested)
    if node not in G:
        raise ValueError(f"--center-isomer {node!r} is not present in the nodes CSV.")
    if G.nodes[node].get("Chirality", "") != "chiral":
        return (node,)
    partner = _clean_id(G.nodes[node].get("Enantiomer_Id", ""))
    if not partner or partner not in G:
        raise ValueError(
            f"--center-isomer {node!r} is chiral but has no valid enantiomer in the nodes CSV."
        )
    if _clean_id(G.nodes[partner].get("Enantiomer_Id", "")) != node:
        raise ValueError(
            f"--center-isomer {node!r} has non-reciprocal enantiomer metadata with {partner!r}."
        )
    return tuple(sorted((node, partner)))


def place_center_targets_at_midpoint(pos, targets):
    """Place the requested node or mirrored pair at the current y midrange."""
    result = dict(pos)
    if not result or not targets:
        return result
    ys = [float(point[1]) for point in result.values()]
    midpoint = 0.5 * (min(ys) + max(ys))
    for node in targets:
        if node in result:
            result[node] = (float(result[node][0]), midpoint)
    return result


def synchronise_sides_from_positions(pos, sides):
    """Return side labels matching the actual coordinates after optional pair flips."""
    result = dict(sides)
    for node, point in pos.items():
        if result.get(node) == "axis":
            continue
        result[node] = "right" if float(point[0]) >= 0.0 else "left"
    return result


def _pair_index(pairs):
    return {node: pid for pid, members in pairs.items() for node in members}


def _orientation_edge_weight(G, u, v, uniform=False):
    if uniform:
        return 1.0
    if not edge_in_length_objective(G, u, v):
        return 0.0
    return float(G.edges[u, v].get("Layout_Spring_Weight", 1.0))


def orientation_proxy_score(G, pos, pairs, uniform=False):
    """Weighted chiral-chiral length used to rank discrete orientation proposals."""
    node_to_pair = _pair_index(pairs)
    total = 0.0
    for u, v in G.edges():
        if u == v or node_to_pair.get(u) is None or node_to_pair.get(v) is None:
            continue
        if node_to_pair[u] == node_to_pair[v]:
            continue
        weight = _orientation_edge_weight(G, u, v, uniform=uniform)
        if weight <= 0.0:
            continue
        distance = math.hypot(
            float(pos[u][0]) - float(pos[v][0]),
            float(pos[u][1]) - float(pos[v][1]),
        )
        total += weight * (distance ** LAYOUT_EDGE_LENGTH_POWER)
    return total


def flip_pair_block(pos, sides, pairs, pair_ids):
    """Swap complete enantiomeric pairs across the axis without moving occupied sites."""
    result = dict(pos)
    for pid in pair_ids:
        A, B = pairs[pid]
        result[A], result[B] = result[B], result[A]
    return result, synchronise_sides_from_positions(result, sides)


def _pair_local_orientation_delta(
    G, pos, pairs, pid, uniform=False, node_to_pair=None
):
    if node_to_pair is None:
        node_to_pair = _pair_index(pairs)
    A, B = pairs[pid]
    swapped = {A: pos[B], B: pos[A]}
    before = 0.0
    after = 0.0
    seen = set()
    for node in (A, B):
        for other in G[node]:
            key = (node, other) if node <= other else (other, node)
            if key in seen or node_to_pair.get(other) == pid:
                continue
            seen.add(key)
            weight = _orientation_edge_weight(G, key[0], key[1], uniform=uniform)
            if weight <= 0.0:
                continue
            pu, pv = pos[key[0]], pos[key[1]]
            cu, cv = swapped.get(key[0], pu), swapped.get(key[1], pv)
            before_distance = math.hypot(
                float(pu[0]) - float(pv[0]), float(pu[1]) - float(pv[1])
            )
            after_distance = math.hypot(
                float(cu[0]) - float(cv[0]), float(cu[1]) - float(cv[1])
            )
            before += weight * (before_distance ** LAYOUT_EDGE_LENGTH_POWER)
            after += weight * (after_distance ** LAYOUT_EDGE_LENGTH_POWER)
    return after - before


def propose_global_orientation(G, pos, sides, pairs, uniform=False, max_sweeps=4):
    """Deterministic whole-graph coordinate descent over enantiomer side assignments."""
    candidate = dict(pos)
    candidate_sides = synchronise_sides_from_positions(candidate, sides)
    flips = []
    sweeps = 0
    node_to_pair = _pair_index(pairs)
    for sweep in range(max(1, int(max_sweeps))):
        ranked = []
        for pid in sorted(pairs):
            delta = _pair_local_orientation_delta(
                G, candidate, pairs, pid, uniform=uniform,
                node_to_pair=node_to_pair,
            )
            if delta < -1e-9:
                ranked.append((delta, pid))
        if not ranked:
            break
        changed = False
        for _, pid in sorted(ranked):
            delta = _pair_local_orientation_delta(
                G, candidate, pairs, pid, uniform=uniform,
                node_to_pair=node_to_pair,
            )
            if delta < -1e-9:
                candidate, candidate_sides = flip_pair_block(
                    candidate, candidate_sides, pairs, {pid}
                )
                flips.append(pid)
                changed = True
        sweeps = sweep + 1
        if not changed:
            break
    return candidate, candidate_sides, {
        "sweeps": sweeps,
        "pair_flips": len(flips),
        "unique_pairs_flipped": len(set(flips)),
    }


def _pair_coupling_graph(G, pairs, uniform=False):
    node_to_pair = _pair_index(pairs)
    coupling = nx.Graph()
    coupling.add_nodes_from(pairs)
    for u, v in G.edges():
        pu, pv = node_to_pair.get(u), node_to_pair.get(v)
        if pu is None or pv is None or pu == pv:
            continue
        weight = _orientation_edge_weight(G, u, v, uniform=uniform)
        if weight <= 0.0:
            continue
        if coupling.has_edge(pu, pv):
            coupling[pu][pv]["strength"] += weight
        else:
            coupling.add_edge(pu, pv, strength=weight)
    return coupling, node_to_pair


def adaptive_block_candidates(G, pos, sides, pairs, uniform=False, max_pairs=128, seed_limit=16):
    """Build nested connected flip candidates from the most costly opposite-side edges."""
    coupling, node_to_pair = _pair_coupling_graph(G, pairs, uniform=uniform)
    costly = []
    for u, v in G.edges():
        pu, pv = node_to_pair.get(u), node_to_pair.get(v)
        if pu is None or pv is None or pu == pv or float(pos[u][0]) * float(pos[v][0]) >= 0.0:
            continue
        weight = _orientation_edge_weight(G, u, v, uniform=uniform)
        distance = math.hypot(float(pos[u][0]) - float(pos[v][0]), float(pos[u][1]) - float(pos[v][1]))
        costly.append((-(weight * distance), str(pu), str(pv), pu, pv))
    seed_edges = []
    seen_seed = set()
    for _, _, _, pu, pv in sorted(costly):
        key = tuple(sorted((pu, pv)))
        if key in seen_seed:
            continue
        seen_seed.add(key)
        seed_edges.append((pu, pv))
        if len(seed_edges) >= max(1, int(seed_limit)):
            break
    if len(seed_edges) < max(1, int(seed_limit)):
        strongest = sorted(
            (
                (-float(data.get("strength", 0.0)), str(a), str(b), a, b)
                for a, b, data in coupling.edges(data=True)
            )
        )
        for _, _, _, pu, pv in strongest:
            key = tuple(sorted((pu, pv)))
            if key in seen_seed:
                continue
            seen_seed.add(key)
            seed_edges.append((pu, pv))
            if len(seed_edges) >= max(1, int(seed_limit)):
                break

    targets = [size for size in (2, 4, 8, 16, 32, 64, 128) if size <= max_pairs]
    if max_pairs not in targets:
        targets.append(max_pairs)
    proposals = {}
    base_proxy = orientation_proxy_score(G, pos, pairs, uniform=uniform)
    for first, excluded in [(a, b) for a, b in seed_edges] + [(b, a) for a, b in seed_edges]:
        block = {first}
        frontier = set(coupling.neighbors(first)) - {excluded}
        previous_improving_delta = None
        for target_size in targets:
            while len(block) < target_size and frontier:
                ranked = []
                for pid in frontier:
                    internal = sum(
                        float(coupling[pid][other].get("strength", 0.0))
                        for other in block if coupling.has_edge(pid, other)
                    )
                    external = sum(
                        float(data.get("strength", 0.0))
                        for other, data in coupling[pid].items() if other not in block
                    )
                    ranked.append((-(internal - 0.25 * external), str(pid), pid))
                _, _, chosen = min(ranked)
                block.add(chosen)
                frontier.discard(chosen)
                frontier.update(set(coupling.neighbors(chosen)) - block - {excluded})
            if len(block) < 2:
                continue
            frozen = frozenset(block)
            if frozen in proposals:
                continue
            cand, cand_sides = flip_pair_block(pos, sides, pairs, frozen)
            proxy = orientation_proxy_score(G, cand, pairs, uniform=uniform)
            proxy_delta = proxy - base_proxy
            if proxy_delta < -1e-9:
                if (
                    previous_improving_delta is not None
                    and proxy_delta >= previous_improving_delta - 1e-9
                ):
                    break
                proposals[frozen] = (proxy_delta, cand, cand_sides)
                previous_improving_delta = proxy_delta
            elif previous_improving_delta is not None:
                break
    ordered = sorted(
        ((delta, tuple(sorted(block)), cand, cand_sides) for block, (delta, cand, cand_sides) in proposals.items()),
        key=lambda item: (item[0], len(item[1]), item[1]),
    )
    return ordered


def coherent_block_relaxation(
    G, pos, sides, pairs, core_pair_ids, iterations=5, halo_layers=1,
    min_sep=60.0, min_x=5.0, uniform=False,
):
    """Propose damped, mirror-safe coordinate relaxation for a connected pair block."""
    if iterations <= 0 or not core_pair_ids:
        return dict(pos), dict(sides), {"active_pairs": len(core_pair_ids), "iterations": 0}
    coupling, _ = _pair_coupling_graph(G, pairs, uniform=uniform)
    core = set(core_pair_ids)
    active = set(core)
    frontier = set(core)
    for _ in range(max(0, int(halo_layers))):
        next_frontier = set()
        for pid in frontier:
            next_frontier.update(coupling.neighbors(pid))
        next_frontier -= active
        active.update(next_frontier)
        frontier = next_frontier

    work = dict(pos)
    cap = max(1.0, 0.25 * float(min_sep))
    for _ in range(max(0, int(iterations))):
        updates = {}
        for pid in sorted(active):
            A, B = pairs[pid]
            forces = []
            for node in (A, B):
                px, py = work[node]
                fx = fy = total_weight = 0.0
                for other in G[node]:
                    weight = _orientation_edge_weight(G, node, other, uniform=uniform)
                    if weight <= 0.0:
                        continue
                    ox, oy = work[other]
                    fx += weight * (float(ox) - float(px))
                    fy += weight * (float(oy) - float(py))
                    total_weight += weight
                if total_weight > 0.0:
                    forces.append((fx / total_weight, fy / total_weight, 1.0 if float(px) >= 0.0 else -1.0))
            if not forces:
                continue
            dmag = sum(sign * fx for fx, _, sign in forces) / len(forces)
            dy = sum(fy for _, fy, _ in forces) / len(forces)
            mobility = 0.12 if pid in core else 0.04
            dmag = max(-cap, min(cap, mobility * dmag))
            dy = max(-cap, min(cap, mobility * dy))
            mag = max(float(min_x), 0.5 * (abs(float(work[A][0])) + abs(float(work[B][0]))) + dmag)
            new_y = 0.5 * (float(work[A][1]) + float(work[B][1])) + dy
            sign_a = 1.0 if float(work[A][0]) >= 0.0 else -1.0
            updates[A] = (sign_a * mag, new_y)
            updates[B] = (-sign_a * mag, new_y)
        work.update(updates)
    work, _ = enforce_strict_mirror(work, pairs, synchronise_sides_from_positions(work, sides))
    return work, synchronise_sides_from_positions(work, sides), {
        "active_pairs": len(active),
        "core_pairs": len(core),
        "iterations": int(iterations),
    }

if __name__ == "__main__":
    main()
