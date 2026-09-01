#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BVGraph3D: convert a 2D BVGraph layout into a
mirror-preserving 3D embedding suitable for visualization and publication.

Main behaviour and features:
 - Reads a GEXF/XGMML graph that already contains 2D coordinates (x,y) for nodes
 - Places achiral nodes on the mirror plane (x == 0) initially on a circle
 - Treats explicitly-labeled chiral pairs as mirrored representatives
 - Uses spring forces for edges, short-range repulsion and a one-sided radial
   container force; supports adaptive container radius shrinking when energy
   decreases
 - Optionally uses scipy.spatial.cKDTree + numba for fast repulsive force
   evaluation; gracefully falls back to pure-Python/numpy if unavailable
 - Enforces exact mirror symmetry for the final positions (left-right mirroring)
 - Performs occasional swap moves of representative nodes to reduce energy
 - Writes the final positions back into a GEXF with viz:position attributes

This file is a cleaned, documented and production-oriented refactor of the
development script. It aims to be readable, well-structured and suitable for
sharing in supplementary materials.

See also the corresponding 2D layout utilities (BVGraph2D.py).

Author: Paul McGonigal
Code development, refactoring, and documentation were carried out with assistance from OpenAI ChatGPT (GPT-5).
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

# Optional accelerated dependencies
try:
    from scipy.spatial import cKDTree
    SCIPY_AVAILABLE = True
except Exception:
    cKDTree = None
    SCIPY_AVAILABLE = False

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except Exception:
    njit = None
    prange = None
    NUMBA_AVAILABLE = False

# Configure a module-level logger
logger = logging.getLogger("BVGraph3D")

# ---------------------------
# I/O and helpers
# ---------------------------

def read_graph(path: str) -> nx.Graph:
    """Read a GEXF or XGMML graph using NetworkX.

    The expectation is that node attributes contain x,y coordinates (or a
    viz:position structure) produced by the 2D BVGraph layout.
    """
    if path.lower().endswith((".gexf", ".xgmml")):
        G = nx.read_gexf(path)
        return G
    raise ValueError("Unsupported input format; provide .gexf or .xgmml")


def attach_nodes_csv_annotations(G: nx.Graph, nodes_csv: str, id_col: str = "id") -> nx.Graph:
    """Attach node annotations from a CSV; column `id_col` used for matching."""
    import pandas as pd

    df = pd.read_csv(nodes_csv, dtype=str)
    # try common alternative id column names if the requested one is not present
    if id_col not in df.columns:
        for alt in ("Id", "ID", "node", "name"):
            if alt in df.columns:
                id_col = alt
                break
    df[id_col] = df[id_col].astype(str)
    df = df.set_index(id_col)
    for n in G.nodes():
        if n in df.index:
            row = df.loc[n]
            for col in row.index:
                G.nodes[n][col] = row[col]
    return G


def detect_pairing_and_type(G: nx.Graph) -> Tuple[Dict[str, str], Set[str]]:
    """Detect enantiomeric pairings and achiral flags.

    Returns:
        pairs: dict mapping node -> partner (only mutual pairs kept)
        achiral: set of nodes flagged as achiral
    """
    pairs: Dict[str, str] = {}
    achiral: Set[str] = set()
    idset = set(G.nodes())

    # prefer explicit attributes commonly used by BVGraph2D: Enantiomer_Id, Enantiomer, Partner
    for n, attrs in G.nodes(data=True):
        for key in list(attrs.keys()):
            if key.lower() in ("enantiomer_id", "enantiomer", "partner", "pair"):
                partner = str(attrs[key]).strip()
                if partner and partner in idset:
                    pairs[n] = partner

    # keep only mutual pairs
    mutual = {}
    for a, b in pairs.items():
        if pairs.get(b) == a:
            mutual[a] = b
    pairs = mutual

    # detect achiral marker
    for n, attrs in G.nodes(data=True):
        for ka in list(attrs.keys()):
            if ka.lower() in ("chirality", "is_achiral", "achiral"):
                val = str(attrs[ka]).strip().lower()
                if val in ("achiral", "ach", "true", "1", "yes", "y"):
                    achiral.add(n)

    # remove pairs if either node marked achiral
    for n in list(pairs.keys()):
        p = pairs.get(n)
        if p in achiral or n in achiral:
            pairs.pop(n, None)
            pairs.pop(p, None)
    return pairs, achiral


def get_2d_positions(G: nx.Graph) -> Dict[str, np.ndarray]:
    """Extract x,y (and optional z) coordinates for all nodes.

    Raises ValueError if x/y are missing for any node.
    """
    pos: Dict[str, np.ndarray] = {}
    for n, attrs in G.nodes(data=True):
        x = attrs.get("x")
        y = attrs.get("y")
        z = attrs.get("z")
        viz = attrs.get("viz")
        if viz and isinstance(viz, dict):
            p = viz.get("position", {})
            x = x if x is not None else p.get("x")
            y = y if y is not None else p.get("y")
            z = z if z is not None else p.get("z")
        # try pos string like 'x,y'
        if (x is None or y is None) and attrs.get("pos"):
            p = attrs.get("pos")
            if isinstance(p, str) and "," in p:
                try:
                    xx, yy = p.split(",")[:2]
                    x = float(xx); y = float(yy)
                except Exception:
                    pass
        if x is None or y is None:
            raise ValueError(f"Node {n!r} lacks x,y coordinates. Provide a layout with x & y.")
        zval = float(z) if z is not None else 0.0
        pos[n] = np.array([float(x), float(y), zval], dtype=float)
    return pos


def write_gexf_with_positions(G: nx.Graph, positions: Dict[str, np.ndarray], path_out: str) -> None:
    """Write node positions back into GEXF with viz:position attributes."""
    for n, p in positions.items():
        G.nodes[n]["x"] = float(p[0]); G.nodes[n]["y"] = float(p[1]); G.nodes[n]["z"] = float(p[2])
        G.nodes[n]["viz"] = {"position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}}
    nx.write_gexf(G, path_out)
    logger.info("Wrote %s", path_out)


# ---------------------------
# Placement helpers
# ---------------------------

def place_achiral_on_circle(positions: Dict[str, np.ndarray], achiral_set: Iterable[str], R_target: float, jitter: float = 0.02) -> None:
    """Place achiral nodes on the mirror-plane circle x=0, y^2+z^2 = R_target^2.

    Modifies positions in-place.
    """
    achiral_list = sorted(list(achiral_set))
    m = len(achiral_list)
    if m == 0:
        return
    angles = np.linspace(0, 2 * np.pi, m, endpoint=False)
    for idx, n in enumerate(achiral_list):
        theta = float(angles[idx])
        y = R_target * math.cos(theta)
        z = R_target * math.sin(theta) + float(np.random.normal(scale=jitter))
        positions[n] = np.array([0.0, y, z], dtype=float)


# ---------------------------
# Forces & energies
# ---------------------------

def edge_spring_forces(positions: Dict[str, np.ndarray], edges_list: Iterable[Tuple[str, str]], k_spring: float = 1.0, rest_lengths: Optional[Dict[Tuple[str, str], float]] = None) -> Dict[str, np.ndarray]:
    forces = {n: np.zeros(3, dtype=float) for n in positions}
    for u, v in edges_list:
        pu = positions[u]; pv = positions[v]
        dvec = pu - pv
        dist = np.linalg.norm(dvec)
        if dist == 0.0:
            dvec = np.random.randn(3) * 1e-9
            dist = np.linalg.norm(dvec)
        rest = None
        if rest_lengths is not None:
            rest = rest_lengths.get((u, v), rest_lengths.get((v, u), None))
        if rest is None:
            rest = dist
        fmag = -k_spring * (dist - rest)
        f = fmag * (dvec / dist)
        forces[u] += f
        forces[v] -= f
    return forces


# Repulsion kernel: use numba accelerated pair loop when available
if NUMBA_AVAILABLE:
    @njit(parallel=True)
    def _compute_pair_forces_numba(pts: np.ndarray, pairs: np.ndarray, min_dist: float, k_repulse: float):
        N = pts.shape[0]
        M = pairs.shape[0]
        forces = np.zeros((N, 3), dtype=np.float64)
        eps = 1e-12
        for idx in prange(M):
            i = pairs[idx, 0]; j = pairs[idx, 1]
            dx0 = pts[i, 0] - pts[j, 0]
            dx1 = pts[i, 1] - pts[j, 1]
            dx2 = pts[i, 2] - pts[j, 2]
            dist = math.sqrt(dx0 * dx0 + dx1 * dx1 + dx2 * dx2)
            if dist < min_dist and dist > eps:
                diff = (min_dist - dist)
                fmag = k_repulse * 2.0 * diff
                invd = 1.0 / dist
                fx = fmag * (dx0 * invd)
                fy = fmag * (dx1 * invd)
                fz = fmag * (dx2 * invd)
                forces[i, 0] += fx; forces[i, 1] += fy; forces[i, 2] += fz
                forces[j, 0] -= fx; forces[j, 1] -= fy; forces[j, 2] -= fz
        return forces


def repulsive_forces(positions: Dict[str, np.ndarray], nodes_list: List[str], min_dist: float, k_repulse: float, pairs: Optional[np.ndarray] = None, pts: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Compute short-range repulsive forces.

    The implementation uses cached `pairs`/`pts` (index pairs over `nodes_list`) when
    provided for speed; otherwise it falls back to cKDTree query -> numba/kernel -> numpy loop.
    """
    N = len(nodes_list)
    if pts is None:
        pts = np.empty((N, 3), dtype=np.float64)
        for idx, n in enumerate(nodes_list):
            pts[idx, :] = positions[n]
    else:
        pts = np.asarray(pts, dtype=np.float64)

    forces = {n: np.zeros(3, dtype=float) for n in positions}

    if pairs is not None:
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.size == 0:
            return forces
        if NUMBA_AVAILABLE:
            forces_arr = _compute_pair_forces_numba(pts, pairs_arr, float(min_dist), float(k_repulse))
            for idx, node in enumerate(nodes_list):
                forces[node] = forces_arr[idx]
            return forces
        # numpy fallback with provided pairs
        i_idx = pairs_arr[:, 0]; j_idx = pairs_arr[:, 1]
        pi = pts[i_idx]; pj = pts[j_idx]
        dvec = pi - pj
        dist = np.linalg.norm(dvec, axis=1)
        zero_mask = dist == 0.0
        if zero_mask.any():
            dvec[zero_mask] = np.random.randn(zero_mask.sum(), 3) * 1e-9
            dist = np.linalg.norm(dvec, axis=1)
        diff = (min_dist - dist)
        fmag = k_repulse * 2.0 * diff
        nvec = (dvec.T / dist).T
        fvec = (fmag[:, None] * nvec)
        for idx, (ii, jj) in enumerate(zip(i_idx, j_idx)):
            ni = nodes_list[ii]; nj = nodes_list[jj]
            f = fvec[idx]
            forces[ni] += f
            forces[nj] -= f
        return forces

    # If no pairs provided, use cKDTree to find nearby pairs
    if SCIPY_AVAILABLE and N > 1:
        try:
            tree = cKDTree(pts)
            pairs_found = tree.query_pairs(r=min_dist, output_type='ndarray')
            if pairs_found.size == 0:
                return forces
            if NUMBA_AVAILABLE:
                forces_arr = _compute_pair_forces_numba(pts, pairs_found, float(min_dist), float(k_repulse))
                for idx, node in enumerate(nodes_list):
                    forces[node] = forces_arr[idx]
                return forces
            # numpy fallback
            i_idx = pairs_found[:, 0]; j_idx = pairs_found[:, 1]
            pi = pts[i_idx]; pj = pts[j_idx]
            dvec = pi - pj
            dist = np.linalg.norm(dvec, axis=1)
            zero_mask = dist == 0.0
            if zero_mask.any():
                dvec[zero_mask] = np.random.randn(zero_mask.sum(), 3) * 1e-9
                dist = np.linalg.norm(dvec, axis=1)
            diff = (min_dist - dist)
            fmag = k_repulse * 2.0 * diff
            nvec = (dvec.T / dist).T
            fvec = (fmag[:, None] * nvec)
            for idx, (ii, jj) in enumerate(zip(i_idx, j_idx)):
                ni = nodes_list[ii]; nj = nodes_list[jj]
                f = fvec[idx]
                forces[ni] += f
                forces[nj] -= f
            return forces
        except Exception:
            # fall through to brute-force pairwise
            pass

    # final brute-force fallback (O(N^2))
    for i in range(N):
        a = nodes_list[i]; pa = positions[a]
        for j in range(i + 1, N):
            b = nodes_list[j]; pb = positions[b]
            dvec = pa - pb
            dist = np.linalg.norm(dvec)
            if dist == 0.0:
                dvec = np.random.randn(3) * 1e-9
                dist = np.linalg.norm(dvec)
            if dist < min_dist:
                diff = (min_dist - dist)
                fmag = k_repulse * 2.0 * diff
                f = fmag * (dvec / dist)
                forces[a] += f
                forces[b] -= f
    return forces


def radial_forces_container(positions: Dict[str, np.ndarray], nodes_list: Iterable[str], R_curr: float, k_radial: float, k_radial_z_factor: float = 1.0) -> Dict[str, np.ndarray]:
    """One-sided soft-wall radial force: pushes nodes inside when they exceed R_curr."""
    forces = {n: np.zeros(3, dtype=float) for n in positions}
    for n in nodes_list:
        p = positions[n]
        r = np.linalg.norm(p)
        if r <= R_curr or r == 0.0:
            continue
        f_base = -k_radial * 2.0 * (r - R_curr) * (p / r)
        f_base[2] *= k_radial_z_factor
        forces[n] += f_base
    return forces


def compute_energy(positions: Dict[str, np.ndarray], edges_list: Iterable[Tuple[str, str]], min_dist: float, k_spring: float, rest_lengths: Dict[Tuple[str, str], float], k_repulse: float, k_radial: float, R_curr: float, nodes_list: List[str], pairs: Optional[np.ndarray] = None, pts: Optional[np.ndarray] = None) -> float:
    """Compute the scalar energy used for acceptance decisions during adaptive shrink.

    Energy terms:
      - springs: 0.5*k_spring*(d - rest)^2
      - repulsion (for d < min_dist): 0.5*k_repulse*(min_dist - d)^2
      - radial container (for r > R_curr): 0.5*k_radial*(r - R_curr)^2
    """
    E = 0.0
    for (u, v) in edges_list:
        duv = np.linalg.norm(positions[u] - positions[v])
        rest = rest_lengths.get((u, v), rest_lengths.get((v, u), duv))
        diff = duv - rest
        E += 0.5 * k_spring * (diff * diff)

    if pts is None:
        pts = np.vstack([positions[n] for n in nodes_list])
    else:
        pts = np.asarray(pts, dtype=np.float64)

    if pairs is not None:
        pairs_arr = np.asarray(pairs, dtype=np.int64)
        if pairs_arr.size > 0:
            pi = pts[pairs_arr[:, 0]]; pj = pts[pairs_arr[:, 1]]
            dvec = pi - pj
            dist = np.linalg.norm(dvec, axis=1)
            mask = dist < min_dist
            if mask.any():
                diffs = (min_dist - dist[mask])
                E += 0.5 * k_repulse * np.sum(diffs * diffs)
    else:
        # try cKDTree to find close pairs
        if SCIPY_AVAILABLE and len(pts) > 1:
            try:
                tree = cKDTree(pts)
                pairs_found = tree.query_pairs(r=min_dist, output_type='ndarray')
                if pairs_found.size > 0:
                    pi = pts[pairs_found[:, 0]]; pj = pts[pairs_found[:, 1]]
                    dvec = pi - pj
                    dist = np.linalg.norm(dvec, axis=1)
                    mask = dist < min_dist
                    if mask.any():
                        diffs = (min_dist - dist[mask])
                        E += 0.5 * k_repulse * np.sum(diffs * diffs)
            except Exception:
                # conservative fallback: skip repulsion energy
                pass

    for n in nodes_list:
        r = np.linalg.norm(positions[n])
        if r > R_curr:
            E += 0.5 * k_radial * (r - R_curr) ** 2
    return float(E)


# ---------------------------
# Representative machinery and symmetry
# ---------------------------

def prepare_optimization(G: nx.Graph, pos2d: Dict[str, np.ndarray], pairs: Dict[str, str], achiral: Set[str], z_perturb: float = 1e-2):
    """Initialise 3D positions and representative lists.

    Returns (positions, reps_info, free_reps)
    - positions: dict node -> np.array([x,y,z])
    - reps_info: list of dicts describing representative nodes (chiral rep or achiral)
    - free_reps: list of representative node ids (nodes we integrate directly)
    """
    positions = {n: pos2d[n].astype(float).copy() for n in pos2d}
    # ensure z present
    for n in positions:
        if positions[n].shape[0] == 2:
            positions[n] = np.array([positions[n][0], positions[n][1], 0.0], dtype=float)
    # small z jitter for numerical breaking of degeneracies
    rng = np.random.RandomState(0)
    for n in positions:
        positions[n][2] += float(rng.normal(scale=z_perturb))

    # build chiral pair representatives (rep, partner) where rep has positive x by convention
    chiral_pairs: List[Tuple[str, str]] = []
    used: Set[str] = set()
    for a, b in pairs.items():
        if a in used or b in used:
            continue
        xa = float(positions[a][0]); xb = float(positions[b][0])
        if xa > 0 and xb < 0:
            rep, partner = a, b
        elif xb > 0 and xa < 0:
            rep, partner = b, a
        else:
            rep = sorted([a, b])[0]
            partner = b if rep == a else a
        chiral_pairs.append((rep, partner))
        used.add(a); used.add(b)

    achiral_list = sorted(list(achiral))

    # ensure achiral x=0 initially
    for n in achiral_list:
        if n in positions:
            positions[n][0] = 0.0

    # small nudge if both pair members have x==0
    for rep, partner in chiral_pairs:
        if abs(positions[rep][0]) < 1e-9 and abs(positions[partner][0]) < 1e-9:
            positions[rep][0] = 1e-3
            positions[partner][0] = -1e-3

    # ensure rep has positive x by convention
    fixed_pairs = []
    for rep, partner in chiral_pairs:
        if positions[rep][0] < 0:
            rep, partner = partner, rep
        fixed_pairs.append((rep, partner))
    chiral_pairs = fixed_pairs

    reps_info = []
    free_reps: List[str] = []
    for rep, partner in chiral_pairs:
        reps_info.append({"type": "chiral_rep", "node": rep, "partner": partner})
        free_reps.append(rep)
    for n in achiral_list:
        reps_info.append({"type": "achiral", "node": n, "partner": None})
        free_reps.append(n)

    return positions, reps_info, free_reps


def enforce_symmetry(positions: Dict[str, np.ndarray], reps_info: List[Dict[str, Optional[str]]]) -> None:
    """Apply mirror symmetry constraints: reflect reps to partners and set achiral x=0.

    Mutates `positions` in-place.
    """
    for info in reps_info:
        if info["type"] == "chiral_rep":
            rep = info["node"]; partner = info["partner"]
            p = positions[rep]
            # ensure rep x positive
            if p[0] < 0:
                p[0] = -p[0]
            positions[rep] = p
            positions[partner] = np.array([-p[0], p[1], p[2]], dtype=float)
        else:
            n = info["node"]
            p = positions[n]
            p[0] = 0.0
            positions[n] = p


# ---------------------------
# Relaxation + adaptive shrink
# ---------------------------

def run_relaxation(positions: Dict[str, np.ndarray], reps_info: List[Dict[str, Optional[str]]], edges_list: List[Tuple[str, str]], rest_lengths: Dict[Tuple[str, str], float], k_spring: float, k_repulse: float, k_radial: float, min_dist: float, R_init: float, lr: float = 0.005, iters: int = 2000, swap_trials: int = 200, k_radial_z_factor: float = 1.0, achiral_lock_iters: int = 0, shrink_step: float = 0.995, shrink_check_interval: int = 50, shrink_tol: float = 1e-6, min_r_factor: float = 0.2) -> Dict[str, np.ndarray]:
    """Main iterative relaxation routine.

    Operates on representative nodes only (reps_info). Enforces symmetry after each
    update. Performs occasional swap moves and an adaptive container shrink that is
    accepted only when total energy decreases.
    """
    all_nodes = list(positions.keys())
    # ensure symmetry initially
    enforce_symmetry(positions, reps_info)

    # caching for repulsion queries
    last_pairs = None
    last_pts = None
    repulse_update_every = 1

    R_curr = float(R_init)
    # compute initial energy
    E = compute_energy(positions, edges_list, min_dist, k_spring, rest_lengths, k_repulse, k_radial, R_curr, all_nodes)
    E_best_for_shrink = E
    shrinking_enabled = True

    for epoch in range(iters):
        t = epoch / max(1, (iters - 1))

        # simple scheduling for parameters (keeps behaviour similar to the original)
        SPRING_RAMP_FACTOR = 8.0
        RADIAL_SHRINK_FACTOR = 0.1
        LR_SHRINK_FACTOR = 0.25

        k_spring_curr = k_spring * (1.0 + (SPRING_RAMP_FACTOR - 1.0) * t)
        k_radial_curr = k_radial * (1.0 - (1.0 - RADIAL_SHRINK_FACTOR) * t)
        lr_curr = lr * (1.0 - (1.0 - LR_SHRINK_FACTOR) * t)

        # springs
        springs = edge_spring_forces(positions, edges_list, k_spring=k_spring_curr, rest_lengths=rest_lengths)

        # build KD-tree & query pairs occasionally (cache pairs for speed)
        nodes_list = list(positions.keys())
        if (epoch % repulse_update_every == 0) or (last_pairs is None):
            try:
                pts_now = np.vstack([positions[n] for n in nodes_list])
                if SCIPY_AVAILABLE:
                    tree = cKDTree(pts_now)
                    last_pairs = tree.query_pairs(r=min_dist, output_type='ndarray')
                    last_pts = pts_now
                else:
                    last_pairs = None
                    last_pts = pts_now
            except Exception:
                last_pairs = None
                last_pts = None

        # repulsion using cached pairs
        repulse = repulsive_forces(positions, nodes_list, min_dist=min_dist, k_repulse=k_repulse, pairs=last_pairs, pts=last_pts)

        radial = radial_forces_container(positions, all_nodes, R_curr=R_curr, k_radial=k_radial_curr, k_radial_z_factor=k_radial_z_factor)

        # total forces on representatives only
        total_forces = {n: np.zeros(3, dtype=float) for n in positions}
        for n in positions:
            total_forces[n] = springs.get(n, np.zeros(3, dtype=float)) + repulse.get(n, np.zeros(3, dtype=float)) + radial.get(n, np.zeros(3, dtype=float))

        # update representatives
        for info in reps_info:
            node = info['node']
            f = total_forces.get(node, np.zeros(3, dtype=float))
            # lock achiral positions on x==0 for initial iterations if requested
            if info['type'] == 'achiral' and epoch < achiral_lock_iters:
                positions[node][0] = 0.0
                continue
            positions[node] = positions[node] + lr_curr * f
            # enforce achiral x=0 and ensure chiral rep x positive
            if info['type'] == 'achiral':
                positions[node][0] = 0.0
            else:
                if positions[node][0] < 1e-12:
                    positions[node][0] = abs(positions[node][0]) + 1e-12

        # reflect to partners and keep mirror-plane constraint
        enforce_symmetry(positions, reps_info)

        # swap moves occasionally (attempts intended to escape shallow local minima)
        if swap_trials > 0 and (epoch % max(1, (iters // 10))) == 0:
            for _ in range(max(1, swap_trials // 10)):
                if len(reps_info) < 2:
                    break
                i, j = random.sample(range(len(reps_info)), 2)
                info_i = reps_info[i]; info_j = reps_info[j]
                if info_i['type'] != info_j['type']:
                    continue
                node_i = info_i['node']; node_j = info_j['node']
                b_i = positions[node_i].copy(); b_j = positions[node_j].copy()
                positions[node_i], positions[node_j] = b_j.copy(), b_i.copy()
                enforce_symmetry(positions, reps_info)
                E_new = compute_energy(positions, edges_list, min_dist, k_spring_curr, rest_lengths, k_repulse, k_radial_curr, R_curr, nodes_list)
                if E_new < E:
                    E = E_new
                else:
                    positions[node_i] = b_i; positions[node_j] = b_j
                    enforce_symmetry(positions, reps_info)

        # Adaptive shrink: attempt a slightly smaller container radius if it decreases energy
        if shrinking_enabled and (epoch % shrink_check_interval == 0) and (epoch >= achiral_lock_iters):
            R_candidate = R_curr * shrink_step
            if R_candidate < (min_r_factor * R_init):
                shrinking_enabled = False
            else:
                E_candidate = compute_energy(positions, edges_list, min_dist, k_spring_curr, rest_lengths, k_repulse, k_radial_curr, R_candidate, nodes_list, pairs=last_pairs, pts=last_pts)
                if E_candidate < E_best_for_shrink - shrink_tol:
                    R_curr = R_candidate
                    E_best_for_shrink = E_candidate
                    E = E_candidate
                else:
                    shrinking_enabled = False

        # light diagnostics at modest frequency
        if epoch % max(1, (iters // 10)) == 0:
            radii = np.array([np.linalg.norm(positions[n]) for n in positions])
            edge_lens = np.array([np.linalg.norm(positions[u] - positions[v]) for (u, v) in edges_list])
            logger.debug("epoch %d: mean_r=%.3f std_r=%.3f mean_edge=%.3f std_edge=%.3f R_curr=%.3f", epoch, radii.mean(), radii.std(), edge_lens.mean(), edge_lens.std(), R_curr)

    return positions


# ---------------------------
# CLI / main
# ---------------------------

def build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BVGraph3D: 3D embedding from 2D BVGraph layout (publication ready)")
    p.add_argument("--input", "-i", required=True, help="Input GEXF/XGMML with 2D layout (x,y)")
    p.add_argument("--output", "-o", required=True, help="Output GEXF path for 3D positions")
    p.add_argument("--nodes", default=None, help="Optional nodes CSV with Chirality/Enantiomer_Id")
    p.add_argument("--iters", type=int, default=3000, help="Number of optimization iterations")
    p.add_argument("--swap-trials", type=int, default=400, help="Swap move attempts per swap epoch")
    p.add_argument("--min-dist", type=float, default=0.6, help="Minimum repulsion distance (relative to L0 if default used)")
    p.add_argument("--k-spring", type=float, default=0.8)
    p.add_argument("--k-repulse", type=float, default=3.0)
    p.add_argument("--k-radial", type=float, default=6.0)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--radius-factor", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--z-perturb", type=float, default=0.02)
    p.add_argument("--k-radial-z-factor", type=float, default=1.0)
    p.add_argument("--achiral-lock-iters", type=int, default=600)
    p.add_argument("--rest-scale", type=float, default=0.75, help="L0 = median_2d_edge_length * rest_scale")
    p.add_argument("--shrink-step", type=float, default=0.995)
    p.add_argument("--shrink-check-interval", type=int, default=50)
    p.add_argument("--shrink-tol", type=float, default=1e-6)
    p.add_argument("--min-r-factor", type=float, default=0.2)
    p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose (DEBUG) logging")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    # logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")

    random.seed(args.seed); np.random.seed(args.seed)

    logger.info("Reading graph from %s", args.input)
    G = read_graph(args.input)
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    if args.nodes:
        logger.info("Attaching node annotations from %s", args.nodes)
        G = attach_nodes_csv_annotations(G, args.nodes, id_col="id")

    pairs, achiral = detect_pairing_and_type(G)
    logger.info("Detected %d chiral pairs and %d achiral nodes (heuristic)", len(pairs) // 2, len(achiral))

    pos2d = get_2d_positions(G)

    positions, reps_info, free_reps = prepare_optimization(G, pos2d, pairs, achiral, z_perturb=args.z_perturb)

    # initial radius estimate
    radii = [np.linalg.norm(positions[n]) for n in positions]
    median_r = float(np.median(radii)) if radii else 10.0
    R_init = args.radius_factor * median_r if median_r > 0 else 10.0
    logger.info("Initial median radius (R_init): %.4f", R_init)

    # place achiral nodes on a small circle at the mirror plane
    achiral_radius_frac = 0.15
    place_achiral_on_circle(positions, achiral, R_init * achiral_radius_frac, jitter=args.z_perturb)
    enforce_symmetry(positions, reps_info)

    # edges list and constant rest length L0
    edges_list = [(u, v) for u, v in G.edges()]
    # compute median 2D edge length
    edge_lengths_2d = []
    for (u, v) in edges_list:
        if u in pos2d and v in pos2d:
            pu = np.asarray(pos2d[u], dtype=float)
            pv = np.asarray(pos2d[v], dtype=float)
            edge_lengths_2d.append(np.linalg.norm(pu[:2] - pv[:2]))
    if len(edge_lengths_2d) == 0:
        edge_lengths_2d = [np.linalg.norm(positions[u] - positions[v]) for (u, v) in edges_list]
    L0 = float(np.median(edge_lengths_2d)) * args.rest_scale
    logger.info("Using constant rest length L0 = %.6f (median_2D * %.3f)", L0, args.rest_scale)

    rest_lengths: Dict[Tuple[str, str], float] = {}
    for (u, v) in edges_list:
        rest_lengths[(u, v)] = L0
        rest_lengths[(v, u)] = L0

    # adjust min_dist relative to L0 if user left default value (heuristic)
    min_dist = (0.6 * L0) if args.min_dist == 0.6 else args.min_dist

    final_pos = run_relaxation(positions, reps_info, edges_list, rest_lengths, k_spring=args.k_spring, k_repulse=args.k_repulse, k_radial=args.k_radial, min_dist=min_dist, R_init=R_init, lr=args.lr, iters=args.iters, swap_trials=args.swap_trials, k_radial_z_factor=args.k_radial_z_factor, achiral_lock_iters=args.achiral_lock_iters, shrink_step=args.shrink_step, shrink_check_interval=args.shrink_check_interval, shrink_tol=args.shrink_tol, min_r_factor=args.min_r_factor)

    # final enforce symmetry & write
    enforce_symmetry(final_pos, reps_info)
    write_gexf_with_positions(G, final_pos, args.output)
    logger.info("Done. Output written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
