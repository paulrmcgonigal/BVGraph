# BVGraph: mirror-preserving layouts for bullvalene isomer networks

BVGraph generates symmetry-aware layouts for bullvalene isomer graphs.

- `BVGraph2D.py` produces a 2D layout in which enantiomeric nodes are exact left/right mirror partners and achiral nodes lie on a central axis.
- `BVGraph3D.py` converts a completed 2D layout to a mirror-symmetric 3D embedding.

BVGraph2D is topological and unweighted by default. Optional edge weighting can make short connections preferentially represent low-energy equilibrium exchange or high transient population flux.

## Package contents

- `BVGraph2D.py`: 2D layout generation and optimisation
- `BVGraph3D.py`: 2D-to-3D conversion
- `README_BVGraph2D.txt`: complete 2D input, method, option, restart, and output reference
- `README_BVGraph3D.txt`: 3D usage guide
- `example/0000001123/`: established unweighted example
- `CITATION.cff`: citation metadata
- `LICENSE`: MIT licence

## Dependencies

```bash
pip install networkx numpy pandas
```

SciPy and Numba are recommended for larger 3D layouts. The separate unpublished kinetics preview also requires SciPy.

## Inputs

BVGraph2D reads a nodes CSV containing `id` and an edges CSV containing `source` and `target` (case-insensitive; `s` and `t` are also accepted). An optional `Energy` or `Relative Energy (kJ/mol)` node column is retained as metadata. Node identifiers are interpreted as bullvalene barcodes to determine canonical rotations, chirality, and enantiomer partners.

### Optional equilibrium-exchange weighting

Use `--edge-weighting equilibrium-exchange` with an edge column named `Relative TS Energy (kJ/mol)` by default. These values are Gibbs free energies of transition states on a common reference scale. For each undirected topological edge, the lowest finite transition-state energy among duplicate or parallel input records is used.

The raw attraction is:

```text
r_e = f + (1-f) exp[-(E_TS,e - E_TS,min)/(RT)]
```

Lower absolute transition-state free energies therefore receive stronger springs, as appropriate for emphasizing equilibrium exchange traffic. Node energies are not used in this layout weighting.

### Optional transient-flux weighting

Use `--edge-weighting transient-flux --transient-flux-file flux.csv`. The flux file contains one undirected edge per row:

```text
source,target,Integrated Absolute Net Flux,Integrated Gross Flux,Integrated Signed Net Flux,Flux Data Status
```

Select `--transient-flux-metric absolute-net` (the default) or `gross`. Absolute net flux emphasizes directed redistribution of population; gross flux emphasizes all forward-plus-reverse reaction traffic. BVGraph uses the selected integrated flux directly:

```text
r_e = f + (1-f) F_e/F_max
```

No second Boltzmann transformation is applied. A valid zero-flux edge receives the floor attraction. An unavailable edge remains explicitly distinguished from a measured zero.

For both modes, `f` is `--spring-weight-floor-ratio` (default 0.05). Non-self-loop spring weights are normalized to mean one. Weighting affects the initial NetworkX spring embedding and every subsequent weighted edge-length term; it does not prescribe absolute edge lengths. Self-loops retain metadata but do not exert geometric attraction.

## Standard workflow

Generate an unweighted 2D layout:

```bash
python BVGraph2D.py \
  --nodes example/0000001123/nodes.csv \
  --edges example/0000001123/edges.csv \
  --out layout.gexf \
  --out-xgmml layout.xgmml \
  --out-nodes-csv nodes_coordinates.csv \
  --out-edges-csv edge_metadata.csv
```

Generate an equilibrium-exchange layout:

```bash
python BVGraph2D.py \
  --nodes nodes.csv \
  --edges edges.csv \
  --out equilibrium_layout.gexf \
  --edge-weighting equilibrium-exchange \
  --edge-objective-scope weighted-data \
  --temperature-k 298.15
```

Generate a transient absolute-net-flux layout:

```bash
python BVGraph2D.py \
  --nodes nodes.csv \
  --edges edges.csv \
  --out transient_layout.gexf \
  --edge-weighting transient-flux \
  --transient-flux-file transient_flux_edges.csv \
  --transient-flux-metric absolute-net \
  --edge-objective-scope weighted-data
```

Convert a completed 2D layout to 3D:

```bash
python BVGraph3D.py --input layout.gexf --output layout_3d.gexf --nodes nodes.csv
```

BVGraph3D accepts weighted and unweighted BVGraph2D outputs without changing its 3D method.

## Layout method

BVGraph2D builds a simple undirected graph and selects one representative from each enantiomeric pair. NetworkX supplies a balanced spring layout of the representative graph. In weighted modes, `Layout_Spring_Weight` is active in this first embedding. The full graph is reconstructed by exact mirroring, and achiral nodes are arranged as connected blocks on the central axis.

Iterative, symmetry-preserving move classes then minimise a combined objective containing edge crossings, edge length, and soft-spacing penalties. With `--objective-mode initial_normalized`, the terms are divided by their starting values before the requested coefficients are applied. `--edge-objective-scope weighted-data` limits edge-length scoring to edges having valid mode-specific data; unavailable edges remain as floor-strength springs in the initial embedding.

The default `--hemisphere-optimization local` uses individual-pair and pair-of-pairs moves. `global` additionally examines side assignments across the complete graph after mirroring. `adaptive` repeats those whole-graph passes and tests connected blocks of enantiomeric pairs, followed by damped mirror-preserving relaxation of accepted blocks and their immediate neighbourhood. Weighted layouts rank these proposals using their active equilibrium or flux weights; unweighted layouts use equal edge importance. Every proposal is accepted against the complete active objective, so crossings, spacing and any central-isomer preference remain in force.

`--center-isomer BARCODE` (also `--centre-isomer`) softly places a selected achiral node, or both members of a selected chiral enantiomeric pair, at the vertical midpoint. The default coefficient is 0.10 when a target is supplied and is inactive otherwise. This affects only the y coordinate: chirality and mirror symmetry continue to determine x. Optional `--edge-length-power` and `--cross-axis-edge-weight` controls can place extra emphasis on long edges or chiral–chiral edges spanning the axis; their neutral defaults are 1.0 and 0.0.

Crossings may be counted exactly or estimated from sampled edge pairs. Dense layouts can progressively raise a mirror-preserving separation threshold. Spatially indexed projection avoids testing every node pair, and the final export is refused unless the requested hard minimum separation is satisfied with exact mirror symmetry. Edge clearance and achiral-axis constraints remain independent of energy or flux weighting.

Vertical span balancing is configurable. `--hemi-span-adjustment expand-smaller` proposes expansion of the shorter axis or chiral group, whereas `compress-larger` proposes contraction of the taller group; refinement accepts either only when the complete active objective improves. The default final chiral-span expansion can be disabled with `--final-span-expansion off` to preserve the objective-optimised coordinates through export. Hard separation and exact mirror validation remain active in either mode.

Long jobs support atomic GEXF/JSON checkpoints, best-layout checkpoints, deterministic cycle numbering, resumption, transactional valid cycles, stage diagnostics, convergence patience, and an optional clean wall-clock limit. Short unique temporary filenames, Windows extended-length paths and automatic recreation make checkpoint writes robust to deep folders, transient missing-file conditions and Windows locks. See `README_BVGraph2D.txt` for full details.

## Outputs

BVGraph2D writes GEXF and optionally XGMML, node-coordinate CSV, edge-metadata CSV, annotated-node CSV, side assignments, metrics JSON, and restart checkpoints. Edge exports include the weighting mode and metric, mode-specific input value and availability status, supplied flux measures, selected transition-state energy, normalized spring weight, source-row count, final cross-axis status and hemisphere mode. Node exports mark any central target. Metrics record central offset and accepted orientation/block statistics.

## Citation and licence

Use `CITATION.cff` when citing the software. BVGraph is distributed under the MIT licence in `LICENSE`.
