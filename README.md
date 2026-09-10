# BVGraph: mirror-preserving layouts for bullvalene isomer networks

BVGraph generates symmetry-aware layouts for bullvalene isomer graphs. `BVGraph2D.py` produces 2D layouts with exact left/right mirror partners for enantiomers and achiral nodes on a central axis. `BVGraph3D.py` converts a completed 2D layout into a mirror-symmetric 3D embedding.

BVGraph2D is topological and energy-blind by default. Optional transition-state-directed modes emphasize equilibrium exchange or rearrangements that are locally preferred in both directions.

## Contents and dependencies

- `BVGraph2D.py`: 2D layout generation and optimization
- `BVGraph3D.py`: 2D-to-3D conversion
- `README_BVGraph2D.txt` and `README_BVGraph3D.txt`: detailed guides
- `example/0000001123/`: established unweighted example
- `CITATION.cff` and `LICENSE`

Install the required libraries with:

```bash
pip install networkx numpy pandas
```

## Inputs and modes

Nodes require `id`. Edges require `source` and `target` (case-insensitive; `s` and `t` are aliases). Energetic modes use `Relative TS Energy (kJ/mol)`, containing Gibbs free energies on a common reference scale. Node energies may be retained as optional metadata.

`--edge-weighting none` is the default and requires no energies. `equilibrium-exchange` ranks direct edges by absolute TS free energy. `reversible-local` ranks each edge by its branching preference relative to other exits from both endpoints, moderated by equilibrium accessibility.

Duplicate or parallel rows become one layout edge using the lowest finite TS energy. Mirror-equivalent values are averaged for geometry while raw values remain in exports. Missing energetic data stay missing, receive floor-strength attraction, and can be excluded from the weighted edge objective.

The default graphical transformation is `r_e = 0.02 + 0.98 q_e^2`, where `q_e` is the tied-midrank favourability percentile. Weights are normalized to mean one. They are visual salience values, not literal rate constants. A Boltzmann transformation remains available for equilibrium-exchange calculations.

## Commands

```bash
python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out layout.gexf
```

```bash
python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out equilibrium.gexf \
  --edge-weighting equilibrium-exchange --edge-objective-scope weighted-data \
  --objective-mode initial_normalized --edge-length-power 2
```

```bash
python BVGraph3D.py --input layout.gexf --output layout_3d.gexf --nodes nodes.csv
```

Direct weights affect both the initial NetworkX spring embedding and subsequent weighted edge-length terms. Adaptive hemisphere optimization, powered lengths, progressive mirror-preserving separation, optional central-isomer placement, objective-gated span handling, atomic checkpoints, deterministic resume, convergence patience and clean wall-clock limits are available. Final export requires exact mirror symmetry and zero hard minimum-separation violations.

Outputs include GEXF and optional XGMML, node-coordinate CSV, edge-metadata CSV, annotations, side assignments, metrics JSON and checkpoints. BVGraph3D accepts weighted and unweighted BVGraph2D outputs without changing its method.

Use `CITATION.cff` when citing the software. BVGraph is distributed under the MIT licence in `LICENSE`.
