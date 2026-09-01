# BVGraph: mirror-preserving layouts for bullvalene isomer networks

BVGraph generates symmetry-aware two- and three-dimensional layouts for
bullvalene isomer networks.  It is designed for barcode-labelled networks that
contain achiral isomers and pairs of enantiomeric isomers.

`BVGraph2D.py` creates a two-dimensional layout with exact left/right mirror
symmetry.  `BVGraph3D.py` converts a completed two-dimensional layout into a
mirror-preserving three-dimensional embedding.

## Contents

- `BVGraph2D.py` — two-dimensional layout generator.
- `BVGraph3D.py` — three-dimensional embedding generator.
- `README_BVGraph2D.txt` — detailed two-dimensional input, method, and option
  reference.
- `README_BVGraph3D.txt` — detailed three-dimensional usage reference.
- `example/0000001123/` — example input and output files.
- `CITATION.cff` and `LICENSE` — citation metadata and MIT license.

## Installation

Python 3.8 or later is required.  Install the packages used by the 2D workflow:

```bash
pip install networkx numpy pandas
```

SciPy is required for the NetworkX spring-layout calculation on larger graphs.
SciPy and Numba are optional accelerators for larger 3D calculations:

```bash
pip install scipy numba
```

## Input files

`BVGraph2D.py` reads a node CSV and an edge CSV.

- The node table requires `id` and may contain an `Energy` column plus existing
  barcode or enantiomer annotations.
- The edge table requires `Source` and `Target` columns, matched
  case-insensitively.  `s` and `t` are accepted aliases.

Barcode annotations, chirality, and enantiomer partners are derived from the
node IDs before the layout is constructed.  Any matching annotation columns in
the node table are retained as graph attributes.

## Standard 2D and 3D workflow

Create a 2D layout:

```bash
python BVGraph2D.py \
  --nodes example/0000001123/nodes.csv \
  --edges example/0000001123/edges.csv \
  --out example/0000001123/my_layout.gexf \
  --out-xgmml example/0000001123/my_layout.xgmml \
  --out-nodes-csv example/0000001123/nodes_coords.csv \
  --out-edges-csv example/0000001123/edges_out.csv \
  --verbose
```

Create a 3D embedding from that layout:

```bash
python BVGraph3D.py \
  --input example/0000001123/my_layout.gexf \
  --output example/0000001123/my_layout_3d.gexf \
  --nodes example/0000001123/nodes.csv \
  --verbose
```

## Energy-aware 2D layouts

When constructing a new layout (rather than resuming coordinates), BVGraph2D
uses a topological NetworkX spring layout and treats all non-self-loop edges
uniformly in its geometric edge-length term by default. If transition-state
energies are available, Boltzmann weighting can make lower-barrier connections
exert stronger relative attraction.

For `--edge-weighting boltzmann`, provide the following default columns:

- node CSV: `Relative Energy (kJ/mol)`
- edge CSV: `Relative TS Energy (kJ/mol)`

Both columns must be in kJ mol⁻¹ and share the globally lowest-energy
ground-state node as their zero.  For an edge between `u` and `v`, BVGraph2D
uses the activation barrier:

```text
B = E_TS - min(E_u, E_v)
```

Run an energy-aware layout with:

```bash
python BVGraph2D.py \
  --nodes nodes.csv \
  --edges edges.csv \
  --out weighted_layout.gexf \
  --out-edges-csv weighted_edges.csv \
  --edge-weighting boltzmann \
  --temperature-k 298.15 \
  --spring-weight-floor-ratio 0.05 \
  --verbose
```

Without `--edge-weighting boltzmann`, energy columns are retained as metadata
when present but do not guide the initial spring layout or layout objective.
The detailed 2D guide describes validation, missing-energy handling, objective
weights, progressive separation, checkpointing, and all exports.

## Outputs

BVGraph2D writes a GEXF layout with `viz:position` coordinates.  Optional
outputs include Cytoscape-compatible XGMML, node-coordinate CSV, edge-metadata
CSV, barcode annotations, side assignments, checkpoint GEXF files, restart
JSON, and a best-layout checkpoint.

In Boltzmann mode, exported edges include the selected relative TS energy,
activation barrier, energy-data status, normalized layout spring weight, and
number of merged input edge rows.  BVGraph3D writes a GEXF containing node
`x`, `y`, and `z` attributes and three-dimensional `viz:position` coordinates.

## Citation and license

Please cite the associated publication and/or use `CITATION.cff` when
referencing this software.  BVGraph is distributed under the MIT license in
`LICENSE`.
