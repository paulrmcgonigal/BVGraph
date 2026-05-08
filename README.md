BVGraph: Mirror-Preserving Layouts for Bullvalene Isomer Networks
=================================================================

## Overview

BVGraph is a two-script workflow for generating symmetry-aware network layouts
for bullvalene isomer graphs.

- `BVGraph2D.py` builds a 2D mirror-preserving layout from node and edge CSV
  files.
- `BVGraph3D.py` converts a completed 2D layout into a 3D embedding while
  preserving enantiomeric symmetry.

The current project layout is centered on these two scripts. The older helper
shell scripts in this folder and in `example/` may still contain legacy script
names and should be treated as templates rather than authoritative entry
points.

## Repository contents

- `BVGraph2D.py`: 2D layout generator for bullvalene networks
- `BVGraph3D.py`: 3D embedding tool for BVGraph outputs
- `README_BVGraph2D.txt`: detailed usage notes for the 2D script
- `README_BVGraph3D.txt`: detailed usage notes for the 3D script
- `example/0000001123/`: bundled example input and output files
- `CITATION.cff`: citation metadata
- `LICENSE.txt`: license text

## Dependencies

Minimum:

```bash
pip install networkx numpy pandas
```

Recommended for `BVGraph3D.py` on larger graphs:

```bash
pip install scipy numba
```

## Input model

### `BVGraph2D.py`

Inputs are two CSV files:

- nodes CSV: must contain an `id` column; `Energy` is optional
- edges CSV: must contain `Source` and `Target` columns (case-insensitive)

`BVGraph2D.py` automatically annotates nodes from the barcode-style node IDs,
including normalized barcodes, chirality classification, and suggested
enantiomer partners. If the nodes table already includes matching annotation
columns, those are attached as node attributes as well.

### `BVGraph3D.py`

Input is a GEXF or XGMML graph that already contains 2D coordinates produced by
the 2D workflow. An optional nodes CSV can be supplied to reinforce chirality
and enantiomer annotations during pair detection.

## Typical workflow

### 1. Generate a 2D layout

```bash
python BVGraph2D.py \
  --nodes example/0000001123/nodes.csv \
  --edges example/0000001123/edges.csv \
  --out example/0000001123/my_layout.gexf \
  --out-xgmml example/0000001123/my_layout.xgmml \
  --out-nodes-csv example/0000001123/nodes_coords.csv \
  --annot-out example/0000001123/annotated_nodes.csv \
  --dump-sides example/0000001123/sides.csv \
  --verbose
```

### 2. Convert the 2D layout into 3D

```bash
python BVGraph3D.py \
  --input example/0000001123/my_layout.gexf \
  --output example/0000001123/my_layout_3d.gexf \
  --nodes example/0000001123/nodes.csv \
  --verbose
```

## Outputs

### `BVGraph2D.py`

Required output:

- GEXF with `viz:position` coordinates for Gephi

Optional outputs:

- XGMML with node graphics coordinates for Cytoscape
- nodes CSV with `id`, `x`, `y`, `Barcode_Normalized`, `Chirality`, `Energy`,
  `DeltaE`, and `Enantiomer_Id`
- edge CSV with `Source`, `Target`, and `TS_Energy`
- annotated nodes CSV from the internal barcode annotation step
- side assignment CSV describing `left`, `right`, or `axis` placement
- checkpoint GEXF files during refinement

### `BVGraph3D.py`

- GEXF with 3D `x`, `y`, `z` coordinates written both as node attributes and
  as `viz:position`

## Method summary

### BVGraph2D

The 2D script builds a simple undirected NetworkX graph, infers chiral pairs
and achiral nodes from the bullvalene barcode scheme, partitions paired
representatives across left and right hemispheres, and arranges achiral nodes on
the central axis. It then refines the layout using multiple symmetry-preserving
move classes, including pair swaps, achiral axis ordering adjustments,
enantiomer lateral swaps, optional chiral relaxation, and post-processing for
node spacing and edge clearance.

The current version supports several crossing-evaluation strategies via
`--crossing-mode`, including exact and sampled estimates for faster optimization
on larger graphs.

### BVGraph3D

The 3D script starts from a completed 2D layout, initializes achiral nodes on
the mirror plane, and optimizes representative coordinates using spring forces,
short-range repulsion, and a one-sided radial container. Exact mirror symmetry
is reimposed during relaxation so the final embedding remains chemically
consistent.

## Notes

- `BVGraph2D.py` is the current 2D entry point. Some older helper scripts in
  the repository still reference legacy filenames such as
  `BVGraph.py` or `BVGraph2D_fastmode_optimise.py`.
- The 2D script reads `TS_Energy`-style edge columns if present in the CSV, but
  the present CSV-loading path does not use edge energies in the layout itself.
- For reproducible runs, keep `--seed` fixed.

## Citation and license

Please cite the associated publication and/or use the metadata in
`CITATION.cff` when referencing this software. The project is distributed under
the license given in `LICENSE.txt`.
