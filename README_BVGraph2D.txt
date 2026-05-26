BVGraph2D README
================

Purpose
-------
BVGraph2D.py generates a mirror-preserving 2D layout for bullvalene isomer
networks. The script is designed for systems with chiral node pairs and achiral
nodes, and it enforces the following layout conventions:

- enantiomeric pairs are placed as left/right mirrors with matched y positions
- achiral nodes are placed on the central vertical axis
- connected achiral nodes are organized as ordered axis blocks
- local refinement steps reduce crossings and improve spacing without breaking
  mirror symmetry

Primary outputs
---------------
- required GEXF with viz:position coordinates
- optional XGMML for Cytoscape
- optional node-coordinate CSV
- optional edge CSV
- optional annotated node CSV
- optional side-assignment CSV
- optional checkpoint GEXF files during refinement

Dependencies
------------
Required:
    pip install networkx numpy pandas

Inputs
------
Nodes CSV:
- required column: id
- optional column: Energy
- optional annotation columns if already available: Chirality, Enantiomer_Id,
  Barcode_Normalized, Enantiomer_Suggested

Edges CSV:
- required columns: source and target (case-insensitive)
- alternative short names s and t are also accepted

The script runs its own annotation pass from the node IDs, then attaches the
resulting Barcode_Normalized, Chirality, Enantiomer_Suggested, and
Enantiomer_Id fields to the graph before pair detection.

Basic usage
-----------
Minimal run:

    python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out layout.gexf

Typical run with optional exports:

    python BVGraph2D.py ^
      --nodes nodes.csv ^
      --edges edges.csv ^
      --out layout.gexf ^
      --out-xgmml layout.xgmml ^
      --out-nodes-csv nodes_coords.csv ^
      --out-edges-csv edges_out.csv ^
      --annot-out annotated_nodes.csv ^
      --dump-sides sides.csv ^
      --verbose

Current command-line options
----------------------------
Required:
- --nodes <path>: input nodes CSV
- --edges <path>: input edges CSV
- --out <path>: output GEXF path

Optional exports and bookkeeping:
- --annot-out <path>: write the internally annotated node table
- --dump-sides <path>: write side assignments for left/right/axis placement
- --out-xgmml <path>: write Cytoscape-compatible XGMML
- --out-nodes-csv <path>: write node coordinates and core attributes
- --out-edges-csv <path>: write edge table
- --checkpoint-gephi <path>: periodically overwrite a checkpoint GEXF
- --checkpoint-every <int>: checkpoint frequency in refinement cycles
- --resume-gephi <path>: resume from an earlier checkpoint GEXF
- --verbose: enable progress logging

Core layout controls:
- --sweeps <int> (default 4): coarse representative-layout sweeps
- --seed <int> (default 42): random seed
- --snap-achiral <int> (default 1): snap achiral nodes to the axis
- --min-sep <float> (default 60.0): minimum node separation target
- --edge-clearance <float> (default 12.0): clearance from unrelated edges
- --overlap-iter <int> (default 200): final overlap-relaxation iterations
- --achiral-gap <float>: axis spacing between adjacent achiral nodes; defaults
  to min-sep
- --swap-iters <int> (default 1000): achiral slot-swap attempts
- --axis-min-gap <float>: minimum gap between achiral axis blocks
- --axis-adjust-iters <int> (default 6): axis block adjustment iterations
- --pair-relocation-trials <int> (default 200): pair-block relocation attempts
- --axis-lateral-gap <float>: minimum non-axis distance from the axis
- --chiral-swap-iters <int> (default 500): chiral pair y-swap attempts
- --pairpair-iters <int> (default 100): pair-of-pairs swap attempts
- --refine-cycles <int> (default 1000): outer refinement cycles
- --achiral-adjacency-weight <float> (default 50.0): penalty for breaking
  adjacency of connected achiral nodes
- --achiral-gap-weight <float> (default 1.0): weight for gap uniformity
- --hemi-span-tol <float> (default 0.10): tolerance in span equalization
- --hemi-max-scale <float> (default 1.5): maximum scale change per equalization
- --hemi-min-nodes <int> (default 2): minimum chiral-node count to equalize
- --hemi-target-ratio <float> (default 1.0): target axis/chiral span ratio
- --enantiomer-swap-iters <int> (default 100): lateral x-swap attempts

Crossing evaluation and speed controls:
- --crossing-mode {exact,estimate,cycle_end,step_interval}: crossing-count
  strategy
- --crossing-exact-every <int> (default 5): exact-count interval when using
  step_interval mode
- --fast-mode: backward-compatible alias for estimate mode
- --fast-sample-size <int> (default 4000): sampled edge-pair count for fast
  crossing estimates

Optional post-refinement adjustments:
- --chiral-relax: enable mirror-preserving relaxation of chiral coordinates
- --chiral-relax-iters <int> (default 150)
- --chiral-relax-x-rate <float> (default 0.35)
- --chiral-relax-y-rate <float> (default 0.35)
- --chiral-relax-spacing-weight <float> (default 0.02)
- --chiral-relax-edge-length-weight <float> (default 0.05)
- --chiral-relax-repulsion-weight <float> (default 0.05)
- --chiral-relax-edge-target-factor <float> (default 1.0)
- --chiral-relax-min-x <float> (default 5.0)
- --cycle-shift-tol <float> (default 0.25): early-stop tolerance for completed
  refinement cycles
- --axis-node-relax: nudge axis nodes away from nearby edges before export
- --axis-node-relax-id <ids>: comma-separated subset of axis nodes to relax
- --axis-node-relax-clearance <float> (default 25.0)
- --axis-node-relax-shift <float> (default 200.0)
- --axis-node-relax-step <float> (default 5.0)

Algorithm summary
-----------------
1. Read the node and edge tables into a simple undirected NetworkX graph.
2. Annotate nodes from the barcode scheme and detect mutual enantiomer pairs.
3. Build a representative graph for paired nodes and partition it across left
   and right hemispheres.
4. Construct an initial coarse layout.
5. Arrange achiral components as ordered blocks on the central axis.
6. Repeatedly refine the layout using mirror-preserving local moves.
7. Optionally run chiral and axis-node relaxation steps.
8. Export the final coordinates in one or more formats.

Outputs in more detail
----------------------
GEXF:
- includes node attributes Energy, DeltaE, Chirality, Enantiomer_Id
- includes edge attribute TS_Energy
- stores 2D coordinates in viz:position with z = 0

XGMML:
- includes Energy, DeltaE, Chirality, Enantiomer_Id, Barcode_Normalized
- stores x and y coordinates in graphics elements

Node-coordinate CSV:
- id, x, y, Barcode_Normalized, Chirality, Energy, DeltaE, Enantiomer_Id

Edge CSV:
- Source, Target, TS_Energy

Notes
-----
- The current CSV-loading path does not use edge energies directly in the 2D
  layout objective.
- Some older shell examples in this repository reference legacy script names;
  use BVGraph2D.py for new runs.
- For quick testing on large graphs, try fewer refine cycles or use estimate
  crossing mode.

Authorship and acknowledgements
-------------------------------
This code was developed by Paul McGonigal. The source file also notes that code
development, refactoring, and documentation were carried out with assistance
from OpenAI ChatGPT (GPT-5).

License
-------
Distributed under the license provided in LICENSE.txt.
