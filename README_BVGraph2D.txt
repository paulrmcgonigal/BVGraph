BVGraph2D reference
===================

Purpose
-------
BVGraph2D.py creates a mirror-preserving two-dimensional layout for a
bullvalene isomer network. Enantiomeric nodes are placed as left/right mirrors
with the same y coordinate. Achiral nodes are arranged on the central axis.

The layout combines a representative-node spring layout with symmetry-
preserving local moves. It can optimize crossings, geometric edge lengths, and
soft node spacing while maintaining a hard minimum separation for final
exports.

Dependencies
------------
Required:
    pip install networkx numpy pandas

Recommended for large sparse 2D networks:
    pip install scipy

Inputs
------
Nodes CSV:
- required column: id
- optional column: Energy
- optional existing annotations: Chirality, Enantiomer_Id, Barcode_Normalized,
  Enantiomer_Suggested

Edges CSV:
- required columns: Source and Target (case-insensitive)
- accepted aliases: s and t

The script annotates node barcodes internally before identifying achiral nodes
and enantiomeric pairs. Input annotation columns are retained where present.

Uniform and energy-aware edge treatment
---------------------------------------
When no `--resume-gephi` layout is supplied, BVGraph2D begins with a topological
NetworkX spring layout of the representative graph. In the default
`--edge-weighting none` mode, all non-self-loop graph edges have equal spring
weight and the edge-length objective is an unweighted sum of geometric edge
lengths. Energy columns may be exported, but do not influence the layout in
this mode.

`--edge-weighting boltzmann` enables relative activation-barrier weighting.
The default energy columns are:

- nodes: Relative Energy (kJ/mol)
- edges: Relative TS Energy (kJ/mol)

Both values must be relative to the globally lowest-energy ground-state node
and expressed in kJ mol⁻¹. Column names are matched case-insensitively and may
be changed with `--node-energy-column` and `--ts-energy-column`.

Energy-specific controls are:

- --edge-weighting {none,boltzmann} (default none)
- --node-energy-column <name> (default `Relative Energy (kJ/mol)`)
- --ts-energy-column <name> (default `Relative TS Energy (kJ/mol)`)
- --temperature-k <float> (default 298.15)
- --spring-weight-floor-ratio <float> (default 0.05; must be greater than 0
  and no greater than 1)
- --missing-node-energy-policy {error,floor} (default error)

For an undirected edge joining u and v:

    B = E_TS - min(E_u, E_v)

Duplicate or parallel rows joining the same nodes are merged using the lowest
available relative TS energy. Negative barriers, malformed non-empty energy
values, missing required node energies, and unknown node references stop the
run with row-specific diagnostics. Self-loop metadata is retained but a
self-loop is excluded from geometric attraction.

For finite barriers, the raw relative spring factor is:

    r = f + (1 - f) exp[-(B - B_min) / (R T)]

where f is `--spring-weight-floor-ratio`, R is the gas constant, and T is
`--temperature-k`. Missing TS energies receive r = f. The non-self-loop
weights are normalized to mean 1, so the weighting changes relative attraction
rather than assigning an absolute target edge length.

`--missing-node-energy-policy error` is the default. Use `floor` only when
blank node-energy records are intentionally unavailable: such nodes remain in
the network, their incident edges receive the minimum raw spring factor, and no
activation barrier is calculated. `--edge-objective-scope finite_energy` limits
the edge-length objective to edges with calculated barriers; all edges remain
in the graph and initial geometry.

Basic use
---------
Minimal layout:

    python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out layout.gexf

Layout with common exports:

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

Energy-aware layout:

    python BVGraph2D.py ^
      --nodes nodes.csv ^
      --edges edges.csv ^
      --out weighted_layout.gexf ^
      --out-edges-csv weighted_edges.csv ^
      --edge-weighting boltzmann ^
      --temperature-k 298.15 ^
      --spring-weight-floor-ratio 0.05 ^
      --verbose

Outputs and restart files
-------------------------
- --out <path>: required GEXF output with 2D viz:position coordinates.
- --out-xgmml <path>: optional Cytoscape XGMML output.
- --out-nodes-csv <path>: node IDs, coordinates, barcode/chirality fields,
  energy metadata, and enantiomer IDs.
- --out-edges-csv <path>: Source, Target, TS energy, relative TS energy,
  activation barrier, energy-data status, layout spring weight, and input-row
  count.
- --annot-out <path>: internally annotated node table.
- --dump-sides <path>: left, right, or axis assignment for each node.
- --out-metrics-json <path>: final objective references and layout metrics.
- --checkpoint-gephi <path> and --checkpoint-every <int>: periodic GEXF
  checkpoints.
- --checkpoint-state-json <path>: atomic JSON restart metadata.
- --best-checkpoint-gephi <path>: best valid layout reached in the run.
- --resume-gephi <path>, --resume-state-json <path>, and --cycle-offset <int>:
  resume coordinates and corresponding global-cycle metadata.

Checkpoint replacement is atomic and retries temporary Windows file locks. If a
lock persists, a uniquely named recovery GEXF is written and the next checkpoint
attempt returns to the requested primary path.

Objective and crossing controls
-------------------------------
`--objective-mode legacy` retains raw scoring coefficients. In
`initial_normalized` mode, the crossing, edge-length, and soft-spacing terms
are divided by their initial references; fixed references can be supplied for a
continued run with:

- --objective-reference-crossings
- --objective-reference-weighted-length
- --objective-reference-spacing-penalty

The corresponding requested coefficients are:

- --objective-crossing-weight (default 0.20)
- --objective-edge-length-weight (default 1.00)
- --objective-spacing-weight (default 8.00)
- --soft-separation-factor (default 1.35 times --min-sep)

Crossing evaluation is selected with `--crossing-mode`:

- exact: exact count for every evaluation
- estimate: sampled estimate for every evaluation
- cycle_end: sampled local evaluations and exact count at cycle end
- step_interval: sampled evaluations with exact counts every
  `--crossing-exact-every` optimization steps

`--fast-mode` is an alias for estimate mode. `--fast-sample-size` controls the
number of sampled edge pairs (default 4000).

Separation, symmetry, and refinement
------------------------------------
`--min-sep` (default 60) is the hard final node-distance requirement.
`--separation-mode legacy_final` applies it only during final processing.
`progressive` raises the active threshold through `--separation-levels`
(default 0.33,0.50,0.67,0.83,0.90,0.95,1.00 fractions of --min-sep).

The progressive projector uses a spatial grid rather than all-pairs scans. It
repairs candidate layouts while preserving mirror symmetry, performs periodic
global projection every `--separation-project-every` cycles, and stops an
unproductive projection after `--separation-stall-passes`. Final export requires
zero violations at the requested full separation.

`--edge-clearance` controls clearance from unrelated edges. Leave
`--overlap-iter` at its scalable default of 0; use
`--final-separation-iters` for mirror-preserving spatial-grid final repair.

The following switches expose the local search and safeguards:

- --sweeps, --layout-scale-factor, and --seed configure initial layout.
- --refine-cycles, --swap-iters, --chiral-swap-iters, --pairpair-iters,
  --pair-flip-iters, --pair-relocation-trials, and --enantiomer-swap-iters
  control search effort.
- --achiral-gap, --axis-min-gap, --axis-adjust-iters, --axis-lateral-gap,
  --achiral-adjacency-weight, --achiral-gap-weight, and the --hemi-* options
  control axis blocks and relative axis/chiral span.
- --chiral-relax and its --chiral-relax-* controls enable a
  mirror-preserving coordinate relaxation.
- --axis-node-relax and its controls nudge selected axis nodes away from
  nearby edges before export.
- --convergence-patience sets the number of consecutive stalled valid cycles
  required for early stopping; --run-all-refine-cycles overrides early stopping.
- --transactional-valid-cycles retains a full-separation cycle only when its
  repaired final layout is valid and improves the cycle-start objective.
- --stage-diagnostics logs retained movement, objective, and separation data
  after each move class.

Exports
-------
GEXF stores node attributes and two-dimensional viz:position coordinates.
XGMML stores equivalent node and edge metadata with graphics coordinates. In
Boltzmann mode, edge attributes include `Relative_TS_Energy`,
`Activation_Barrier`, `Energy_Data_Status`, `Layout_Spring_Weight`, and
`Input_Edge_Rows`. The node-coordinate CSV includes `Relative Energy (kJ/mol)`
and energy-data status where available.

BVGraph3D.py accepts either an unweighted or energy-aware 2D GEXF/XGMML as its
starting layout; it does not apply the 2D energy-weighting model itself.
