BVGraph2D reference guide
=========================

Purpose
-------
BVGraph2D.py generates a mirror-preserving 2D layout for bullvalene isomer
networks. Enantiomeric pairs have opposite x coordinates and identical y
coordinates. Achiral nodes lie on x = 0. The default operation is entirely
unweighted: every topological edge has the same spring strength and every edge
enters the geometric edge-length score equally.

Dependencies
------------
Required:
    pip install networkx numpy pandas

Inputs and barcode annotation
-----------------------------
Nodes CSV:
- required: id
- optional metadata: Energy, Relative Energy (kJ/mol)
- optional pre-existing annotation: Chirality, Enantiomer_Id,
  Barcode_Normalized, Enantiomer_Suggested

Edges CSV:
- required: source and target (case-insensitive)
- s and t are accepted aliases
- optional: Relative TS Energy (kJ/mol)

BVGraph canonicalises the three arms encoded by each node barcode, classifies
the node as chiral or achiral, and finds a mutual enantiomer identifier when it
is present. Unknown edge endpoints stop a weighted run with a row-specific
diagnostic. A simple undirected graph is constructed. Duplicate or parallel
input rows are represented by one layout edge, while their row count and
selected metadata are retained. Self-loops retain metadata but exert no
geometric attraction.

Minimal usage
-------------
    python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out layout.gexf

Typical exports:
    python BVGraph2D.py ^
      --nodes nodes.csv ^
      --edges edges.csv ^
      --out layout.gexf ^
      --out-xgmml layout.xgmml ^
      --out-nodes-csv nodes_coordinates.csv ^
      --out-edges-csv edge_metadata.csv ^
      --out-metrics-json metrics.json

Initial layout and uniform edge treatment
-----------------------------------------
The default --edge-weighting none assigns each non-self-loop edge a spring
weight of one. A balanced NetworkX spring layout positions one representative
of every enantiomeric pair. The full network is then reconstructed by exact
mirroring, and connected achiral components are arranged along the central
axis. Subsequent local and global edge-length terms are also unweighted.

Equilibrium-exchange weighting
------------------------------
Select:
    --edge-weighting equilibrium-exchange

The default edge input heading is Relative TS Energy (kJ/mol), overridable by
--ts-energy-column. It contains transition-state Gibbs free energies on one
common reference scale. Node energies are not required for this mode and do
not enter the weighting.

For duplicate or parallel input records, the lowest finite E_TS is used. Let
E_TS,min be the lowest selected finite transition-state energy in the network.
The raw edge attraction is:

    r_e = f + (1-f) exp[-(E_TS,e - E_TS,min)/(R T)]

where f is --spring-weight-floor-ratio (default 0.05), T is
--temperature-k (default 298.15 K), and R is in kJ mol-1 K-1. Missing TS
energies receive r_e = f and retain status missing_ts_energy.

This mode emphasizes connections that contribute strongly to exchange at
thermal equilibrium. It uses absolute transition-state free energy, not an
activation barrier measured from either endpoint.

Transient-flux weighting
------------------------
Select:
    --edge-weighting transient-flux
    --transient-flux-file transient_flux_edges.csv

The flux CSV contains one undirected edge per row:

    source
    target
    Integrated Absolute Net Flux
    Integrated Gross Flux
    Integrated Signed Net Flux
    Flux Data Status

Source and target order is immaterial. Duplicate undirected rows, negative
absolute/gross values, malformed values, unknown nodes, and edges absent from
the topology are rejected with row-specific diagnostics.

Choose --transient-flux-metric absolute-net (default) to emphasize accumulated
population redistribution, or gross to emphasize all forward-plus-reverse
reaction traffic. --transient-flux-column overrides the selected value heading
when a non-standard file is unavoidable. The raw attraction is:

    r_e = f + (1-f) F_e/F_max

The selected integrated flux is used directly; no additional exponential is
applied. A finite zero is valid data and receives the floor. Blank or explicitly
unavailable data also receive the floor, but remain distinguishable by their
status and availability attributes.

Spring normalization and use
----------------------------
In either weighted mode, all non-self-loop raw spring weights, including floor
weights, are divided by their mean. This retains the average attraction scale
while changing relative spring strengths. Layout_Spring_Weight affects:

1. the initial representative NetworkX spring embedding; and
2. every subsequent local and global weighted edge-length term.

It does not prescribe an absolute edge length and does not turn off crossing,
spacing, clearance, axis, or mirror constraints.

Layout objective
----------------
The iterative score combines:

    crossing contribution + edge-length contribution + soft-spacing contribution
    + optional centrality contribution + optional cross-axis contribution

Use --objective-mode initial_normalized to divide crossings, weighted edge
length, and soft-spacing penalty by their respective values in the initial
layout. The requested coefficients then describe their nominal contributions
at the starting point. The soft-spacing coefficient is independent; the three
coefficients need not sum to one.

Relevant options:
- --objective-crossing-weight
- --objective-edge-length-weight
- --objective-spacing-weight
- --soft-separation-factor
- --objective-reference-crossings
- --objective-reference-weighted-length
- --objective-reference-spacing-penalty
- --objective-reference-center-penalty
- --edge-length-power
- --cross-axis-edge-weight

--edge-objective-scope all includes every non-self-loop edge in edge-length
scoring. --edge-objective-scope weighted-data includes only edges with a valid
mode-specific input value. Unavailable edges remain in the graph and retain
floor-strength attraction during the initial spring embedding.

--edge-length-power is 1.0 by default, giving the standard linear distance
score. Values above one increasingly penalise unusually long edges.
--cross-axis-edge-weight is zero by default. A positive value adds the weighted
fraction of chiral-chiral edges whose endpoints occupy opposite sides of the
axis. Both controls are independent of the scientific source of edge weights.

Crossing strategies
-------------------
--crossing-mode exact evaluates all eligible edge pairs.
--crossing-mode estimate uses --fast-sample-size random edge pairs.
--crossing-mode cycle_end uses sampled local decisions and exact cycle-end
counts. --crossing-mode step_interval additionally uses
--crossing-exact-every. --fast-mode remains an alias for estimate.

Sampled counts are reproducible for a fixed seed, cycle number, and command,
but are estimates rather than the exact number of crossings.

Mirror-preserving separation and clearance
-------------------------------------------
--min-sep sets the hard final node separation. In legacy_final mode, hard
separation is imposed after refinement. In progressive mode,
--separation-levels raises an active mirror-preserving threshold in stages.
Completed coordinate-changing proposals can be spatially projected to the
active threshold before being scored. Once a stage is feasible, accepted moves
may not violate it.

The spatial index groups nearby coordinates into grid cells so only nearby
cells are checked; it avoids an all-pairs comparison between every node and
every other node. Corrections move complete enantiomeric pairs symmetrically,
and achiral nodes remain on the axis. Final strict mirroring is followed by a
full mirror-preserving separation pass. Export stops with an error if any pair
remains closer than --min-sep. --edge-clearance controls clearance from
unrelated edges; --overlap-iter controls the optional all-pairs relaxation and
can remain zero for scalable runs.

Progressive options:
- --separation-mode {legacy_final,progressive}
- --separation-levels
- --separation-project-every
- --separation-project-iters
- --separation-stall-passes
- --final-separation-iters

The achiral axis and chiral cloud can have different vertical spans because
achiral nodes must share one line while chiral nodes can use two dimensions.
--hemi-target-ratio and --hemi-span-tol define the desired range.
--hemi-span-adjustment expand-smaller (default) proposes expanding the shorter
group; compress-larger proposes contracting the taller group instead. Every
such proposal made during refinement is repaired for active separation and is
accepted only when the complete objective improves. --hemi-max-scale limits
the size of each proposed change.

By default, --final-span-expansion on expands a shorter chiral span to the
axis target immediately before final separation. This final operation is not
objective-gated. Use --final-span-expansion off when the exported coordinates
must retain the objective-optimised spans exactly; the final mirror-preserving
hard-separation validation still runs.

Local refinement controls
-------------------------
The move classes preserve the chemical symmetry conventions. Their per-cycle
attempt counts are controlled independently by:

- --swap-iters (achiral-axis slots)
- --chiral-swap-iters
- --pairpair-iters
- --enantiomer-swap-iters
- --pair-relocation-trials
- --pair-flip-iters
- --chiral-relax and --chiral-relax-iters

--transactional-valid-cycles applies after full minimum separation is reached.
A complete cycle is repaired and rescored; it is retained only if the resulting
layout is valid and improves the cycle-start objective. --stage-diagnostics
prints movement, score, and separation measurements after each move class.
--convergence-patience is the number of consecutive eligible cycles without a
new best valid objective before clean early termination.

Hemisphere optimisation
-----------------------
--hemisphere-optimization controls the scale at which BVGraph assigns the two
members of each enantiomeric pair to the left and right sides:

- local (default) uses the standard individual-pair and pair-of-pairs
  refinement only.
- global adds a deterministic whole-graph orientation pass immediately after
  the representative layout is mirrored.
- adaptive also repeats the whole-graph pass during refinement and searches
  for connected blocks that should change sides coherently.

Adaptive candidate blocks are seeded by costly long, opposite-side, or strongly
coupled chiral-chiral connections. Connected candidates grow through 2, 4, 8,
16, 32, 64 and at most 128 enantiomeric pairs by default, stopping when further
growth weakens the predicted improvement. Promising candidates are evaluated
against the complete objective. An accepted block can undergo five damped,
mirror-preserving relaxation steps together with a one-neighbour halo, followed
by the active separation repair before final acceptance.

Relevant controls:
- --hemisphere-pass-every (default 5 cycles)
- --hemisphere-stall-trigger (default 3 stalled cycles)
- --hemisphere-block-max-pairs (default 128)
- --hemisphere-block-seeds (default 16)
- --hemisphere-block-relax-iters (default 5)
- --hemisphere-block-relax-halo (default 1)

Equilibrium and transient-flux layouts use their active scientific weights to
rank orientation and block proposals. Unweighted layouts give every ordinary
edge equal importance. If the continuous edge-length coefficient is zero,
uniform length is used only to select proposals and break ties; acceptance may
not worsen the crossing, spacing and centrality objective.

Central-isomer placement
------------------------
--center-isomer BARCODE, with spelling alias --centre-isomer, adds a soft
preference for a chosen isomer at the vertical midpoint of the full layout.
Leading zeroes are preserved and the barcode must match a node identifier
exactly. An achiral target is placed on the axis. For a chiral target, its
recorded reciprocal enantiomer is selected automatically and the pair shares
one target y coordinate; x remains determined by mirror symmetry.

The dimensionless offset is:

    q = 2 |y_target - (y_min + y_max)/2| / (y_max - y_min)
    P_center = q^2

--center-weight defaults to 0.10 when a target is present and is inactive
otherwise. In initial_normalized mode, the reference denominator is never
smaller than 0.01, avoiding unstable amplification when the initial target is
already close to the midpoint. Central placement is a preference, not a hard
constraint; final absolute and fractional offsets are reported.

Checkpoints, resumption, and runtime limits
------------------------------------------
--checkpoint-gephi writes atomic coordinate checkpoints at the cadence set by
--checkpoint-every. --checkpoint-state-json records completed global cycle,
seed, objective references, separation stage, best rank, hemisphere settings,
central target and accepted orientation/block statistics. A resume that
requests incompatible new settings is rejected explicitly.
--best-checkpoint-gephi retains the best state at the most advanced separation
stage. Every write uses a unique short temporary filename and Windows
extended-length filesystem syntax where needed. Missing temporary files are
recreated before retry, Windows file locks are retried, and a uniquely named
recovery GEXF is kept if the primary destination remains locked.

Resume coordinates with --resume-gephi and state with --resume-state-json.
--cycle-offset prevents global cycle numbering and deterministic random streams
from restarting at one. --max-runtime-hours is disabled at zero; otherwise the
active cycle is completed, checkpoints are written, and the run exits cleanly
once the limit has been reached. Final validation and exports still run.

Outputs
-------
Required:
- GEXF with viz:position coordinates

Optional:
- XGMML (--out-xgmml)
- node-coordinate CSV (--out-nodes-csv)
- edge-metadata CSV (--out-edges-csv)
- annotated nodes CSV (--annot-out)
- left/right/axis assignments (--dump-sides)
- metrics JSON (--out-metrics-json)
- checkpoint and best-layout GEXF/JSON files

Edge outputs include Relative_TS_Energy when supplied,
Layout_Weighting_Mode, Layout_Weight_Input_Metric,
Layout_Weight_Input_Value, Layout_Weight_Data_Available,
Weight_Data_Status, Integrated_Absolute_Net_Flux,
Integrated_Gross_Flux, Integrated_Signed_Net_Flux,
Layout_Spring_Weight, and Input_Edge_Rows.
Node outputs also identify Central_Layout_Target. Node and edge outputs record
Hemisphere_Optimization, and edges identify Cross_Axis_At_Final.

The metrics JSON records objective references and effective coefficients,
cycle-by-cycle crossings, weighted distance, soft-spacing penalty, span and
separation diagnostics, final validation, runtime-limit status, and the
Spearman association between the selected weighting input and final edge
length where enough data are present. It also records weighted and unweighted
opposite-side edge fractions, central offset, and orientation/block proposals
and acceptances.
