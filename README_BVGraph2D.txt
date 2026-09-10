BVGraph2D reference guide
=========================

BVGraph2D.py generates mirror-preserving 2D bullvalene-network layouts.
Enantiomeric nodes have opposite x coordinates and identical y coordinates;
achiral nodes lie on x = 0. The default mode is energy-blind.

Dependencies: pip install networkx numpy pandas

Inputs
------
Nodes require id. Edges require source and target (case-insensitive; s and t
are aliases). Energetic modes read Relative TS Energy (kJ/mol), overridable by
--ts-energy-column. Values are TS Gibbs free energies on a common reference
scale. Node energies are optional metadata in this release.

Parallel rows become one geometric edge using the lowest finite TS energy.
Mirror-equivalent edges use their available orbit-average value for geometry;
raw values stay unchanged in exports. Missing complete orbits receive the
spring floor and are excluded by --edge-objective-scope weighted-data.
Self-loops retain metadata but exert no attraction.

Weighting modes
---------------
none is the default and gives every ordinary edge unit attraction.

equilibrium-exchange orders edges by absolute TS free energy, equivalent to
ordering equilibrium exchange conductance C_e proportional to
exp[-G_TS,e/(RT)].

reversible-local calculates P(u->v) relative to other finite exits from u,
then uses sqrt[P(u->v) P(v->u)]. It multiplies this mutual preference by
g + (1-g)q_equilibrium, where g defaults to 0.5. The gate prevents a very
high-TS edge from ranking highly only because its alternatives are worse.

The default transformation is r_e = f + (1-f)q_e^gamma with tied midranks,
f = 0.02 and gamma = 2. Weights are normalized to mean one. These percentiles
are graphical salience values, not physical rates. The boltzmann transform is
available only for equilibrium-exchange.

Layout and optimization
-----------------------
Mode-specific spring weights guide the initial NetworkX embedding and every
subsequent weighted edge-length term. --objective-mode initial_normalized
normalizes crossings, weighted edge length and soft spacing to their initial
references. --edge-length-power 2 gives additional cost to isolated long
important edges.

--hemisphere-optimization accepts local, global and adaptive. Adaptive mode
adds global side assignment and connected orientation blocks. Proxies propose
and rank moves; the complete normalized objective and mirror-preserving
separation repair decide acceptance.

Progressive separation uses a spatial index rather than all node pairs.
--min-sep is a hard export requirement. --final-span-expansion off prevents an
unconditional final span change. --center-isomer optionally favours a selected
node or enantiomeric pair at the vertical midpoint.

Atomic GEXF/JSON checkpoints record compatible configuration and progress.
--max-runtime-hours exits after a complete cycle and writes final/checkpoint
outputs. Transactional cycles, stage diagnostics and best checkpoints support
long refinements.

Outputs
-------
GEXF can be supplemented by XGMML, node-coordinate CSV, edge-metadata CSV,
annotated nodes, side assignments and metrics JSON. Energy exports distinguish
raw and orbit-symmetrized TS values, conductance or local-preference evidence,
percentile salience, normalized spring weight and data status.

Examples
--------
    python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out layout.gexf

    python BVGraph2D.py --nodes nodes.csv --edges edges.csv --out equilibrium.gexf \
      --edge-weighting equilibrium-exchange --edge-objective-scope weighted-data \
      --objective-mode initial_normalized --edge-length-power 2
