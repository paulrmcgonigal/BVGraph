BVGraph3D reference
===================

Purpose
-------
BVGraph3D.py converts a completed BVGraph2D layout into a mirror-preserving
three-dimensional embedding suitable for visualization and publication. It
accepts GEXF or XGMML layouts produced with either uniform or energy-aware
BVGraph2D settings; the 3D method itself does not apply transition-state-energy
weighting.

The script reads the two-dimensional coordinates and node annotations, detects
enantiomeric pairs and achiral nodes, initializes a three-dimensional geometry,
and relaxes representative coordinates with spring, short-range repulsive, and
radial container forces. Exact mirror symmetry is reimposed throughout the
relaxation.

Dependencies
------------
Required:
    pip install networkx numpy

Recommended for larger graphs:
    pip install scipy numba

Inputs
------
- --input, -i <path>: GEXF or XGMML with existing 2D coordinates. Coordinates
  may be x/y attributes, viz:position, or a supported pos string.
- --output, -o <path>: required 3D GEXF output.
- --nodes <path>: optional node CSV used to reinforce Chirality and
  Enantiomer_Id annotations before pair detection.

Typical use
-----------

    python BVGraph3D.py ^
      --input layout_2d.gexf ^
      --output layout_3d.gexf ^
      --nodes nodes.csv ^
      --verbose

Key controls
------------
- --iters <int> (default 3000): relaxation iterations.
- --swap-trials <int> (default 400): swap attempts during relaxation.
- --min-dist <float> (default 0.6): repulsion distance; the default is scaled
  relative to the inferred 2D edge length.
- --k-spring, --k-repulse, --k-radial: spring, short-range repulsion, and
  radial-container strengths.
- --lr <float>: learning rate.
- --radius-factor <float>: initial container-radius scaling.
- --seed <int>: random seed.
- --z-perturb <float>: initial out-of-plane perturbation.
- --k-radial-z-factor <float>: z weighting in the radial force.
- --achiral-lock-iters <int>: initial iterations with achiral nodes constrained
  to the mirror plane.
- --rest-scale <float>: spring rest length relative to the median 2D edge
  length.
- --shrink-step, --shrink-check-interval, --shrink-tol, and --min-r-factor:
  radial-container shrink controls.
- --verbose, -v: progress logging.

Output
------
The output GEXF contains x, y, and z node attributes and a three-dimensional
viz:position block for every node. It can be opened directly in software that
understands GEXF position metadata, including Gephi.

License
-------
Distributed under the MIT license in LICENSE.
