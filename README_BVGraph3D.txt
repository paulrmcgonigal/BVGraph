BVGraph3D README
================

Purpose
-------
BVGraph3D.py converts a completed BVGraph 2D layout into a mirror-preserving 3D
embedding suitable for visualization and publication.

The input may be an unweighted or edge-weighted BVGraph2D layout. BVGraph3D
uses the supplied coordinates and retains graph attributes; its 3D method does
not reinterpret transition-state energies or flux weights.

The script:
- reads a GEXF or XGMML graph that already contains 2D coordinates
- detects enantiomeric pairs and achiral nodes from node annotations
- initializes a 3D geometry from the 2D layout
- optimizes representative coordinates with spring, repulsive, and radial
  container forces
- re-enforces exact mirror symmetry during the optimization
- writes the final 3D coordinates back to GEXF

Dependencies
------------
Required:
    pip install networkx numpy

Optional but recommended for larger graphs:
    pip install scipy numba

Inputs
------
Required input graph:
- GEXF or XGMML with 2D coordinates already present
- node coordinates may be stored as x/y attributes, in viz:position, or as a
  simple pos string understood by the parser

Optional nodes CSV:
- can be supplied with --nodes
- useful for attaching Chirality and Enantiomer_Id information before pair
  detection

Basic usage
-----------
Minimal run:

    python BVGraph3D.py --input layout.gexf --output layout_3d.gexf

Typical run:

    python BVGraph3D.py ^
      --input layout.gexf ^
      --output layout_3d.gexf ^
      --nodes nodes.csv ^
      --verbose

Current command-line options
----------------------------
- --input, -i <path>: input GEXF or XGMML with 2D coordinates
- --output, -o <path>: output GEXF path for 3D coordinates
- --nodes <path>: optional nodes CSV for annotation transfer
- --iters <int> (default 3000): optimization iterations
- --swap-trials <int> (default 400): swap attempts during relaxation
- --min-dist <float> (default 0.6): minimum repulsion distance; if left at the
  default value, it is rescaled internally relative to the inferred 2D edge
  length scale
- --k-spring <float> (default 0.8): spring force strength
- --k-repulse <float> (default 3.0): short-range repulsion strength
- --k-radial <float> (default 6.0): radial container strength
- --lr <float> (default 0.005): learning rate
- --radius-factor <float> (default 1.0): scaling applied to the initial
  container radius
- --seed <int> (default 42): random seed
- --z-perturb <float> (default 0.02): initial z perturbation
- --k-radial-z-factor <float> (default 1.0): z weighting in the radial force
- --achiral-lock-iters <int> (default 600): initial epochs during which
  achiral nodes are kept on the mirror plane
- --rest-scale <float> (default 0.75): sets the spring rest length as
  median_2d_edge_length * rest_scale
- --shrink-step <float> (default 0.995): factor used when shrinking the radial
  container
- --shrink-check-interval <int> (default 50): how often to test container
  shrinking
- --shrink-tol <float> (default 1e-6): tolerance for accepting a smaller
  container
- --min-r-factor <float> (default 0.2): lower bound on container shrinkage
- --verbose, -v: enable verbose logging

Algorithm summary
-----------------
1. Read the input graph and optionally attach node annotations from CSV.
2. Detect mutual enantiomer pairs and achiral nodes heuristically from node
   attributes.
3. Extract the existing 2D coordinates.
4. Build representative optimization objects for chiral pairs and achiral
   nodes.
5. Initialize achiral nodes on a circle in the mirror plane.
6. Estimate a characteristic edge length from the 2D layout and use it to set a
   constant spring rest length scale.
7. Run iterative relaxation with spring forces, short-range repulsion, swap
   moves, and a one-sided radial container.
8. Reapply exact symmetry and export the final 3D coordinates.

Output
------
The script writes a GEXF in which each node has:
- x, y, z node attributes
- a viz:position block containing x, y, z

This output can be opened directly in tools that understand GEXF position
metadata, such as Gephi.

Notes
-----
- BVGraph3D.py assumes the input graph already has a valid 2D layout.
- If node annotations are incomplete in the GEXF/XGMML file, provide the
  original nodes CSV with --nodes.
- Installing scipy enables cKDTree-based neighbor searches; installing numba
  enables an accelerated pair-force kernel.

Authorship and acknowledgements
-------------------------------
This code was developed by Paul McGonigal. The source file also notes that code
development, refactoring, and documentation were carried out with assistance
from OpenAI ChatGPT (GPT-5).

License
-------
Distributed under the license provided in LICENSE.txt.
