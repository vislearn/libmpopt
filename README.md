# libmpopt – Library for Messaging Passing Optimization Techniques

This repository contains code for optimizing combinatorial optimization
problems by message passing techniques. The solvers work on the Lagrange
decomposition of the specific optimization problem and monotonously improve the
bound of the Lagrange dual function. A solution of the original primal problem
is computed by rounding strategies.

Supported combinatorial optimization problems:

  - Quadratic Assignment Problems
  - (REMOVED) Cell Tracking Problems
  - (REMOVED) Graphical Models (by reimplementing the TRW-S technique, see references)

## References

L. Hutschenreiter, S. Haller, L. Feineis, C. Rother, D. Kainmüller, B. Savchynskyy.\
**“Fusion Moves for Graph Matching”**.\
*arXiv preprint arXiv:2101.12085*. [[pdf][arxiv2021]]

[arxiv2021]: https://arxiv.org/pdf/2101.12085

# MODIFIED TO USE AS LIBRARY
Not useable as standalone project anymore. Modified to be included into multi-graph matching solver.

This port is intended to be used as a subproject in a meson. Builds as a static library.

Build within parent project by linking the dependency via

    libmpopt_proj  = subproject('libmpopt')
    libmpopt_dep   = libmpopt_proj.get_variable('libmpopt_dep')

See Meson documentation on subprojects for details.
(https://mesonbuild.com/Subprojects.html)

# Dependencies
The project has a dependency to ``libqpbo`` (Quadratic Pseudo Boolean Optimization (QPBO) Library).
Download from: https://gitlab.com/sebastianstricker/libqpbo

In order for the ``libmpopt`` to work as intended, place ``libqpbo`` as another subproject next to it.

    Parent Project
    ├── meson.build                   # Include dependency from libmpopt 
    ├── ...
    └── subprojects             
        ├── libqpbo
        │   ├── meson.build           # Defines libqpbo dependency
        │   └── ...              
        └── libmpopt  
            ├── meson.build           # Includes dependency from libqpbo
            └── ...              