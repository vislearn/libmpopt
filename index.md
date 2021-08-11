## About

This repository contains code for optimizing combinatorial optimization problems by message passing techniques.
The solvers work on the Lagrange decomposition of the specific optimization problem and monotonously improve the bound of the Lagrange dual function.
A solution of the original primal problem is computed by rounding strategies.

Supported combinatorial optimization problems:

- Graphical Models (by reimplementing the TRW-S technique, see references)
- Cell Tracking Problems
- Quadratic Assignment Problems


## References

L. Hutschenreiter, S. Haller, L. Feineis, C. Rother, D. Kainmüller, B. Savchynskyy.\
**“Fusion Moves for Graph Matching”**.\
*arXiv preprint arXiv:2101.12085*. [[pdf][arxiv2021]]

S. Haller, M. Prakash, L. Hutschenreiter, T. Pietzsch, C. Rother, F. Jug, P. Swoboda, B. Savchynskyy.\
**“A Primal-Dual Solver for Large-Scale Tracking-by-Assignment”**.\
*AISTATS 2020*. [[pdf][aistats2020]]

V. Kolmogorov.\
**“Convergent tree-reweighted message passing for energy minimization”**.\
*PAMI 2006*. [[pdf][pami2006]]

[arxiv2021]: https://arxiv.org/pdf/2101.12085
[aistats2020]: https://hci.iwr.uni-heidelberg.de/vislearn/HTML/people/stefan_haller/pdf/A%20Primal-Dual%20Solver%20for%20Large-Scale%20Tracking-by-Assignment%20-%20AISTATS2020.pdf
[pami2006]: https://pub.ist.ac.at/~vnk/papers/trw_maxproduct_tr2.pdf
