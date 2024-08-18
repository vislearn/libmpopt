## About

This repository contains code for optimizing combinatorial optimization problems by message passing techniques.
The solvers work on the Lagrange decomposition of the specific optimization problem and monotonously improve the bound of the Lagrange dual function.
A solution of the original primal problem is computed by rounding strategies.

Supported combinatorial optimization problems:

- Graphical Models (by reimplementing the TRW-S technique, see references)
- Cell Tracking Problems
- Quadratic Assignment Problems


## References

- S. Haller, B. Savchynskyy.<br>
  **“A Bregman-Sinkhorn Algorithm for the Maximum Weight Independent Set Problem”**.<br>
  arXiv Pre-Print 2024. [[PDF][arxiv2024]] [[Website & Paper Specific Code][arxiv2024_website]]

- S. Haller, L. Feineis, L. Hutschenreiter, F. Bernard, C. Rother, D. Kainmüller, P. Swoboda, B. Savchynskyy.<br>
  **“A Comparative Study of Graph Matching Algorithms in Computer Vision”**.<br>
  ECCV 2022 [[PDF][eccv2022]] [[Website][eccv2022_website]]

- L. Hutschenreiter, S. Haller, L. Feineis, C. Rother, D. Kainmüller, B. Savchynskyy.<br>
  **“Fusion Moves for Graph Matching”**.<br>
  ICCV 2021. [[PDF][iccv2021]]

- S. Haller, M. Prakash, L. Hutschenreiter, T. Pietzsch, C. Rother, F. Jug, P. Swoboda, B. Savchynskyy.<br>
  **“A Primal-Dual Solver for Large-Scale Tracking-by-Assignment”**.<br>
  *AISTATS 2020*. [[pdf][aistats2020]]

- V. Kolmogorov.<br>
  **“Convergent tree-reweighted message passing for energy minimization”**.<br>
  *PAMI 2006*. [[pdf][pami2006]]

[pami2006]: https://pub.ist.ac.at/~vnk/papers/trw_maxproduct_tr2.pdf
[aistats2020]: https://arxiv.org/pdf/2004.06375.pdf
[iccv2021]: https://arxiv.org/pdf/2101.12085.pdf
[eccv2022]: https://arxiv.org/pdf/2207.00291.pdf
[eccv2022_website]: https://vislearn.github.io/gmbench/
[arxiv2024]: https://arxiv.org/pdf/2408.02086
[arxiv2024_website]: https://vislearn.github.io/libmpopt/mwis2024/
