# Code for “Fusion Moves for Graph Matching” ICCV 2021 Paper

This branch is dedicated to our ICCV 2021 publication “Fusion Moves for Graph Matching”.
We try our best to make the results reproducible and keep them available in the future.
Therefore we keep this historic `iccv2021` branch.
For more information also visit the [project page][website].

## Citation

L. Hutschenreiter, S. Haller, L. Feineis, C. Rother, D. Kainmüller, B. Savchynskyy.
**“Fusion Moves for Graph Matching”**.
*Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2021*. [[pdf][arxiv2021]]

BibTeX entry:

```
@inproceedings{hutschenreiter2021fusion,
  title={Fusion Moves for Graph Matching},
  author={Hutschenreiter, Lisa and Haller, Stefan and Feineis, Lorenz and Rother, Carsten and Kainm{\"u}ller, Dagmar and Savchynskyy, Bogdan},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Building and running the provided container

To run the application inside a development container via Docker:

```
docker build -t qapopt .
docker run --rm -ti qapopt /bin/bash -l
```

To run the application inside a development container via Podman:

```
podman build -t qapopt .
podman run --rm -ti qapopt /bin/bash -l
```

Inside the development environment all tools are locally installed and they can
be simply executed from the interative shell session.

## Recreating the results

The container image is pre-populated with the `worms` dataset.
To run bca-greedy and afterwards fuse the solution proposals with qpbo-i run:

```
qap_dd_greedy_gen --verbose --max-batches 1000 --batch-size 1 --generate 1 worms/worm01-16-03-11-1745.dd proposals.txt
qap_dd_fusion --solver qpbo-i --output fused.txt worms/worm01-16-03-11-1745.dd proposals.txt
```

The above parameters are comparable to the ones that we have used for preparing
the submission. The parameters will generate one solution proposal after every
dual iteration. So in total there will be generated 1000 solution proposals. As
discussed in the manuscript, usually a much smaller number of proposals is
sufficient to obtain high-quality solutions.

## License

```
Copyright (c) 2018-2020 Stefan Haller
Copyright (c) 2021 Stefan Haller, Lisa Hutschenreiter

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

[website]: https://vislearn.github.io/libmpopt/iccv2021
[arxiv2021]: https://arxiv.org/pdf/2101.12085
