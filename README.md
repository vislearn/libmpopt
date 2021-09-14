# libmpopt – Library for Messaging Passing Optimization Techniques

This repository contains code for optimizing combinatorial optimization
problems by message passing techniques. The solvers work on the Lagrange
decomposition of the specific optimization problem and monotonously improve the
bound of the Lagrange dual function. A solution of the original primal problem
is computed by rounding strategies.

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


## Setting up a Development Environment

### Using a Development Container via Docker/Podman

The easiest way to get started is by using [Docker][docker] or
[podman][podman]. Usage of podman is recommended as it allows to create
root-less containers. The advantage is that with podman the local user is able
to create a container that contains the full development environment without
requiring root access.

[podman]: https://podman.io/
[docker]: https://www.docker.com/

On Ubuntu 20.10 and later you can install podman via:

```sh
sudo apt update && sudo apt install podman
```

To spawn a container containing the development environment you can simply run
the following command in the root of the repository:

```sh
# for a debug build:
./scripts/setup-local-dev-env debug
# or alternatively for a release build:
./scripts/setup-local-dev-env release
```

After building the container image and spawning a new container instance you
will be greeted by a shell inside the container. The software is already built
and installed. You can use it by executing `gm_uai`, `ct_jug`, `qap_dd`, etc.

### Using Your Host System

If you want to set up the development environment on your host system you
should first install the necessary dependencies. On Ubuntu you can install them
by issuing:

```sh
sudo apt install build-essential curl meson ninja-build pkg-config python3 python3-dev python3-numpy swig
```

There is a helper script to automate the process of creating a development
environment. You can can execute the following command from the root of the
repository:

```sh
# for debug build:
./scripts/setup-local-dev-env debug
# alternatively for release build:
./scripts/setup-local-dev-env release
```

The script will create a temporary directory, build and install some
dependencies to the temporary install root and build the project in the
temporary directory. Note that the script aims to not change or install
software on your host system. Everything is done in temporary directories that
are automatically deleted when you leave the development environment. If the
scripts finishes successfully, you will be greeted by a shell in the build
directory of libmpopt. Everything is installed to a temporary install prefix
which is automatically added to your `$PATH` environment variable. You can run
the software by executing `gm_uai`, `ct_jug`, `qap_dd`, etc.

After making changes to the source code run `ninja install` in the shell of the
development environment.

If you want to switch between debug or release mode you can execute `meson
configure -Dbuildtype=debug` or `meson configure -Dbuildtype=release` followed
by `ninja install`.


## Building and Installing the Software Manually

You will need build tools like meson, ninja, a C++17 compatible compiler, Python
and the SWIG binding generator.

The following command should install all necessary dependencies on an Ubuntu machine:

```sh
sudo apt install build-essential curl meson ninja-build pkg-config python3 python3-dev python3-numpy swig
```

Building is done by:

```sh
mkdir /path/to/build/directory
meson setup /path/to/build/directory
ninja -C /path/to/build/directory
```

Installation is done by:

```sh
ninja -C /path/to/build/directory install
```
