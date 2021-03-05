import ctypes
import numpy as np

from ..common.solver import (DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES, BaseSolver)
from . import libmpopt_mwis as lib


class Solver(BaseSolver):

    def __init__(self):
        super().__init__(lib)

    def add_node(self, cost):
        lib.solver_add_node(self.solver, -cost)

    def add_clique(self, indices):
        indices = np.asarray(sorted(indices), dtype=np.int32)
        array = ctypes.cast(np.ctypeslib.as_ctypes(indices), ctypes.c_void_p)
        lib.solver_add_clique(self.solver, array.value, indices.size)

    def finalize(self):
        lib.solver_finalize(self.solver)


def construct_solver(model):
    s = Solver()

    for c in model.nodes:
        s.add_node(c)

    for nodes in model.cliques:
        s.add_clique(nodes)

    s.finalize()
    return s
