import ctypes
import numpy as np

from ..common.solver import (DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES, BaseSolver)
from . import libmpopt_mwis as lib


class Solver(BaseSolver):

    def __init__(self):
        super().__init__(lib)

    def add_node(self, cost):
        lib.solver_add_node(self.solver, cost)

    def add_clique(self, indices):
        indices = np.asarray(sorted(indices), dtype=np.int32)
        array = ctypes.cast(np.ctypeslib.as_ctypes(indices), ctypes.c_void_p)
        lib.solver_add_clique(self.solver, array.value, indices.size)

    def finalize(self):
        lib.solver_finalize(self.solver)

    def constant(self, c=None):
        if c is None:
            return lib.solver_get_constant(self.solver)
        else:
            lib.solver_set_constant(self.solver, c)

    def node_cost(self, node_idx, cost=None):
        if cost is None:
            return lib.solver_get_node_cost(self.solver, node_idx)
        else:
            lib.solver_set_node_cost(self.solver, node_idx, cost)

    def clique_cost(self, clique_idx, cost=None):
        if cost is None:
            return lib.solver_get_clique_cost(self.solver, clique_idx)
        else:
            lib.solver_set_clique_cost(self.solver, clique_cost, cost)


def construct_solver(model):
    s = Solver()

    for c in model.nodes:
        s.add_node(c)

    for nodes in model.cliques:
        s.add_clique(nodes)

    s.finalize()
    return s
