import ctypes
import numpy as np

from ..common.solver import (DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES, BaseSolver)
from . import libmpopt_mwis as lib

DEFAULT_GREEDY_GENERATIONS = 10


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

    def run(self, batch_size=DEFAULT_BATCH_SIZE, max_batches=DEFAULT_MAX_BATCHES, greedy_generations=DEFAULT_GREEDY_GENERATIONS):
        return self.lib.solver_run(self.solver, batch_size, max_batches, greedy_generations)

    def limit_runtime(self, seconds):
        lib.solver_limit_runtime(self.solver, seconds)

    def limit_integer_primal_gap(self, percentage):
        lib.solver_limit_integer_primal_gap(self.solver, percentage)

    def limit_integer_primal_stagnation(self, seconds):
        lib.solver_limit_integer_primal_stagnation(self.solver, seconds)

    def iterations(self):
        return lib.solver_get_iterations(self.solver)

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

    def gamma(self, g=None):
        if gamma is None:
            return lib.solver_get_gamma(self.solver)
        else:
            lib.solver_set_gamma(self.solver, g)

    def temperature(self, t=None):
        if t is None:
            return lib.solver_get_temperature(self.solver)
        else:
            lib.solver_set_temperature(self.solver, t)


def construct_solver(model):
    s = Solver()

    for c in model.nodes:
        s.add_node(c)

    for nodes in model.cliques:
        s.add_clique(nodes)

    s.finalize()
    return s
