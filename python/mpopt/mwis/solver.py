import ctypes
import numpy as np

from ..common.solver import (DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES, BaseSolver)

from . import libmpopt_mwis_original as lib_original
from . import libmpopt_mwis_bregman_exp as lib_bregman_exp
from . import libmpopt_mwis_bregman_log as lib_bregman_log

DEFAULT_GREEDY_GENERATIONS = 10


class Solver(BaseSolver):

    def __init__(self, lib):
        super().__init__(lib)

    def add_node(self, cost):
        self.lib.solver_add_node(self.solver, cost)

    def add_clique(self, indices):
        indices = np.asarray(sorted(indices), dtype=np.int32)
        array = ctypes.cast(np.ctypeslib.as_ctypes(indices), ctypes.c_void_p)
        self.lib.solver_add_clique(self.solver, array.value, indices.size)

    def finalize(self):
        self.lib.solver_finalize(self.solver)

    def run(self, batch_size=DEFAULT_BATCH_SIZE, max_batches=DEFAULT_MAX_BATCHES, greedy_generations=DEFAULT_GREEDY_GENERATIONS):
        return self.lib.solver_run(self.solver, batch_size, max_batches, greedy_generations)

    def limit_runtime(self, seconds):
        self.lib.solver_limit_runtime(self.solver, seconds)

    def limit_integer_primal_gap(self, percentage):
        self.lib.solver_limit_integer_primal_gap(self.solver, percentage)

    def limit_integer_primal_stagnation(self, seconds):
        self.lib.solver_limit_integer_primal_stagnation(self.solver, seconds)

    def iterations(self):
        return self.lib.solver_get_iterations(self.solver)

    def constant(self, c=None):
        if c is None:
            return self.lib.solver_get_constant(self.solver)
        else:
            self.lib.solver_set_constant(self.solver, c)

    def node_cost(self, node_idx, cost=None):
        if cost is None:
            return self.lib.solver_get_node_cost(self.solver, node_idx)
        else:
            self.lib.solver_set_node_cost(self.solver, node_idx, cost)

    def clique_cost(self, clique_idx, cost=None):
        if cost is None:
            return self.lib.solver_get_clique_cost(self.solver, clique_idx)
        else:
            self.lib.solver_set_clique_cost(self.solver, clique_cost, cost)

    def _getter_setter(self, k, v=None):
        if v is None:
            return getattr(lib, 'solver_get_' + k)(self.solver)
        else:
            getattr(self.lib, 'solver_set_' + k)(self.solver, v)

    def temperature(self, v=None):
        return self._getter_setter('temperature', v)

    def threshold_optimality(self, v=None):
        return self._getter_setter('threshold_optimality', v)

    def threshold_stability(self, v=None):
        return self._getter_setter('threshold_stability', v)

    def temperature_drop_factor(self, v=None):
        return self._getter_setter('temperature_drop_factor', v)


def construct_solver(model, library):
    assert library in ('original', 'bregman_exp', 'bregman_log')
    lib = globals()[f'lib_{library}']
    s = Solver(lib)

    for c in model.nodes:
        s.add_node(c)

    for nodes in model.cliques:
        s.add_clique(nodes)

    s.finalize()
    return s
