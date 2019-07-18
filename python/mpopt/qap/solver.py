from . import libmpopt_qap as lib
from .model import sort_ids

import numpy


class Solver:

    def __init__(self):
        self.solver = lib.solver_create()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.solver is not None:
            lib.solver_destroy(self.solver)
            self.solver = None

    def lower_bound(self):
        return lib.solver_lower_bound(self.solver)

    def upper_bound(self):
        return lib.solver_upper_bound(self.solver)

    def run(self, max_iterations=1000):
        lib.solver_run(self.solver, max_iterations)


def construct_gm_solver(model):
    from ..gm import libmpopt_gm as lib
    from ..gm.solver import Solver as Solver

    # construct edges
    edges = {}
    for i, (idx1, idx2) in enumerate(model._pairwise_left):
        data = numpy.zeros((len(model._unaries_left[idx1]) + 1,
            len(model._unaries_left[idx2]) + 1))
        data[:len(model._unaries_left[idx1]),:len(model._unaries_left[idx2])] = model._pairwise_left[idx1, idx2]
        edges[idx1, idx2] = data

    # get stats for existing edges
    no_forward = list(model._no_forward_left)
    no_backward = list(model._no_backward_left)

    # insert infinities, add necessary edges
    # TODO: for new edges uv, adapt no_forward[u], no_backward[v]
    infinity_cost = 1e20
    for label in model._unaries_right:
        assigned_in = model._unaries_right[label]
        for i in range(len(assigned_in) - 1):
            for j in range(i+1, len(assigned_in)):
                ass1 = assigned_in[i]
                ass2 = assigned_in[j]
                node1 = model._assignments[ass1].left
                node2 = model._assignments[ass2].left
                pos_in_node1 = model._unaries_left[node1].index(ass1)
                pos_in_node2 = model._unaries_left[node2].index(ass2)

                idx1, idx2, pos1, pos2 = sort_ids(node1, node2, pos_in_node1, pos_in_node2)
                if (idx1, idx2) in edges:
                    edges[idx1, idx2][pos1, pos2] = infinity_cost
                else:
                    data = numpy.zeros((len(model._unaries_left[idx1]) + 1,
                        len(model._unaries_left[idx2]) + 1))
                    data[pos1, pos2] = infinity_cost
                    edges[idx1, idx2] = data
                    no_forward[idx1] += 1
                    no_backward[idx2] += 1

    # construct solver
    s = Solver()
    g = lib.solver_get_graph(s.solver)

    u_map = []
    idx_map = {}
    for idx in range(model._no_left):
        if idx in model._unaries_left:
            idx_map[idx] = len(u_map)
            u_map.append(idx)

    # insert unaries
    for u, idx in enumerate(u_map):
        f = lib.graph_add_unary(g, u, len(model._unaries_left[idx]) + 1,
            no_forward[idx], no_backward[idx])
        i = 0
        for ass_id in model._unaries_left[idx]:
            lib.unary_set_cost(f, i, model._assignments[ass_id].cost)
            i += 1
        lib.unary_set_cost(f, i, 0.0)

    # insert pairwise
    for i, (idx1, idx2) in enumerate(edges):
        f = lib.graph_add_pairwise(g, i, edges[idx1, idx2].shape[0], edges[idx1, idx2].shape[1])
        lib.graph_add_pairwise_link(g, idx_map[idx1], idx_map[idx2], i)
        for l_u in range(len(model._unaries_left[idx1]) + 1):
            for l_v in range(len(model._unaries_left[idx2]) + 1):
                lib.pairwise_set_cost(f, l_u, l_v, edges[idx1, idx2][l_u, l_v])

    lib.solver_finalize(s.solver)
    return s
