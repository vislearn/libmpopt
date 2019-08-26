from ..common.solver import BaseSolver
from . import libmpopt_qap as lib
from .model import sort_ids

import numpy


INFINITY_COST = 1e99


class Solver(BaseSolver):

    def __init__(self):
        super().__init__(lib)


def construct_gm_model(model):
    from ..gm.model import Model as GmModel

    # We ignore the no_forward and no_backward info, as the GM model
    # does this bookkeeping action anyway.
    edges = create_pairwise_data(model, create_new_edges=True)[0]

    u_map = []
    idx_map = {}
    for idx in range(model._no_left):
        if idx in model._unaries_left:
            idx_map[idx] = len(u_map)
            u_map.append(idx)

    gm_model = GmModel()
    for u, idx in enumerate(u_map):
        costs = [model._assignments[ass_id].cost for ass_id in model._unaries_left[idx]]
        costs.append(0.0)
        gm_model.add_unary(costs)

    for (idx1, idx2), costs in edges.items():
        gm_model.add_pairwise(idx1, idx2, costs)

    return gm_model


def construct_solver(model, with_uniqueness=True):
    s = Solver()
    g = lib.solver_get_graph(s.solver)

    # We ignore the no_forward and no_backward return values, as they are not
    # needed for the QAP library.
    edges = create_pairwise_data(model, create_new_edges=not with_uniqueness)[0]

    u_map = []
    idx_map = {}
    for idx in range(model._no_left):
        if idx in model._unaries_left:
            idx_map[idx] = len(u_map)
            u_map.append(idx)

    # insert unary factors
    for u, idx in enumerate(u_map):
        f = lib.graph_add_unary(g, u, len(model._unaries_left[idx]) + 1)
        for i, ass_id in enumerate(model._unaries_left[idx]):
            lib.unary_set_cost(f, i, model._assignments[ass_id].cost)
        lib.unary_set_cost(f, i+1, 0.0)

    # insert uniqueness factors
    if with_uniqueness:
        for idx_uniqueness, (right, assigned_in) in enumerate(model._unaries_right.items()):
            f = lib.graph_add_uniqueness(g, idx_uniqueness, len(assigned_in))
            for slot, assignment_idx in enumerate(assigned_in):
                assignment = model._assignments[assignment_idx]
                assert assignment.right == right
                label = model._unaries_left[assignment.left].index(assignment_idx) # FIXME: O(n) is best avoided.
                lib.graph_add_uniqueness_link(g, idx_map[assignment.left], label, idx_uniqueness, slot)

    # insert pairwise factors
    for i, (idx1, idx2) in enumerate(edges): # use items()
        f = lib.graph_add_pairwise(g, i, edges[idx1, idx2].shape[0], edges[idx1, idx2].shape[1])
        lib.graph_add_pairwise_link(g, idx_map[idx1], idx_map[idx2], i)
        for l_u in range(len(model._unaries_left[idx1]) + 1):
            for l_v in range(len(model._unaries_left[idx2]) + 1):
                lib.pairwise_set_cost(f, l_u, l_v, edges[idx1, idx2][l_u, l_v])

    lib.solver_finalize(s.solver)
    return s


def create_pairwise_data(model, create_new_edges=False):
    edges = {}
    for i, (idx1, idx2) in enumerate(model._pairwise_left):
        data = numpy.zeros((len(model._unaries_left[idx1]) + 1,
            len(model._unaries_left[idx2]) + 1))
        data[:len(model._unaries_left[idx1]),:len(model._unaries_left[idx2])] = model._pairwise_left[idx1, idx2]
        edges[idx1, idx2] = data

    # Copy forward/backward edge stats from original model.
    # If we insert additional infinity edges, we will update these counters.
    no_forward = list(model._no_forward_left)
    no_backward = list(model._no_backward_left)

    # Insert infinities, add necessary edges.
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
                    edges[idx1, idx2][pos1, pos2] = INFINITY_COST
                elif create_new_edges:
                    data = numpy.zeros((len(model._unaries_left[idx1]) + 1,
                        len(model._unaries_left[idx2]) + 1))
                    data[pos1, pos2] = INFINITY_COST
                    edges[idx1, idx2] = data
                    no_forward[idx1] += 1
                    no_backward[idx2] += 1

    if not create_new_edges:
        assert no_forward == model._no_forward_left
        assert no_backward == model._no_backward_left

    return edges, no_forward, no_backward
