from ..common.solver import BaseSolver, DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES
from . import libmpopt_qap as lib
from . primals import Primals

import numpy


DEFAULT_GREEDY_GENERATIONS = 10
INFINITY_COST = 1e99
DEFAULT_ALPHA = 0.25


class ModelDecomposition:

    def __init__(self, model, with_uniqueness, unary_side='left'):
        """A helper for gathering information useful for a Lagrange-dual based
        decomposition of a graph matching problem.

        The argument `unary_side` (either `left` or `right`) determines which
        point set of the underlying model is used as a unary node set. The
        other one is used as label universe for all unary nodes. When a unary
        node is created only labels for which a possible assignment is
        available will be added to the node's label space. A bidirectional
        mapping is remembered.

        The argument `with_uniqueness` determines whether uniqueness constraints
        will be inserted. If uniqueness constraints are inserted then no
        additional infinity edges will be included in the decomposition, i.e.
        the call to `add_infinity_edge_arcs` will not add new edges.
        Without uniqueness constraints we have to insert additional infinity
        edges to disallow impossible configurations that would have been
        otherwise forbidden by the uniqueness constraints.

        Note that currently `with_uniqueness` has no further impact. The
        downstream code must check `with_uniqueness` to see if it should insert
        uniqueness constraints into the solver or not.
        """
        assert unary_side in ('left', 'right')

        self.model = model
        self.with_uniqueness = with_uniqueness
        self.unary_side = unary_side
        self.label_side = 'right' if unary_side == 'left' else 'left'

        self.unary_to_nodeside = [] # unary solver index -> graph matching node index (left or right)
        self.nodeside_to_unary = {} # same as above, only the other way around
        for node in range(self.number_of_nodes):
            if node in self.unary_set:
                self.nodeside_to_unary[node] = len(self.unary_to_nodeside)
                self.unary_to_nodeside.append(node)

        self.pairwise = {}
        self.no_forward = [0] * self.number_of_nodes
        self.no_backward = [0] * self.number_of_nodes

        for edge in self.model.edges:
            self._insert_pairwise(*edge)

        self._add_infinity_edge_arcs()

    @property
    def unary_set(self):
        """Returns the unary node set of the underlying graph matching model.

        The result is a dict of lists of assignment-ids, detailing for each
        unary in which assignments it is involved.

        The unaries are indexed as in the original model, indices not occuring
        in any assignment do also not occur in the dict.
        """
        return getattr(self.model, self.unary_side)

    @property
    def number_of_nodes(self):
        """Returns the number of nodes of the underlying graph matching model.

        The result is either the number of left or right points depending on
        the side where we build the model.
        """
        return getattr(self.model, 'no_' + self.unary_side)

    @property
    def label_set(self):
        """Returns the label set of the underlying graph matching model.

        The result is a dict of lists of assignment-ids, detailing for each
        label in which assignments it is involved.

        The labels are indexed as in the original model, indices not occuring
        in any assignment do also not occur in the dict.
        """
        return getattr(self.model, self.label_side)

    def _insert_pairwise(self, id_assignment1, id_assignment2, cost, create_new_edges=True):
        """Inserts a pairwise cost between the two assignments.

        This method finds the corresponding unary nodes and creates an empty
        edge if necessary. Afterwards the cost on the corresponding arc of the
        edge are set to the cost.

        New edges are only inserted if `create_new_edges` is set to `True`. If this
        method creates a new edge, the forward and backward edge counter is updated.

        Note that depending on `self.unary_side` the quadratic term will be
        inserted between two left points or two right points of the underlying
        graph matching model.
        """
        node1 = getattr(self.model.assignments[id_assignment1], self.unary_side)
        node2 = getattr(self.model.assignments[id_assignment2], self.unary_side)
        pos_in_node1 = self.unary_set[node1].index(id_assignment1)
        pos_in_node2 = self.unary_set[node2].index(id_assignment2)
        idx1, idx2, pos1, pos2 = sort_ids(node1, node2, pos_in_node1, pos_in_node2)

        if (idx1, idx2) not in self.pairwise and create_new_edges:
            self.no_forward[idx1] += 1
            self.no_backward[idx2] += 1
            # The node set does not contain the dummy label. The pairwise edges
            # will insert later need to have space for the dummy label. Hence
            # we add +1 here for both dimensions.
            shape = (len(self.unary_set[idx1]) + 1, len(self.unary_set[idx2]) + 1)
            self.pairwise[idx1, idx2] = numpy.zeros(shape)

        if (idx1, idx2) in self.pairwise:
            assert self.pairwise[idx1, idx2][pos1, pos2] == 0.0
            self.pairwise[idx1, idx2][pos1, pos2] = cost

    def _add_infinity_edge_arcs(self):
        """Set pairwise edge cost to infinity for prohibiting assignment constraints.

        If `self.with_uniqueness` is set to `True`, only existing edges will
        get updated. Otherwise, non-existing edges will get created (costs are
        initialized to zero). The later is needed for building a purely pairwise
        graphical model.
        """
        for label in self.label_set:
            assigned_in = self.label_set[label]
            for i in range(len(assigned_in) - 1):
                for j in range(i+1, len(assigned_in)):
                    ass1 = assigned_in[i]
                    ass2 = assigned_in[j]
                    self._insert_pairwise(ass1, ass2, INFINITY_COST, create_new_edges=not self.with_uniqueness)


class Solver(BaseSolver):

    def __init__(self):
        super().__init__(lib)

    def set_fusion_moves_enabled(self, enabled):
        self.lib.solver_set_fusion_moves_enabled(self.solver, enabled)

    def set_local_search_enabled(self, enabled):
        self.lib.solver_set_local_search_enabled(self.solver, enabled)

    def set_dual_updates_enabled(self, enabled):
        self.lib.solver_set_dual_updates_enabled(self.solver, enabled)

    def set_grasp_alpha(self, alpha):
        assert(0 < alpha <= 1)
        self.lib.solver_set_grasp_alpha(self.solver, alpha)

    def use_grasp(self):
        self.lib.solver_use_grasp(self.solver)

    def use_greedy(self):
        self.lib.solver_use_greedy(self.solver)

    def set_random_seed(self, seed):
        return self.lib.solver_set_random_seed(self.solver, seed)

    def run(self, batch_size=DEFAULT_BATCH_SIZE, max_batches=DEFAULT_MAX_BATCHES, greedy_generations=DEFAULT_GREEDY_GENERATIONS):
        return self.lib.solver_run(self.solver, batch_size, max_batches, greedy_generations)

    def compute_greedy_assignment(self):
        return self.lib.solver_compute_greedy_assignment(self.solver)


def construct_gm_model(deco):
    from ..gm.model import Model as GmModel

    edges = deco.pairwise

    gm_model = GmModel()
    for u, idx in enumerate(deco.unary_to_nodeside):
        costs = [deco.model.assignments[ass_id].cost for ass_id in deco.unary_set[idx]]
        costs.append(0.0)
        gm_model.add_unary(costs)

    for (idx1, idx2), costs in deco.pairwise.items():
        gm_model.add_pairwise(idx1, idx2, costs)

    return gm_model


def construct_solver(deco):
    s = Solver()
    g = lib.solver_get_graph(s.solver)

    # insert unary factors
    for u, idx in enumerate(deco.unary_to_nodeside):
        f = lib.graph_add_unary(g, u, len(deco.unary_set[idx]) + 1, deco.no_forward[idx], deco.no_backward[idx])
        for i, ass_id in enumerate(deco.unary_set[idx]):
            lib.unary_set_cost(f, i, deco.model.assignments[ass_id].cost)
        lib.unary_set_cost(f, i+1, 0.0)

    # insert uniqueness factors
    if deco.with_uniqueness:
        for idx_uniqueness, (label_idx, assigned_in) in enumerate(deco.label_set.items()):
            f = lib.graph_add_uniqueness(g, idx_uniqueness, len(assigned_in), label_idx)
            for slot, assignment_idx in enumerate(assigned_in):
                assignment = deco.model.assignments[assignment_idx]
                assert getattr(assignment, deco.label_side) == label_idx
                label = deco.unary_set[getattr(assignment, deco.unary_side)].index(assignment_idx) # FIXME: O(n) is best avoided.
                lib.graph_add_uniqueness_link(g, deco.nodeside_to_unary[getattr(assignment, deco.unary_side)], label, idx_uniqueness, slot)

    # insert pairwise factors
    for i, ((idx1, idx2), cost) in enumerate(deco.pairwise.items()): # use items()
        f = lib.graph_add_pairwise(g, i, cost.shape[0], cost.shape[1])
        lib.graph_add_pairwise_link(g, deco.nodeside_to_unary[idx1], deco.nodeside_to_unary[idx2], i)
        for l_u in range(len(deco.unary_set[idx1]) + 1):
            for l_v in range(len(deco.unary_set[idx2]) + 1):
                lib.pairwise_set_cost(f, l_u, l_v, cost[l_u, l_v])

    lib.solver_finalize(s.solver)
    return s


def extract_primals(deco, solver):
    primals = Primals(deco.model)
    g = lib.solver_get_graph(solver.solver)

    for u, idx in enumerate(deco.unary_to_nodeside):
        lib_primal = lib.unary_get_primal(lib.graph_get_unary(g, u))
        if lib_primal < len(deco.unary_set[idx]):
            assignment_idx = deco.unary_set[idx][lib_primal]
            assignment = deco.model.assignments[assignment_idx]
            primals[idx] = getattr(assignment, deco.label_side)
        else:
            assert lib_primal == len(deco.unary_set[idx])

    return primals


def sort_ids(id1, id2, pos1, pos2):
    if id1 < id2:
        return id1, id2, pos1, pos2
    else:
        return id2, id1, pos2, pos1
