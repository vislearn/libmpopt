from ..common.solver import BaseSolver, DEFAULT_BATCH_SIZE, DEFAULT_MAX_BATCHES
from . import libqapopt as lib
from .model import Model
from . primals import Primals

import ctypes
import numpy as np


INFINITY_COST = 1e99


class ModelDecomposition:

    def __init__(self, model, with_uniqueness, unary_side='left', no_dummy=False, add_infinity_edge_arcs=True):
        """A helper for gathering information useful for a Lagrange-dual based
        decomposition of a graph matching problem.

        The argument `no_dummy` determines whether dummy-nodes are added
        in the unaries. By default, dummies are added.

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
        self.no_dummy = no_dummy
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

        if not with_uniqueness or add_infinity_edge_arcs:
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
            d = 0 if (self.model.dummies_included or self.no_dummy) else 1
            shape = (len(self.unary_set[idx1]) + d, len(self.unary_set[idx2]) + d)
            self.pairwise[idx1, idx2] = np.zeros(shape)

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

    def run_rounding_only(self):
        self.lib.solver_run_rounding_only(self.solver)

    def run_quiet(self, batch_size=DEFAULT_BATCH_SIZE, max_batches=DEFAULT_MAX_BATCHES):
        self.lib.solver_run_quiet(self.solver, batch_size, max_batches)

    def run_no_rounding(self, batch_size=DEFAULT_BATCH_SIZE, max_batches=DEFAULT_MAX_BATCHES):
        self.lib.solver_run_no_rounding(self.solver, batch_size, max_batches)

    def _execute_fusion_move_helper(self, method, solution0, solution1, *args):
        assert(len(solution0) == len(solution1))

        np0 = np.asarray(solution0, dtype=np.uint32)
        carray0 = np.ctypeslib.as_ctypes(np0)
        ptr0 = ctypes.cast(carray0, ctypes.c_void_p)

        np1 = np.asarray(solution1, dtype=np.uint32)
        carray1 = np.ctypeslib.as_ctypes(np1)
        ptr1 = ctypes.cast(carray1, ctypes.c_void_p)

        f = getattr(self.lib, 'solver_execute_' + method)
        return f(self.solver, ptr0.value, ptr1.value, len(solution0), *args)

    def execute_fusion_move(self, solution0, solution1):
        return self._execute_fusion_move_helper('fusion_move', solution0, solution1)

    def execute_qpbo(self, solution0, solution1, enable_weak_persistency, enable_probe, enable_improve):
        return self._execute_fusion_move_helper('qpbo', solution0, solution1, enable_weak_persistency, enable_probe, enable_improve)

    def execute_lsatr(self, solution0, solution1):
        return self._execute_fusion_move_helper('lsatr', solution0, solution1)

    def compute_greedy_assignment(self):
        return self.lib.solver_compute_greedy_assignment(self.solver)

    def get_times(self):
        return (self.lib.solver_get_fm_ms_build(self.solver),
                self.lib.solver_get_fm_ms_solve(self.solver))


def construct_solver(deco):
    s = Solver()
    g = lib.solver_get_graph(s.solver)

    d = 0 if (deco.model.dummies_included or deco.no_dummy) else 1

    # insert unary factors
    for u, idx in enumerate(deco.unary_to_nodeside):
        f = lib.graph_add_unary(g, u, len(deco.unary_set[idx]) + d, deco.no_forward[idx], deco.no_backward[idx])
        for i, ass_id in enumerate(deco.unary_set[idx]):
            lib.unary_set_cost(f, i, deco.model.assignments[ass_id].cost)
        if not (deco.model.dummies_included or deco.no_dummy):
            lib.unary_set_cost(f, i+1, 0.0)

    # insert uniqueness factors
    if deco.with_uniqueness:
        for idx_uniqueness, (label_idx, assigned_in) in enumerate(deco.label_set.items()):
            f = lib.graph_add_uniqueness(g, idx_uniqueness, len(assigned_in))
            for slot, assignment_idx in enumerate(assigned_in):
                assignment = deco.model.assignments[assignment_idx]
                assert getattr(assignment, deco.label_side) == label_idx
                label = deco.unary_set[getattr(assignment, deco.unary_side)].index(assignment_idx) # FIXME: O(n) is best avoided.
                lib.graph_add_uniqueness_link(g, deco.nodeside_to_unary[getattr(assignment, deco.unary_side)], label, idx_uniqueness, slot)

    # insert pairwise factors
    for i, ((idx1, idx2), cost) in enumerate(deco.pairwise.items()): # use items()
        f = lib.graph_add_pairwise(g, i, cost.shape[0], cost.shape[1])
        lib.graph_add_pairwise_link(g, deco.nodeside_to_unary[idx1], deco.nodeside_to_unary[idx2], i)
        for l_u in range(len(deco.unary_set[idx1]) + d):
            for l_v in range(len(deco.unary_set[idx2]) + d):
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


def prepare_import_primal(deco, primal):
    solution = [None] * len(deco.unary_set)

    for left, right in enumerate(primal):
        hack = {'left': left, 'right': right}
        unary = hack[deco.unary_side]
        label = hack[deco.label_side]
        assert(unary == deco.unary_to_nodeside[unary])

        if label is not None:
            assignment_idx = deco.model.get_assignment_id(left, right)
            # FIXME: linear complexity
            label = deco.unary_set[unary].index(assignment_idx)
        else:
            label = len(deco.unary_set[unary]) # dummy label

        solution[unary] = label

    assert(all(x is not None for x in solution))
    return solution


def export_model(deco, solver):
    g = lib.solver_get_graph(solver.solver)

    original_assignments = []
    dummy_assignments_left = {}
    dummy_assignments_right = {}
    normal_edges = []
    dummy_connected_edges = []

    labellist = list(deco.label_set)

    # get assignments (node-label-assignments) with costs, make sure that numbering of labels and nodes remains the same as in original model
    for ass_id, assignment in enumerate(deco.model.assignments):
        idx_unary_side = getattr(assignment, deco.unary_side)
        idx_label_side = getattr(assignment, deco.label_side)
        # get current cost of assignment in unary:
        u = deco.nodeside_to_unary[idx_unary_side]
        idx = deco.unary_set[idx_unary_side].index(ass_id)
        cost = lib.unary_get_cost(lib.graph_get_unary(g, u), idx)
        # add cost of assignment in uniqueness:
        if deco.with_uniqueness:
            v = labellist.index(idx_label_side)
            jdx = deco.label_set[idx_label_side].index(ass_id)
            cost += lib.uniqueness_get_cost(lib.graph_get_uniqueness(g, v), jdx)

        original_assignments.append((assignment.left, assignment.right, cost))

    # get additional dummy assignments on unaries
    for (unary_idx, label_set) in deco.unary_set.items():
        u = deco.nodeside_to_unary[unary_idx]
        idx = len(label_set)
        cost = lib.unary_get_cost(lib.graph_get_unary(g, u), idx)
        if deco.unary_side == 'left':
            dummy_assignments_left[unary_idx] = cost
        else:
            dummy_assignments_right[unary_idx] = cost

    # get additional dummy assignments on uniqueness
    if deco.with_uniqueness:
        for v, (label_idx, unary_set) in enumerate(deco.label_set.items()):
            jdx = len(unary_set)
            cost = lib.uniqueness_get_cost(lib.graph_get_uniqueness(g, v), jdx)
            if deco.unary_side == 'left':
                dummy_assignments_right[label_idx] = cost
            else:
                dummy_assignments_left[label_idx] = cost

    for i, (idx1, idx2) in enumerate(deco.pairwise):
        for l_u in range(len(deco.unary_set[idx1])+1):
            for l_v in range(len(deco.unary_set[idx2])+1):
                cost = lib.pairwise_get_cost(lib.graph_get_pairwise(g, i), l_u, l_v)
                if cost != 0.0:
                    ass1, ass2 = -1, -1
                    if l_u < len(deco.unary_set[idx1]):
                        ass1 = deco.unary_set[idx1][l_u]
                    if l_v < len(deco.unary_set[idx2]):
                        ass2 = deco.unary_set[idx2][l_v]
                    if ass1 < 0 or ass2 < 0:
                        dummy_connected_edges.append((deco.unary_side, idx1, idx2, ass1, ass2, cost))
                        assert cost < INFINITY_COST
                    else:
                        if cost < INFINITY_COST:
                            normal_edges.append((ass1, ass2, cost))
                        else:
                            assert getattr(deco.model.assignments[ass1], deco.label_side) == getattr(deco.model.assignments[ass2], deco.label_side)

    return original_assignments, dummy_assignments_left, dummy_assignments_right, normal_edges, dummy_connected_edges


def statistics(deco, solver, primals=[]):
    eps = 0.5

    if len(primals) == 2:
        initialPrimals = primals[0]
        repaPrimals = primals[1]
        print('Initial primal better: {}'.format(deco.model.get_sol_cost(initialPrimals) < deco.model.get_sol_cost(repaPrimals)))
        primalsCorrespond = []
        for i in range(len(initialPrimals)):
            primalsCorrespond.append(initialPrimals[i] == repaPrimals[i])
        print('Primals differ in {} unaries.'.format(primalsCorrespond.count(False)))

        original_assignments, dummy_assignments_left, dummy_assignments_right, standard_edges_repa, dummy_connected_edges_repa = export_model(deco, solver)
        print('Number of labels with similar (or smaller) cost in reparametrized problem (compared to primal):')
        similarPrimals = [(0,0,0)] * deco.model.no_left
        for i, assigned_in in deco.model.left.items():
            possible_costs = []
            primal_cost = None
            for ass_id in assigned_in:
                possible_costs.append(original_assignments[ass_id][2])
                if original_assignments[ass_id][1] == repaPrimals[i]:
                    primal_cost = original_assignments[ass_id][2]
            possible_costs.append(dummy_assignments_left[i])
            if primal_cost == None:
                assert repaPrimals[i] == -1
                primal_cost = dummy_assignments_left[i]

            counterLess = 0
            counter = 0
            counterMore = 0
            counter10 = 0
            counter20 = 0
            for c in possible_costs:
                if c < primal_cost:
                    counterLess += 1
                if c <= primal_cost:
                    counter += 1
                if c <= primal_cost + eps:
                    counterMore += 1
                if c <= primal_cost + 10*eps:
                    counter10 += 1
                if c <= primal_cost + 20*eps:
                    counter20 += 1
            similarPrimals[i] = (counterLess, counter, counterMore)
        for i, (n1,n2,n3) in enumerate(similarPrimals):
            m = len(deco.model.left[i])+1-n3
            # m = len(deco.model.left[i])+1-n1
            print('{: >4d}  {}{}{}{}'.format(repaPrimals[i], '-'*n1, 'o'*(n2-n1), '+'*(n3-n2), '_'*m))
            # print('{: >6.1f}  {: >6.1f}  {: >6.1f}  {}{}{}{}'.format(100*n1/(m+n3), 100*n2/(m+n3), 100*n3/(m+n3), '-'*n1, 'o'*(n2-n1), '+'*(n3-n2), '_'*m))
            # print('{: >6.1f}  {}/{}'.format(100*n1/(m+n1), 'x'*n1, '-'*m))


def check_dummy_costs(deco, solver):
    g = lib.solver_get_graph(solver.solver)
    eps = 10e-6
    counter = 0
    for u, idx in enumerate(deco.unary_to_nodeside):
        dummy = len(deco.unary_set[idx])
        if not -eps < lib.unary_get_cost(lib.graph_get_unary(g, u), dummy) < eps:
            counter += 1
    return counter


def sort_ids(id1, id2, pos1, pos2):
    if id1 < id2:
        return id1, id2, pos1, pos2
    else:
        return id2, id1, pos2, pos1


def build_model_from_solutions(deco, sol1, sol2):
    assert len(sol1) == len(sol2) == deco.model.no_left
    # to initialise model need: no_left, no_right, no_assignments, no_edges
    no_left = deco.model.no_left
    no_right = deco.model.no_right + sol1.count(-1) + sol2.count(-1)

    nodes = []
    assignments = []
    edges = []
    labelling_to_assignment = {}

    no_assignments = 0
    no_edges = 0
    for i in range(no_left):
        no_assignments = no_assignments + 2 - (sol1[i] == sol2[i])
        nodes.append([sol1[i]])
        if sol2[i] != sol1[i]:
            nodes[-1].append(sol2[i])

        for l1 in nodes[-1]:
            assignments.append([i,l1,0.0])
            if l1 != -1:
                id1 = deco.model.get_assignment_id(i,l1)
                assignments[-1][2] = deco.model.get_assignment_cost(id1)
                for j in range(i):
                    for l2 in nodes[j]:
                        if l2 != -1:
                            id2 = deco.model.get_assignment_id(j,l2)
                            edge_cost = deco.model.get_edge_costs([id1, id2])
                            if edge_cost != 0:
                                no_edges += 1
                                edges.append(((i,l1),(j,l2),edge_cost))

    model = Model(no_left, no_right, no_assignments, no_edges)

    # add assignments
    assignment_counter = 0
    dummy_counter = deco.model.no_right
    for [n,l,c] in assignments:
        if l != -1:
            model.add_assignment(assignment_counter, n, l, c)
            labelling_to_assignment[(n,l)] = assignment_counter
        else:
            model.add_dummy(assignment_counter, n, dummy_counter, c)
        assignment_counter += 1

    assert assignment_counter == no_assignments

    # add edges
    edge_counter = 0
    for (p1,p2,edge_cost) in edges:
        model.add_edge(labelling_to_assignment[p1], labelling_to_assignment[p2], edge_cost)
        edge_counter += 1

    assert edge_counter == no_edges

    return model
