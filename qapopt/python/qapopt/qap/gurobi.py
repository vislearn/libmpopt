import time

from ..common import gurobi
from ..common.gurobi import GRB
from .primals import Primals

class FusionMovesModel:

    def __init__(self, model, solution1, solution2):
        assert len(solution1) == len(solution2) == model.no_left
        self.model = model
        self.solution1 = solution1
        self.solution2 = solution2


class GurobiBase:

    def __init__(self, model, ilp_mode=True, times=None):
        self.model = model
        self.gurobi = None
        self.ilp_mode = ilp_mode
        if times is None:
            self.times = {'build': 0.0, 'solve': 0.0, 'a': 0.0, 'b': 0.0, 'c': 0.0, 'd': 0.0}
        else:
            assert 'build' in times
            assert 'solve' in times
            self.times = times

    def _add_gurobi_variable(self, obj=0.0, disable_ub=False):
        """Adds a single Gurobi variable.

        State of `ilp_mode` determines if the variable is continuous or binary.
        """
        kwargs = {'lb': 0.0,
                  'ub': 1.0,
                  'obj': obj,
                  'vtype': GRB.BINARY if self.ilp_mode else GRB.CONTINUOUS}
        if disable_ub:
            del kwargs['ub']
        return self.gurobi.addVar(**kwargs)

    def construct(self):
        raise NotImplementedError()

    def update_upper_bound(self, tracker):
        ub = tracker.evaluate_primal()
        self.gurobi.Params.CutOff = ub

    def run(self, output=1):
        if self.gurobi is None:
            tik = time.perf_counter()
            self.construct()
            tak = time.perf_counter()
            self.times['build'] += tak - tik
        self.gurobi.setParam('OutputFlag', output)

        # Keep this in sync with C++ part.
        self.gurobi_.setParam('Thread', 1)
        self.gurobi_.setParam('Method', 1) # dual simplex
        self.gurobi.setParam('MIPGap', 0)
        self.gurobi.setParam('MIPGapAbs', 0)
        tik = time.perf_counter()
        self.gurobi.optimize()
        tak = time.perf_counter()
        self.times['solve'] += tak - tik

    def get_primals(self):
        raise NotImplementedError()

    def get_times(self):
        return self.times


class GurobiStandardModel(GurobiBase):

    # NOTE: the construction below leads to a weaker relaxation
    # + does not need equality constraints
    # + does only add a minimal number of variables
    def construct(self):
        self.gurobi = gurobi.Model()
        self._nodes = {}
        self._edges = {}

        # node variables and constraints
        print('...Adding nodes')
        for i, assignments in self.model.left.items():
            assert len(assignments) > 0
            nodes = []
            # get all possible labels `s` for `i` with their `assignment_cost`
            for assignment_id in assignments:
                s = self.model.assignments[assignment_id].right
                assert s >= 0
                cost = self.model.assignments[assignment_id].cost
                self._nodes[i,s] = self._add_gurobi_variable(cost)
                nodes.append(self._nodes[i,s])
            # add constraint that all variables for all labels sum up to at most one
            self.gurobi.addConstr(sum(nodes) <= 1)

        # uniqueness constraints
        print('...Adding uniqueness')
        for label, assignments in self.model.right.items():
            if len(assignments) > 1:
                # add constraint that each label occurs at most once
                self.gurobi.addConstr(sum([self._nodes[self.model.assignments[assignment_id].left,label] for assignment_id in assignments]) <= 1)

        # edge constraints
        print('...Adding edges')
        for (id1,id2,cost) in self.model.edges:
            var = self._add_gurobi_variable(cost)
            self._edges[id1, id2] = var
            assignment1 = self.model.assignments[id1]
            assignment2 = self.model.assignments[id2]
            var1 = self._nodes[assignment1.left,assignment1.right]
            var2 = self._nodes[assignment2.left,assignment2.right]
            self.gurobi.addConstr(var <= var1)
            self.gurobi.addConstr(var <= var2)
            self.gurobi.addConstr(var >= var1 + var2 - 1)

    def get_primals(self):
        labeling = [-2] * self.model.model.no_left
        for (i,s), v in self._nodes.items():
            if v.X > .9:
                assert labeling[i] == -2
                labeling[i] = s
        assert -2 not in labeling
        primals = Primals(self.model.model, labeling)
        return primals


class Gurobi(GurobiStandardModel):

    def construct(self):
        self.gurobi = gurobi.Model()
        self._nodes = {}
        self._edges = {}

        # node variables and constraints
        print('...Adding nodes')
        for i, assignments in self.model.left.items():
            assert len(assignments) > 0
            nodes = []
            # get all possible labels `s` for `i` with their `assignment_cost`
            for assignment_id in assignments:
                s = self.model.assignments[assignment_id].right
                assert self.model.assignments[assignment_id].right >= 0
                cost = self.model.assignments[assignment_id].cost
                self._nodes[assignment_id] = self._add_gurobi_variable(cost)
                nodes.append(self._nodes[assignment_id])
            # also add dummy label with `cost=0`
            self._nodes[i,-1] = self._add_gurobi_variable(0)
            nodes.append(self._nodes[i,-1])
            # add constraint that all variables for all labels sum up to one
            self.gurobi.addConstr(sum(nodes) == 1)

        # uniqueness constraints
        print('...Adding uniqueness')
        for label, assignments in self.model.right.items():
            if len(assignments) > 1:
                # add constraint that each label occurs at most once
                self.gurobi.addConstr(sum([self._nodes[assignment_id] for assignment_id in assignments]) <= 1)

        edges = {}
        # edge constraints
        print('...Adding edges')
        for (id1,id2,cost) in self.model.edges:
            self._edges[id1,id2] = self._add_gurobi_variable(cost)
            assignment1 = self.model.assignments[id1]
            assignment2 = self.model.assignments[id2]
            assert assignment1.left < assignment2.left
            edges.setdefault((assignment1.left, assignment2.left), []).append((id1, id2))
        for (node1, node2), assignment_pairs in edges.items():
            assignments_node1 = self.model.left[node1]
            assignments_node2 = self.model.left[node2]
            for id1 in assignments_node1:
                for id2 in assignments_node2:
                    if (id1,id2) not in assignment_pairs:
                        self._edges[id1,id2] = self._add_gurobi_variable(0)
            for id1 in assignments_node1:
                self._edges[id1,node2,-1] = self._add_gurobi_variable(0)
                self.gurobi.addConstr(self._edges[id1,node2,-1] + sum([self._edges[id1,id2] for id2 in assignments_node2]) == self._nodes[id1])
            for id2 in assignments_node2:
                self._edges[node1,-1,id2] = self._add_gurobi_variable(0)
                self.gurobi.addConstr(self._edges[node1,-1,id2] + sum([self._edges[id1,id2] for id1 in assignments_node1]) == self._nodes[id2])
            self._edges[node1,-1,node2,-1] = self._add_gurobi_variable(0)
            self.gurobi.addConstr(self._edges[node1,-1,node2,-1] + sum([self._edges[node1,-1,id2] for id2 in assignments_node2]) == self._nodes[node1,-1])
            self.gurobi.addConstr(self._edges[node1,-1,node2,-1] + sum([self._edges[id1,node2,-1] for id1 in assignments_node1]) == self._nodes[node2,-1])


class GurobiFusionModel(GurobiStandardModel):

    def construct(self):
        self.gurobi = gurobi.Model()
        self._nodes = {}
        self._edges = {}

        assignments = {}
        occurrence = {}

        # tik = time.perf_counter()

        # node variables and constraints
        for i in range(self.model.model.no_left):
            s1 = self.model.solution1[i]
            s2 = self.model.solution2[i]
            if s1 == -1 or s1 is None:
                assignment_cost = 0
            else:
                occurrence.setdefault(s1, []).append(i)
                id_assignment = self.model.model.get_assignment_id(i, s1)
                assignments[id_assignment] = (i, s1)
                assignment_cost = self.model.model.get_assignment_cost(id_assignment)
            self._nodes[i, s1] = self._add_gurobi_variable(assignment_cost)
            if s2 != s1:
                if s2 == -1 or s2 is None:
                    assignment_cost = 0
                else:
                    occurrence.setdefault(s2, []).append(i)
                    id_assignment = self.model.model.get_assignment_id(i, s2)
                    assignments[id_assignment] = (i, s2)
                    assignment_cost = self.model.model.get_assignment_cost(id_assignment)
                self._nodes[i, s2] = self._add_gurobi_variable(assignment_cost)
                self.gurobi.addConstr(self._nodes[i,s1] + self._nodes[i,s2] == 1)
            else:
                self.gurobi.addConstr(self._nodes[i,s1] == 1)

        # tak = time.perf_counter()
        # self.times['a'] += tak - tik

        # tik = time.perf_counter()

        # uniqueness constraints
        for label, nodes in occurrence.items():
            if len(nodes) > 1:
                assert len(nodes) == 2
                n1, n2 = nodes
                self.gurobi.addConstr(self._nodes[n1,label] + self._nodes[n2, label] <= 1)

        # tak = time.perf_counter()
        # self.times['b'] += tak - tik

        # tik = time.perf_counter()

        # edge constraints
        for (id1, id2, cost) in self.model.model.edges:
            if id1 in assignments and id2 in assignments:
                var1 = self._nodes[assignments[id1]]
                var2 = self._nodes[assignments[id2]]
                var = self._add_gurobi_variable(cost)
                # tik2 = time.perf_counter()
                #self._edges[id1, id2] = var
                self.gurobi.addConstr(var <= var1)
                self.gurobi.addConstr(var <= var2)
                self.gurobi.addConstr(var >= var1 + var2 - 1)
        #         tak2 = time.perf_counter()
        #         self.times['d'] += tak2 - tik2

        # tak = time.perf_counter()
        # self.times['c'] += tak - tik


class SlowGurobiFusionModel(GurobiBase):

    def construct(self):
        self.gurobi = gurobi.Model()

        # shortcuts
        g = self.gurobi
        m = self.model.model
        s1 = self.model.solution1
        s2 = self.model.solution2
        vtype = GRB.BINARY if self.ilp_mode else GRB.CONTINUOUS

        assert len(s1) == m.no_left
        assert len(s2) == m.no_left

        # somehow the solutions can have a weird format, let's fix this here
        def cleanup_solution(sol):
            return [x if x != -1 else None for x in sol]
        s1 = cleanup_solution(s1)
        s2 = cleanup_solution(s2)

        dummy_counter = m.no_right

        def assign_dummy_if_necessary(x):
            nonlocal dummy_counter
            if x is None:
                dummy_counter += 1
                return dummy_counter - 1
            else:
                return x

        def fetch_assignment_cost(i, x):
            if x is None:
                return 0
            else:
                a = m.get_assignment_id(i, x)
                return m.get_assignment_cost(a)

        tmp = {}
        for i, (x1, x2) in enumerate(zip(s1, s2)):
            tmp[i, assign_dummy_if_necessary(x1)] = fetch_assignment_cost(i, x1)
            if x1 != x2:
                tmp[i, assign_dummy_if_necessary(x2)] = fetch_assignment_cost(i, x2)

        indices, objectives = gurobi.multidict(tmp)
        self._nodes = g.addVars(indices, obj=objectives, lb=0, ub=1, vtype=vtype)

        # uniqueness constraints
        g.addConstrs(self._nodes.sum(i, '*') == 1 for i in range(m.no_left))
        g.addConstrs(self._nodes.sum('*', i) <= 1 for i in range(m.no_right))

        tik = time.perf_counter()
        tmp = {}
        for assignment_id1, assignment_id2, cost in m.edges:
            assignment1 = m.assignments[assignment_id1]
            assignment2 = m.assignments[assignment_id2]

            key1_present = (assignment1.left, assignment1.right) in indices
            key2_present = (assignment2.left, assignment2.right) in indices
            if key1_present and key2_present:
                tmp[assignment1.left, assignment1.right, assignment2.left, assignment2.right] = cost
        self.times['a'] = time.perf_counter() - tik

        indices, objectives = gurobi.multidict(tmp)
        self._edges = g.addVars(indices, obj=objectives, lb=0, ub=1, vtype=vtype)
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] <= self._nodes[idx[0], idx[1]] for idx in indices)
        self.times['b'] += time.perf_counter() - tik
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] <= self._nodes[idx[2], idx[3]] for idx in indices)
        self.times['c'] += time.perf_counter() - tik
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] >= self._nodes[idx[0], idx[1]] + self._nodes[idx[2], idx[3]] - 1 for idx in indices)
        self.times['d'] += time.perf_counter() - tik

    def get_primals(self):
        # shortcuts
        m = self.model.model

        labeling = [None] * m.no_left
        for (left, right), variable in self._nodes.items():
            if variable.X > .9:
                assert labeling[left] == None
                labeling[left] = right if right < m.no_right else None

        return Primals(m, labeling)


class FastGurobiFusionModel(GurobiBase):

    def construct(self):
        self.gurobi = gurobi.Model()

        # shortcuts
        g = self.gurobi
        m = self.model.model
        s1 = self.model.solution1
        s2 = self.model.solution2
        vtype = gurobi.GRB.BINARY if self.ilp_mode else gurobi.GRB.CONTINUOUS

        assert len(s1) == m.no_left
        assert len(s2) == m.no_left

        # somehow the solutions can have a weird format, let's fix this here
        def cleanup_solution(sol):
            return [x if x != -1 else None for x in sol]
        s1 = cleanup_solution(s1)
        s2 = cleanup_solution(s2)

        dummy_counter = m.no_right

        def assign_dummy_if_necessary(x):
            nonlocal dummy_counter
            if x is None:
                dummy_counter += 1
                return dummy_counter - 1
            else:
                return x

        def fetch_assignment_cost(i, x):
            if x is None:
                return 0
            else:
                a = m.get_assignment_id(i, x)
                return m.get_assignment_cost(a)

        tmp = {}
        for i, (x1, x2) in enumerate(zip(s1, s2)):
            tmp[i, assign_dummy_if_necessary(x1)] = fetch_assignment_cost(i, x1)
            if x1 != x2:
                tmp[i, assign_dummy_if_necessary(x2)] = fetch_assignment_cost(i, x2)

        indices, objectives = gurobi.multidict(tmp)
        self._nodes = g.addVars(indices, obj=objectives, lb=0, ub=1, vtype=vtype)

        # uniqueness constraints
        g.addConstrs(self._nodes.sum(i, '*') == 1 for i in range(m.no_left))
        g.addConstrs(self._nodes.sum('*', i) <= 1 for i in range(m.no_right))

        tik = time.perf_counter()
        tmp = {}
        for assignment_id1, assignment_id2, cost in m.edges:
            assignment1 = m.assignments[assignment_id1]
            assignment2 = m.assignments[assignment_id2]

            key1_present = (assignment1.left, assignment1.right) in indices
            key2_present = (assignment2.left, assignment2.right) in indices
            if key1_present and key2_present:
                tmp[assignment1.left, assignment1.right, assignment2.left, assignment2.right] = cost
        self.times['a'] = time.perf_counter() - tik

        indices, objectives = gurobi.multidict(tmp)
        self._edges = g.addVars(indices, obj=objectives, lb=0, ub=1, vtype=vtype)
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] <= self._nodes[idx[0], idx[1]] for idx in indices)
        self.times['b'] += time.perf_counter() - tik
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] <= self._nodes[idx[2], idx[3]] for idx in indices)
        self.times['c'] += time.perf_counter() - tik
        tik = time.perf_counter()
        g.addConstrs(self._edges[idx] >= self._nodes[idx[0], idx[1]] + self._nodes[idx[2], idx[3]] - 1 for idx in indices)
        self.times['d'] += time.perf_counter() - tik

    def get_primals(self):
        # shortcuts
        m = self.model.model

        labeling = [None] * m.no_left
        for (left, right), variable in self._nodes.items():
            if variable.X > .9:
                assert labeling[left] == None
                labeling[left] = right if right < m.no_right else None

        return Primals(m, labeling)
