from ..common import gurobi
from ..common.gurobi import GRB


class GurobiModel():

    def __init__(self, model, quadratic_objective=False, ilp_mode=True):
        self.model = model
        self.ilp_mode = ilp_mode
        self.gurobi = gurobi.Model()
        vtype = GRB.BINARY if ilp_mode else GRB.CONTINUOUS

        if quadratic_objective:
            self.assignments = self.gurobi.addVars(model.no_assignments, lb=0.0, ub=1.0, vtype=vtype)

            self.gurobi.setObjective(
                sum(a.cost * self.assignments[i] for i, a in enumerate(model.assignments)) +
                sum(e.cost * self.assignments[e.assignment1] * self.assignments[e.assignment2] for e in model.edges))

            for assignment_indices in model.left.values():
                self.gurobi.addConstr(sum(self.assignments[i] for i in assignment_indices) <= 1)

            for assignment_indices in model.right.values():
                self.gurobi.addConstr(sum(self.assignments[i] for i in assignment_indices) <= 1)
        else:
            self.assignments = self.gurobi.addVars(model.no_assignments, lb=0.0, ub=1.0, vtype=vtype)
            self.edges = self.gurobi.addVars(model.no_edges, lb=0.0, ub=1.0, vtype=vtype)

            self.gurobi.setObjective(
                sum(a.cost * self.assignments[i] for i, a in enumerate(model.assignments)) +
                sum(e.cost * self.edges[i] for i, e in enumerate(model.edges)))

            for assignment_indices in model.left.values():
                self.gurobi.addConstr(sum(self.assignments[i] for i in assignment_indices) <= 1)

            for assignment_indices in model.right.values():
                self.gurobi.addConstr(sum(self.assignments[i] for i in assignment_indices) <= 1)

            for i, edge in enumerate(model.edges):
                a = self.assignments[edge.assignment1]
                b = self.assignments[edge.assignment2]
                c = self.edges[i]
                self.gurobi.addConstr(a + b - 1 <= c)
                self.gurobi.addConstr(a >= c)
                self.gurobi.addConstr(b >= c)

    def optimize(self):
        self.gurobi.Params.Threads = 1
        self.gurobi.Params.Method = 1
        self.gurobi.optimize()
