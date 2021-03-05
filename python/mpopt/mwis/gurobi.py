from mpopt.utils import if_debug
from mpopt.common import gurobi
from mpopt.mwis.model import Model


class Gurobi:

    def __init__(self, model, ilp_mode=True, pairwise_relaxation=False, silent=True):
        self._constructed = False
        self.model = model
        self.ilp_mode = ilp_mode
        self.pairwise_relaxation = pairwise_relaxation
        self.gurobi = gurobi.Model()

        if silent:
            self.gurobi.Params.OutputFlag = 0

        self.gurobi.Params.Method = 1
        self.gurobi.Params.DisplayInterval = 1
        self.gurobi.Params.MIPGap = 0
        self.gurobi.Params.MIPGapAbs = 0

    def construct(self):
        if self._constructed:
            return

        vtype = gurobi.GRB.CONTINUOUS
        if self.ilp_mode:
            vtype = gurobi.GRB.BINARY

        self.node_vars = self.gurobi.addVars(len(self.model.nodes),
                                             lb=0.0,
                                             ub=1.0,
                                             vtype=vtype,
                                             obj=self.model.nodes)
        self.clique_vars = self.gurobi.addVars(len(self.model.cliques),
                                               lb=0.0,
                                               ub=1.0,
                                               vtype=vtype,
                                               obj=0.0)
        for cidx, clique in enumerate(self.model.cliques):
            lhs = sum(self.node_vars[nidx] for nidx in clique) + self.clique_vars[cidx]
            self.gurobi.addConstr(lhs == 1)

        self.gurobi.ObjCon = self.model.constant
        self._constructed = True

    def use_mapping(self, mapping):
        for v, x in zip(self.node_vars.values(), mapping):
            if x in (0, 1):
                self.gurobi.addConstr(v == x)

    def solve(self):
        self.construct()
        self.gurobi.optimize()
        self.assignment = [var.X for var in self.node_vars.values()]
        if self.ilp_mode:
            self.assignment = [int(x > .5) for x in self.assignment]

    def write(self, filename):
        self.construct()
        self.gurobi.write(filename)
