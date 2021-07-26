import pulp

from mpopt.utils import if_debug
from mpopt.common import gurobi
from mpopt.mwis.model import Model


class CBC:

    def __init__(self, model, ilp_mode=True, silent=True, presolve=True):
        self._constructed = False
        self.model = model
        self.ilp_mode = ilp_mode
        self.solver = pulp.COIN_CMD()
        self.solver.msg = 0 if silent else 1
        self.solver.presolve = 1 if presolve else 0
        self.lp = pulp.LpProblem('WSP', pulp.LpMaximize)

    def construct(self):
        if self._constructed:
            return

        no_nodes = len(self.model.nodes)
        no_confs = len(self.model.cliques)
        cat = pulp.LpBinary if self.ilp_mode else pulp.LpContinuous

        self.node_vars = [pulp.LpVariable(f'n{i}', 0, 1, cat) for i in range(no_nodes)]
        self.conf_vars = [pulp.LpVariable(f'd{i}', 0, 1, cat) for i in range(no_confs)]

        self.lp += sum(c * v for c, v in zip(self.model.nodes, self.node_vars)) + self.model.constant

        for cidx, clique in enumerate(self.model.cliques):
            lhs = sum(self.node_vars[nidx] for nidx in clique) + self.conf_vars[cidx]
            self.lp += lhs == 1

        self._constructed = True

    def use_mapping(self, mapping):
        for v, x in zip(self.node_vars, mapping):
            if x in (0, 1):
                self.lp += v == x

    def solve(self):
        self.construct()
        self.solver.solve(self.lp)
        self.assignment = [var.varValue for var in self.node_vars]
        if self.ilp_mode:
            self.assignment = [int(x > .5) for x in self.assignment]

    def write(self, filename):
        self.construct()
        self.lp.writeLP(filename)
