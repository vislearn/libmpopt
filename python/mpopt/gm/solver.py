from . import libmpopt_gm as lib


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

    def solve_ilp(self):
        lib.solver_solve_ilp(self.solver)


def construct_solver(model):
    s = Solver()
    g = lib.solver_get_graph(s.solver)

    for u, data in enumerate(model.unaries):
        f = lib.graph_add_unary(g, u, len(data), model.no_forward[u], model.no_backward[u])
        for i, c in enumerate(data):
            lib.unary_set_cost(f, i, c)

    for i, (u, v, data) in enumerate(model.pairwise):
        f = lib.graph_add_pairwise(g, i, len(model.unaries[u]), len(model.unaries[v]))
        lib.graph_add_pairwise_link(g, u, v, i)
        for l_u in range(len(model.unaries[u])):
            for l_v in range(len(model.unaries[v])):
                lib.pairwise_set_cost(f, l_u, l_v, data[l_u, l_v])

    lib.solver_finalize(s.solver)
    return s
