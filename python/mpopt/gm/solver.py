from ..common.solver import BaseSolver
from . import libmpopt_gm as lib


class Solver(BaseSolver):

    def __init__(self):
        super().__init__(lib)


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


def extract_primals(model, solver):
    labeling = [None] * len(model.unaries)
    g = lib.solver_get_graph(solver.solver)

    for u, _ in enumerate(model.unaries):
        f = lib.graph_get_unary(g, u)
        labeling[u] = lib.unary_get_primal(f)

    return labeling
