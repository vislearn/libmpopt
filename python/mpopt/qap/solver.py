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


def construct_solver(model):
    raise NotImplementedError
