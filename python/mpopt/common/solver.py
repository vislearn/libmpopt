class BaseSolver:

    def __init__(self, lib):
        self.lib = lib
        self.solver = lib.solver_create()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.solver is not None:
            self.lib.solver_destroy(self.solver)
            self.solver = None

    def lower_bound(self):
        return self.lib.solver_lower_bound(self.solver)

    def evaluate_primal(self):
        return self.lib.solver_evaluate_primal(self.solver)

    def run(self, max_iterations=1000):
        self.lib.solver_run(self.solver, max_iterations)

    def solve_ilp(self):
        self.lib.solver_solve_ilp(self.solver)

    def execute_combilp(self):
        self.lib.solver_execute_combilp(self.solver)
