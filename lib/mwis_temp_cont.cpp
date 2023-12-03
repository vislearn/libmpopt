#include <mpopt/mwis.hpp>
#include <mpopt/mwis_temp_cont.h>

using solver_type = mpopt::mwis::original::solver;

struct mpopt_mwis_solver_t { solver_type solver; };

extern "C" {

//
// solver API
//

mpopt_mwis_solver* mpopt_mwis_solver_create() { return new mpopt_mwis_solver; }
void mpopt_mwis_solver_destroy(mpopt_mwis_solver* s) { delete s; }
void mpopt_mwis_solver_finalize(mpopt_mwis_solver* s) { s->solver.finalize(); }

int mpopt_mwis_solver_add_node(mpopt_mwis_solver* s, double cost) { return s->solver.add_node(cost); }
int mpopt_mwis_solver_add_clique(mpopt_mwis_solver* s, int* indices, int size)
{
  std::vector<mpopt::index> tmp(indices, indices + size);
  return s->solver.add_clique(tmp);
}

void mpopt_mwis_solver_run(mpopt_mwis_solver* s, int batch_size, int max_batches, int greedy_iterations) { s->solver.run(batch_size, max_batches, greedy_iterations); }
int mpopt_mwis_solver_get_iterations(mpopt_mwis_solver* s) { return s->solver.iterations(); }

double mpopt_mwis_solver_get_dual_relaxed(mpopt_mwis_solver* s) { return s->solver.dual_relaxed(); }
double mpopt_mwis_solver_get_primal(mpopt_mwis_solver* s) { return s->solver.primal(); }

void mpopt_mwis_solver_get_assignment(mpopt_mwis_solver* s, int* assignment, int size) { s->solver.assignment(assignment, assignment + size); }
int mpopt_mwis_solver_get_node_assignment(mpopt_mwis_solver* s, int node) { return s->solver.assignment(node); }

double mpopt_mwis_solver_get_constant(mpopt_mwis_solver* s) { return s->solver.constant(); }
void mpopt_mwis_solver_set_constant(mpopt_mwis_solver* s, double c) { s->solver.constant(c); }

double mpopt_mwis_solver_get_node_cost(mpopt_mwis_solver* s, int i) { return s->solver.node_cost<false>(i); }
double mpopt_mwis_solver_get_reduced_node_cost(mpopt_mwis_solver* s, int i) { return s->solver.node_cost<true>(i); }
void mpopt_mwis_solver_set_node_cost(mpopt_mwis_solver* s, int i, double c) { s->solver.node_cost(i, c); }

double mpopt_mwis_solver_get_clique_cost(mpopt_mwis_solver* s, int i) { return s->solver.clique_cost<false>(i); }
double mpopt_mwis_solver_get_reduced_clique_cost(mpopt_mwis_solver* s, int i) { return s->solver.clique_cost<true>(i); }

double mpopt_mwis_solver_get_temperature(mpopt_mwis_solver* s) { return s->solver.temperature(); }
void mpopt_mwis_solver_set_temperature(mpopt_mwis_solver* s, double v) { s->solver.temperature(v); }

//double mpopt_mwis_solver_get_threshold_optimality(mpopt_mwis_solver* s) { return s->solver.threshold_optimality(); }
//void mpopt_mwis_solver_set_threshold_optimality(mpopt_mwis_solver* s, double v) { s->solver.threshold_optimality(v); }

//double mpopt_mwis_solver_get_threshold_stability(mpopt_mwis_solver* s) { return s->solver.threshold_stability(); }
//void mpopt_mwis_solver_set_threshold_stability(mpopt_mwis_solver* s, double v) { s->solver.threshold_stability(v); }

double mpopt_mwis_solver_get_temperature_drop_factor(mpopt_mwis_solver* s) { return s->solver.temperature_drop_factor(); }
void mpopt_mwis_solver_set_temperature_drop_factor(mpopt_mwis_solver* s, double v) { s->solver.temperature_drop_factor(v); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
