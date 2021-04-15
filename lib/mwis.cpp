#include <mpopt/mwis.hpp>
#include <mpopt/mwis.h>

using solver_type = mpopt::mwis::solver;

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

void mpopt_mwis_solver_run(mpopt_mwis_solver* s, int batch_size, int max_batches) { s->solver.run(batch_size, max_batches); }

double mpopt_mwis_solver_get_dual_relaxed(mpopt_mwis_solver* s) { return s->solver.dual_relaxed(); }
double mpopt_mwis_solver_get_primal_relaxed(mpopt_mwis_solver* s) { return s->solver.primal_relaxed(); }
double mpopt_mwis_solver_get_primal(mpopt_mwis_solver* s) { return s->solver.primal(); }

void mpopt_mwis_solver_get_assignment(mpopt_mwis_solver* s, int* assignment, int size) { s->solver.assignment(assignment, assignment + size); }
int mpopt_mwis_solver_get_node_assignment(mpopt_mwis_solver* s, int node) { return s->solver.assignment(node); }

double mpopt_mwis_solver_get_constant(mpopt_mwis_solver* s) { return s->solver.constant(); }
void mpopt_mwis_solver_set_constant(mpopt_mwis_solver* s, double c) { s->solver.constant(c); }

double mpopt_mwis_solver_get_node_cost(mpopt_mwis_solver* s, int i) { return s->solver.node_cost(i); }
void mpopt_mwis_solver_set_node_cost(mpopt_mwis_solver* s, int i, double c) { s->solver.node_cost(i, c); }

double mpopt_mwis_solver_get_clique_cost(mpopt_mwis_solver* s, int i) { return s->solver.clique_cost(i); }
void mpopt_mwis_set_clique_cost(mpopt_mwis_solver* s, int i, double c) { s->solver.clique_cost(i, c); }

double mpopt_mwis_solver_get_gamma(mpopt_mwis_solver* s) { return s->solver.gamma(); }
void mpopt_mwis_solver_set_gamma(mpopt_mwis_solver* s, double g) { s->solver.gamma(g); }

double mpopt_mwis_solver_get_temperature(mpopt_mwis_solver* s) { return s->solver.temperature(); }
void mpopt_mwis_solver_set_temperature(mpopt_mwis_solver* s, double t) { s->solver.temperature(t); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
