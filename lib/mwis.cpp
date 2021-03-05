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
//double mpopt_mwis_solver_lower_bound(mpopt_mwis_solver* s) { return s->solver.lower_bound(); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
