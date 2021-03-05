#ifndef LIBMPOPT_MWIS_H
#define LIBMPOPT_MWIS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mpopt_mwis_solver_t mpopt_mwis_solver;

//
// solver API
//

mpopt_mwis_solver* mpopt_mwis_solver_create();
void mpopt_mwis_solver_destroy(mpopt_mwis_solver* s);
void mpopt_mwis_solver_finalize(mpopt_mwis_solver* s);

int mpopt_mwis_solver_add_node(mpopt_mwis_solver* s, double cost);
int mpopt_mwis_solver_add_clique(mpopt_mwis_solver* s, int* indices, int size);

void mpopt_mwis_solver_run(mpopt_mwis_solver* s, int batch_size, int max_batches);
//double mpopt_mwis_solver_lower_bound(mpopt_mwis_solver* s);

#ifdef __cplusplus
}
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=c: */
