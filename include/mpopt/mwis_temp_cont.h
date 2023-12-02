#ifndef LIBMPOPT_MWIS_TEMP_CONT_H
#define LIBMPOPT_MWIS_TEMP_CONT_H

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

void mpopt_mwis_solver_limit_runtime(mpopt_mwis_solver* s, double seconds);
void mpopt_mwis_solver_limit_integer_primal_gap(mpopt_mwis_solver* s, double percentage);
void mpopt_mwis_solver_limit_integer_primal_stagnation(mpopt_mwis_solver* s, double seconds);

void mpopt_mwis_solver_run(mpopt_mwis_solver* s, int batch_size, int max_batches, int greedy_iterations);
int mpopt_mwis_solver_get_iterations(mpopt_mwis_solver* s);

double mpopt_mwis_solver_get_dual_relaxed(mpopt_mwis_solver* s);
double mpopt_mwis_solver_get_primal_relaxed(mpopt_mwis_solver* s);
double mpopt_mwis_solver_get_primal(mpopt_mwis_solver* s);
void mpopt_mwis_solver_get_assignment(mpopt_mwis_solver* s, int* assignment, int size);
int mpopt_mwis_solver_get_node_assignment(mpopt_mwis_solver*, int node);

double mpopt_mwis_solver_get_constant(mpopt_mwis_solver* s);
void mpopt_mwis_solver_set_constant(mpopt_mwis_solver* s, double c);

double mpopt_mwis_solver_get_node_cost(mpopt_mwis_solver* s, int i);
void mpopt_mwis_solver_set_node_cost(mpopt_mwis_solver* s, int i, double c);

double mpopt_mwis_solver_get_clique_cost(mpopt_mwis_solver* s, int i);

double mpopt_mwis_solver_get_temperature(mpopt_mwis_solver* s);
void mpopt_mwis_solver_set_temperature(mpopt_mwis_solver* s, double v);

//double mpopt_mwis_solver_get_threshold_optimality(mpopt_mwis_solver* s);
//void mpopt_mwis_solver_set_threshold_optimality(mpopt_mwis_solver* s, double v);

//double mpopt_mwis_solver_get_threshold_stability(mpopt_mwis_solver* s);
//void mpopt_mwis_solver_set_threshold_stability(mpopt_mwis_solver* s, double v);

double mpopt_mwis_solver_get_temperature_drop_factor(mpopt_mwis_solver* s);
void mpopt_mwis_solver_set_temperature_drop_factor(mpopt_mwis_solver* s, double v);

#ifdef __cplusplus
}
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=c: */
