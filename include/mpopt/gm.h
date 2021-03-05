#ifndef LIBMPOPT_GM_H
#define LIBMPOPT_GM_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mpopt_gm_solver_t mpopt_gm_solver;
typedef struct mpopt_gm_graph_t mpopt_gm_graph;
typedef struct mpopt_gm_unary_node_t mpopt_gm_unary_node;
typedef struct mpopt_gm_pairwise_node_t mpopt_gm_pairwise_node;

mpopt_gm_solver* mpopt_gm_solver_create();
void mpopt_gm_solver_destroy(mpopt_gm_solver* s);
void mpopt_gm_solver_finalize(mpopt_gm_solver* s);
mpopt_gm_graph* mpopt_gm_solver_get_graph(mpopt_gm_solver* s);
mpopt_gm_unary_node* mpopt_gm_graph_add_unary(mpopt_gm_graph* graph, int idx, int number_of_labels, int number_of_forward, int number_of_backward);
mpopt_gm_unary_node* mpopt_gm_graph_get_unary(mpopt_gm_graph* graph, int idx);
mpopt_gm_pairwise_node* mpopt_gm_graph_add_pairwise(mpopt_gm_graph* graph, int idx, int number_of_labels0, int number_of_labels1);
void mpopt_gm_graph_add_pairwise_link(mpopt_gm_graph* graph, int idx_unary0, int idx_unary1, int idx_pairwise);
void mpopt_gm_solver_run(mpopt_gm_solver* s, int batch_size, int max_batches);
void mpopt_gm_solver_solve_ilp(mpopt_gm_solver* s);
void mpopt_gm_solver_execute_combilp(mpopt_gm_solver* s);
double mpopt_gm_solver_runtime(mpopt_gm_solver* s);
double mpopt_gm_solver_lower_bound(mpopt_gm_solver* s);
double mpopt_gm_solver_evaluate_primal(mpopt_gm_solver* s);

void mpopt_gm_unary_set_cost(mpopt_gm_unary_node* n, int label, double cost);
int mpopt_gm_unary_get_primal(mpopt_gm_unary_node* n);
void mpopt_gm_pairwise_set_cost(mpopt_gm_pairwise_node* n, int l0, int l1, double cost);

#ifdef __cplusplus
}
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=c: */
