#ifndef LIBQAPOPT_QAP_H
#define LIBQAPOPT_QAP_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct qapopt_solver_t qapopt_solver;
typedef struct qapopt_graph_t qapopt_graph;
typedef struct qapopt_unary_node_t qapopt_unary_node;
typedef struct qapopt_uniqueness_node_t qapopt_uniqueness_node;
typedef struct qapopt_pairwise_node_t qapopt_pairwise_node;

qapopt_solver* qapopt_solver_create();
void qapopt_solver_destroy(qapopt_solver* s);
void qapopt_solver_finalize(qapopt_solver* s);
qapopt_graph* qapopt_solver_get_graph(qapopt_solver* s);
qapopt_unary_node* qapopt_graph_add_unary(qapopt_graph* graph, int idx, int number_of_labels, int number_of_forward, int number_of_backward);
qapopt_uniqueness_node* qapopt_graph_add_uniqueness(qapopt_graph* graph, int idx, int number_of_unaries);
qapopt_pairwise_node* qapopt_graph_add_pairwise(qapopt_graph* graph, int idx, int number_of_labels0, int number_of_labels1);
void qapopt_graph_add_pairwise_link(qapopt_graph* graph, int idx_unary0, int idx_unary1, int idx_pairwise);
void qapopt_graph_add_uniqueness_link(qapopt_graph* graph, int idx_unary, int label, int idx_uniqueness, int slot);
qapopt_unary_node* qapopt_graph_get_unary(qapopt_graph* graph, int idx);
qapopt_uniqueness_node* qapopt_graph_get_uniqueness(qapopt_graph* graph, int idx);
qapopt_pairwise_node* qapopt_graph_get_pairwise(qapopt_graph* graph, int idx);
void qapopt_solver_run(qapopt_solver* s, int batch_size, int max_batches);
void qapopt_solver_run_rounding_only(qapopt_solver* s);
void qapopt_solver_run_quiet(qapopt_solver* s, int batch_size, int max_batches);
void qapopt_solver_run_no_rounding(qapopt_solver* s, int batch_size, int max_batches);
void qapopt_solver_solve_ilp(qapopt_solver* s);
void qapopt_solver_execute_fusion_move(qapopt_solver* s, int* solution0, int* solution1, size_t length);
void qapopt_solver_execute_qpbo(qapopt_solver* s, int* solution0, int* solution1, size_t length, int enable_weak_persistency, int enable_probe, int enable_improve);
void qapopt_solver_execute_lsatr(qapopt_solver* s, int* solution0, int* solution1, size_t length);
void qapopt_solver_compute_greedy_assignment(qapopt_solver* s);
double qapopt_solver_runtime(qapopt_solver* t);
double qapopt_solver_lower_bound(qapopt_solver* s);
double qapopt_solver_evaluate_primal(qapopt_solver* s);
double qapopt_solver_get_fm_ms_build(qapopt_solver* s);
double qapopt_solver_get_fm_ms_solve(qapopt_solver* s);

void qapopt_unary_set_cost(qapopt_unary_node* n, int label, double cost);
double qapopt_unary_get_cost(qapopt_unary_node* n, int label);
int qapopt_unary_get_primal(qapopt_unary_node* n);

void qapopt_uniqueness_set_cost(qapopt_uniqueness_node* n, int unary, double cost);
double qapopt_uniqueness_get_cost(qapopt_uniqueness_node* n, int unary);
int qapopt_uniqueness_get_primal(qapopt_uniqueness_node* n);

void qapopt_pairwise_set_cost(qapopt_pairwise_node* n, int l0, int l1, double cost);
double qapopt_pairwise_get_cost(qapopt_pairwise_node* n, int l0, int l1);
int qapopt_pairwise_get_primal(qapopt_pairwise_node* n, char left_side);


#ifdef __cplusplus
}
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=c: */
