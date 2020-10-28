#ifndef LIBMPOPT_CT_H
#define LIBMPOPT_CT_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mpopt_ct_tracker_t mpopt_ct_tracker;
typedef struct mpopt_ct_graph_t mpopt_ct_graph;
typedef struct mpopt_ct_detection_t mpopt_ct_detection;
typedef struct mpopt_ct_conflict_t mpopt_ct_conflict;

//
// tracker API
//

mpopt_ct_tracker* mpopt_ct_tracker_create();
void mpopt_ct_tracker_destroy(mpopt_ct_tracker* t);
void mpopt_ct_tracker_finalize(mpopt_ct_tracker* t);

mpopt_ct_graph* mpopt_ct_tracker_get_graph(mpopt_ct_tracker* t);
mpopt_ct_detection* mpopt_ct_graph_add_detection(mpopt_ct_graph* g, int timestep, int detection, int number_of_incoming, int number_of_outgoing, int number_of_conflicts);
mpopt_ct_conflict* mpopt_ct_graph_add_conflict(mpopt_ct_graph* g, int timestep, int conflict, int number_of_detections);
mpopt_ct_detection* mpopt_ct_graph_get_detection(mpopt_ct_graph* g, int timestep, int detection);
void mpopt_ct_graph_add_transition(mpopt_ct_graph* g, int timestep_from, int detection_from, int index_from, int detection_to, int index_to);
void mpopt_ct_graph_add_division(mpopt_ct_graph* g, int timestep_from, int detection_from, int index_from, int detection_to_1, int index_to_1, int detection_to_2, int index_to_2);
void mpopt_ct_graph_add_conflict_link(mpopt_ct_graph* g, int timestep, int conflict, int conflict_slot, int detection, int detection_slot);
mpopt_ct_conflict* mpopt_ct_graph_get_conflict(mpopt_ct_graph* g, int timestep, int conflict);

void mpopt_ct_tracker_run(mpopt_ct_tracker* t, int batch_size, int max_batches);
double mpopt_ct_tracker_runtime(mpopt_ct_tracker* t);
double mpopt_ct_tracker_lower_bound(mpopt_ct_tracker* t);
double mpopt_ct_tracker_evaluate_primal(mpopt_ct_tracker* t);
void mpopt_ct_tracker_forward_step(mpopt_ct_tracker* t, int timestep);
void mpopt_ct_tracker_backward_step(mpopt_ct_tracker* t, int timestep);

void mpopt_ct_tracker_solve_ilp(mpopt_ct_tracker* t);
void mpopt_ct_tracker_execute_combilp(mpopt_ct_tracker* t);

//
// detection API
//

void mpopt_ct_detection_set_detection_cost(mpopt_ct_detection* d, double on);
void mpopt_ct_detection_set_appearance_cost(mpopt_ct_detection* d, double c);
void mpopt_ct_detection_set_disappearance_cost(mpopt_ct_detection* d, double c);
void mpopt_ct_detection_set_incoming_cost(mpopt_ct_detection* d, int idx, double c);
void mpopt_ct_detection_set_outgoing_cost(mpopt_ct_detection* d, int idx, double c);

double mpopt_ct_detection_get_detection_cost(mpopt_ct_detection* d);
double mpopt_ct_detection_get_appearance_cost(mpopt_ct_detection* d);
double mpopt_ct_detection_get_disappearance_cost(mpopt_ct_detection* d);
double mpopt_ct_detection_get_incoming_cost(mpopt_ct_detection* d, int idx);
double mpopt_ct_detection_get_outgoing_cost(mpopt_ct_detection* d, int idx);

int mpopt_ct_detection_get_incoming_primal(mpopt_ct_detection* d);
int mpopt_ct_detection_get_outgoing_primal(mpopt_ct_detection* d);

//
// conflict API
//

void mpopt_ct_conflict_set_cost(mpopt_ct_conflict* c, int idx, double cost);
double mpopt_ct_conflict_get_cost(mpopt_ct_conflict* c, int idx);
int mpopt_ct_conflict_get_primal(mpopt_ct_conflict* c);

#ifdef __cplusplus
}
#endif

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=c: */
