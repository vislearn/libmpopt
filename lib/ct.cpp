#include <mpopt/ct.hpp>
#include <mpopt/ct.h>

using allocator_type = mpopt::block_allocator<mpopt::cost>;
using tracker_type = mpopt::ct::tracker<allocator_type>;
using graph_type = mpopt::ct::graph<allocator_type>;
using detection_type = mpopt::ct::detection_node<allocator_type>;
using conflict_type = mpopt::ct::conflict_node<allocator_type>;

struct mpopt_ct_tracker_t {
  mpopt::memory_block memory;
  allocator_type allocator;
  tracker_type tracker;

  mpopt_ct_tracker_t()
  : memory()
  , allocator(memory)
  , tracker(allocator)
  { }
};

inline auto* to_graph(graph_type* g) { return reinterpret_cast<mpopt_ct_graph*>(g); }
inline auto* from_graph(mpopt_ct_graph* g) { return reinterpret_cast<graph_type*>(g); }

inline auto* to_detection(detection_type* d) { return reinterpret_cast<mpopt_ct_detection*>(d); }
inline auto* from_detection(mpopt_ct_detection* d) { return reinterpret_cast<detection_type*>(d); }

inline auto* to_conflict(conflict_type* d) { return reinterpret_cast<mpopt_ct_conflict*>(d); }
inline auto* from_conflict(mpopt_ct_conflict* d) { return reinterpret_cast<conflict_type*>(d); }

extern "C" {

//
// tracker API
//

mpopt_ct_tracker* mpopt_ct_tracker_create() { return new mpopt_ct_tracker; }
void mpopt_ct_tracker_destroy(mpopt_ct_tracker* t) { delete t; }
void mpopt_ct_tracker_finalize(mpopt_ct_tracker* t) { t->memory.finalize(); }

mpopt_ct_graph* mpopt_ct_tracker_get_graph(mpopt_ct_tracker* t) { return to_graph(&t->tracker.get_graph()); }

mpopt_ct_detection* mpopt_ct_graph_add_detection(mpopt_ct_graph* g, int timestep, int detection, int number_of_incoming, int number_of_outgoing, int number_of_conflicts)
{
  auto* d = from_graph(g)->add_detection(timestep, detection, number_of_incoming, number_of_outgoing, number_of_conflicts);
  return to_detection(d);
}

mpopt_ct_conflict* mpopt_ct_graph_add_conflict(mpopt_ct_graph* g, int timestep, int conflict, int number_of_detections)
{
  auto* c = from_graph(g)->add_conflict(timestep, conflict, number_of_detections);
  return to_conflict(c);
}

mpopt_ct_detection* mpopt_ct_graph_get_detection(mpopt_ct_graph* g, int timestep, int detection)
{
  auto* d = from_graph(g)->detection(timestep, detection);
  return to_detection(d);
}

void mpopt_ct_graph_add_transition(mpopt_ct_graph* g, int timestep_from, int detection_from, int index_from, int detection_to, int index_to)
{
  from_graph(g)->add_transition(timestep_from, detection_from, index_from, detection_to, index_to);
}

void mpopt_ct_graph_add_division(mpopt_ct_graph* g, int timestep_from, int detection_from, int index_from, int detection_to_1, int index_to_1, int detection_to_2, int index_to_2)
{
  from_graph(g)->add_division(timestep_from, detection_from, index_from, detection_to_1, index_to_1, detection_to_2, index_to_2);
}

void mpopt_ct_graph_add_conflict_link(mpopt_ct_graph* g, int timestep, int conflict, int conflict_slot, int detection, int detection_slot)
{
  from_graph(g)->add_conflict_link(timestep, conflict, conflict_slot, detection, detection_slot);
}

mpopt_ct_conflict* mpopt_ct_graph_get_conflict(mpopt_ct_graph* g, int timestep, int conflict)
{
  auto* e = from_graph(g)->conflict(timestep, conflict);
  return to_conflict(e);
}

void mpopt_ct_tracker_run(mpopt_ct_tracker* t, int max_iterations) { t->tracker.run(max_iterations); }
double mpopt_ct_tracker_runtime(mpopt_ct_tracker* t) { return t->tracker.runtime(); }
double mpopt_ct_tracker_lower_bound(mpopt_ct_tracker* t) { return t->tracker.lower_bound(); }
double mpopt_ct_tracker_evaluate_primal(mpopt_ct_tracker* t) { return t->tracker.evaluate_primal(); }
void mpopt_ct_tracker_forward_step(mpopt_ct_tracker* t, int timestep) { t->tracker.single_step<true>(timestep); }
void mpopt_ct_tracker_backward_step(mpopt_ct_tracker* t, int timestep) { t->tracker.single_step<false>(timestep); }

void mpopt_ct_tracker_solve_ilp(mpopt_ct_tracker* t) { t->tracker.solve_ilp(); }
void mpopt_ct_tracker_execute_combilp(mpopt_ct_tracker* t) { t->tracker.execute_combilp(); }

//
// detection API
//

void mpopt_ct_detection_set_detection_cost(mpopt_ct_detection* d, double on) { from_detection(d)->factor.set_detection_cost(on); }
void mpopt_ct_detection_set_appearance_cost(mpopt_ct_detection* d, double c) { from_detection(d)->factor.set_appearance_cost(c); }
void mpopt_ct_detection_set_disappearance_cost(mpopt_ct_detection* d, double c) { from_detection(d)->factor.set_disappearance_cost(c); }
void mpopt_ct_detection_set_incoming_cost(mpopt_ct_detection* d, int idx, double c) { from_detection(d)->factor.set_incoming_cost(idx, c); }
void mpopt_ct_detection_set_outgoing_cost(mpopt_ct_detection* d, int idx, double c) { from_detection(d)->factor.set_outgoing_cost(idx, c); }

double mpopt_ct_detection_get_detection_cost(mpopt_ct_detection* d) { return from_detection(d)->factor.detection(); }
double mpopt_ct_detection_get_appearance_cost(mpopt_ct_detection* d) { return from_detection(d)->factor.appearance(); }
double mpopt_ct_detection_get_disappearance_cost(mpopt_ct_detection* d) { return from_detection(d)->factor.disappearance(); }
double mpopt_ct_detection_get_incoming_cost(mpopt_ct_detection* d, int idx) { return from_detection(d)->factor.incoming(idx); }
double mpopt_ct_detection_get_outgoing_cost(mpopt_ct_detection* d, int idx) { return from_detection(d)->factor.outgoing(idx); }

int mpopt_ct_detection_get_incoming_primal(mpopt_ct_detection* d)
{
  auto p = from_detection(d)->factor.primal().incoming();
  if (p == mpopt::ct::detection_primal::undecided || p == mpopt::ct::detection_primal::off)
    return -1;
  else
    return p;
}

int mpopt_ct_detection_get_outgoing_primal(mpopt_ct_detection* d)
{
  auto p = from_detection(d)->factor.primal().outgoing();
  if (p == mpopt::ct::detection_primal::undecided || p == mpopt::ct::detection_primal::off)
    return -1;
  else
    return p;
}

//
// conflict API
//

void mpopt_ct_conflict_set_cost(mpopt_ct_conflict* c, int idx, double cost) { from_conflict(c)->factor.set(idx, cost); }
double mpopt_ct_conflict_get_cost(mpopt_ct_conflict* c, int idx) { return from_conflict(c)->factor.get(idx); }
int mpopt_ct_conflict_get_primal(mpopt_ct_conflict* c) { return from_conflict(c)->factor.primal().get(); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
