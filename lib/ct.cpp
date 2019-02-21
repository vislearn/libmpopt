#include <ct/all.hpp>
#include <ct/ct.h>

using allocator_type = ct::block_allocator<ct::cost>;
using tracker_type = ct::tracker<allocator_type>;
using detection_type = ct::detection_factor<allocator_type>;
using conflict_type = ct::conflict_factor<allocator_type>;

struct ct_tracker_t {
  ct::memory_block memory;
  allocator_type allocator;
  tracker_type tracker;

  ct_tracker_t()
  : memory()
  , allocator(memory)
  , tracker(allocator)
  { }
};

inline auto* to_detection(detection_type* d) { return reinterpret_cast<ct_detection*>(d); }
inline auto* from_detection(ct_detection* d) { return reinterpret_cast<detection_type*>(d); }

inline auto* to_conflict(conflict_type* d) { return reinterpret_cast<ct_conflict*>(d); }
inline auto* from_conflict(ct_conflict* d) { return reinterpret_cast<conflict_type*>(d); }

extern "C" {

//
// tracker API
//

ct_tracker* ct_tracker_create() { return new ct_tracker; }
void ct_tracker_destroy(ct_tracker* t) { delete t; }
void ct_tracker_finalize(ct_tracker* t) { t->memory.finalize(); }

ct_detection* ct_tracker_add_detection(ct_tracker* t, int timestep, int detection, int number_of_incoming, int number_of_outgoing)
{
  auto* d = t->tracker.add_detection(timestep, detection, number_of_incoming, number_of_outgoing);
  return to_detection(d);
}

ct_conflict* ct_tracker_add_conflict(ct_tracker* t, int timestep, int conflict, int number_of_detections)
{
  auto* e = t->tracker.add_conflict(timestep, conflict, number_of_detections);
  return to_conflict(e);
}

ct_detection* ct_tracker_get_detection(ct_tracker* t, int timestep, int detection)
{
  auto* d = t->tracker.detection(timestep, detection);
  return to_detection(d);
}

void ct_tracker_add_transition(ct_tracker* t, int timestep_from, int detection_from, int index_from, int detection_to, int index_to)
{
  t->tracker.add_transition(timestep_from, detection_from, index_from, detection_to, index_to);
}

void ct_tracker_add_division(ct_tracker* t, int timestep_from, int detection_from, int index_from, int detection_to_1, int index_to_1, int detection_to_2, int index_to_2)
{
  t->tracker.add_division(timestep_from, detection_from, index_from, detection_to_1, index_to_1, detection_to_2, index_to_2);
}

void ct_tracker_add_conflict_link(ct_tracker* t, const int timestep, const int conflict, const int slot, const int detection, const double weight)
{
  t->tracker.add_conflict_link(timestep, conflict, slot, detection, weight);
}

ct_conflict* ct_tracker_get_conflict(ct_tracker* t, int timestep, int conflict)
{
  auto* e = t->tracker.conflict(timestep, conflict);
  return to_conflict(e);
}

void ct_tracker_run(ct_tracker* t, int max_iterations) { t->tracker.run(max_iterations); }
double ct_tracker_lower_bound(ct_tracker* t) { return t->tracker.lower_bound(); }
double ct_tracker_upper_bound(ct_tracker* t) { return t->tracker.upper_bound(); }
void ct_tracker_forward_step(ct_tracker* t, int timestep) { t->tracker.single_step<true>(timestep); }
void ct_tracker_backward_step(ct_tracker* t, int timestep) { t->tracker.single_step<false>(timestep); }

//
// detection API
//

void ct_detection_set_detection_cost(ct_detection* d, double on) { from_detection(d)->set_detection_cost(on); }
void ct_detection_set_appearance_cost(ct_detection* d, double c) { from_detection(d)->set_appearance_cost(c); }
void ct_detection_set_disappearance_cost(ct_detection* d, double c) { from_detection(d)->set_disappearance_cost(c); }
void ct_detection_set_incoming_cost(ct_detection* d, int idx, double c) { from_detection(d)->set_incoming_cost(idx, c); }
void ct_detection_set_outgoing_cost(ct_detection* d, int idx, double c) { from_detection(d)->set_outgoing_cost(idx, c); }

double ct_detection_get_detection_cost(ct_detection* d) { return from_detection(d)->detection(); }
double ct_detection_get_appearance_cost(ct_detection* d) { return from_detection(d)->appearance(); }
double ct_detection_get_disappearance_cost(ct_detection* d) { return from_detection(d)->disappearance(); }
double ct_detection_get_incoming_cost(ct_detection* d, int idx) { return from_detection(d)->incoming(idx); }
double ct_detection_get_outgoing_cost(ct_detection* d, int idx) { return from_detection(d)->outgoing(idx); }

int ct_detection_get_incoming_primal(ct_detection* d)
{
  auto p = from_detection(d)->primal().incoming();
  if (p == ct::detection_primal::undecided || p == ct::detection_primal::off)
    return -1;
  else
    return p;
}

int ct_detection_get_outgoing_primal(ct_detection* d)
{
  auto p = from_detection(d)->primal().outgoing();
  if (p == ct::detection_primal::undecided || p == ct::detection_primal::off)
    return -1;
  else
    return p;
}

//
// conflict API
//

void ct_conflict_set_cost(ct_conflict* c, int idx, double cost) { from_conflict(c)->set(idx, cost); }
double ct_conflict_get_cost(ct_conflict* c, int idx) { return from_conflict(c)->get(idx); }
int ct_conflict_get_primal(ct_conflict* c) { return from_conflict(c)->primal(); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
