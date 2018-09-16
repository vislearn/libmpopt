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

void ct_tracker_add_conflict_link(ct_tracker* t, const int timestep, const int conflict, const int slot, const int detection)
{
  t->tracker.add_conflict_link(timestep, conflict, slot, detection);
}

ct_conflict* ct_tracker_get_conflict(ct_tracker* t, int timestep, int conflict)
{
  auto* e = t->tracker.conflict(timestep, conflict);
  return to_conflict(e);
}

void ct_tracker_run(ct_tracker* t, int max_iterations) { t->tracker.run(max_iterations); }
double ct_tracker_lower_bound(ct_tracker* t) { return t->tracker.lower_bound(); }

//
// detection API
//

void ct_detection_set_detection_cost(ct_detection* d, double on, double off) { from_detection(d)->set_detection_cost(on, off); }
void ct_detection_set_appearance_cost(ct_detection* d, double c) { from_detection(d)->set_appearance_cost(c); }
void ct_detection_set_disappearance_cost(ct_detection* d, double c) { from_detection(d)->set_disappearance_cost(c); }
void ct_detection_set_incoming_cost(ct_detection* d, int idx, double c) { from_detection(d)->set_incoming_cost(idx, c); }
void ct_detection_set_outgoing_cost(ct_detection* d, int idx, double c) { from_detection(d)->set_outgoing_cost(idx, c); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
