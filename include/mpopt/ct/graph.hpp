#ifndef LIBMPOPT_CT_GRAPH_HPP
#define LIBMPOPT_CT_GRAPH_HPP

namespace mpopt {
namespace ct {

constexpr index max_number_of_detection_edges = 128;

class factor_counter {
public:
#ifndef NDEBUG
  factor_counter()
  : timestep_(0)
  , detection_(0)
  , conflict_(0)
  { }
#endif

  void new_detection(index timestep, index detection)
  {
#ifndef NDEBUG
    if (timestep_ != timestep)
      advance_timestep();

    assert(timestep_ == timestep);
    assert(conflict_ == 0);
    assert(detection_ == detection);
    ++detection_;
#endif
  }

  void new_conflict(index timestep, index conflict)
  {
#ifndef NDEBUG
    if (timestep_ != timestep)
      advance_timestep();

    assert(timestep_ == timestep);
    assert(conflict_ == conflict);
    ++conflict_;
#endif
  }

protected:
  void advance_timestep()
  {
#ifndef NDEBUG
    ++timestep_;
    detection_ = 0;
    conflict_ = 0;
#endif
  }

#ifndef NDEBUG
  index timestep_;
  index detection_;
  index conflict_;
#endif
};


template<typename> struct detection_node;
template<typename> struct conflict_node;

//
// NOTE: In `is_prepared` we also check for cycle inconsistencies in the
//       graph. We only check the pointers to the node structs, but not check
//       the slot indices explicitly. (As long as every edge is only there
//       once, it should be equivalent. We do not want to have the same edge
//       multiple times anyway.)
//

template<typename NODE_TYPE>
struct transition_edge {
  using node_type = NODE_TYPE;

  transition_edge()
  : node1(nullptr)
  , node2(nullptr)
  { }

  bool is_division() const
  {
    assert(node1 != nullptr);
    return node2 != nullptr;
  }

  bool is_prepared() const
  {
    return node1 != nullptr;
  }

  node_type* node1;
  node_type* node2;
  index slot1, slot2;
};


template<typename NODE_TYPE>
struct conflict_edge {
  using node_type = NODE_TYPE;

  conflict_edge()
  : node(nullptr)
  { }

  bool is_prepared() const
  {
    return node != nullptr;
  }

  node_type* node;
  index slot;
};


template<typename ALLOCATOR>
struct detection_node {
  using allocator_type = ALLOCATOR;
  using node_type = detection_node<ALLOCATOR>;
  using detection_type = detection_factor<ALLOCATOR>;
  using conflict_type = conflict_node<ALLOCATOR>;

  mutable detection_type detection;
  fixed_vector_alloc_gen<transition_edge<node_type>, ALLOCATOR> incoming;
  fixed_vector_alloc_gen<transition_edge<node_type>, ALLOCATOR> outgoing;
  fixed_vector_alloc_gen<conflict_edge<conflict_type>, ALLOCATOR> conflicts;

  detection_node(index number_of_incoming, index number_of_outgoing, index number_of_conflicts, const allocator_type& allocator)
  : detection(number_of_incoming, number_of_outgoing, allocator)
  , incoming(number_of_incoming, allocator)
  , outgoing(number_of_outgoing, allocator)
  , conflicts(number_of_conflicts, allocator)
  { }

  template<bool to_right>
  auto& transitions() const
  {
    return to_right ? outgoing : incoming;
  }

  template<bool to_right, typename FUNCTOR>
  void traverse_transitions(FUNCTOR f) const
  {
    index slot = 0;
    for (const auto& edge : transitions<to_right>()) {
      f(edge, slot);
      ++slot;
    }
  }

  void check_structure() const
  {
    assert(detection.is_prepared());

    for (const auto& edge : incoming) {
      assert(edge.is_prepared());
      assert(edge.node1->outgoing[edge.slot1].node1 == this ||
             edge.node1->outgoing[edge.slot1].node2 == this);
      assert(!edge.is_division() ||
             edge.node2->incoming[edge.slot2].node1 == edge.node1);
      assert(!edge.is_division() ||
             edge.node2->incoming[edge.slot2].node2 == this);
    }

    for (const auto& edge : outgoing)
    {
      assert(edge.is_prepared());
      assert(edge.node1->incoming[edge.slot1].node1 == this);
      assert(!edge.is_division() ||
             edge.node2->incoming[edge.slot2].node1 == this);
    }

    for (const auto& edge : conflicts) {
      assert(edge.is_prepared());
      assert(edge.node->detections[edge.slot].node == this);
    }
  }
};


template<typename ALLOCATOR>
struct conflict_node {
  using allocator_type = ALLOCATOR;
  using node_type = conflict_node<ALLOCATOR>;
  using conflict_type = conflict_factor<ALLOCATOR>;
  using detection_type = detection_node<ALLOCATOR>;

  mutable conflict_type conflict;
  fixed_vector_alloc_gen<conflict_edge<detection_type>, ALLOCATOR> detections;

  conflict_node(index number_of_detections, const allocator_type& allocator)
  : conflict(number_of_detections, allocator)
  , detections(number_of_detections, allocator)
  { }

  template<typename FUNCTOR>
  void traverse_detections(FUNCTOR f) const
  {
    index slot = 0;
    for (const auto& edge : detections) {
      f(edge, slot);
      ++slot;
    }
  }

  void check_structure() const
  {
    assert(conflict.is_prepared());

    for (const auto& edge : detections) {
      assert(edge.is_prepared());
      assert(edge.node->conflicts[edge.slot].node == this);
    }
  }
};


template<typename ALLOCATOR>
struct timestep {
  using allocator_type = ALLOCATOR;
  using detection_type = detection_node<ALLOCATOR>;
  using conflict_type = conflict_node<ALLOCATOR>;

  std::vector<detection_type*> detections;
  std::vector<conflict_type*> conflicts;
};


template<typename ALLOCATOR>
class graph {
public:
  using allocator_type = ALLOCATOR;
  using detection_node_type = detection_node<ALLOCATOR>;
  using detection_type = typename detection_node_type::detection_type;
  using conflict_node_type = conflict_node<ALLOCATOR>;
  using conflict_type = typename conflict_node_type::conflict_type;
  using timestep_type = timestep<ALLOCATOR>;

  graph(const ALLOCATOR& allocator = ALLOCATOR())
  : allocator_(allocator)
  { }

  const auto& timesteps() const { return timesteps_; }

  detection_node_type* add_detection(index timestep, index detection, index number_of_incoming, index number_of_outgoing, index number_of_conflicts)
  {
    assert(number_of_incoming >= 0 && number_of_incoming <= max_number_of_detection_edges);
    assert(number_of_outgoing >= 0 && number_of_outgoing <= max_number_of_detection_edges);
    factor_counter_.new_detection(timestep, detection);

    if (timestep >= timesteps_.size())
      timesteps_.resize(timestep + 1);
    auto& detections = timesteps_[timestep].detections;

    if (detection >= detections.size())
      detections.resize(detection + 1);
    auto& node = detections[detection];

    typename std::allocator_traits<allocator_type>::template rebind_alloc<detection_node_type> a(allocator_);
    node = a.allocate();
    new (node) detection_node_type(number_of_incoming, number_of_outgoing, number_of_conflicts, allocator_);
#ifndef NDEBUG
    node->detection.set_debug_info(timestep, detection);
#endif

    return node;
  }

  conflict_node_type* add_conflict(index timestep, index conflict, index number_of_detections)
  {
    assert(number_of_detections >= 2);
    factor_counter_.new_conflict(timestep, conflict);

    auto& conflicts = timesteps_[timestep].conflicts;
    if (conflict >= conflicts.size())
      conflicts.resize(conflict + 1);
    auto& node = conflicts[conflict];

    typename std::allocator_traits<allocator_type>::template rebind_alloc<conflict_node_type> a(allocator_);
    node = a.allocate();
    new (node) conflict_node_type(number_of_detections, allocator_); // FIXME: Dtor is never called.
#ifndef NDEBUG
    node->conflict.set_debug_info(timestep, conflict);
#endif

    return node;
  }

  void add_transition(index timestep_from, index detection_from, index slot_from, index detection_to, index slot_to)
  {
    auto* from_node = timesteps_[timestep_from].detections[detection_from];
    auto* to_node = timesteps_[timestep_from+1].detections[detection_to];

    assert(from_node->outgoing[slot_from].node1 == nullptr);
    assert(from_node->outgoing[slot_from].node2 == nullptr);
    from_node->outgoing[slot_from].node1 = to_node;
    from_node->outgoing[slot_from].slot1 = slot_to;

    assert(to_node->incoming[slot_to].node1 == nullptr);
    assert(to_node->incoming[slot_to].node2 == nullptr);
    to_node->incoming[slot_to].node1 = from_node;
    to_node->incoming[slot_to].slot1 = slot_from;
  }

  void add_division(index timestep_from, index detection_from, index slot_from, index detection_to_1, index slot_to_1, index detection_to_2, index slot_to_2)
  {
    auto* from_node = timesteps_[timestep_from].detections[detection_from];
    auto* to_node_1 = timesteps_[timestep_from+1].detections[detection_to_1];
    auto* to_node_2 = timesteps_[timestep_from+1].detections[detection_to_2];

    assert(from_node->outgoing[slot_from].node1 == nullptr);
    assert(from_node->outgoing[slot_from].node2 == nullptr);
    from_node->outgoing[slot_from].node1 = to_node_1;
    from_node->outgoing[slot_from].slot1 = slot_to_1;
    from_node->outgoing[slot_from].node2 = to_node_2;
    from_node->outgoing[slot_from].slot2 = slot_to_2;

    assert(to_node_1->incoming[slot_to_1].node1 == nullptr);
    assert(to_node_1->incoming[slot_to_1].node2 == nullptr);
    to_node_1->incoming[slot_to_1].node1 = from_node;
    to_node_1->incoming[slot_to_1].slot1 = slot_from;
    to_node_1->incoming[slot_to_1].node2 = to_node_2;
    to_node_1->incoming[slot_to_1].slot2 = slot_to_2;

    assert(to_node_2->incoming[slot_to_2].node1 == nullptr);
    assert(to_node_2->incoming[slot_to_2].node2 == nullptr);
    to_node_2->incoming[slot_to_2].node1 = from_node;
    to_node_2->incoming[slot_to_2].slot1 = slot_from;
    to_node_2->incoming[slot_to_2].node2 = to_node_1;
    to_node_2->incoming[slot_to_2].slot2 = slot_to_1;
  }

  void add_conflict_link(index timestep, index conflict, index conflict_slot, index detection, index detection_slot)
  {
    auto* node_conflict = timesteps_[timestep].conflicts[conflict];
    auto* node_detection = timesteps_[timestep].detections[detection];

    assert(node_conflict->detections[conflict_slot].node == nullptr);
    node_conflict->detections[conflict_slot].node = node_detection;
    node_conflict->detections[conflict_slot].slot = detection_slot;

    assert(node_detection->conflicts[detection_slot].node == nullptr);
    node_detection->conflicts[detection_slot].node = node_conflict;
    node_detection->conflicts[detection_slot].slot = conflict_slot;
  }

  detection_node_type* detection(index timestep, index detection)
  {
    return timesteps_[timestep].detections[detection];
  }

  conflict_node_type* conflict(index timestep, index conflict)
  {
    return timesteps_[timestep].conflicts[conflict];
  }

  size_t number_of_detections()
  {
    return std::accumulate(
      timesteps_.cbegin(), timesteps_.cend(), 0,
      [](size_t acc, const auto& t) {
        return acc + t.detections.size();
      });
  }

  size_t number_of_conflicts()
  {
    return std::accumulate(
      timesteps_.cbegin(), timesteps_.cend(), 0,
      [](size_t acc, const auto& t) {
        return acc + t.conflicts.size();
      });
  }

  void check_structure() const
  {
    for (auto& timestep : timesteps_) {
      for (auto* node : timestep.detections)
        node->check_structure();

      for (auto* node : timestep.conflicts)
        node->check_structure();
    }
  }

protected:
  allocator_type allocator_;
  factor_counter factor_counter_;
  std::vector<timestep_type> timesteps_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
