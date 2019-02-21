#ifndef LIBCT_TRACKER_HPP
#define LIBCT_TRACKER_HPP

namespace ct {

class factor_counter {
public:
#ifndef NDEBUG
  factor_counter()
  : timestep_(0)
  , detection_(0)
  , conflict_(0)
  { }
#endif

  void new_detection(const index timestep, const index detection)
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

  void new_conflict(const index timestep, const index conflict)
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

template<typename ALLOCATOR = std::allocator<cost>>
class tracker {
public:
  using detection_type = detection_factor<ALLOCATOR>;
  using conflict_type = conflict_factor<ALLOCATOR>;
  using transition_type = transition_messages<ALLOCATOR>;
  using conflict_messages_type = conflict_messages<ALLOCATOR>;

  static constexpr int config_batch = 100;

  tracker(const ALLOCATOR& allocator = ALLOCATOR())
  : allocator_(allocator)
  , iterations_(0)
  { }

  detection_type* add_detection(const index timestep, const index detection, const index number_of_incoming, const index number_of_outgoing)
  {
    assert(number_of_incoming >= 0 && number_of_incoming <= max_number_of_detection_edges);
    assert(number_of_outgoing >= 0 && number_of_outgoing <= max_number_of_detection_edges);
    factor_counter_.new_detection(timestep, detection);

    if (timestep >= timesteps_.size())
      timesteps_.resize(timestep + 1);
    auto& detections = timesteps_[timestep].detections;

    if (detection >= detections.size())
      detections.resize(detection + 1);
    auto& [d, t] = detections[detection];

    {
      using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<detection_type>;
      allocator_type a(allocator_);
      d = a.allocate();
      std::allocator_traits<allocator_type>::construct(a, d, number_of_incoming, number_of_outgoing, allocator_); // FIXME: Dtor is never called.
    }

    {
      using allocator_type = typename std::allocator_traits<ALLOCATOR>::template rebind_alloc<transition_type>;
      allocator_type a(allocator_);
      t = a.allocate();
      std::allocator_traits<allocator_type>::construct(a, t, d, number_of_incoming, number_of_outgoing, allocator_); // FIXME: Dtor is never caleld.
    }

    return d;
  }

  conflict_type* add_conflict(const index timestep, const index conflict, const index number_of_detections)
  {
    assert(number_of_detections >= 2 && number_of_detections <= max_number_of_conflict_edges);
    factor_counter_.new_conflict(timestep, conflict);

    auto& conflicts = timesteps_[timestep].conflicts;
    if (conflict >= conflicts.size())
      conflicts.resize(conflict + 1);
    auto& [c, m] = conflicts[conflict];

    {
      typename ALLOCATOR::template rebind<conflict_type>::other a(allocator_);
      c = a.allocate();
      ::new (c) conflict_type(number_of_detections, allocator_); // FIXME: Dtor is never called.
    }

    {
      typename ALLOCATOR::template rebind<conflict_messages_type>::other a(allocator_);
      m = a.allocate();
      ::new (m) conflict_messages_type(c, number_of_detections, allocator_); // FIXME: Dtor is never called.
    }

    return c;
  }

  void add_transition(const index timestep_from, const index detection_from, const index index_from, const index detection_to, const index index_to)
  {
    auto& [from_detection, from_messages] = timesteps_[timestep_from].detections[detection_from];
    auto& [to_detection, to_messages] = timesteps_[timestep_from+1].detections[detection_to];

    from_messages->set_right_transition(index_from, to_detection, index_to);
    to_messages->set_left_transition(index_to, from_detection, index_from);
  }

  void add_division(const index timestep_from, const index detection_from, const index index_from, const index detection_to_1, const index index_to_1, const index detection_to_2, const index index_to_2)
  {
    auto& [from_detection, from_messages] = timesteps_[timestep_from].detections[detection_from];
    auto& [to_detection_1, to_messages_1] = timesteps_[timestep_from+1].detections[detection_to_1];
    auto& [to_detection_2, to_messages_2] = timesteps_[timestep_from+1].detections[detection_to_2];

    from_messages->set_right_transition(index_from, to_detection_1, index_to_1, to_detection_2, index_to_2);
    to_messages_1->set_left_transition(index_to_1, from_detection, index_from, to_detection_2, index_to_2);
    to_messages_2->set_left_transition(index_to_2, from_detection, index_from, to_detection_1, index_to_1);
  }

  void add_conflict_link(const index timestep, const index conflict, const index slot, const index detection, const cost weight)
  {
    auto [d, _1] = timesteps_[timestep].detections[detection];
    auto [_2, m] = timesteps_[timestep].conflicts[conflict];
    m->add_link(slot, d, weight);
  }

  detection_type* detection(const index timestep, const index detection)
  {
    return std::get<0>(timesteps_[timestep].detections[detection]);
  }

  conflict_type* conflict(const index timestep, const index conflict)
  {
    return std::get<0>(timesteps_[timestep].conflicts[conflict]);
  }

  cost lower_bound() const
  {
    cost result = 0;
    for (auto& timestep : timesteps_) {
      for (auto [factor, _] : timestep.detections)
        result += factor->lower_bound();

      for (auto [factor, _] : timestep.conflicts)
        result += factor->lower_bound();
    }

    return result;
  }

  cost evaluate_primal() const
  {
    const cost inf = std::numeric_limits<cost>::infinity();
    cost result = 0;

    for (auto& timestep : timesteps_) {
      for (auto [factor, messages] : timestep.detections) {
        if (!messages->check_primal_consistency())
          result += inf;
        result += factor->evaluate_primal();
      }

      for (auto [factor, messages] : timestep.conflicts) {
        if (!messages->check_primal_consistency())
          result += inf;
        result += factor->evaluate_primal();
      }
    }

    return result;
  }

  cost upper_bound() const { return evaluate_primal(); }

  void reset_primal()
  {
    for (auto& timestep : timesteps_) {
      for (auto [factor, _] : timestep.detections)
        factor->reset_primal();

      for (auto [factor, _] : timestep.conflicts)
        factor->reset_primal();
    }
  }

  // This method is only needed for external rounding code.
  template<bool forward>
  void single_step(const index timestep_idx)
  {
    assert(timestep_idx >= 0 && timestep_idx < timesteps_.size());
    const auto& t = timesteps_[timestep_idx];
    single_step<forward, false>(t); // Rounding is disabled here.
  }

  // FIXME: No data locality here!
  template<bool forward, bool rounding>
  void single_pass()
  {
#ifndef NDEBUG
    auto lb_before = lower_bound();
#endif

    auto runner = [&](auto begin, auto end) {
      for (auto it = begin; it != end; ++it) {
        this->single_step<forward, rounding>(*it);
      }
    };

    if constexpr (forward)
      runner(timesteps_.begin(), timesteps_.end());
    else
      runner(timesteps_.rbegin(), timesteps_.rend());

    if constexpr (rounding) {
      for (const auto& timestep : timesteps_) {
        for (auto [factor, _] : timestep.detections)
          factor->fix_primal();

        for (auto [factor, _] : timestep.conflicts)
          factor->fix_primal();
      }
    }

#ifndef NDEBUG
    auto lb_after = lower_bound();
    assert(lb_before <= lb_after + epsilon);
#endif
  }

  template<bool rounding=false> void forward_pass() { single_pass<true, rounding>(); }
  template<bool rounding=false> void backward_pass() { single_pass<false, rounding>(); }

  void run(const int max_batches = 1000 / config_batch)
  {
#ifndef NDEBUG
    auto assert_prepared = [](auto& vec) {
      for (auto [factor, messages] : vec)
        assert(factor->is_prepared() && messages->is_prepared());
    };

    for (auto& timestep : timesteps_) {
      assert_prepared(timestep.detections);
      assert_prepared(timestep.conflicts);
    }
#endif

    const size_t number_of_detections = std::accumulate(
      timesteps_.cbegin(), timesteps_.cend(), 0,
      [](size_t acc, const auto& t) {
        return acc + t.detections.size();
      });

    const size_t number_of_conflicts = std::accumulate(
      timesteps_.cbegin(), timesteps_.cend(), 0,
      [](size_t acc, const auto& t) {
        return acc + t.conflicts.size();
      });

    std::vector<detection_primal> best_detection_primals(number_of_detections);
    std::vector<conflict_primal> best_conflict_primals(number_of_conflicts);
    cost best_ub = std::numeric_limits<cost>::infinity();

    auto visit_primal_storage = [&](auto functor) {
      auto it_detection = best_detection_primals.begin();
      auto it_conflict = best_conflict_primals.begin();
      for (const auto& t : timesteps_) {
        for (auto [factor, _] : t.detections) {
          assert(it_detection != best_detection_primals.end());
          functor(it_detection++, factor);
        }
        for (auto [factor, _ ] : t.conflicts) {
          assert(it_conflict != best_conflict_primals.end());
          functor(it_conflict++, factor);
        }
      }
      assert(it_detection == best_detection_primals.end());
      assert(it_conflict == best_conflict_primals.end());
    };

    auto remember_best_primals = [&]() {
      auto ub = evaluate_primal();
      if (ub < best_ub) {
        best_ub = ub;
        visit_primal_storage([&](auto it, auto* f) { *it = f->primal(); });
      }
    };

    auto restore_best_primals = [&] () {
      visit_primal_storage([&](auto it, auto* f) { f->primal() = *it++; });
    };

    signal_handler h;
    using clock_type = std::chrono::high_resolution_clock;
    const auto clock_start = clock_type::now();
    std::cout.precision(std::numeric_limits<cost>::max_digits10);
    for (int i = 0; i < max_batches && !h.signaled(); ++i) {
      for (int j = 0; j < config_batch-1; ++j) {
        forward_pass<false>();
        backward_pass<false>();
      }

      reset_primal();
      forward_pass<true>();
      remember_best_primals();

      reset_primal();
      backward_pass<true>();
      remember_best_primals();

      const auto clock_now = clock_type::now();
      const std::chrono::duration<double> seconds = clock_now - clock_start;

      const auto lb = lower_bound();
      iterations_ += config_batch;
      std::cout << "it=" << iterations_ << " "
                << "lb=" << lb << " "
                << "ub=" << best_ub << " "
                << "gap=" << static_cast<float>(100.0 * (best_ub - lb) / std::abs(lb)) << "% "
                << "t=" << seconds.count() << std::endl;
    }

    restore_best_primals();
  }

protected:
  struct timestep {
    std::vector<std::tuple<detection_type*, transition_type*>> detections;
    std::vector<std::tuple<conflict_type*, conflict_messages_type*>> conflicts;
  };

  template<bool forward, bool rounding>
  void single_step(const timestep& t)
  {
    // Checks that all messages are either consistent or unknown but not
    // inconsitent. This property must be invariant during rounding, so we
    // verify that it is the case.
    auto check_messages = [&]() {
#ifndef NDEBUG
      for (const auto& timestep : timesteps_) {
        for (auto [_, messages] : t.detections)
          assert(messages->check_primal_consistency().is_not_inconsistent());
        for (auto [_, messages] : t.conflicts)
          assert(messages->check_primal_consistency().is_not_inconsistent());
      }
#endif
    };

    for (auto [_, messages] : t.conflicts)
      messages->send_message_to_conflict();

    for (auto [_, messages] : t.conflicts)
      messages->send_message_to_detection();

    if constexpr (rounding) {
      // FIXME: Pre-allocate scratch space and do not resort to dynamic
      // memory allocation.
      auto sorted_detections = t.detections;
      std::sort(sorted_detections.begin(), sorted_detections.end(),
        [](auto a, auto b) {
          return std::get<0>(a)->min_detection() < std::get<0>(b)->min_detection();
        });

      for (auto [factor, messages] : sorted_detections) {
        std::array<bool, max_number_of_detection_edges + 1> possible;
        messages->template get_primal_possibilities<forward>(possible);

        factor->template round_primal<forward>(possible);
        check_messages();
        messages->template propagate_primal<!forward>();
        check_messages();

        // FIXME: Make the other direction explicit and only push specific
        // messages.
        for (auto [_, messages] : t.conflicts) {
          messages->propagate_primal_to_conflict();
          check_messages();
          messages->propagate_primal_to_detections();
          check_messages();
        }
      }
    }

    for (auto [_, messages] : t.detections)
      messages->template send_messages<forward>();
  }

  const ALLOCATOR allocator_;
  std::vector<timestep> timesteps_; // FIXME: Use flat allocator
  factor_counter factor_counter_;
  int iterations_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
