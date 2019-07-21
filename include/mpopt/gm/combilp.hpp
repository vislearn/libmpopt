#ifndef LIBMPOPT_GM_COMBILP_HPP
#define LIBMPOPT_GM_COMBILP_HPP

namespace mpopt {
namespace gm {

template<typename ALLOCATOR>
struct sac_detector {
  using allocator_type = ALLOCATOR;
  using graph_type = graph<allocator_type>;
  using unary_node_type = typename graph_type::unary_node_type;
  using pairwise_node_type = typename graph_type::pairwise_node_type;

  sac_detector(const graph_type& graph)
  : graph_(&graph)
  { }

  void update()
  {
    for (const auto* node : graph_->unaries())
      unaries_[node] = true;

    for (const auto* node : graph_->pairwise())
      pairwise_[node] = true;

    for (const auto* node : graph_->unaries()) {
      assert(messages::check_unary_primal_consistency(node).is_known());
      if (!messages::check_unary_primal_consistency(node))
        unaries_.at(node) = false;
    }

    for (const auto* node : graph_->pairwise()) {
      assert(messages::check_pairwise_primal_consistency(node).is_known());
      if (!messages::check_pairwise_primal_consistency(node))
        pairwise_.at(node) = false;
    }
  }

  void merge(const sac_detector<ALLOCATOR>& other)
  {
    for (auto it = unaries_.begin(); it != unaries_.end(); ++it) {
      if (!other.unaries_.at(it->first))
        it->second = false;
    }
    for (auto it = pairwise_.begin(); it != pairwise_.end(); ++it) {
      if (!other.pairwise_.at(it->first))
        it->second = false;
    }
  }

  auto get_stats() const
  {
    size_t no_inconsistent = 0;
    size_t no_factors = 0;
    for (auto it = unaries_.cbegin(); it != unaries_.cend(); ++it) {
      ++no_factors;
      if (!it->second)
        ++no_inconsistent;
    }
    for (auto it = pairwise_.cbegin(); it != pairwise_.cend(); ++it) {
      ++no_factors;
      if (!it->second)
        ++no_inconsistent;
    }

    return std::make_tuple(no_inconsistent, no_factors);
  };

  bool get(const unary_node_type* node) const { return unaries_.at(node); }
  bool get(const pairwise_node_type* node) const { return pairwise_.at(node); }

  const graph_type* graph_;
  std::map<const unary_node_type*, bool> unaries_;
  std::map<const pairwise_node_type*, bool> pairwise_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
