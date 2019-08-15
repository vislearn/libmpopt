#ifndef LIBMPOPT_QAP_GRAPH_HPP
#define LIBMPOPT_QAP_GRAPH_HPP

namespace mpopt {
namespace qap {

template<typename> struct unary_node;
template<typename> struct pairwise_node;
template<typename> struct uniqueness_node;


template<typename NODE_TYPE>
struct link_info {
  using node_type = NODE_TYPE;

  link_info()
  : node(nullptr)
  , slot(-1)
  { }

  bool is_prepared() const { return node != nullptr; }

  const NODE_TYPE* node;
  index slot;
};


template<typename ALLOCATOR>
struct unary_node {
  using allocator_type = ALLOCATOR;
  using unary_node_type = unary_node<allocator_type>;
  using pairwise_node_type = pairwise_node<allocator_type>;
  using uniqueness_node_type = uniqueness_node<allocator_type>;

  mutable unary_factor<allocator_type> unary;
  fixed_vector_alloc_gen<link_info<uniqueness_node_type>, allocator_type> uniqueness;

  unary_node(index number_of_labels, const allocator_type& allocator)
  : unary(number_of_labels, allocator)
  , uniqueness(number_of_labels, allocator)
  { }

  void check_structure() const
  {
    assert(unary.is_prepared());

    index slot = 0;
    for (const auto& link : uniqueness) {
      // assert(link.is_prepared());
      // FIXME: The handling of unary and uniqueness factors is inconsistent.
      //        The uniqueness factor knows about the dummy. For the unary we
      //        add +1 to the number of labels to model the dummy.
      //        Unfortunately we allocate as many uniqueness links as labels.
      //        This means that the last uniqueness link will never be prepared.
      //        The best solution is to unify the behaviour of unary and
      //        uniqueness factors and uncomment the check above, again.
      assert(link.node == nullptr || link.node->unaries[link.slot].node == this);
      assert(link.node == nullptr || link.node->unaries[link.slot].slot == slot);
      ++slot;
    }
  }

  template<typename FUNCTOR>
  void traverse_uniqueness(FUNCTOR f) const
  {
    index slot = 0;
    for (const auto& link : uniqueness) {
      // TODO: Would be awesome if we could execute things in this loop
      //       unconditionally.
      if (link.node != nullptr)
        f(link, slot);
      ++slot;
    }
  }
};


template<typename ALLOCATOR>
struct uniqueness_node {
  using allocator_type = ALLOCATOR;
  using uniqueness_node_type = uniqueness_node<allocator_type>;
  using unary_node_type = unary_node<allocator_type>;

  mutable uniqueness_factor<allocator_type> uniqueness;
  fixed_vector_alloc_gen<link_info<unary_node_type>, allocator_type> unaries;

  uniqueness_node(index number_of_unaries, const allocator_type& allocator)
  : uniqueness(number_of_unaries, allocator)
  , unaries(number_of_unaries, allocator)
  { }

  void check_structure() const
  {
    assert(uniqueness.is_prepared());

    index slot = 0;
    for (const auto& link : unaries) {
      assert(link.is_prepared());
      assert(link.node == nullptr || link.node->uniqueness[link.slot].node == this);
      assert(link.node == nullptr || link.node->uniqueness[link.slot].slot == slot);
      ++slot;
    }
  }

  template<typename FUNCTOR>
  void traverse_unaries(FUNCTOR f) const
  {
    index slot = 0;
    for (const auto& link : unaries) {
      assert(link.is_prepared());
      f(link, slot);
      ++slot;
    }
  }
};


template<typename ALLOCATOR>
struct pairwise_node {
  using allocator_type = ALLOCATOR;
  using unary_node_type = unary_node<allocator_type>;
  using pairwise_node_type = pairwise_node<allocator_type>;

  mutable pairwise_factor<allocator_type> pairwise;
  unary_node_type* unary0;
  unary_node_type* unary1;

  pairwise_node(index number_of_labels0, index number_of_labels1, const allocator_type& allocator)
  : pairwise(number_of_labels0, number_of_labels1, allocator)
  , unary0(nullptr)
  , unary1(nullptr)
  { }

  template<bool forward>
  auto* unary() const
  {
    return forward ? unary1 : unary0;
  }

  void check_structure() const
  {
    assert(pairwise.is_prepared());
    assert(unary0 != nullptr);
    assert(unary1 != nullptr);
  }
};


template<typename ALLOCATOR>
class graph {
public:
  using allocator_type = ALLOCATOR;
  using unary_node_type = unary_node<ALLOCATOR>;
  using uniqueness_node_type = uniqueness_node<allocator_type>;
  using pairwise_node_type = pairwise_node<ALLOCATOR>;

  graph(const ALLOCATOR& allocator = ALLOCATOR())
  : allocator_(allocator)
  { }

  const auto& unaries() const { return unaries_; }
  const auto& pairwise() const { return pairwise_; }
  const auto& uniqueness() const { return uniqueness_; }

  unary_node_type* add_unary(index idx, index number_of_labels)
  {
    assert(number_of_labels >= 0);
    assert(idx == unaries_.size());
    assert(uniqueness_.size() == 0);
    assert(pairwise_.size() == 0);

    unaries_.push_back(nullptr);
    auto& node = unaries_.back();
    typename std::allocator_traits<allocator_type>::template rebind_alloc<unary_node_type> a(allocator_);
    node = a.allocate();
    new (node) unary_node_type(number_of_labels, allocator_);
#ifndef NDEBUG
    node->unary.set_debug_info(idx);
#endif

    return node;
  }

  uniqueness_node_type* add_uniqueness(index idx, index number_of_unaries)
  {
    assert(number_of_unaries >= 0);
    assert(idx == uniqueness_.size());
    assert(pairwise_.size() == 0);

    uniqueness_.push_back(nullptr);
    auto& node = uniqueness_.back();
    typename std::allocator_traits<allocator_type>::template rebind_alloc<uniqueness_node_type> a(allocator_);
    node = a.allocate();
    new (node) uniqueness_node_type(number_of_unaries, allocator_);
#ifndef NDEBUG
    node->uniqueness.set_debug_info(idx);
#endif

    return node;
  }

  pairwise_node_type* add_pairwise(index idx, index number_of_labels0, index number_of_labels1)
  {
    assert(number_of_labels0 >= 0 && number_of_labels1 >= 0);
    assert(idx == pairwise_.size());

    pairwise_.push_back(nullptr);
    auto& node = pairwise_.back();
    typename std::allocator_traits<allocator_type>::template rebind_alloc<pairwise_node_type> a(allocator_);
    node = a.allocate();
    new (node) pairwise_node_type(number_of_labels0, number_of_labels1, allocator_);

    return node;
  }

  void add_pairwise_link(index idx_unary0, index idx_unary1, index idx_pairwise)
  {
    assert(idx_unary0 >= 0 && idx_unary0 < unaries_.size());
    assert(idx_unary1 >= 0 && idx_unary1 < unaries_.size());
    assert(idx_pairwise >= 0 && idx_pairwise < pairwise_.size());

    auto* unary0 = unaries_[idx_unary0];
    auto* unary1 = unaries_[idx_unary1];
    auto* pairwise = pairwise_[idx_pairwise];
    assert(std::tuple(unary0->unary.size(), unary1->unary.size()) == pairwise->pairwise.size());

    assert(pairwise->unary0 == nullptr && pairwise->unary1 == nullptr);
    pairwise->unary0 = unary0;
    pairwise->unary1 = unary1;

#ifndef NDEBUG
    pairwise->pairwise.set_debug_info(idx_unary0, idx_unary1);
#endif
  }

  void add_uniqueness_link(index idx_unary, index label, index idx_uniqueness, index slot)
  {
    assert(idx_unary >= 0 && idx_unary < unaries_.size());
    assert(idx_uniqueness >= 0 && idx_uniqueness < uniqueness_.size());

    auto* unary = unaries_[idx_unary];
    auto* uniqueness = uniqueness_[idx_uniqueness];

    assert(label >= 0 && label < unary->uniqueness.size());
    assert(!unary->uniqueness[label].is_prepared());
    unary->uniqueness[label].node = uniqueness;
    unary->uniqueness[label].slot = slot;

    assert(slot >= 0 && slot < uniqueness->unaries.size());
    assert(!uniqueness->unaries[slot].is_prepared());
    uniqueness->unaries[slot].node = unary;
    uniqueness->unaries[slot].slot = label;
  }

  void check_structure() const
  {
    bool result = true;

    for (auto* node : unaries_)
      node->check_structure();

    for (auto* node : uniqueness_)
      node->check_structure();

    for (auto* node : pairwise_)
      node->check_structure();
  }

protected:
  allocator_type allocator_;
  std::vector<unary_node_type*> unaries_;
  std::vector<pairwise_node_type*> pairwise_;
  std::vector<uniqueness_node_type*> uniqueness_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
