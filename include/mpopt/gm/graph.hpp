#ifndef LIBMPOPT_GM_GRAPH_HPP
#define LIBMPOPT_GM_GRAPH_HPP

namespace mpopt {
namespace gm {

template<typename> struct unary_node;
template<typename> struct pairwise_node;


template<typename ALLOCATOR>
struct unary_node {
  using allocator_type = ALLOCATOR;
  using unary_node_type = unary_node<allocator_type>;
  using pairwise_node_type = pairwise_node<allocator_type>;

  mutable unary_factor<allocator_type> unary;
  fixed_vector_alloc_gen<const pairwise_node_type*, allocator_type> forward;
  fixed_vector_alloc_gen<const pairwise_node_type*, allocator_type> backward;

  unary_node(index number_of_labels, index number_of_forward, index number_of_backward, const allocator_type& allocator)
  : unary(number_of_labels, allocator)
  , forward(number_of_forward, allocator)
  , backward(number_of_backward, allocator)
  { }

  template<bool FORWARD>
  auto& edges() const
  {
    return FORWARD ? forward : backward;
  }

  bool is_prepared() const
  {
    bool result = unary.is_prepared();

    for (const auto* pairwise : forward)
      result = result && pairwise != nullptr;

    for (const auto* pairwise : backward)
      result = result && pairwise != nullptr;

    return result;
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

  bool is_prepared() const
  {
    return pairwise.is_prepared() && unary0 != nullptr && unary1 != nullptr;
  }
};


template<typename ALLOCATOR>
class graph {
public:
  using allocator_type = ALLOCATOR;
  using unary_node_type = unary_node<ALLOCATOR>;
  using pairwise_node_type = pairwise_node<ALLOCATOR>;

  graph(const ALLOCATOR& allocator = ALLOCATOR())
  : allocator_(allocator)
  { }

  const auto& unaries() const { return unaries_; }
  const auto& pairwise() const { return pairwise_; }

  unary_node_type* add_unary(index idx, index number_of_labels, index number_of_forward, index number_of_backward)
  {
    assert(number_of_labels >= 0);
    assert(idx == unaries_.size());
    assert(pairwise_.size() == 0);

    unaries_.push_back(nullptr);
    auto& node = unaries_.back();
    typename std::allocator_traits<allocator_type>::template rebind_alloc<unary_node_type> a(allocator_);
    node = a.allocate();
    new (node) unary_node_type(number_of_labels, number_of_forward, number_of_backward, allocator_);
#ifndef NDEBUG
    node->unary.set_debug_info(idx);
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

    auto find_free_slot = [](auto& vector) -> auto& {
      auto it = std::find(vector.begin(), vector.end(), nullptr);
      assert(it != vector.end());
      return *it;
    };

    find_free_slot(unary0->forward) = pairwise;
    find_free_slot(unary1->backward) = pairwise;

    assert(pairwise->unary0 == nullptr && pairwise->unary1 == nullptr);
    pairwise->unary0 = unary0;
    pairwise->unary1 = unary1;

#ifndef NDEBUG
    pairwise->pairwise.set_debug_info(idx_unary0, idx_unary1);
#endif
  }

  bool is_prepared() const
  {
    bool result = true;

    for (auto* node : unaries_)
      result = result && node->is_prepared();

    for (auto* node : pairwise_)
      result = result && node->is_prepared();

    return result;
  }

protected:
  allocator_type allocator_;
  std::vector<unary_node_type*> unaries_;
  std::vector<pairwise_node_type*> pairwise_;
};

}
}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
