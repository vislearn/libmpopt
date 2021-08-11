#ifndef LIBQAPOPT_COMMON_ARRAY_ACCESSOR_HPP
#define LIBQAPOPT_COMMON_ARRAY_ACCESSOR_HPP

namespace qapopt {

class two_dimension_array_accessor {
public:
  two_dimension_array_accessor(index dim0, index dim1)
  : dim0_(dim0)
  , dim1_(dim1)
  { }

  auto dimensions() const { return std::tuple(dim0_, dim1_); }

  size_t to_linear(const index idx0, const index idx1) const
  {
    assert_index(idx0, idx1);
    size_t result = idx0 * dim1_ + idx1;
    assert(result >= 0 && result < dim0_ * dim1_);
    return result;
  }

  std::tuple<index, index> to_nonlinear(const index idx) const
  {
    assert(idx >= 0 && idx < dim0_ * dim1_);
    const index idx0 = idx / dim1_;
    const index idx1 = idx % dim1_;
    assert(idx0 >= 0 && idx0 < dim0_);
    assert(idx1 >= 0 && idx1 < dim1_);
    assert(to_linear(idx0, idx1) == idx);
    return std::tuple(idx0, idx1);
  }

protected:
  void assert_index(const index idx0, const index idx1) const
  {
    assert(idx0 >= 0 && idx0 < dim0_);
    assert(idx1 >= 0 && idx1 < dim1_);
  }

  index dim0_, dim1_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
