#ifndef LIBMPOPT_COMMON_ALLOCATOR_HPP
#define LIBMPOPT_COMMON_ALLOCATOR_HPP

namespace mpopt {

class memory_block {
public:
  static constexpr size_t size_mib = size_t(1) * 1024 * 1024;
  static constexpr size_t size_512mib = size_mib * 512;
  static constexpr size_t size_gib = size_mib * 1024;
  static constexpr size_t size_1024gib = size_gib * 1024;

  memory_block()
  : memory_(0)
  , size_(size_1024gib)
  , current_(0)
  , finalized_(false)
  {
    while (memory_ == 0 && size_ >= size_512mib) {
      auto result = reinterpret_cast<uintptr_t>(std::malloc(size_));
      if (result == 0)
        size_ -= size_512mib;
      else
        memory_ = result;
    }

#ifndef NDEBUG
      //std::cout << "[mem] ctor: size=" << size_ << "B (" << (1.0f * size_ / size_gib) << "GiB) -> memory_=" << reinterpret_cast<void*>(memory_) << std::endl;
#endif

    if (memory_ == 0)
      throw std::bad_alloc();

    current_ = memory_;
  }

  ~memory_block()
  {
    if (memory_ != 0) {
      std::free(reinterpret_cast<void*>(memory_));
    }
  }

  void align(size_t a)
  {
    if (memory_ % a != 0)
      allocate(a - memory_ % a);
  }

  uintptr_t allocate(size_t s)
  {
    assert(!finalized_);
    if (current_ + s >= memory_ + size_)
      throw std::bad_alloc();

    auto result = current_;
    current_ += s;
    return result;
  }

  bool is_finalized() const { return finalized_; }

  void finalize()
  {
    assert(!finalized_);
    auto current_size = current_ - memory_;
    auto result = reinterpret_cast<uintptr_t>(std::realloc(reinterpret_cast<void*>(memory_), current_size));
    // result could be 0 (error) or memory could have been relocated
    if (result != memory_)
      throw std::bad_alloc();
    size_ = current_size;
#ifndef NDEBUG
      //std::cout << "[mem] finalize: size=" << size_ << " (" << (1.0f * size_ / size_mib) << " MiB)" << std::endl;
#endif
    finalized_ = true;
  }

protected:
  uintptr_t memory_;
  size_t size_;
  uintptr_t current_;
  bool finalized_;
};

template<typename T>
class block_allocator {
public:
  using value_type = T;
  template<typename U> struct rebind { using other = block_allocator<U>; };

  block_allocator(memory_block& block)
  : block_(&block)
  { }

  template<typename U>
  block_allocator(const block_allocator<U>& other)
  : block_(other.block_)
  { }

  T* allocate(size_t n = 1)
  {
    block_->align(alignof(T));
    auto mem = block_->allocate(sizeof(T) * n);
    assert(reinterpret_cast<std::uintptr_t>(mem) % alignof(T) == 0);
    return reinterpret_cast<T*>(mem);
  }

  void deallocate(T* ptr, size_t n = 1) { }

protected:
  memory_block* block_;

  template<typename U> friend class block_allocator;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
