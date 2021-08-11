#ifndef LIBQAPOPT_COMMON_CONSISTENCY_HPP
#define LIBQAPOPT_COMMON_CONSISTENCY_HPP

namespace qapopt {

struct consistency {
public:
  enum value {
    consistent,
    inconsistent,
    unknown
  };

  consistency()
  : value_(consistent)
  { }

  void reset() { value_ = consistent; }

  void mark_unknown()
  {
    if (value_ == inconsistent)
      return;

    value_ = unknown;
  }

  void mark_inconsistent()
  {
    value_ = inconsistent;
  }

  bool is_known() const { return value_ != unknown; }
  bool is_unknown() const { return value_ == unknown; }

  bool is_consistent() const { return value_ == consistent; }
  bool is_inconsistent() const { return value_ == inconsistent; }
  bool is_not_inconsistent() const { return value_ != inconsistent; }

  operator bool() const { return is_consistent(); }

  void merge(const consistency& other)
  {
    switch (other.value_) {
      case consistent:                        break;
      case inconsistent: mark_inconsistent(); break;
      case unknown:      mark_unknown();      break;
    }
  }

protected:
  value value_;
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
