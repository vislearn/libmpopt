#ifndef LIBQAPOPT_COMMON_SIGNAL_HANDLER_HPP
#define LIBQAPOPT_COMMON_SIGNAL_HANDLER_HPP

extern "C" {
  static volatile sig_atomic_t qapopt_signaled;

  void qapopt_signal_handler(int sig)
  {
    qapopt_signaled = 1;
  }
}

namespace qapopt {

class signal_handler {
public:
  signal_handler()
  {
    qapopt_signaled = 0;
    old_handler_ = std::signal(SIGINT, qapopt_signal_handler);
  }

  ~signal_handler()
  {
    std::signal(SIGINT, old_handler_);
    if (signaled())
      std::raise(SIGINT);
  }

  bool signaled() const { return qapopt_signaled != 0; };

protected:
  void(*old_handler_)(int);
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
