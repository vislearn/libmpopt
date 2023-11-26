#ifndef LIBMPOPT_COMMON_SIGNAL_HANDLER_HPP
#define LIBMPOPT_COMMON_SIGNAL_HANDLER_HPP

#include <unistd.h>

extern "C" {
  static volatile sig_atomic_t mpopt_signaled;

  void mpopt_signal_handler(int sig)
  {
    if (sig == SIGALRM) {
      write(1, "\nTimeout.\n", 10);
      std::signal(sig, SIG_DFL);
      std::raise(sig);
    }

    mpopt_signaled = 1;
  }
}

namespace mpopt {

class signal_handler {
public:
  signal_handler()
  {
    mpopt_signaled = 0;
    old_handler_ = std::signal(SIGINT, mpopt_signal_handler);
    std::signal(SIGALRM, mpopt_signal_handler);
  }

  ~signal_handler()
  {
    std::signal(SIGINT, old_handler_);
    if (signaled())
      std::raise(SIGINT);
  }

  bool signaled() const { return mpopt_signaled != 0; };

protected:
  void(*old_handler_)(int);
};

}

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
