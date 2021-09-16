#ifndef LIBMPOPT_QAP_HPP
#define LIBMPOPT_QAP_HPP

#include <mpopt/common/system_includes.hpp>

#include <mpopt/common/types.hpp>
#include <mpopt/common/allocator.hpp>
#include <mpopt/common/debug.hpp>
#include <mpopt/common/fixed_vector.hpp>
#include <mpopt/common/signal_handler.hpp>
#include <mpopt/common/misc.hpp>
#include <mpopt/common/consistency.hpp>
#include <mpopt/common/array_accessor.hpp>

#include <mpopt/common/factors/unary.hpp>
#include <mpopt/common/factors/pairwise.hpp>
#include <mpopt/common/graph.hpp>
#include <mpopt/common/solver.hpp>

#include <mpopt/qap/unary_factor.hpp>
#include <mpopt/qap/uniqueness_factor.hpp>
#include <mpopt/qap/pairwise_messages.hpp>
#include <mpopt/qap/unary_messages.hpp>
#include <mpopt/qap/uniqueness_messages.hpp>
#include <mpopt/qap/graph.hpp>
#include <mpopt/qap/primal_storage.hpp>
#include <mpopt/qap/gurobi.hpp>
#include <mpopt/qap/qpbo.hpp>
#include <mpopt/qap/combilp.hpp>
#include <mpopt/qap/local_search.hpp>
#include <mpopt/qap/grasp.hpp>
#include <mpopt/qap/greedy.hpp>
#include <mpopt/qap/solver.hpp>

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
