#ifndef LIBQAPOPT_QAP_HPP
#define LIBQAPOPT_QAP_HPP

#include <qapopt/common/system_includes.hpp>

#include <qapopt/common/types.hpp>
#include <qapopt/common/allocator.hpp>
#include <qapopt/common/debug.hpp>
#include <qapopt/common/fixed_vector.hpp>
#include <qapopt/common/signal_handler.hpp>
#include <qapopt/common/misc.hpp>
#include <qapopt/common/consistency.hpp>
#include <qapopt/common/array_accessor.hpp>

#include <qapopt/common/factors/unary.hpp>
#include <qapopt/common/factors/pairwise.hpp>
#include <qapopt/common/solver.hpp>

#include <qapopt/qap/unary_factor.hpp>
#include <qapopt/qap/uniqueness_factor.hpp>
#include <qapopt/qap/graph.hpp>
#include <qapopt/qap/pairwise_messages.hpp>
#include <qapopt/qap/unary_messages.hpp>
#include <qapopt/qap/uniqueness_messages.hpp>
#include <qapopt/qap/primal_storage.hpp>
#include <qapopt/qap/gurobi.hpp>
#include <qapopt/qap/qpbo.hpp>
#include <qapopt/qap/lsa_tr.hpp>
#include <qapopt/qap/greedy.hpp>
#include <qapopt/qap/solver.hpp>

#endif

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
