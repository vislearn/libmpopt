#include <qapopt/qap.hpp>
#include <qapopt/qap.h>

using allocator_type = qapopt::block_allocator<qapopt::cost>;
using solver_type = qapopt::qap::solver<allocator_type>;
using graph_type = qapopt::qap::graph<allocator_type>;
using unary_node_type = qapopt::qap::unary_node<allocator_type>;
using uniqueness_node_type = qapopt::qap::uniqueness_node<allocator_type>;
using pairwise_node_type = qapopt::qap::pairwise_node<allocator_type>;

struct qapopt_solver_t {
  qapopt::memory_block memory;
  allocator_type allocator;
  solver_type solver;

  qapopt_solver_t()
  : memory()
  , allocator(memory)
  , solver(allocator)
  { }
};

inline auto* to_graph(graph_type* g) { return reinterpret_cast<qapopt_graph*>(g); }
inline auto* from_graph(qapopt_graph* g) { return reinterpret_cast<graph_type*>(g); }

inline auto* to_unary(unary_node_type* n) { return reinterpret_cast<qapopt_unary_node*>(n); }
inline auto* from_unary(qapopt_unary_node* n) { return reinterpret_cast<unary_node_type*>(n); }

inline auto* to_uniqueness(uniqueness_node_type* n) { return reinterpret_cast<qapopt_uniqueness_node*>(n); }
inline auto* from_uniqueness(qapopt_uniqueness_node* n) { return reinterpret_cast<uniqueness_node_type*>(n); }

inline auto* to_pairwise(pairwise_node_type* n) { return reinterpret_cast<qapopt_pairwise_node*>(n); }
inline auto* from_pairwise(qapopt_pairwise_node* n) { return reinterpret_cast<pairwise_node_type*>(n); }

extern "C" {

//
// solver API
//

qapopt_solver* qapopt_solver_create() { return new qapopt_solver; }
void qapopt_solver_destroy(qapopt_solver* s) { delete s; }
void qapopt_solver_finalize(qapopt_solver* s) { s->memory.finalize(); }

qapopt_graph* qapopt_solver_get_graph(qapopt_solver* s) { return to_graph(&s->solver.get_graph()); }

qapopt_unary_node* qapopt_graph_add_unary(qapopt_graph* graph, int idx, int number_of_labels, int number_of_forward, int number_of_backward)
{
  auto* node = from_graph(graph)->add_unary(idx, number_of_labels, number_of_forward, number_of_backward);
  return to_unary(node);
}

qapopt_uniqueness_node* qapopt_graph_add_uniqueness(qapopt_graph* graph, int idx, int number_of_unaries)
{
  auto* node = from_graph(graph)->add_uniqueness(idx, number_of_unaries);
  return to_uniqueness(node);
}

qapopt_pairwise_node* qapopt_graph_add_pairwise(qapopt_graph* graph, int idx, int number_of_labels0, int number_of_labels1)
{
  auto* node = from_graph(graph)->add_pairwise(idx, number_of_labels0, number_of_labels1);
  return to_pairwise(node);
}

void qapopt_graph_add_pairwise_link(qapopt_graph* graph, int idx_unary0, int idx_unary1, int idx_pairwise)
{
  from_graph(graph)->add_pairwise_link(idx_unary0, idx_unary1, idx_pairwise);
}

void qapopt_graph_add_uniqueness_link(qapopt_graph* graph, int idx_unary, int label, int idx_uniqueness, int slot)
{
  from_graph(graph)->add_uniqueness_link(idx_unary, label, idx_uniqueness, slot);
}

qapopt_unary_node* qapopt_graph_get_unary(qapopt_graph* graph, int idx) { return to_unary(from_graph(graph)->get_unary(idx)); }
qapopt_uniqueness_node* qapopt_graph_get_uniqueness(qapopt_graph* graph, int idx) { return to_uniqueness(from_graph(graph)->get_uniqueness(idx)); }
qapopt_pairwise_node* qapopt_graph_get_pairwise(qapopt_graph* graph, int idx) { return to_pairwise(from_graph(graph)->get_pairwise(idx)); }

void qapopt_solver_run(qapopt_solver* s, int batch_size, int max_batches) { s->solver.run(batch_size, max_batches); }
void qapopt_solver_run_rounding_only(qapopt_solver* s) { s->solver.run_rounding_only(); }
void qapopt_solver_run_quiet(qapopt_solver* s, int batch_size, int max_batches) { s->solver.run<false>(batch_size, max_batches); }
void qapopt_solver_run_no_rounding(qapopt_solver* s, int batch_size, int max_batches) { s->solver.run<false, false>(batch_size, max_batches); }

void qapopt_solver_solve_ilp(qapopt_solver* s) { s->solver.solve_ilp(); }
void qapopt_solver_compute_greedy_assignment(qapopt_solver* s) { s->solver.compute_greedy_assignment(); }
double qapopt_solver_runtime(qapopt_solver* s) { return s->solver.runtime(); }
double qapopt_solver_lower_bound(qapopt_solver* s) { return s->solver.lower_bound(); }
double qapopt_solver_evaluate_primal(qapopt_solver* s) { return s->solver.evaluate_primal(); }

void qapopt_solver_execute_fusion_move(qapopt_solver* s, int* solution0, int* solution1, size_t length)
{
  std::vector<::qapopt::index> tmp0(solution0, solution0 + length);
  std::vector<::qapopt::index> tmp1(solution1, solution1 + length);
  s->solver.execute_fusion_move(tmp0, tmp1);
}

void qapopt_solver_execute_qpbo(qapopt_solver* s, int* solution0, int* solution1, size_t length, int enable_weak_persistency, int enable_probe, int enable_improve)
{
  std::vector<::qapopt::index> tmp0(solution0, solution0 + length);
  std::vector<::qapopt::index> tmp1(solution1, solution1 + length);
  s->solver.execute_qpbo(tmp0, tmp1, enable_weak_persistency, enable_probe, enable_improve);
}

void qapopt_solver_execute_lsatr(qapopt_solver* s, int* solution0, int* solution1, size_t length)
{
  std::vector<::qapopt::index> tmp0(solution0, solution0 + length);
  std::vector<::qapopt::index> tmp1(solution1, solution1 + length);
  s->solver.execute_lsatr(tmp0, tmp1);
}


double qapopt_solver_get_fm_ms_build(qapopt_solver* s) { return s->solver.get_fm_ms_build(); }
double qapopt_solver_get_fm_ms_solve(qapopt_solver* s) { return s->solver.get_fm_ms_solve(); }

//
// unary factor API
//

void qapopt_unary_set_cost(qapopt_unary_node* n, int label, double cost) { from_unary(n)->factor.set(label, cost); }
double qapopt_unary_get_cost(qapopt_unary_node* n, int label) { return from_unary(n)->factor.get(label); }

int qapopt_unary_get_primal(qapopt_unary_node* n)
{
  auto& f = from_unary(n)->factor;
  if (f.is_primal_set())
    return f.primal();
  else
    return -1;
}

//
// uniqueness factor API
//

void qapopt_uniqueness_set_cost(qapopt_uniqueness_node* n, int unary, double cost) { from_uniqueness(n)->factor.set(unary, cost); }
double qapopt_uniqueness_get_cost(qapopt_uniqueness_node* n, int unary) { return from_uniqueness(n)->factor.get(unary); }

int qapopt_uniqueness_get_primal(qapopt_uniqueness_node* n) {
  auto& f = from_uniqueness(n)->factor;
  if (f.is_primal_set())
    return f.primal();
  else
    return -1;
}

//
// pairwise factor API
//

void qapopt_pairwise_set_cost(qapopt_pairwise_node* n, int l0, int l1, double cost) { from_pairwise(n)->factor.set(l0, l1, cost); }
double qapopt_pairwise_get_cost(qapopt_pairwise_node* n, int l0, int l1) { return from_pairwise(n)->factor.get(l0, l1); }

int qapopt_pairwise_get_primal(qapopt_pairwise_node* n, char left_side) {
  auto& f = from_pairwise(n)->factor;
  auto [p0, p1] = f.primal();
  auto p = left_side ? p0 : p1;
  if (p == std::decay_t<decltype(f)>::primal_unset)
    return -1;
  return p;
}

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
