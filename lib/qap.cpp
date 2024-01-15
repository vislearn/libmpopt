#include <mpopt/qap.hpp>
#include <mpopt/qap.h>

using allocator_type = mpopt::block_allocator<mpopt::cost>;
using solver_type = mpopt::qap::solver<allocator_type>;
using graph_type = mpopt::qap::graph<allocator_type>;
using unary_node_type = mpopt::qap::unary_node<allocator_type>;
using uniqueness_node_type = mpopt::qap::uniqueness_node<allocator_type>;
using pairwise_node_type = mpopt::qap::pairwise_node<allocator_type>;

struct mpopt_qap_solver_t {
  mpopt::memory_block memory;
  allocator_type allocator;
  solver_type solver;

  mpopt_qap_solver_t()
  : memory()
  , allocator(memory)
  , solver(allocator)
  { }
};

inline auto* to_graph(graph_type* g) { return reinterpret_cast<mpopt_qap_graph*>(g); }
inline auto* from_graph(mpopt_qap_graph* g) { return reinterpret_cast<graph_type*>(g); }

inline auto* to_unary(unary_node_type* n) { return reinterpret_cast<mpopt_qap_unary_node*>(n); }
inline auto* from_unary(mpopt_qap_unary_node* n) { return reinterpret_cast<unary_node_type*>(n); }

inline auto* to_uniqueness(uniqueness_node_type* n) { return reinterpret_cast<mpopt_qap_uniqueness_node*>(n); }
inline auto* from_uniqueness(mpopt_qap_uniqueness_node* n) { return reinterpret_cast<uniqueness_node_type*>(n); }

inline auto* to_pairwise(pairwise_node_type* n) { return reinterpret_cast<mpopt_qap_pairwise_node*>(n); }
inline auto* from_pairwise(mpopt_qap_pairwise_node* n) { return reinterpret_cast<pairwise_node_type*>(n); }

extern "C" {

//
// solver API
//

mpopt_qap_solver* mpopt_qap_solver_create() { return new mpopt_qap_solver; }
void mpopt_qap_solver_destroy(mpopt_qap_solver* s) { delete s; }
void mpopt_qap_solver_finalize(mpopt_qap_solver* s) { s->memory.finalize(); }

mpopt_qap_graph* mpopt_qap_solver_get_graph(mpopt_qap_solver* s) { return to_graph(&s->solver.get_graph()); }

mpopt_qap_unary_node* mpopt_qap_graph_add_unary(mpopt_qap_graph* graph, int idx, int number_of_labels, int number_of_forward, int number_of_backward)
{
  auto* node = from_graph(graph)->add_unary(idx, number_of_labels, number_of_forward, number_of_backward);
  return to_unary(node);
}

mpopt_qap_uniqueness_node* mpopt_qap_graph_add_uniqueness(mpopt_qap_graph* graph, int idx, int number_of_unaries, int label_idx)
{
  auto* node = from_graph(graph)->add_uniqueness(idx, number_of_unaries, label_idx);
  return to_uniqueness(node);
}

mpopt_qap_pairwise_node* mpopt_qap_graph_add_pairwise(mpopt_qap_graph* graph, int idx, int number_of_labels0, int number_of_labels1)
{
  auto* node = from_graph(graph)->add_pairwise(idx, number_of_labels0, number_of_labels1);
  return to_pairwise(node);
}

void mpopt_qap_graph_add_pairwise_link(mpopt_qap_graph* graph, int idx_unary0, int idx_unary1, int idx_pairwise)
{
  from_graph(graph)->add_pairwise_link(idx_unary0, idx_unary1, idx_pairwise);
}

void mpopt_qap_graph_add_uniqueness_link(mpopt_qap_graph* graph, int idx_unary, int label, int idx_uniqueness, int slot)
{
  from_graph(graph)->add_uniqueness_link(idx_unary, label, idx_uniqueness, slot);
}

mpopt_qap_unary_node* mpopt_qap_graph_get_unary(mpopt_qap_graph* graph, int idx) { return to_unary(from_graph(graph)->get_unary(idx)); }
mpopt_qap_uniqueness_node* mpopt_qap_graph_get_uniqueness(mpopt_qap_graph* graph, int idx) { return to_uniqueness(from_graph(graph)->get_uniqueness(idx)); }
mpopt_qap_pairwise_node* mpopt_qap_graph_get_pairwise(mpopt_qap_graph* graph, int idx) { return to_pairwise(from_graph(graph)->get_pairwise(idx)); }
void mpopt_qap_solver_set_fusion_moves_enabled(mpopt_qap_solver* s, bool enabled) { s->solver.set_fusion_moves_enabled(enabled); }
void mpopt_qap_solver_set_dual_updates_enabled(mpopt_qap_solver* s, bool enabled) { s->solver.set_dual_updates_enabled(enabled); }
void mpopt_qap_solver_set_local_search_enabled(mpopt_qap_solver* s, bool enabled) { s->solver.set_local_search_enabled(enabled); }
void mpopt_qap_solver_set_grasp_alpha(mpopt_qap_solver* s, double alpha) { s->solver.set_grasp_alpha(alpha); }
void mpopt_qap_solver_use_grasp(mpopt_qap_solver* s) { s->solver.use_grasp(); }
void mpopt_qap_solver_use_greedy(mpopt_qap_solver* s) { s->solver.use_greedy(); }
void mpopt_qap_solver_set_random_seed(mpopt_qap_solver *s, const unsigned long seed) { s->solver.set_random_seed(seed); }
void mpopt_qap_solver_run(mpopt_qap_solver* s, int batch_size, int max_batches, int greedy_generations) { s->solver.run(batch_size, max_batches, greedy_generations); }
void mpopt_qap_solver_solve_ilp(mpopt_qap_solver* s) { s->solver.solve_ilp(); }
void mpopt_qap_solver_execute_combilp(mpopt_qap_solver* s) { s->solver.execute_combilp(); }
void mpopt_qap_solver_compute_greedy_assignment(mpopt_qap_solver* s) { s->solver.compute_greedy_assignment(); }
double mpopt_qap_solver_runtime(mpopt_qap_solver* s) { return s->solver.runtime(); }
double mpopt_qap_solver_lower_bound(mpopt_qap_solver* s) { return s->solver.lower_bound(); }
double mpopt_qap_solver_evaluate_primal(mpopt_qap_solver* s) { return s->solver.evaluate_primal(); }

//
// unary factor API
//

void mpopt_qap_unary_set_cost(mpopt_qap_unary_node* n, int label, double cost) { from_unary(n)->factor.set(label, cost); }
double mpopt_qap_unary_get_cost(mpopt_qap_unary_node* n, int label) { return from_unary(n)->factor.get(label); }

int mpopt_qap_unary_get_primal(mpopt_qap_unary_node* n)
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

void mpopt_qap_uniqueness_set_cost(mpopt_qap_uniqueness_node* n, int unary, double cost) { from_uniqueness(n)->factor.set(unary, cost); }
double mpopt_qap_uniqueness_get_cost(mpopt_qap_uniqueness_node* n, int unary) { return from_uniqueness(n)->factor.get(unary); }

int mpopt_qap_uniqueness_get_primal(mpopt_qap_uniqueness_node* n) {
  auto& f = from_uniqueness(n)->factor;
  if (f.is_primal_set())
    return f.primal();
  else
    return -1;
}

//
// pairwise factor API
//

void mpopt_qap_pairwise_set_cost(mpopt_qap_pairwise_node* n, int l0, int l1, double cost) { from_pairwise(n)->factor.set(l0, l1, cost); }
double mpopt_qap_pairwise_get_cost(mpopt_qap_pairwise_node* n, int l0, int l1) { return from_pairwise(n)->factor.get(l0, l1); }

int mpopt_qap_pairwise_get_primal(mpopt_qap_pairwise_node* n, char left_side) {
  auto& f = from_pairwise(n)->factor;
  auto [p0, p1] = f.primal();
  auto p = left_side ? p0 : p1;
  if (p == std::decay_t<decltype(f)>::primal_unset)
    return -1;
  return p;
}

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
