#include <mpopt/gm.hpp>
#include <mpopt/gm.h>

using allocator_type = mpopt::block_allocator<mpopt::cost>;
using solver_type = mpopt::gm::solver<allocator_type>;
using graph_type = mpopt::gm::graph<allocator_type>;
using unary_node_type = mpopt::gm::unary_node<allocator_type>;
using pairwise_node_type = mpopt::gm::pairwise_node<allocator_type>;

struct mpopt_gm_solver_t {
  mpopt::memory_block memory;
  allocator_type allocator;
  solver_type solver;

  mpopt_gm_solver_t()
  : memory()
  , allocator(memory)
  , solver(allocator)
  { }
};

inline auto* to_graph(graph_type* g) { return reinterpret_cast<mpopt_gm_graph*>(g); }
inline auto* from_graph(mpopt_gm_graph* g) { return reinterpret_cast<graph_type*>(g); }

inline auto* to_unary(unary_node_type* g) { return reinterpret_cast<mpopt_gm_unary_node*>(g); }
inline auto* from_unary(mpopt_gm_unary_node* g) { return reinterpret_cast<unary_node_type*>(g); }

inline auto* to_pairwise(pairwise_node_type* g) { return reinterpret_cast<mpopt_gm_pairwise_node*>(g); }
inline auto* from_pairwise(mpopt_gm_pairwise_node* g) { return reinterpret_cast<pairwise_node_type*>(g); }

extern "C" {

//
// solver API
//

mpopt_gm_solver* mpopt_gm_solver_create() { return new mpopt_gm_solver; }
void mpopt_gm_solver_destroy(mpopt_gm_solver* s) { delete s; }
void mpopt_gm_solver_finalize(mpopt_gm_solver* s) { s->memory.finalize(); }

mpopt_gm_graph* mpopt_gm_solver_get_graph(mpopt_gm_solver* s) { return to_graph(&s->solver.get_graph()); }

mpopt_gm_unary_node* mpopt_gm_graph_add_unary(mpopt_gm_graph* graph, int idx, int number_of_labels, int number_of_forward, int number_of_backward)
{
  auto* node = from_graph(graph)->add_unary(idx, number_of_labels, number_of_forward, number_of_backward);
  return to_unary(node);
}

mpopt_gm_unary_node* mpopt_gm_graph_get_unary(mpopt_gm_graph* graph, int idx)
{
  return to_unary(from_graph(graph)->unaries()[idx]);
}

mpopt_gm_pairwise_node* mpopt_gm_graph_add_pairwise(mpopt_gm_graph* graph, int idx, int number_of_labels0, int number_of_labels1)
{
  auto* node = from_graph(graph)->add_pairwise(idx, number_of_labels0, number_of_labels1);
  return to_pairwise(node);
}

void mpopt_gm_graph_add_pairwise_link(mpopt_gm_graph* graph, int idx_unary0, int idx_unary1, int idx_pairwise)
{
  from_graph(graph)->add_pairwise_link(idx_unary0, idx_unary1, idx_pairwise);
}

void mpopt_gm_solver_run(mpopt_gm_solver* s, int batch_size, int max_batches) { s->solver.run(batch_size, max_batches); }
void mpopt_gm_solver_solve_ilp(mpopt_gm_solver* s) { s->solver.solve_ilp(); }
void mpopt_gm_solver_execute_combilp(mpopt_gm_solver* s) {s->solver.execute_combilp(); }
double mpopt_gm_solver_runtime(mpopt_gm_solver* s) { return s->solver.runtime(); }
double mpopt_gm_solver_lower_bound(mpopt_gm_solver* s) { return s->solver.lower_bound(); }
double mpopt_gm_solver_evaluate_primal(mpopt_gm_solver* s) { return s->solver.evaluate_primal(); }

//
// factor API
//

void mpopt_gm_unary_set_cost(mpopt_gm_unary_node* n, int label, double cost) { from_unary(n)->factor.set(label, cost); }

int mpopt_gm_unary_get_primal(mpopt_gm_unary_node* n)
{
  auto &f = from_unary(n)->factor;
  if (f.is_primal_set())
    return f.primal();
  else
    return -1;
}

void mpopt_gm_pairwise_set_cost(mpopt_gm_pairwise_node* n, int l0, int l1, double cost) { from_pairwise(n)->factor.set(l0, l1, cost); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
