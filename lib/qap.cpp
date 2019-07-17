#include <mpopt/qap.hpp>
#include <mpopt/qap.h>

using allocator_type = mpopt::block_allocator<mpopt::cost>;
using solver_type = mpopt::qap::solver<allocator_type>;
using graph_type = mpopt::qap::graph<allocator_type>;
using unary_node_type = mpopt::qap::unary_node<allocator_type>;
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

inline auto* to_unary(unary_node_type* g) { return reinterpret_cast<mpopt_qap_unary_node*>(g); }
inline auto* from_unary(mpopt_qap_unary_node* g) { return reinterpret_cast<unary_node_type*>(g); }

inline auto* to_pairwise(pairwise_node_type* g) { return reinterpret_cast<mpopt_qap_pairwise_node*>(g); }
inline auto* from_pairwise(mpopt_qap_pairwise_node* g) { return reinterpret_cast<pairwise_node_type*>(g); }

extern "C" {

//
// solver API
//

mpopt_qap_solver* mpopt_qap_solver_create() { return new mpopt_qap_solver; }
void mpopt_qap_solver_destroy(mpopt_qap_solver* s) { delete s; }
void mpopt_qap_solver_finalize(mpopt_qap_solver* s) { s->memory.finalize(); }

mpopt_qap_graph* mpopt_qap_solver_get_graph(mpopt_qap_solver* s) { return to_graph(&s->solver.get_graph()); }

mpopt_qap_unary_node* mpopt_qap_graph_add_unary(mpopt_qap_graph* graph, int idx, int number_of_labels)
{
  auto* node = from_graph(graph)->add_unary(idx, number_of_labels);
  return to_unary(node);
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

void mpopt_qap_solver_run(mpopt_qap_solver* s, int max_iterations) { s->solver.run(max_iterations); }
double mpopt_qap_solver_lower_bound(mpopt_qap_solver* s) { return s->solver.lower_bound(); }

//
// factor API
//

void mpopt_qap_unary_set_cost(mpopt_qap_unary_node* n, int label, double cost) { from_unary(n)->unary.set(label, cost); }
void mpopt_qap_pairwise_set_cost(mpopt_qap_pairwise_node* n, int l0, int l1, double cost) { from_pairwise(n)->pairwise.set(l0, l1, cost); }

}

/* vim: set ts=8 sts=2 sw=2 et ft=cpp: */
