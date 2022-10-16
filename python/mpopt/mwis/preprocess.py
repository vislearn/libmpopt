import random

from mpopt.mwis.model import Model, reduce_model


def _cliques_for_nodes(model):
    result = [[] for _ in model.nodes]
    for cidx, clique in enumerate(model.cliques):
        for nidx in clique:
            result[nidx].append(cidx)
    return result


def _node_optimal_percentage(model):
    cliques_for_node = _cliques_for_nodes(model)
    optimal = [0] * len(model.nodes)
    total = [len(cliques) for cliques in cliques_for_node]

    for nidx, cliques in enumerate(cliques_for_node):
        for cidx in cliques:
            clique = model.cliques[cidx]
            costs = [model.nodes[nidx] for nidx in clique]
            if abs(model.nodes[nidx] - min(costs)) < 1e-3:
                optimal[nidx] += 1

    nan = float('nan')
    return [o / t * 100.0 if t > 0 else nan for o, t in zip(optimal, total)]


def prune_positive_nodes(model):
    mapping = [0 if cost >= 0 else None for cost in model.nodes]
    return reduce_model(model, mapping), mapping


def persistency(model):
    no_nodes = len(model.nodes)
    mapping = [None] * no_nodes

    for nidx in range(no_nodes):
        # Fix nodes with positive cost to 0.
        # Otherwise check if node participates in any clique and if not set
        # it to 1 (cost known to be < 0).
        if model.nodes[nidx] >= 0:
            mapping[nidx] = 0
        elif not model.edges[nidx]:
            mapping[nidx] = 1

    for nidx in range(no_nodes):
        if mapping[nidx] is not None:
            continue

        assert all(mapping[nidx] != 1 for nidx2 in model.edges[nidx])

        cost_nb = sum(model.nodes[nidx2] for nidx2 in model.edges[nidx] if nidx != nidx2 and model.nodes[nidx2] <= 0)
        if model.nodes[nidx] < cost_nb:
            for nidx2 in model.edges[nidx]:
                mapping[nidx2] = 0
            mapping[nidx] = 1

    for nidx0 in range(no_nodes):
        if mapping[nidx0] is not None:
            continue

        for nidx1 in model.edges[nidx0]:
            assert mapping[nidx] is None

            if nidx0 < nidx1:
                cost0, cost1 = model.nodes[nidx0], model.nodes[nidx1]
                nb0, nb1 = model.edges[nidx0], model.edges[nidx1]

                # If one of the clique sets is empty, the nodes are
                # non-comparable as they do not share any cliques.
                if not nb0 or not nb1:
                    continue

                if cost0 <= cost1 and nb0 <= nb1:
                    mapping[nidx1] = 0
                elif cost1 <= cost0 and nb1 <= nb0:
                    mapping[nidx0] = 0

    return reduce_model(model, mapping), mapping


def sample_cliques(model):
    edges = [set(edges) for edges in model.edges]
    uncovered = [set(edges) for edges in model.edges]
    queue = set(range(len(model.nodes)))

    m = Model()
    for cost in model.nodes:
        m.add_node(cost)

    while queue:
        u = random.choice(tuple(queue))
        vs = uncovered[u]
        if not vs:
            queue.remove(u)
            continue

        v = random.choice(tuple(vs))
        clique = {u, v}
        all_neighs = edges[u] & edges[v]
        uncovered_neighs = uncovered[u] & uncovered[v]

        while uncovered_neighs:
            v = random.choice(tuple(uncovered_neighs))
            clique.add(v)
            all_neighs.intersection_update(edges[v])
            uncovered_neighs.intersection_update(uncovered[v])

        while all_neighs:
            v = random.choice(tuple(all_neighs))
            clique.add(v)
            all_neighs.intersection_update(edges[v])

        m.add_clique(tuple(clique))
        for u in clique:
            for v in clique:
                uncovered[u].discard(v)

        print(len(clique), sum(len(edges) for edges in uncovered))

    print()
    print('orig:', len(model.cliques))
    print('redu:', len(m.cliques))

    return m
