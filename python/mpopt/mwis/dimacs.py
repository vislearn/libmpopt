from mpopt.mwis.preprocess import prune_positive_nodes


DEFAULT_SCALING_FACTOR = 1000


def write_dimacs(model, f, scaling_factor=DEFAULT_SCALING_FACTOR):
    model, _ = prune_positive_nodes(model)
    no_nodes = len(model.nodes)

    tmp = sum(len(nb) for nb in model.edges)
    assert tmp % 2 == 0
    no_edges = tmp // 2

    # 10 = node weights + no edge weights
    f.write(f'{no_nodes} {no_edges} 10\n')
    for nidx in range(no_nodes):
        cost = int(- model.nodes[nidx] * scaling_factor)
        assert cost >= 0
        neigh = ' '.join(str(x+1) for x in sorted(model.edges[nidx]))
        f.write(f'{cost} {neigh}\n')
