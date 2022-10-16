from mpopt.mwis.preprocess import prune_negative_nodes

MAXIMUM_TOTAL_COSTS = 2**32


def compute_scaling_factor(model):
    assert all(cost >= 0 for cost in model.nodes)
    total_costs = sum(model.nodes)
    if total_costs < (MAXIMUM_TOTAL_COSTS >> 1) and all(isinstance(cost, int) for cost in model.nodes):
        return 1
    else:
        return (MAXIMUM_TOTAL_COSTS >> 1) / total_costs


def write_dimacs(model, f, scaling_factor=None):
    if any(cost < 0 for cost in model.nodes):
        model, _ = prune_negative_nodes(model)

    if scaling_factor is None:
        scaling_factor = compute_scaling_factor(model)
        print('Automatic scaling factor:', scaling_factor)

    no_nodes = len(model.nodes)
    no_edges = sum(len(nb) for nb in model.edges)
    assert no_edges % 2 == 0
    no_edges = no_edges // 2

    total_cost = sum(int(cost * scaling_factor) for cost in model.nodes)
    assert sum(int(cost * scaling_factor) for cost in model.nodes) < MAXIMUM_TOTAL_COSTS
    if scaling_factor != 1:
        f.write(f'%scaling_factor {scaling_factor}\n')

    # 10 = node weights + no edge weights
    f.write(f'{no_nodes} {no_edges} 10\n')
    for nidx in range(no_nodes):
        cost = int(model.nodes[nidx] * scaling_factor)
        assert cost >= 0
        neigh = ' '.join(str(x+1) for x in sorted(model.edges[nidx]))
        f.write(f'{cost} {neigh}\n')
