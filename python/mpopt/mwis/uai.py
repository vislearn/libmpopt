import itertools


COST_INFINITY_STR = '9999'


def write_uai(model, f):
    # header
    f.write('MARKOV\n')

    # number of nodes
    f.write('{}\n'.format(len(model.nodes)))

    # label space (two labels per node)
    for i, _ in enumerate(model.nodes):
        if i != 0:
            f.write(' ')
        f.write('2')
    f.write('\n')

    # number of factors (one per node + one per clique)
    f.write('{}\n'.format(len(model.nodes) + len(model.cliques)))

    # the node indices associated with each factor
    for i, _ in enumerate(model.nodes):
        f.write('1 {}\n'.format(i))

    for i, clique in enumerate(model.cliques):
        f.write('{} {}\n'.format(
            len(clique),
            ' '.join(str(x) for x in clique)))

    # the costs for each factor, prefixed by its total size of costs
    for node_cost in model.nodes:
        f.write('\n2\n0 {:f}\n'.format(node_cost))

    for clique in model.cliques:
        states = [(0, 1)] * len(clique)
        f.write('\n{}\n'.format(2 ** len(clique)))
        for i, state in enumerate(itertools.product(*states)):
            if i != 0:
                f.write(' ')
            if sum(x for x in state) > 1:
                f.write(COST_INFINITY_STR)
            else:
                f.write('0')
        f.write('\n')
