import random
from .model import Model
from .preprocess import sample_cliques


def parse_snap_file(f):
    model = Model()
    no_nodes = 0
    edges = []

    for line in f:
        if line[0] == '#':
            continue

        ustr, vstr = line.rstrip().split('\t')
        u, v = int(ustr), int(vstr)

        if u == v:
            print(f'WARN: Skipping loop {u} -> {v}')
            continue

        no_nodes = max(u+1, v+1, no_nodes)
        edges.append((u, v))

    for u in range(no_nodes):
        #model.add_node(random.randint(1, 200))
        model.add_node(u % 200 + 1)

    for e in edges:
        model.add_clique(e)

    return model
