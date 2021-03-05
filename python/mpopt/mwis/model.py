import itertools
import json
import math
import random

from mpopt.utils import if_debug


class Model:

    def __init__(self):
        self.constant = 0.0
        self.nodes = []
        self.cliques = []
        self.edges = []

    def add_node(self, cost):
        v = len(self.nodes)
        self.nodes.append(cost)
        self.edges.append(set())
        return v

    def add_clique(self, nodes):
        assert len(nodes) >= 2 
        assert all(i >= 0 and i < len(self.nodes) for i in nodes)

        self.cliques.append(tuple(nodes))

        for nidx0 in nodes:
            for nidx1 in nodes:
                if nidx0 != nidx1:
                    self.edges[nidx0].add(nidx1)

    def evaluate(self, assignment):
        if any(x is None for x in assignment):
            return math.inf
        assert all(x in (0, 1, False, True) for x in assignment)
        assert len(assignment) == len(self.nodes)
        assert all(sum(assignment[i] for i in clique) <= 1 for clique in self.cliques)
        return sum(c for c, a in zip(self.nodes, assignment) if a) + self.constant


    @if_debug
    def check_integrity(self):
        for clique in self.cliques:
            assert len(clique) >= 2
            assert all(nidx >= 0 and nidx < len(self.nodes) for nidx in clique)

            dups = {}
            for nidx in clique:
                dups[nidx] = dups.get(nidx, 0) + 1
            for k, v in dups.items():
                assert v == 1

        assert self.edges == _recompute_edges(self)


def random_assignment(model):
    assignment = [None] * len(model.nodes)
    for clique in model.cliques:
        nindices = [nidx for nidx in clique if assignment[nidx] != 0]
        nindices.append(-1)

        active = random.choice(nindices)
        for nidx in clique:
            assignment[nidx] = 1 if nidx == active else 0

    for nidx in range(len(model.nodes)):
        if assignment[nidx] is None:
            assignment[nidx] = random.randint(0, 1)

    return assignment


def _recompute_edges(model):
    edges = [set() for _ in model.nodes]
    for clique in model.cliques:
        for nidx0 in clique:
            for nidx1 in clique:
                if nidx0 != nidx1:
                    edges[nidx0].add(nidx1)
    return edges


@if_debug
def _verify_mapping(model, mapping):
    count_ones = lambda c : sum(1 for i in c if mapping[i] == 1)
    count_nones = lambda c : sum(1 for i in c if mapping[i] is None)

    assert len(model.nodes) == len(mapping)
    assert all(x in (0, 1) or x is None for x in mapping)
    assert all(count_ones(c) <= 1 for c in model.cliques)
    assert all(count_ones(c) == 0 or count_nones(c) == 0 for c in model.cliques)


@if_debug
def _verify_equivalence(old_model, new_model, mapping):
    print('[DBG] Verifying model equivalence... ', end='', flush=True)
    for _ in range(10 * 1000):
        new_assignment = random_assignment(new_model)
        new_objective = new_model.evaluate(new_assignment)
        old_assignemnt = original_assignment(mapping, new_assignment)
        old_objective = old_model.evaluate(old_assignemnt)
        assert(abs(old_objective - new_objective) < 1e-4)
    print('ok.')


def serialize(model, f, store_edges=True):
    obj = {}
    obj['constant'] = model.constant
    obj['nodes'] = model.nodes
    obj['cliques'] = model.cliques
    if store_edges:
        obj['edges'] = [list(nb) for nb in model.edges]
    json.dump(obj, f)


def deserialize(f):
    obj = json.load(f)
    model = Model()
    model.nodes = obj['nodes']
    model.cliques = [tuple(clique) for clique in obj['cliques']]

    if 'constant' in obj:
        model.constant = obj['constant']
    else:
        model.constant = 0.0

    if 'edges' in obj:
        model.edges = [set(nb) for nb in obj['edges']]
    else:
        model.edges = _recompute_edges(model)

    model.check_integrity()
    return model


def reduce_model(model, mapping):
    _verify_mapping(model, mapping)

    # Compute new indices.
    # mapping : old node idx -> fixed label (or None)
    # indices : old node idx -> new node idx (or None)
    it = itertools.count()
    indices = [None if x is not None else next(it) for x in mapping]

    # Helper for clique mapping.
    f = lambda clique: [v for idx in clique if (v := indices[idx]) is not None]

    m = Model()
    m.constant = sum(cost for cost, label in zip(model.nodes, mapping) if label == 1)
    m.nodes = [cost for cost, label in zip(model.nodes, mapping) if label is None]
    m.cliques = [v for c in model.cliques if len(v := f(c)) >= 2]
    m.edges = _recompute_edges(m)

    m.check_integrity()
    _verify_equivalence(model, m, mapping)
    return m


def merge_mappings(old_mapping, new_mapping):
    assert sum(1 for x in old_mapping if x is None) == len(new_mapping)
    it = iter(new_mapping)
    return [x if x is not None else next(it) for x in old_mapping]


def original_assignment(mapping, assignment):
    assert all(x in (0, 1) for x in assignment)
    return merge_mappings(mapping, assignment)
