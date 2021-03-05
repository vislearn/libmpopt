import math
import numpy as np
import time

from mpopt.utils import if_debug
from mpopt.mwis.model import random_assignment
from mpopt import gm


INFINITY = 1e10


def are_mwis_and_gm_primal_equal(mwis_model, gm_model, mwis_primal, gm_primal):
    # Some nodes are not "reachable" in the graphical model, because they
    # are not part of any conflict. We search those nodes here.
    referenced = [False] * len(mwis_model.nodes)
    for clique in mwis_model.cliques:
        for nidx in clique:
            referenced[nidx] = True

    # Accumulate the state of all unreachable nodes into a constant. Note
    # that the optimal configuration of unreachable nodes got added as a
    # constant while creating the GM.
    constant = 0
    for nidx, cost in enumerate(mwis_model.nodes):
        if not referenced[nidx]:
            if cost >= 0 and mwis_primal[nidx]:
                constant += cost
            elif cost <= 0 and not mwis_primal[nidx]:
                constant -= cost

    try:
        mwis_obj = mwis_model.evaluate(mwis_primal)
    except:
        mwis_obj = math.inf

    gm_obj = gm_model.evaluate(gm_primal) + constant
    if gm_obj > 1e4:
        gm_obj = math.inf

    if math.isinf(mwis_obj) and math.isinf(gm_obj):
        return True
    else:
        return abs(mwis_obj - gm_obj) < 1e-4


def mwis_primal_to_gm_primal(mwis_model, gm_model, mwis_primal):
    gm_primal = [None] * len(gm_model.unaries)

    # If the GM has one additional node, it corresponds to a constant (node
    # with single label).
    if len(gm_model.unaries) == len(mwis_model.cliques) + 1:
        assert len(gm_model.unaries[-1]) == 1
        gm_primal[-1] = 0

    for cidx, clique in enumerate(mwis_model.cliques):
        label = None
        for l, nidx in enumerate(clique):
            if mwis_primal[nidx]:
                assert label is None
                label = l
        if label is None:
            label = len(clique)
        gm_primal[cidx] = label

    assert all(l is not None for l in gm_primal)
    #assert are_mwis_and_gm_primal_equal(mwis_model, gm_model, mwis_primal, gm_primal)
    return gm_primal


def gm_primal_to_mwis_primal(mwis_model, gm_model, gm_primal):
    mwis_primal = [None] * len(mwis_model.nodes)
    for nidx, cost in enumerate(mwis_model.nodes):
        if not mwis_model.edges[nidx]:
            mwis_primal[nidx] = 1 if cost < 0 else 0

    for cidx, clique in enumerate(mwis_model.cliques):
        l = gm_primal[cidx]
        assert l >= 0 and l <= len(clique)
        if l < len(clique):
            mwis_primal[clique[l]] = 1

    mwis_primal = [x if x is not None else 0 for x in mwis_primal]

    assert all(x in (0, 1) for x in mwis_primal)
    #assert are_mwis_and_gm_primal_equal(mwis_model, gm_model, mwis_primal, gm_primal)
    return mwis_primal


@if_debug
def _verify_equivalence(mwis_model, gm_model):
    print('Verifying equivalence...', end=' ', flush=True)

    #print('-----------------------')
    #print(mwis_model.nodes)
    #print(mwis_model.cliques)
    #print('-----------------------')
    #print(gm_model.unaries)
    #print(gm_model.pairwise)
    #print('-----------------------')

    for i in range(1 * 1000):
        print('.', end='', flush=True)
        mwis_primal = random_assignment(mwis_model)
        gm_primal = mwis_primal_to_gm_primal(mwis_model, gm_model, mwis_primal)
        assert are_mwis_and_gm_primal_equal(mwis_model, gm_model, mwis_primal, gm_primal)

    for i in range(10 * 1000):
        gm_primal = gm.random_labeling(gm_model)
        print(gm_primal)
        mwis_primal = gm_primal_to_mwis_primal(mwis_model, gm_model, gm_primal)
        assert are_mwis_and_gm_primal_equal(mwis_model, gm_model, mwis_primal, gm_primal)

    print('ok')


def convert_to_gm(model):
    # Create copy of the model costs. Different labels can refer to the same
    # node idx and we initially assign the first label the full costs and set
    # all other labels of the same node idx to zero.
    costs = list(model.nodes)
    assert costs == model.nodes

    m = gm.Model()
    for clique in model.cliques:
        unary = m.add_unary([costs[idx] for idx in clique] + [0.0])

        # Further labels that refer to the same node share the cost with the
        # current label. We set the cost for newly created labels hence to
        # zero.
        for idx in clique:
            costs[idx] = 0

    left_overs = [x for x in costs if x < 0]
    if left_overs:
        m.add_unary([sum(left_overs)])

    cliques = [set(x) for x in model.cliques]
    no_cliques = len(model.cliques)
    for u in range(no_cliques):
        for v in range(u + 1, no_cliques):
            X_u = model.cliques[u]
            X_v = model.cliques[v]
            X_both = set.intersection(cliques[u], cliques[v])

            if not X_both:
                continue

            costs = np.zeros((len(X_u) + 1, len(X_v) + 1), dtype=np.int)
            for s, x_u in enumerate(X_u):
                for t, x_v in enumerate(X_v):
                    if (x_u != x_v) and (x_u in X_both or x_v in X_both):
                        costs[s, t] = INFINITY

            # FIXME: This is ugly.
            s = len(X_u)
            for t, x_v in enumerate(X_v):
                if x_v in X_both:
                    costs[s, t] = INFINITY

            # FIXME
            t = len(X_v)
            for s, x_u in enumerate(X_u):
                if x_u in X_both:
                    costs[s, t] = INFINITY

            assert np.count_nonzero(costs)
            m.add_pairwise(u, v, costs)

    #_verify_equivalence(model, m)
    return m
