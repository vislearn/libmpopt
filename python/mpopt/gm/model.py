import numpy

from ..utils import create_permutation


class Model:

    def __init__(self):
        self.unaries = []
        self.pairwise = []
        self.no_forward = []
        self.no_backward = []

    def add_unary(self, costs):
        index = len(self.unaries)
        self.unaries.append(numpy.asarray(costs, dtype=numpy.float64))
        self.no_forward.append(0)
        self.no_backward.append(0)
        assert len(self.unaries) == len(self.no_forward)
        assert len(self.unaries) == len(self.no_backward)
        return index

    def add_pairwise(self, u, v, costs):
        assert u < v
        l_u = len(self.unaries[u])
        l_v = len(self.unaries[v])
        index = len(self.pairwise)
        self.pairwise.append((u, v, numpy.asarray(costs, dtype=numpy.float64).reshape(l_u, l_v)))
        self.no_forward[u] += 1
        self.no_backward[v] += 1

    def reorder_unaries(self, new_order):
        new_to_old = new_order
        old_to_new = create_permutation(new_order)

        self.unaries = [self.unaries[i] for i in new_to_old]
        self.pairwise = [(old_to_new[u], old_to_new[v], c) for u, v, c in self.pairwise]
        self.no_forward = [self.no_forward[i] for i in new_to_old]
        self.no_backward = [self.no_backward[i] for i in new_to_old]

        for i, (u, v, c) in enumerate(self.pairwise):
            if u > v:
                self.pairwise[i] = (v, u, c.transpose())

                self.no_forward[u] -= 1
                self.no_backward[u] += 1

                self.no_forward[v] += 1
                self.no_backward[v] -= 1
