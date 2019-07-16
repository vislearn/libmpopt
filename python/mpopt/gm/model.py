import numpy


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
