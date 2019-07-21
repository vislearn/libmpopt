from collections import namedtuple
import numpy

Assignment = namedtuple('Assignment', 'left right cost')
Edge = namedtuple('Edge', 'assignment1 assignment2 cost')

class Model:

    def __init__(self, no_left, no_right, no_assignments, no_edges):
        self._no_left  = no_left
        self._no_right = no_right

        self._no_assignments = no_assignments
        self._no_edges = no_edges

        self._assignments = []
        self._edges = []
        self._unaries_left = {}
        self._unaries_right = {}

        self._pairwise_left = {}
        self._pairwise_right = {}

        self._no_forward_left = [0] * self._no_left
        self._no_backward_left = [0] * self._no_left
        self._no_forward_right = [0] * self._no_right
        self._no_backward_right = [0] * self._no_right

    def add_assignment(self, id_assignment, id_left, id_right, cost):
        assert id_left  < self._no_left
        assert id_right < self._no_right
        assert id_assignment == len(self._assignments)

        self._assignments.append(Assignment(left=id_left, right=id_right, cost=cost))

        self._unaries_left.setdefault(id_left, []).append(id_assignment)
        self._unaries_right.setdefault(id_right, []).append(id_assignment)

    def add_edge(self, id_assignment1, id_assignment2, cost):
        assert len(self._edges) < self._no_edges

        id_edge = len(self._edges)
        self._edges.append(Edge(assignment1=id_assignment1, assignment2=id_assignment2, cost=cost))

        left1, right1 = self._assignments[id_assignment1][:2]
        left2, right2 = self._assignments[id_assignment2][:2]

        pos_in_left1 = self._unaries_left[left1].index(id_assignment1)
        pos_in_left2 = self._unaries_left[left2].index(id_assignment2)
        pos_in_right1 = self._unaries_right[right1].index(id_assignment1)
        pos_in_right2 = self._unaries_right[right2].index(id_assignment2)

        self._insert_pairwise('left', left1, left2, pos_in_left1, pos_in_left2, cost)
        self._insert_pairwise('right', right1, right2, pos_in_right1, pos_in_right2, cost)

    def _insert_pairwise(self, side, node1, node2, pos_in_node1, pos_in_node2, cost):
        idx1, idx2, pos1, pos2 = sort_ids(node1, node2, pos_in_node1, pos_in_node2)

        pw = getattr(self, '_pairwise_' + side)
        un = getattr(self, '_unaries_' + side)

        if (idx1, idx2) not in pw:
            getattr(self, '_no_forward_' + side)[idx1] += 1
            getattr(self, '_no_backward_' + side)[idx2] += 1
            pw[idx1, idx2] = numpy.zeros((len(un[idx1]), len(un[idx2])))

        assert pw[idx1, idx2][pos1, pos2] == 0.0
        pw[idx1, idx2][pos1, pos2] = cost


def sort_ids(id1, id2, pos1, pos2):
    if id1 < id2:
        return id1, id2, pos1, pos2
    else:
        return id2, id1, pos2, pos1
