from collections import namedtuple
import numpy


Assignment = namedtuple('Assignment', 'left right cost')
Edge = namedtuple('Edge', 'assignment1 assignment2 cost')


class Model:

    def __init__(self, no_left, no_right, no_assignments, no_edges):
        self.no_left  = no_left
        self.no_right = no_right

        self.no_assignments = no_assignments
        self.no_edges = no_edges

        self.assignments = [] # [Assignment]
        self.edges = []       # [Edge]
        self.left = {}        # idx_left  -> [idx_assignment]
        self.right = {}       # idx_right -> [idx_assignment]

    def add_assignment(self, id_assignment, id_left, id_right, cost):
        assert id_left  < self.no_left
        assert id_right < self.no_right
        assert id_assignment == len(self.assignments)

        self.assignments.append(Assignment(left=id_left, right=id_right, cost=cost))

        self.left.setdefault(id_left, []).append(id_assignment)
        self.right.setdefault(id_right, []).append(id_assignment)

    def add_edge(self, id_assignment1, id_assignment2, cost):
        assert len(self.edges) < self.no_edges
        self.edges.append(Edge(assignment1=id_assignment1, assignment2=id_assignment2, cost=cost))
