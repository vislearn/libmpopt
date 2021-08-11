class Primals:

    def __init__(self, model, labeling=None):
        self.model = model

        if labeling is None:
            self.labeling = [None] * self.model.no_left
        else:
            assert len(labeling) == self.model.no_left
            assert all(right is None or right < self.model.no_right for right in labeling)
            self.labeling = labeling

    def __getitem__(self, key):
        return self.labeling[key]

    def __setitem__(self, key, value):
        assert value < self.model.no_right
        self.labeling[key] = value
        assert len(self.labeling) == self.model.no_left

    def __len__(self):
        return len(self.labeling)

    def check_consistency(self):
        for label in self.labeling:
            if label is not None:
                assert label < self.model.no_right
                if sum(1 for x in self.labeling if x == label) != 1:
                    print('Too frequent label: {}'.format(label))
                    return False
        return True

    def evaluate(self):
        assert self.check_consistency()

        result = 0.0
        active_assignments = [False] * len(self.model.assignments)

        for idx, assignment in enumerate(self.model.assignments):
            if assignment.right == self.labeling[assignment.left]:
                result += assignment.cost
                active_assignments[idx] = True

        for edge in self.model.edges:
            if all(active_assignments[idx] for idx in (edge.assignment1, edge.assignment2)):
                result += edge.cost

        return result
