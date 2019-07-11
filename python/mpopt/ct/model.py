from .tracker import Tracker


class Model:

    def __init__(self):
        self._detections = {}        # (timestep, index) -> (detection_cost, appearance_cost, disappearance_cost)
        self._no_detections = {}     # timestep -> count

        self._conflicts = {}         # (timestep, index) -> [(detection_index, slot)]
        self._no_conflicts = {}      # timestep -> count

        self._transitions = {}       # (timestep, index_from, index_to) -> (out_slot, in_slot, cost)
        self._divisions = {}         # (timestep, index_from, index_to_1, index_to_2) -> (out_slot, in_slot_1, in_slot_2, cost)
        self._no_incoming_edges = {} # (timestep, index) -> count
        self._no_outgoing_edges = {} # (timestep, index) -> count

    def add_detection(self, timestep, detection=None, appearance=None, disappearance=None):
        assert timestep >= 0
        index = self.no_detections(timestep)
        self._detections[timestep, index] = (detection, appearance, disappearance)
        self._no_detections[timestep] = index + 1
        return index

    def set_detection_cost(self, timestep, index, detection=None, appearance=None, disappearance=None):
        costs = list(self._detections[timestep, index])

        if detection is not None:
            costs[0] = detection

        if appearance is not None:
            costs[1] = appearance

        if disappearance is not None:
            costs[2] = disappearance

        self._detections[timestep, index] = tuple(costs)

    def add_conflict(self, timestep, detections):
        assert timestep >= 0
        for detection in detections:
            assert (timestep, detection) in self._detections

        if __debug__:
            # Verify that subset of new conflict set is not already present.
            for k, v in self._conflicts.items():
                if timestep == k[0]:
                    s1, s2 = set(detections), set(v)
                    assert not s1.issubset(s2)
                    assert not s2.issubset(s1)

        index = self.no_conflicts(timestep)
        self._conflicts[timestep, index] = detections
        self._no_conflicts[timestep] = index + 1

    def add_transition(self, timestep, index_from, index_to, cost):
        k_left = (timestep, index_from)
        k_right = (timestep + 1, index_to)
        assert k_left in self._detections
        assert k_right in self._detections

        slot_left = self._inc_dict(self._no_outgoing_edges, k_left)
        slot_right = self._inc_dict(self._no_incoming_edges, k_right)

        k = (timestep, index_from, index_to)
        assert k not in self._transitions
        self._transitions[k] = (slot_left, slot_right, cost)

    def add_division(self, timestep, index_from, index_to_1, index_to_2, cost):
        k_left = (timestep, index_from)
        k_right1 = (timestep + 1, index_to_1)
        k_right2 = (timestep + 1, index_to_2)
        assert k_left in self._detections
        assert k_right1 in self._detections
        assert k_right2 in self._detections

        slot_left = self._inc_dict(self._no_outgoing_edges, k_left)
        slot_right1 = self._inc_dict(self._no_incoming_edges, k_right1)
        slot_right2 = self._inc_dict(self._no_incoming_edges, k_right2)

        k = (timestep, index_from, index_to_1, index_to_2)
        assert k not in self._divisions
        self._divisions[k] = (slot_left, slot_right1, slot_right2, cost)

    def no_timesteps(self):
        return max(self._no_detections.keys()) + 1

    def no_detections(self, timestep):
        return self._no_detections.get(timestep, 0)

    def no_incoming_edges(self, timestep, detection):
        return self._no_incoming_edges.get((timestep, detection), 0)

    def no_outgoing_edges(self, timestep, detection):
        return self._no_outgoing_edges.get((timestep, detection), 0)

    def no_conflicts(self, timestep):
        return self._no_conflicts.get(timestep, 0)

    def dump(self):
        lines = ['m = ct.Model()']

        for k, v in self._detections.items():
            lines.append('m.add_detection({}, {}, {}, {}, {})'.format(*k, *v))

        for k, v in self._conflicts.items():
            lines.append('m.add_conflict({}, {})'.format(k[0], v))

        for k, v in self._transitions.items():
            lines.append('m.add_transition({}, {}, {}, {})'.format(*k, v))

        for k, v in self._divisions.items():
            lines.append('m.add_division({}, {}, {}, {}, {})'.format(*k, v))

        return '\n'.join(lines)

    @staticmethod
    def _inc_dict(d, k):
        v = d.get(k, 0)
        d[k] = v + 1
        return v
