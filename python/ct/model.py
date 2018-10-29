from .tracker import Tracker
from . import libct


class Model:

    def __init__(self):
        self._detections = {}        # (timestep, index) -> cost
        self._no_detections = {}     # timestep -> count

        self._conflicts = {}         # (timestep, index) -> [detection_index]
        self._no_conflicts = {}      # timestep -> count

        self._transitions = {}       # (timestep, index_from, index_to) -> (out_idx, in_idx, cost)
        self._divisions = {}         # (timestep, index_from, index_to_1, index_to_2) -> (out_idx, in_idx_1, in_idx_2, cost)
        self._no_incoming_edges = {} # (timestep, index) -> count
        self._no_outgoing_edges = {} # (timestep, index) -> count

    def add_detection(self, timestep, detection=None, appearance=None, disappearance=None):
        assert(timestep >= 0)
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
        assert(timestep >= 0)
        for detection in detections:
            assert((timestep, detection) in self._detections)

        if __debug__:
            # Verify that subset of new conflict set is not already present.
            for k, v in self._conflicts.items():
                if timestep == k[0]:
                    s1, s2 = set(detections), set(v)
                    assert(not s1.issubset(s2))
                    assert(not s2.issubset(s1))

        index = self.no_conflicts(timestep)
        self._conflicts[timestep, index] = detections
        self._no_conflicts[timestep] = index + 1

    def add_transition(self, timestep, index_from, index_to, cost):
        k_left = (timestep, index_from)
        k_right = (timestep + 1, index_to)
        assert(k_left in self._detections)
        assert(k_right in self._detections)

        slot_left = self._inc_dict(self._no_outgoing_edges, k_left)
        slot_right = self._inc_dict(self._no_incoming_edges, k_right)

        k = (timestep, index_from, index_to)
        assert(k not in self._transitions)
        self._transitions[k] = (slot_left, slot_right, cost)

    def add_division(self, timestep, index_from, index_to_1, index_to_2, cost):
        k_left = (timestep, index_from)
        k_right1 = (timestep + 1, index_to_1)
        k_right2 = (timestep + 1, index_to_2)
        assert(k_left in self._detections)
        assert(k_right1 in self._detections)
        assert(k_right2 in self._detections)

        slot_left = self._inc_dict(self._no_outgoing_edges, k_left)
        slot_right1 = self._inc_dict(self._no_incoming_edges, k_right1)
        slot_right2 = self._inc_dict(self._no_incoming_edges, k_right2)

        k = (timestep, index_from, index_to_1, index_to_2)
        assert(k not in self._divisions)
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

    def construct_tracker(self):
        t = Tracker()

        detection_map = {} # (timestep, detection) -> detection_object
        for timestep in range(self.no_timesteps()):
            for detection in range(self.no_detections(timestep)):
                d = libct.tracker_add_detection(t.tracker, timestep, detection,
                        self.no_incoming_edges(timestep, detection),
                        self.no_outgoing_edges(timestep, detection))

                c_det, c_app, c_dis = self._detections[timestep, detection]
                assert(c_det <= 0 and c_app >= 0 and c_dis >= 0)
                libct.detection_set_detection_cost(d, c_det, 0)
                libct.detection_set_appearance_cost(d, c_app)
                libct.detection_set_disappearance_cost(d, c_dis)

                detection_map[timestep, detection] = d

            conflict_counter = {}
            for conflict in range(self.no_conflicts(timestep)):
                detections = self._conflicts[timestep, conflict]
                for d in detections:
                    self._inc_dict(conflict_counter, d)

            for conflict in range(self.no_conflicts(timestep)):
                detections = self._conflicts[timestep, conflict]
                c = libct.tracker_add_conflict(t.tracker, timestep, conflict, len(detections))
                for i, d in enumerate(detections):
                    conflict_count = conflict_counter[d]
                    assert(conflict_count >= 1)
                    libct.tracker_add_conflict_link(t.tracker, timestep, conflict, i, d, 1.0 / conflict_count)
                    conflict_counter[d] = conflict_count - 1

            if __debug__:
                for k, v in conflict_counter.items():
                    assert(v == 0)

        for k, v in self._transitions.items():
            timestep, index_from, index_to = k
            slot_left, slot_right, cost = v
            assert(cost >= 0)

            libct.detection_set_outgoing_cost(detection_map[timestep, index_from],
                    slot_left, cost * .5)
            libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to],
                    slot_right, cost * .5)
            libct.tracker_add_transition(t.tracker, timestep, index_from, slot_left,
                    index_to, slot_right)

        for k, v in self._divisions.items():
            timestep, index_from, index_to_1, index_to_2 = k
            slot_left, slot_right_1, slot_right_2, cost = v
            assert(cost >= 0)

            libct.detection_set_outgoing_cost(detection_map[timestep, index_from],
                    slot_left, cost / 3.0)
            libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to_1],
                    slot_right_1, cost / 3.0)
            libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to_2],
                    slot_right_2, cost / 3.0)
            libct.tracker_add_division(t.tracker, timestep, index_from, slot_left,
                    index_to_1, slot_right_1, index_to_2, slot_right_2)

        return t

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
