from .libct import *


class Tracker:

    def __init__(self):
        self.tracker = tracker_create()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.tracker is not None:
            tracker_destroy(self.tracker)
            self.tracker = None

    def lower_bound(self):
        return tracker_lower_bound(self.tracker)

    def run(self, max_iterations=1000):
        tracker_run(self.tracker, max_iterations)


class Model:

    def __init__(self):
        self._detections = {}        # (timestep, index) -> cost
        self._no_detections = {}     # timestep -> count

        self._conflicts = {}         # (timestep, index) -> [detection_index]
        self._no_conflicts = {}      # timestep -> count

        self._transitions = {}       # timestep -> (index_from, index_to, cost)
        self._divisions = {}         # timestep -> (index_from, index_to_1, index_to_2, cost)
        self._no_incoming_edges = {} # (timestep, index) -> count
        self._no_outgoing_edges = {} # (timestep, index) -> count

    def add_detection(self, timestep, detection=None, appearance=None, disappearance=None):
        assert(timestep >= 0)
        index = self._no_detections.get(timestep, 0)
        self._detections[(timestep, index)] = (detection, appearance, disappearance)
        self._no_detections[timestep] = index + 1
        return index

    def set_detection_cost(self, timestep, index, detection=None, appearance=None, disappearance=None):
        costs = list(self._detections[(timestep, index)])

        if detection is not None:
            costs[0] = detection

        if appearance is not None:
            costs[1] = appearance

        if disappearance is not None:
            costs[2] = disappearance

        self._detections[(timestep, index)] = tuple(costs)

    def add_conflict(self, timestep, detections):
        assert(timestep >= 0)
        for detection in detections:
            assert((timestep, detection) in self._detections)

        index = self._no_conflicts.get(timestep, 0)
        self._conflicts[(timestep, index)] = detections
        self._no_conflicts[timestep] = index + 1

    def add_transition(self, timestep, index_from, index_to, cost):
        k_left = (timestep, index_from)
        k_right = (timestep + 1, index_to)
        assert(k_left in self._detections)
        assert(k_right in self._detections)

        self._inc_dict(self._no_outgoing_edges, k_left)
        self._inc_dict(self._no_incoming_edges, k_right)

        self._transitions.setdefault(timestep, []).append((index_from, index_to, cost))

    def add_division(self, timestep, index_from, index_to_1, index_to_2, cost):
        k_left = (timestep, index_from)
        k_right1 = (timestep + 1, index_to_1)
        k_right2 = (timestep + 1, index_to_2)
        assert(k_left in self._detections)
        assert(k_right1 in self._detections)
        assert(k_right2 in self._detections)

        self._inc_dict(self._no_outgoing_edges, k_left)
        self._inc_dict(self._no_incoming_edges, k_right1)
        self._inc_dict(self._no_incoming_edges, k_right2)

        self._divisions.setdefault(timestep, []).append((index_from, index_to_1, index_to_2, cost))

    def construct_tracker(self):
        no_timesteps = max(self._no_detections.keys()) + 1
        t = Tracker()

        detection_map = {} # (timestep, detection) -> detection_object
        for timestep in range(no_timesteps):
            for detection in range(self._no_detections.get(timestep, 0)):
                d = tracker_add_detection(t.tracker, timestep, detection,
                        self._no_incoming_edges.get((timestep, detection), 0),
                        self._no_outgoing_edges.get((timestep, detection), 0))

                c_det, c_app, c_dis = self._detections[(timestep, detection)]
                detection_set_detection_cost(d, c_det, 0)
                detection_set_appearance_cost(d, c_app)
                detection_set_disappearance_cost(d, c_dis)

                detection_map[(timestep, detection)] = d

            for conflict in range(self._no_conflicts.get(timestep, 0)):
                detections = self._conflicts[(timestep, conflict)]
                c = tracker_add_conflict(t.tracker, timestep, conflict, len(detections))
                for i, d in enumerate(detections):
                    tracker_add_conflict_link(t.tracker, timestep, conflict, i, d)

        slot_map = {}      # (timestep, detection, left/right) -> slot_index
        for timestep, transitions in self._transitions.items():
            for index_from, index_to, cost in transitions:
                slot_left = self._inc_dict(slot_map, (timestep, index_from, 1))
                slot_right = self._inc_dict(slot_map, (timestep + 1, index_to, 0))
                detection_set_outgoing_cost(detection_map[(timestep, index_from)],
                        slot_left, cost * .5)
                detection_set_incoming_cost(detection_map[(timestep + 1, index_to)],
                        slot_right, cost * .5)
                tracker_add_transition(t.tracker, timestep, index_from, slot_left,
                        index_to, slot_right)

        for timestep, divisions in self._divisions.items():
            for index_from, index_to_1, index_to_2, cost in divisions:
                slot_left = self._inc_dict(slot_map, (timestep, index_from, 1))
                slot_right_1 = self._inc_dict(slot_map, (timestep + 1, index_to_1, 0))
                slot_right_2 = self._inc_dict(slot_map, (timestep + 1, index_to_2, 0))
                detection_set_outgoing_cost(detection_map[(timestep, index_from)],
                        slot_left, cost / 3.0)
                detection_set_incoming_cost(detection_map[(timestep + 1, index_to_1)],
                        slot_right_1, cost / 3.0)
                detection_set_incoming_cost(detection_map[(timestep + 1, index_to_2)],
                        slot_right_2, cost / 3.0)
                tracker_add_division(t.tracker, timestep, index_from, slot_left,
                        index_to_1, slot_right_1, index_to_2, slot_right_2)

        return t

    @staticmethod
    def _inc_dict(d, k):
        v = d.get(k, 0)
        d[k] = v + 1
        return v
