from . import libct

class Tracker:

    def __init__(self):
        self.tracker = libct.tracker_create()

    def __del__(self):
        self.destroy()

    def destroy(self):
        if self.tracker is not None:
            libct.tracker_destroy(self.tracker)
            self.tracker = None

    def lower_bound(self):
        return libct.tracker_lower_bound(self.tracker)

    def run(self, max_iterations=1000):
        libct.tracker_run(self.tracker, max_iterations)

    def forward_step(self, timestep):
        libct.tracker_forward_step(self.tracker, timestep)

    def backward_step(self, timestep):
        libct.tracker_backward_step(self.tracker, timestep)


def construct_tracker(model):
    t = Tracker()

    detection_map = {} # (timestep, detection) -> detection_object
    for timestep in range(model.no_timesteps()):
        for detection in range(model.no_detections(timestep)):
            d = libct.tracker_add_detection(t.tracker, timestep, detection,
                    model.no_incoming_edges(timestep, detection),
                    model.no_outgoing_edges(timestep, detection))

            c_det, c_app, c_dis = model._detections[timestep, detection]
            assert c_det <= 0 and c_app >= 0 and c_dis >= 0
            libct.detection_set_detection_cost(d, c_det)
            libct.detection_set_appearance_cost(d, c_app)
            libct.detection_set_disappearance_cost(d, c_dis)

            detection_map[timestep, detection] = d

        conflict_counter = {}
        for conflict in range(model.no_conflicts(timestep)):
            detections = model._conflicts[timestep, conflict]
            for d in detections:
                model._inc_dict(conflict_counter, d)

        for conflict in range(model.no_conflicts(timestep)):
            detections = model._conflicts[timestep, conflict]
            c = libct.tracker_add_conflict(t.tracker, timestep, conflict, len(detections))
            for i, d in enumerate(detections):
                conflict_count = conflict_counter[d]
                assert conflict_count >= 1
                libct.tracker_add_conflict_link(t.tracker, timestep, conflict, i, d, 0.5 / conflict_count)
                conflict_counter[d] = conflict_count - 1

        if __debug__:
            for k, v in conflict_counter.items():
                assert v == 0

    for k, v in model._transitions.items():
        timestep, index_from, index_to = k
        slot_left, slot_right, cost = v
        assert cost >= 0

        libct.detection_set_outgoing_cost(detection_map[timestep, index_from],
                slot_left, cost * .5)
        libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to],
                slot_right, cost * .5)
        libct.tracker_add_transition(t.tracker, timestep, index_from, slot_left,
                index_to, slot_right)

    for k, v in model._divisions.items():
        timestep, index_from, index_to_1, index_to_2 = k
        slot_left, slot_right_1, slot_right_2, cost = v
        assert cost >= 0

        libct.detection_set_outgoing_cost(detection_map[timestep, index_from],
                slot_left, cost / 3.0)
        libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to_1],
                slot_right_1, cost / 3.0)
        libct.detection_set_incoming_cost(detection_map[timestep + 1, index_to_2],
                slot_right_2, cost / 3.0)
        libct.tracker_add_division(t.tracker, timestep, index_from, slot_left,
                index_to_1, slot_right_1, index_to_2, slot_right_2)

    libct.tracker_finalize(t.tracker)

    return t
