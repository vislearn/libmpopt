from ..common.solver import BaseSolver
from . import libmpopt_ct as lib
from .primals import Primals

for member in dir(lib):
    if member[:8] == 'tracker_':
        new_member = 'solver_' + member[8:]
        setattr(lib, new_member, getattr(lib, member))


class Tracker(BaseSolver):

    def __init__(self):
        super().__init__(lib)

    @property
    def tracker(self):
        return self.solver

    def forward_step(self, timestep):
        lib.tracker_forward_step(self.tracker, timestep)

    def backward_step(self, timestep):
        lib.tracker_backward_step(self.tracker, timestep)


def construct_tracker(model):
    t = Tracker()
    g = lib.tracker_get_graph(t.tracker)

    detection_map = {} # (timestep, detection) -> detection_object
    for timestep in range(model.no_timesteps()):
        # FIXME: This is not very efficient.
        conflict_counter = {}
        conflict_slots = {}
        for conflict in range(model.no_conflicts(timestep)):
            detections = model._conflicts[timestep, conflict]
            for d in detections:
                conflict_slots[conflict, d] = conflict_counter.get(d, 0)
                model._inc_dict(conflict_counter, d)

        for detection in range(model.no_detections(timestep)):
            d = lib.graph_add_detection(
                g, timestep, detection, model.no_incoming_edges(timestep,
                detection), model.no_outgoing_edges(timestep, detection),
                conflict_counter.get(detection, 0))

            c_det, c_app, c_dis = model._detections[timestep, detection]
            lib.detection_set_detection_cost(d, c_det)
            lib.detection_set_appearance_cost(d, c_app)
            lib.detection_set_disappearance_cost(d, c_dis)
            detection_map[timestep, detection] = d

        for conflict in range(model.no_conflicts(timestep)):
            detections = model._conflicts[timestep, conflict]
            c = lib.graph_add_conflict(g, timestep, conflict, len(detections))
            for i, d in enumerate(detections):
                conflict_count = conflict_counter[d]
                assert conflict_count >= 1
                lib.graph_add_conflict_link(g, timestep, conflict, i, d, conflict_slots[conflict, d])
                conflict_counter[d] = conflict_count - 1

        if __debug__:
            for k, v in conflict_counter.items():
                assert v == 0

    for k, v in model._transitions.items():
        timestep, index_from, index_to = k
        slot_left, slot_right, cost = v

        lib.detection_set_outgoing_cost(detection_map[timestep, index_from], slot_left, cost * .5)
        lib.detection_set_incoming_cost(detection_map[timestep + 1, index_to], slot_right, cost * .5)
        lib.graph_add_transition(g, timestep, index_from, slot_left, index_to, slot_right)

    for k, v in model._divisions.items():
        timestep, index_from, index_to_1, index_to_2 = k
        slot_left, slot_right_1, slot_right_2, cost = v

        lib.detection_set_outgoing_cost(detection_map[timestep, index_from], slot_left, cost / 3.0)
        lib.detection_set_incoming_cost(detection_map[timestep + 1, index_to_1], slot_right_1, cost / 3.0)
        lib.detection_set_incoming_cost(detection_map[timestep + 1, index_to_2], slot_right_2, cost / 3.0)
        lib.graph_add_division(g, timestep, index_from, slot_left, index_to_1, slot_right_1, index_to_2, slot_right_2)

    lib.tracker_finalize(t.tracker)

    return t


def extract_primals_from_tracker(model, tracker):
    incoming_slot_to_transition = {}
    outgoing_slot_to_transition = {}
    for key, (slot_left, slot_right, _) in model._transitions.items():
        timestep, detection_left, detection_right = key
        outgoing_slot_to_transition[timestep, detection_left, slot_left] = key
        incoming_slot_to_transition[timestep+1, detection_right, slot_right] = key

    incoming_slot_to_division = {}
    outgoing_slot_to_division = {}
    for key, (slot_left, slot_right_1, slot_right_2, _) in model._divisions.items():
        timestep, detection_left, detection_right_1, detection_right_2 = key
        outgoing_slot_to_division[timestep, detection_left, slot_left] = key
        incoming_slot_to_division[timestep+1, detection_right_1, slot_right_1] = key
        incoming_slot_to_division[timestep+1, detection_right_2, slot_right_2] = key

    g = lib.tracker_get_graph(tracker.tracker)
    primals = Primals(model)
    for timestep in range(model.no_timesteps()):
        for detection in range(model.no_detections(timestep)):
            factor = lib.graph_get_detection(g, timestep, detection)
            incoming_primal = lib.detection_get_incoming_primal(factor)
            outgoing_primal = lib.detection_get_outgoing_primal(factor)

            assert (incoming_primal == -1) == (outgoing_primal == -1)
            primals.detection(timestep, detection, (incoming_primal != -1) and (outgoing_primal != -1))

            if incoming_primal >= 0 and incoming_primal < model.no_incoming_edges(timestep, detection):
                key = incoming_slot_to_transition.get((timestep, detection, incoming_primal))
                if key is not None:
                    primals.transition(*key, True)

                key = incoming_slot_to_division.get((timestep, detection, incoming_primal))
                if key is not None:
                    primals.division(*key, True)

            if outgoing_primal >= 0 and outgoing_primal < model.no_outgoing_edges(timestep, detection):
                key = outgoing_slot_to_transition.get((timestep, detection, outgoing_primal))
                if key is not None:
                    primals.transition(*key, True)

                key = outgoing_slot_to_division.get((timestep, detection, outgoing_primal))
                if key is not None:
                    primals.division(*key, True)

    assert primals.check_consistency()
    return primals
