import copy
import itertools

from . import libct
from .gurobi import Gurobi

#
# Primal class
#

class Primals:

    def __init__(self, model):
        self.model = model
        self._incomings = {}
        self._detections = {}
        self._outgoings = {}

    def __eq__(self, other):
        return (self.model == other.model and
            self._incomings == other._incomings and
            self._detections == other._detections and
            self._outgoings == other._outgoings)

    def helper(self, attr, timestep, detection, value=''):
        d = getattr(self, attr)
        if value == '':
            return d.get((timestep, detection))
        else:
            if value is None:
                d.pop((timestep, detection), None)
            else:
                d[timestep, detection] = value
        self.check_conflicts()

    def incoming(self, timestep, detection, value=''):
        return self.helper('_incomings', timestep, detection, value)

    def detection(self, timestep, detection, value=''):
        return self.helper('_detections', timestep, detection, value)

    def outgoing(self, timestep, detection, value=''):
        return self.helper('_outgoings', timestep, detection, value)

    def evaluate(self):
        result = 0
        for timestep in range(self.model.no_timesteps()):
            for detection in range(self.model.no_detections(timestep)):
                assert self.detection(timestep, detection) in (False, True)
                if self.detection(timestep, detection):
                    c_det, c_app, c_dis = self.model._detections[timestep, detection]
                    result += c_det

                    if self.incoming(timestep, detection) is None:
                        result += c_app

                    if self.outgoing(timestep, detection) is None:
                        result += c_dis

            for conflict in range(self.model.no_conflicts(timestep)):
                detections = self.model._conflicts[timestep, conflict]
                assert sum(self.detection(timestep, d) for d in detections) <= 1

        for k, v in self.model._transitions.items():
            timestep, detection_left, detection_right = k
            slot_left, slot_right, cost = v

            is_left_active = self.outgoing(timestep, detection_left) == slot_left
            is_right_active = self.incoming(timestep+1, detection_right) == slot_right
            assert is_left_active == is_right_active

            if is_left_active and is_right_active:
                result += cost

        for k, v in self.model._divisions.items():
            timestep, detection_left, detection_right_1, detection_right_2 = k
            slot_left, slot_right_1, slot_right_2, cost = v

            is_left_active = self.outgoing(timestep, detection_left) == slot_left
            is_right_active_1 = self.incoming(timestep+1, detection_right_1) == slot_right_1
            is_right_active_2 = self.incoming(timestep+1, detection_right_2) == slot_right_2
            assert is_left_active == is_right_active_1
            assert is_left_active == is_right_active_2

            if is_left_active and is_right_active_1 and is_right_active_2:
                result += cost

        return result

    def check_conflicts(self):
        if __debug__:
            for timestep in range(self.model.no_timesteps()):
                for conflict in range(self.model.no_conflicts(timestep)):
                    detections = self.model._conflicts[timestep, conflict]
                    assert sum(self.detection(timestep, d) == True for d in detections) <= 1


def copy_primals(primals):
    result = copy.copy(primals)
    for attr in ('_incomings', '_detections', '_outgoings'):
        setattr(result, attr, copy.deepcopy(getattr(primals, attr)))
    return result


def extract_primals_from_tracker(model, tracker):
    g = libct.tracker_get_graph(tracker.tracker)
    primals = Primals(model)
    for timestep in range(model.no_timesteps()):
        for detection in range(model.no_detections(timestep)):
            factor = libct.graph_get_detection(g, timestep, detection)
            incoming_primal = libct.detection_get_incoming_primal(factor)
            outgoing_primal = libct.detection_get_outgoing_primal(factor)

            assert (incoming_primal == -1) == (outgoing_primal == -1)
            primals.detection(timestep, detection, (incoming_primal != -1) and (outgoing_primal != -1))

            if incoming_primal >= 0 and incoming_primal < model.no_incoming_edges(timestep, detection):
                primals.incoming(timestep, detection, incoming_primal)

            if outgoing_primal >= 0 and outgoing_primal < model.no_outgoing_edges(timestep, detection):
                primals.outgoing(timestep, detection, outgoing_primal)

    return primals


def extract_primals_from_gurobi(gurobi):
    model = gurobi._model
    primals = Primals(model)
    for timestep in range(model.no_timesteps()):
        for detection in range(model.no_detections(timestep)):
            variables = gurobi._detections[timestep, detection]

            primals.detection(timestep, detection, variables.detection.X > .5)

            for i in range(model.no_incoming_edges(timestep, detection)):
                if variables.incoming[i].X > .5:
                    primals.incoming(timestep, detection, i)

            for i in range(model.no_outgoing_edges(timestep, detection)):
                if variables.outgoing[i].X > .5:
                    primals.outgoing(timestep, detection, i)

    return primals


#
# Build Gurobi problem consisting only of neighboring timesteps and solve
# them to obtain a rounded solution.
#

class ExactNeighbourRounding:

    def __init__(self, model, tracker):
        self.model = model
        self.tracker = tracker
        self.best_primals = None

    def _build_ilp(self, timestep, direction):
        assert timestep >= 0 and timestep < self.model.no_timesteps()
        assert direction in ('forward', 'backward')
        forward = direction == 'forward'
        next_timestep = timestep + (1 if forward else -1)

        timesteps = [timestep]
        if next_timestep >= 0 and next_timestep < self.model.no_timesteps():
            timesteps.append(next_timestep)

        solver = Gurobi()
        solver.construct(self.model, self.tracker, ilp_mode=True,
                         timesteps=timesteps)
        #solver._gurobi.setParam('OutputFlag', False)
        return solver

    def _restrict_ilp_to_primal(self, timestep, direction, solver):
        assert direction in ('forward', 'backward')
        forward = direction == 'forward'
        which_one = 'incoming' if forward else 'outgoing'
        for detection in range(self.model.no_detections(timestep)):
            p = getattr(self._primals, which_one)(timestep, detection)
            detection_vars = solver._detections[timestep, detection]
            if p is not None:
                solver._gurobi.addConstr(getattr(detection_vars, which_one)[p] == 1)
            else:
                solver._gurobi.addConstr(sum(getattr(detection_vars, which_one)[:-1]) == 0)

    def _update_primals_from_ilp(self, timestep, direction, solver):
        assert direction in ('forward', 'backward')
        forward = direction == 'forward'

        if forward:
            next_timestep = timestep + 1
        else:
            next_timestep = timestep - 1

        assert timestep >= 0 and timestep < self.model.no_timesteps()
        #assert next_timestep >= 0 and next_timestep < self.model.no_timesteps()
        # FIXME: Okay, we don't need this because the stuff below just does not
        # fail, but I think we should handle this case more gracefully.

        t_left = min(timestep, next_timestep)
        t_right = max(timestep, next_timestep)

        # left
        for detection in range(self.model.no_detections(t_left)):
            variables = solver._detections[t_left, detection]
            self._primals.detection(t_left, detection, value=variables.detection.X > .5)

            for i, var in enumerate(variables.outgoing[:-1]):
                if var.X > .5:
                    self._primals.outgoing(t_left, detection, value=i)

        # right
        for detection in range(self.model.no_detections(t_right)):
            variables = solver._detections[t_right, detection]
            self._primals.detection(t_right, detection, value=variables.detection.X > .5)

            for i, var in enumerate(variables.incoming[:-1]):
                if var.X > .5:
                    self._primals.incoming(t_right, detection, value=i)

    def _loosen_primals(self, direction, timestep):
        assert direction in ('forward', 'backward')
        if direction == 'forward':
            f = self._primals.incoming
        else:
            f = self._primals.outgoing

        if timestep >= 0 and timestep < self.model.no_timesteps():
            for detection in range(self.model.no_detections(timestep)):
                p_a = f(timestep, detection)
                p_d = self._primals.detection(timestep, detection)
                if (p_d and p_a is None) or not p_d:
                    self._primals.incoming(timestep, detection, None)
                    self._primals.detection(timestep, detection, None)
                    self._primals.outgoing(timestep, detection, None)

    def _single_pass(self, direction):
        assert direction in ('forward', 'backward')
        forward = direction == 'forward'

        timesteps = range(self.model.no_timesteps())
        if forward:
            stepping_func = self.tracker.forward_step
        else:
            timesteps = reversed(timesteps)
            stepping_func = self.tracker.backward_step

        self._primals = Primals(self.model)
        for timestep in timesteps:
            stepping_func(timestep)
            print('{} rounding pass at timestep {}...'.format(direction, timestep))
            solver = self._build_ilp(timestep, direction)
            self._restrict_ilp_to_primal(timestep, direction, solver)
            solver.run()
            self.ilp_time += solver._gurobi.Runtime
            self._update_primals_from_ilp(timestep, direction, solver)
            self._loosen_primals(direction, timestep + (1 if forward else -1))

        if not self.best_primals or self._primals.evaluate() < self.best_primals.evaluate():
            self.best_primals = self._primals

    def run(self):
        self.ilp_time = 0

        self._single_pass('forward')
        fw_primals, fw_ub = self._primals, self._primals.evaluate()

        if False:
            self._single_pass('backward')
            bw_primals, bw_ub = self._primals, self._primals.evaluate()

            print('fw_ub={} bw_ub={} delta={}'.format(fw_ub, bw_ub, abs(fw_ub - bw_ub)))
            print('total ilp runtime for rounding = {}'.format(self.ilp_time))
            return fw_primals if fw_ub < bw_ub else bw_primals
        else:
            return fw_primals
