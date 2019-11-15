import copy
import itertools

from . import libmpopt_ct as lib
from .gurobi import Gurobi


def extract_primals_from_tracker(model, tracker):
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
