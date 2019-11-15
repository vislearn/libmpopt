class Primals:

    def __init__(self, model):
        self.model = model

        # Storage for active variables.
        self._detections = set()
        self._transitions = set()
        self._divisions = set()

        # We count how many active incoming/outgoing edges each detection has.
        # This helps us checking if we have to pay dis-/appearance costs.
        self._counter_incoming = {k: 0 for k in self.model._detections.keys()}
        self._counter_outgoing = {k: 0 for k in self.model._detections.keys()}

    def appearance(self, timestep, detection):
        return self.detection(timestep, detection) and \
               self._counter_incoming[timestep, detection] == 0

    def disappearance(self, timestep, detection):
        return self.detection(timestep, detection) and \
               self._counter_outgoing[timestep, detection] == 0

    def detection(self, timestep, detection, value=None):
        return self._access_helper('_detections', (timestep, detection), value)

    def transition(self, timestep, detection_left, detection_right, value=None):
        return self._access_helper('_transitions', (timestep, detection_left, detection_right), value)

    def division(self, timestep, detection_left, detection_right_1, detection_right_2, value=None):
        return self._access_helper('_divisions', (timestep, detection_left, detection_right_1, detection_right_2), value)

    def check_consistency(self):
        result = True

        # First we count the number of active detections per conflict. They
        # should all be at most 1 as otherwise the primal is not consistent.
        if __debug__:
            counter_conflict = {k: 0 for k in self.model._conflicts.keys()}
            for (timestep, conflict), detections in self.model._conflicts.items():
                for detection in detections:
                    if self.detection(timestep, detection):
                        counter_conflict[timestep, conflict] += 1
            result = result and all(v <= 1 for v in counter_conflict.values())

        # Counters of activated incoming and outgoing transitions/divisions per
        # detection we maintain already. Again, the values should be at most 1.
        # Additionally, we can only have incoming/outgoing edges, if the
        # detection itself is active. We need the value nonetheless as we later
        # will add appearance and disappearance costs if a detection has zero
        # incoming/outgoing edges assigned.
        result = result and all(v <= int(self.detection(*k)) for k, v in self._counter_incoming.items())
        result = result and all(v <= int(self.detection(*k)) for k, v in self._counter_outgoing.items())

        return result

    def evaluate(self):
        assert self.check_consistency()

        result = 0
        for key in self._detections:
            cost_det, cost_app, cost_dis = self.model._detections[key]
            result += cost_det
            if self.appearance(*key):
                result += cost_app
            if self.disappearance(*key):
                result += cost_dis

        for key in self._transitions:
            _, _, cost = self.model._transitions[key]
            result += cost

        for key in self._divisions:
            _, _, _, cost = self.model._divisions[key]
            result += cost

        return result

    def _access_helper(self, attr, key, value=None):
        attr_set = getattr(self, attr)
        if value is None:
            return key in attr_set
        else:
            state_before = key in attr_set
            if value:
                attr_set.add(key)
            else:
                attr_set.discard(key)
            state_after = key in attr_set
            if state_after != state_before:
                getattr(self, '_update' + attr + '_counters')(*key, 1 if value else -1)

    def _update_detections_counters(self, timestep, detection, value):
        pass

    def _update_transitions_counters(self, timestep, detection_left, detection_right, value):
        self._counter_outgoing[timestep, detection_left] += value
        self._counter_incoming[timestep+1, detection_right] += value

    def _update_divisions_counters(self, timestep, detection_left, detection_right_1, detection_right_2, value):
        self._counter_outgoing[timestep, detection_left] += value
        self._counter_incoming[timestep+1, detection_right_1] += value
        self._counter_incoming[timestep+1, detection_right_2] += value
