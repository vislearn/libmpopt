from collections import namedtuple
import re

from .model import Model


class JugBase:

    def __repr__(self):
        args = ['{}={!r}'.format(x, self.__dict__[x]) for x in sorted(self.__dict__.keys())]
        return '{}({})'.format(self.__class__.__name__, ', '.join(args))

    def _yes(self):
        return True

    def _no(self):
        return False

    is_variable = _no
    is_detection = _no
    is_disappearance = _no
    is_appearance = _no
    is_disappearance = _no
    is_move = _no
    is_division = _no
    is_conf_set = _no

    def is_transition(self):
        return self.is_move() or self.is_division()


class JugVariable(JugBase):

    is_variable = JugBase._yes

    def __init__(self, unique_id, cost):
        self.unique_id = unique_id
        self.cost = cost


class JugDetection(JugVariable):

    is_detection = JugVariable._yes

    def __init__(self, timestep, unique_id, cost):
        super().__init__(unique_id, cost)
        self.timestep = timestep


class JugAppearance(JugVariable):

    is_appearance = JugVariable._yes

    def __init__(self, unique_id, detection_id, cost):
        super().__init__(unique_id, cost)
        self.detection_id = detection_id


class JugDisappearance(JugVariable):

    is_disappearance = JugVariable._yes

    def __init__(self, unique_id, detection_id, cost):
        super().__init__(unique_id, cost)
        self.detection_id = detection_id


class JugMove(JugVariable):

    def __init__(self, unique_id, id_from, id_to, cost):
        super().__init__(unique_id, cost)
        self.id_from = id_from
        self.id_to = id_to

    is_move = JugVariable._yes


class JugDivision(JugVariable):

    def __init__(self, unique_id, id_from, id_to_1, id_to_2, cost):
        super().__init__(unique_id, cost)
        self.id_from = id_from
        self.id_to_1 = id_to_1
        self.id_to_2 = id_to_2

    is_division = JugVariable._yes


class JugConfSet(JugBase):

    def __init__(self, detections):
        self.detections = detections

    is_conf_set = JugBase._yes


def parse_jug_model(f):
    re_comment = re.compile(r'^#|^$')
    re_hypothesis = re.compile(r'^H +([0-9]+) +([0-9]+) +([-.0-9]+)')
    re_app = re.compile(r'^(DIS)?APP +([0-9]+) +([0-9]+) +([-.0-9]+)')
    re_move = re.compile(r'^MOVE +([0-9]+) +([0-9]+) +([0-9]+) +([-.0-9]+)')
    re_div = re.compile(r'^DIV +([0-9]+) +([0-9]+) +([0-9]+) +([0-9]+) +([-.0-9]+)')
    re_confset = re.compile(r'^CONFSET +(.+) +<= 1$')

    for line in f:
        line = line.rstrip('\r\n')

        m = re_comment.search(line)
        if m:
            continue

        m = re_hypothesis.search(line)
        if m:
            yield JugDetection(
                    timestep=int(m.group(1)),
                    unique_id=int(m.group(2)),
                    cost=float(m.group(3)))
            continue

        m = re_app.search(line)
        if m:
            ctor = JugDisappearance if m.group(1) == 'DIS' else JugAppearance
            yield ctor(
                    unique_id=int(m.group(2)),
                    detection_id=int(m.group(3)),
                    cost=float(m.group(4)))
            continue

        m = re_move.search(line)
        if m:
            yield JugMove(
                    unique_id=int(m.group(1)),
                    id_from=int(m.group(2)),
                    id_to=int(m.group(3)),
                    cost=float(m.group(4)))
            continue

        m = re_div.search(line)
        if m:
            yield JugDivision(
                    unique_id=int(m.group(1)),
                    id_from=int(m.group(2)),
                    id_to_1=int(m.group(3)),
                    id_to_2=int(m.group(4)),
                    cost=float(m.group(5)))
            continue

        m = re_confset.search(line)
        if m:
            detections = [int(x) for x in re.split(r' *\+ *', m.group(1))]
            yield JugConfSet(detections=detections)
            continue

        raise RuntimeError('Unhandled input line: {}'.format(line))


def convert_jug_to_ct(jug_model):
    m = Model()
    model_to_id = {}
    id_to_model = {}

    def update_bimap(model_key, id_key):
        #assert model_key not in model_to_id
        #assert id_key not in id_to_model
        model_to_id[model_key] = id_key
        id_to_model[id_key] = model_key

    for item in jug_model:
        if item.is_detection():
            r = m.add_detection(item.timestep, detection=item.cost)
            update_bimap(('H', item.timestep, r), ('H', item.unique_id))
        elif item.is_appearance():
            tag, timestep, detection = id_to_model['H', item.detection_id]
            assert(tag == 'H')

            m.set_detection_cost(timestep, detection, appearance=item.cost)
            update_bimap(('APP', timestep, detection), ('A', item.unique_id))
        elif item.is_disappearance():
            tag, timestep, detection = id_to_model['H', item.detection_id]
            assert(tag == 'H')

            m.set_detection_cost(timestep, detection, disappearance=item.cost)
            update_bimap(('DISAPP', timestep, detection), ('A', item.unique_id))
        elif item.is_move():
            tag_from, timestep_from, detection_from = id_to_model['H', item.id_from]
            assert(tag_from == 'H')

            tag_to, timestep_to, detection_to = id_to_model['H', item.id_to]
            assert(tag_to == 'H')
            assert(timestep_from + 1 == timestep_to)

            m.add_transition(timestep_from, detection_from, detection_to, item.cost)
            update_bimap(('MOVE', timestep_from, detection_from, detection_to), ('A', item.unique_id))
        elif item.is_division():
            tag_from, timestep_from, detection_from = id_to_model['H', item.id_from]
            assert(tag_from == 'H')

            tag_to_1, timestep_to_1, detection_to_1 = id_to_model['H', item.id_to_1]
            assert(tag_to_1 == 'H')
            assert(timestep_from + 1 == timestep_to_1)

            tag_to_2, timestep_to_2, detection_to_2 = id_to_model['H', item.id_to_2]
            assert(tag_to_2 == 'H')
            assert(timestep_from + 1 == timestep_to_2)

            m.add_division(timestep_from, detection_from, detection_to_1, detection_to_2, item.cost)
            update_bimap(('DIV', timestep_from, detection_from, detection_to_1, detection_to_2), ('A', item.unique_id))
        elif item.is_conf_set():
            detections = [id_to_model['H', x] for x in item.detections]
            timestep = detections[0][1]
            assert(all(x[0] == 'H' for x in detections))
            assert(all(x[1] == detections[0][1] for x in detections)) # check that timesteps are all identical
            detections = [x[2] for x in detections]
            m.add_conflict(timestep, detections)
        else:
            assert(False)

    return m, (model_to_id, id_to_model)


def format_jug_primals(primals, bimaps, out):
    model_to_id, id_to_model = bimaps

    def unique_id(*args):
        tag, unique_id = model_to_id[args]
        assert tag == 'H' if args[0] == 'H' else 'A'
        return unique_id

    for timestep in range(primals.model.no_timesteps()):
        for detection in range(primals.model.no_detections(timestep)):
            if primals.detection(timestep, detection):
                if primals.incoming(timestep, detection) is None:
                    out.write('APP {}\n'.format(unique_id('APP', timestep, detection)))
                out.write('H {}\n'.format(unique_id('H', timestep, detection)))
                if primals.outgoing(timestep, detection) is None:
                    out.write('DISAPP {}\n'.format(unique_id('DISAPP', timestep, detection)))

    for k, v in primals.model._transitions.items():
        timestep, detection_left, detection_right = k
        slot_left, slot_right, cost = v
        is_left_active = primals.outgoing(timestep, detection_left) == slot_left
        is_right_active = primals.incoming(timestep+1, detection_right) == slot_right
        assert is_left_active == is_right_active
        if is_left_active and is_right_active:
            out.write('MOVE {}\n'.format(unique_id('MOVE', timestep, detection_left, detection_right)))

    for k, v in primals.model._divisions.items():
        timestep, detection_left, detection_right_1, detection_right_2 = k
        slot_left, slot_right_1, slot_right_2, cost = v
        is_left_active = primals.outgoing(timestep, detection_left) == slot_left
        is_right_1_active = primals.incoming(timestep+1, detection_right_1) == slot_right_1
        is_right_2_active = primals.incoming(timestep+1, detection_right_2) == slot_right_2
        assert is_left_active == is_right_1_active and is_left_active == is_right_2_active
        if is_left_active and is_right_1_active and is_right_2_active:
            out.write('DIV {}\n'.format(unique_id('DIV', timestep, detection_left, detection_right_1, detection_right_2)))
