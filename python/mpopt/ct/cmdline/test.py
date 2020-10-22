#!/usr/bin/env python3

import sys
import random

from mpopt import ct


def random_cost():
    return random.uniform(-300, 300)


def gen_transitions_only(no_timesteps, no_detections):
    model = ct.Model()

    for t in range(no_timesteps):
        for i in range(no_detections):
            model.add_detection(t, random_cost(), random_cost(), random_cost())

    for t in range(no_timesteps-1):
        for i in range(no_detections):
            for j in range(no_detections):
                model.add_transition(t, i, j, random_cost())

    return model


def gen_transitions_divisions_only(no_timesteps, no_detections):
    model = gen_transitions_only(no_timesteps, no_detections)

    for t in range(no_timesteps-1):
        for i in range(no_detections):
            for j in range(no_detections):
                for k in range(j+1, no_detections):
                    model.add_division(t, i, j, k, random_cost())

    return model


def gen_everything(no_timesteps, no_detections, no_conflicts):
    model = gen_transitions_only(no_timesteps, no_detections)

    conflicts = []
    while len(conflicts) < no_conflicts:
        t = random.randrange(no_timesteps)
        i = random.randrange(no_detections-1)
        j = random.randrange(i+1, no_detections)

        conflict = (t, i, j)
        if conflict not in conflicts:
            conflicts.append(conflict)
    print(conflicts)

    for t in range(no_timesteps-1):
        for i in range(no_detections):
            for j in range(no_detections):
                for k in range(j+1, no_detections):
                    if (t+1, j, k) not in conflicts:
                        model.add_division(t, i, j, k, random_cost())

    for t, i, j in conflicts:
        model.add_conflict(t, [i, j])

    return model


def random_models():
    def gen(generator, *args):
        for i in range(5000):
            s = random.getstate()
            n = '{}{}'.format(generator.__name__, repr(tuple(args)))
            m = generator(*args)
            yield s, n, m

    sizes = ((2, 2), (3, 2), (3, 3), (3, 4), (5, 5))

    for size in sizes:
        yield from gen(gen_transitions_only, *size)
        yield from gen(gen_transitions_divisions_only, *size)

    for no_t, no_d in sizes:
        for no_c in range(no_t * no_d // 2):
            yield from gen(gen_everything, no_t, no_d, no_c)


def runner(model):
    tracker = ct.construct_tracker(model)
    tracker.run(1)


if __name__ == '__main__':
    for state, generator, model in random_models():
        print()
        print()
        print('state = {}'.format(state))
        print('generator = {}'.format(generator))
        print(flush=True)
        runner(model)
