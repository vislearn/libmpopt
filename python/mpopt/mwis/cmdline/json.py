#!/usr/bin/env python3

import argparse
import math
import signal

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_json', description='Optimizer for *.json maximum weighted independent set files.')
    parser.add_argument('-l', '--library', choices=('bregman_exp', 'bregman_log', 'original'), default='bregman_exp')
    parser.add_argument('-B', '--batch-size', type=int, default=mwis.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=mwis.DEFAULT_MAX_BATCHES)
    parser.add_argument('-g', '--greedy-generations', type=int, default=mwis.DEFAULT_GREEDY_GENERATIONS, help='Specify number of greedy generation passes per batch.')
    parser.add_argument('-p', '--python', action='store_true', help='Use Python bregman implementation.')
    parser.add_argument('-t', '--timeout', default=None)
    parser.add_argument('--initial-temperature', type=float)
    parser.add_argument('--threshold-optimality', type=float)
    parser.add_argument('--threshold-stability', type=float)
    parser.add_argument('--temperature-drop-factor', type=float)
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)

    solver = mwis.construct_solver(model, args.library)

    if args.initial_temperature:
        solver.temperature(args.initial_temperature)

    if args.threshold_optimality:
        solver.threshold_optimality(args.threshold_optimality)

    if args.threshold_stability:
        solver.threshold_stability(args.threshold_stability)

    if args.temperature_drop_factor:
        solver.temperature_drop_factor(args.temperature_drop_factor)

    if args.timeout:
        if args.timeout.endswith('h'):
            timeout = float(args.timeout[:-1]) * 3600
        elif args.timeout.endswith('m'):
            timeout = float(args.timeout[:-1]) * 60
        elif args.timeout.endswith('s'):
            timeout = float(args.timeout[:-1])
        else:
            timeout = float(args.timeout)
        signal.alarm(math.ceil(timeout))

    solver.run(args.batch_size, args.max_batches, args.greedy_generations)
    print('\nOk.')
