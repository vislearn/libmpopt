#!/usr/bin/env python3

import argparse
import sys

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_json', description='Optimizer for *.json maximum weighted independent set files.')
    parser.add_argument('-B', '--batch-size', type=int, default=mwis.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=mwis.DEFAULT_MAX_BATCHES)
    parser.add_argument('-p', '--preprocess', action='store_true', help='Ignores nodes with positive cost.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)

    mapping = None
    if args.preprocess:
        model, mapping = mwis.persistency(model)

        original = len(mapping)
        reduced = sum(1 for x in mapping if x is None)
        print(f'Preprocessing: {original} -> {reduced} nodes ({reduced/original*100:.2f}%)')

    solver = mwis.construct_solver(model)
    solver.run(args.batch_size, args.max_batches)
