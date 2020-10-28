#!/usr/bin/env python3

import argparse
import sys

from mpopt import gm, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='gm_uai', description='Optimizer for *.uai graphical model files.')
    parser.add_argument('-B', '--batch-size', type=int, default=gm.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=gm.DEFAULT_MAX_BATCHES)
    parser.add_argument('--bfs', type=bool, default=True, help='Order nodes by breadth first search.')
    parser.add_argument('--ilp', choices=('standard', 'decomposed'), help='Solves the ILP after reparametrizing.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = gm.parse_uai_file(f)

    if args.bfs:
        ordered = utils.breadth_first_search(
            len(model.unaries),
            list((u, v) for u, v, c in model.pairwise))
        model.reorder_unaries(ordered)

    solver = gm.construct_solver(model)
    print('initial lower bound: {}'.format(solver.lower_bound()))
    solver.run(args.batch_size, args.max_batches)
