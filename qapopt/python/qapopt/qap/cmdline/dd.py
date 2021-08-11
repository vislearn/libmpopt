#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys

from qapopt import qap, utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimizer for *.dd quadratic assignment models.')
    parser.add_argument('-r', '--relaxation', choices=('qap', 'qap-pw'), default='qap')
    parser.add_argument('-B', '--batch-size', type=int, default=qap.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=qap.DEFAULT_MAX_BATCHES)
    parser.add_argument('-g', '--greedy-generations', type=int, default=10, help='Specify number of greedy generation passes per batch.')
    parser.add_argument('-u', '--unary-side', choices=('left', 'right'), default='left', help='Choose side where quadratic terms will be instantiated.')
    parser.add_argument('-n', '--no-dummy', action='store_true')
    parser.add_argument('-o', '--output', help='Output file for resulting labeling.')
    parser.add_argument('--ilp', action='store_true', help='Solves the ILP after reparametrizing.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    return parser.parse_args()


def open_maybe(filename, *args, **kwargs):
    if filename is None:
        return contextlib.nullcontext()
    else:
        return open(filename, *args, **kwargs)


def write_primals(deco, solver, f):
    if f:
        primals = qap.extract_primals(deco, solver)
        energy = solver.evaluate_primal()
        obj = {'labeling': primals.labeling,
               'energy': energy}
        json.dump(obj, f)
        f.write('\n')


def main():
    args = parse_arguments()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    deco = qap.ModelDecomposition(model, with_uniqueness=args.relaxation=='qap',
                                  unary_side=args.unary_side,
                                  no_dummy=args.no_dummy)

    solver = qap.construct_solver(deco)
    print('initial lower bound: {}'.format(solver.lower_bound()), flush=True)

    with open_maybe(args.output, 'wt') as f:
        solver.run_rounding_only()
        write_primals(deco, solver, f)

        for i in range(args.max_batches):
            solver.run(args.batch_size, 1)
            write_primals(deco, solver, f)

        if args.ilp:
            print('Running ILP solver...', flush=True)
            solver.solve_ilp()
            print('exact solution: {}'.format(solver.evaluate_primal()))

        write_primals(deco, solver, f)


if __name__ == '__main__':
    main()
