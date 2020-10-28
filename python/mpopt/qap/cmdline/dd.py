#!/usr/bin/env python3

import argparse
import contextlib
import json
import sys

from mpopt import gm, qap, utils


def parse_arguments():
    parser = argparse.ArgumentParser(description='Optimizer for *.dd quadratic assignment models.')
    parser.add_argument('-r', '--relaxation', choices=('gm', 'gm-unordered', 'qap-pw', 'qap'), default='qap')
    parser.add_argument('-B', '--batch-size', type=int, default=qap.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=qap.DEFAULT_MAX_BATCHES)
    parser.add_argument('-u', '--unary-side', choices=('left', 'right'), default='left', help='Choose side where quadratic terms will be instantiated.')
    parser.add_argument('-o', '--output', help='Output file for resulting labeling.')
    parser.add_argument('--ilp', action='store_true', help='Solves the ILP after reparametrizing.')
    parser.add_argument('--combilp', action='store_true', help='Solves the problem using combilp after reparametrizing.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    return parser.parse_args()


def construct_solver(deco, args):
    if args.relaxation in ('gm', 'gm-unordered'):
        gm_model = qap.construct_gm_model(deco)
        if args.relaxation != 'gm-unordered':
            ordered = utils.breadth_first_search(
                len(gm_model.unaries),
                list((u, v) for u, v, c in gm_model.pairwise))
            gm_model.reorder_unaries(ordered)
        return gm.construct_solver(gm_model)
    elif args.relaxation in ('qap', 'qap-pw'):
        return qap.construct_solver(deco)
    else:
        raise NotImplementedError


def write_primals(deco, solver, args, f):
    if args.relaxation.startswith('gm'):
        raise NotImplementedError('Export of primals for GM currently not implemented')

    primals = qap.extract_primals(solver)
    json.dump(primals.labeling, f)
    f.write('\n')


def main():
    args = parse_arguments()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    deco = qap.ModelDecomposition(model, with_uniqueness=args.relaxation=='qap',
                                  unary_side=args.unary_side)

    solver = construct_solver(deco, args)
    print('initial lower bound: {}'.format(solver.lower_bound()), flush=True)
    solver.run(args.batch_size, args.max_batches)

    if args.ilp:
        print('Running ILP solver...', flush=True)
        solver.solve_ilp()
        print('exact solution: {}'.format(solver.evaluate_primal()))

    if args.combilp:
        print('Running CombiLP solver...', flush=True)
        solver.execute_combilp()
        print('exact solution: {}'.format(solver.evaluate_primal()))

    if args.output:
        with open(args.output, 'w') as f:
            write_primals(deco, solver, args, f)


if __name__ == '__main__':
    main()
