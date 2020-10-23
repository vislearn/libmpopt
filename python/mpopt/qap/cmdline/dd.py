#!/usr/bin/env python3

import argparse
import sys

from mpopt import gm, qap, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='qap_dd', description='Optimizer for *.dd quadratic assignment models.')
    parser.add_argument('--relaxation', choices=('gm', 'gm-unordered', 'qap-pw', 'qap'), default='qap')
    parser.add_argument('--maxIterations', type=int, default=1000)
    parser.add_argument('--ilp', action='store_true', help='Solves the ILP after reparametrizing.')
    parser.add_argument('--unary-side', choices=('left', 'right'), default='left', help='Choose side where quadratic terms will be instantiated.')
    parser.add_argument('--combilp', action='store_true', help='Runs CombiLP after reparametrizing.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    deco = qap.ModelDecomposition(model, with_uniqueness=args.relaxation=='qap', unary_side=args.unary_side)
    if args.relaxation in ('gm', 'gm-unordered'):
        gm_model = qap.construct_gm_model(deco)
        if args.relaxation != 'gm-unordered':
            ordered = utils.breadth_first_search(
                len(gm_model.unaries),
                list((u, v) for u, v, c in gm_model.pairwise))
            gm_model.reorder_unaries(ordered)
        solver = gm.construct_solver(gm_model)
    elif args.relaxation in ('qap', 'qap-pw'):
        solver = qap.construct_solver(deco)
    else:
        raise NotImplementedError

    print('initial lower bound: {}'.format(solver.lower_bound()), flush=True)
    solver.run(args.maxIterations)

    if args.ilp:
        solver.solve_ilp()
        print('exact solution: {}'.format(solver.evaluate_primal()))

    if args.combilp:
        solver.execute_combilp()
        print('exact solution: {}'.format(solver.evaluate_primal()))

    primals = qap.extract_primals(deco, solver)
    print('primals consistent: {}'.format('yes' if primals.check_consistency else 'no'))
    print('primals.evaluate: {}'.format(primals.evaluate()))
    print('primals.labeling: {}'.format(primals.labeling))