#!/usr/bin/env python3

import argparse
import json
import sys
import time

from qapopt import qap, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='qap_dd_greedy_gen', description='Greedy assignment generation for *.dd quadratic assignment models.')
    parser.add_argument('--relaxation', choices=('qap', 'qap-pw'), default='qap')
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--max-batches', type=int, default=1000)
    parser.add_argument('--generate', type=int, default=10)
    parser.add_argument('--unary-side', choices=('left', 'right'), default='left', help='Choose side where quadratic terms will be instantiated.')
    parser.add_argument('--no-dummy', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    parser.add_argument('output_filename', metavar='OUTPUT', help='Output file for the resulting labelings.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    deco = qap.ModelDecomposition(model, with_uniqueness=args.relaxation=='qap', unary_side=args.unary_side, no_dummy=args.no_dummy)
    solver = qap.construct_solver(deco)

    if args.verbose:
        print('initial lower bound: {}'.format(solver.lower_bound()), flush=True)

    iterations = 0
    time_start = time.monotonic()

    with open(args.output_filename, 'wt') as f:
        def generate_labelings():
            vals = []
            for i in range(args.generate):
                solver.compute_greedy_assignment()
                energy = solver.evaluate_primal()
                if args.verbose:
                    vals.append(energy)
                primals = qap.extract_primals(deco, solver)
                obj = {'labeling': primals.labeling,
                       'energy': energy}
                json.dump(obj, f)
                f.write('\n')

            if args.verbose:
                lb = solver.lower_bound()
                runtime = time.monotonic() - time_start
                print(f'it={iterations} t={runtime} lb={lb} ubs={vals}', flush=True)

        generate_labelings()
        for i in range(args.max_batches):
            solver.run_no_rounding(args.batch_size, 1)
            solver.compute_greedy_assignment()
            iterations += args.batch_size
            generate_labelings()
