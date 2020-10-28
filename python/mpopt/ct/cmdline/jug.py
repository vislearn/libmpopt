#!/usr/bin/env python3

import argparse
import sys

from mpopt import ct, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ct_jug', description='Optimizer for *.jug cell tracking models.')
    parser.add_argument('-B', '--batch-size', type=int, default=ct.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=ct.DEFAULT_MAX_BATCHES)
    parser.add_argument('-o', '--output', default=None, help='Specifies the output file.')
    parser.add_argument('--ilp', choices=('standard', 'decomposed'), help='Solves the ILP after reparametrizing.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model, bimap = ct.convert_jug_to_ct(ct.parse_jug_model(f))

    tracker = ct.construct_tracker(model)
    tracker.run(args.batch_size, args.max_batches)

    if args.ilp:
        if args.ilp == 'standard':
            gurobi = ct.GurobiStandardModel(model)
            gurobi.construct()
            gurobi.update_upper_bound(tracker)
        else:
            gurobi = ct.GurobiDecomposedModel(model, tracker)
            gurobi.construct()
            gurobi.update_upper_bound()
        gurobi.run()
        primals = gurobi.get_primals()
    else:
        primals = ct.extract_primals_from_tracker(model, tracker)

    print('final solution:', primals.evaluate())
    if args.output:
        with open(args.output, 'w') as f:
            ct.format_jug_primals(primals, bimap, f)
