#!/usr/bin/env python3

import argparse

from mpopt import mwis, gm, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_to_uai', description='Converter for Maximum Weighted Independent Set Problems.')
    parser.add_argument('-B', '--batch-size', type=int, default=gm.DEFAULT_BATCH_SIZE)
    parser.add_argument('-b', '--max-batches', type=int, default=gm.DEFAULT_MAX_BATCHES)
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.mwis input file.')
    args = parser.parse_args()

    print('Loading file...', end=' ', flush=True)
    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)
    print('ok', flush=True)

    print('Converting model...', end=' ', flush=True)
    gm_model = mwis.convert_to_gm(model)
    print('ok', flush=True)

    solver = gm.construct_solver(gm_model)
    print('initial lower bound: {}'.format(solver.lower_bound()))
    solver.run(args.batch_size, args.max_batches)

    labeling = gm.extract_primals(gm_model, solver)
    print(labeling)
    print(gm_model.evaluate(labeling))

    assignment = mwis.gm_primal_to_mwis_primal(model, gm_model, labeling)
    print(assignment)
    print(model.evaluate(assignment))
