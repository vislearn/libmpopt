#!/usr/bin/env python3

import argparse
import json
import sys

from mpopt import qap, utils


def main():
    parser = argparse.ArgumentParser(description='Optimizer for *.dd quadratic assignment models.')
    parser.add_argument('-o', '--output', help='Output file for resulting labeling.')
    parser.add_argument('--quadratic-objective', action='store_true', help='Use quadratic objective and to reduce number of linear constraints.')
    parser.add_argument('--mode', choices=('ilp', 'lp'), help='Select ILP or LP-relaxation mode.')
    parser.add_argument('--linear-assignment-only', action='store_true', help='Solve LAP by removing pairwise edge terms.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    if args.linear_assignment_only:
        model.edges = []

    gurobi_model = qap.GurobiModel(model, quadratic_objective=args.quadratic_objective, ilp_mode=(args.mode == 'ilp'))
    gurobi_model.optimize()

    if args.output:
        with open(args.output, 'wt') as f:
            json.dump(gurobi_model.assignment(), f)
            f.write('\n')


if __name__ == '__main__':
    main()
