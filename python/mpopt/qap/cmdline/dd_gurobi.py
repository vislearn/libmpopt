#!/usr/bin/env python3

import argparse

from mpopt import qap, utils


def main():
    parser = argparse.ArgumentParser(description='Optimizer for *.dd quadratic assignment models.')
    parser.add_argument('--quadratic-objective', action='store_true', help='Use quadratic objective and to reduce number of linear constraints.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    gurobi_model = qap.GurobiModel(model, quadratic_objective=args.quadratic_objective, ilp_mode=True)
    gurobi_model.optimize()


if __name__ == '__main__':
    main()
