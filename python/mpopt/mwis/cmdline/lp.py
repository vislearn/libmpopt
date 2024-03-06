#!/usr/bin/env python3

import argparse

from mpopt import utils, mwis


def parse_time(s):
    if s.endswith('h'):
        return float(s[:-1]) * 3600
    elif s.endswith('m'):
        return float(s[:-1]) * 60
    elif s.endswith('s'):
        return float(s[:-1])
    else:
        return float(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_lp', description='Optimizer for *.jug weighted set packing files.')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Ignores nodes with positive cost.')
    parser.add_argument('-s', '--solver', choices=('cbc', 'cbc-pre', 'gurobi'), default='gurobi')
    parser.add_argument('-i', '--ilp-mode', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-w', '--write')
    parser.add_argument('-t', '--timeout', type=parse_time, default=None)
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)

    #model, _ = mwis.prune_positive_nodes(model)
    #model.nodes = [int(c * 1000) / 1000 for c in model.nodes]

    mapping = None
    if args.preprocess:
        model, mapping = mwis.persistency(model)

        original = len(mapping)
        reduced = sum(1 for x in mapping if x is None)
        print(f'Preprocessing: {original} -> {reduced} nodes ({reduced/original*100:.2f}%)')

    if args.solver == 'cbc':
        solver = mwis.CBC(model, ilp_mode=args.ilp_mode, presolve=False, silent=not args.verbose)
    elif args.solver == 'cbc-pre':
        solver = mwis.CBC(model, ilp_mode=args.ilp_mode, presolve=True, silent=not args.verbose)
    elif args.solver == 'gurobi':
        solver = mwis.Gurobi(model, ilp_mode=args.ilp_mode, silent=not args.verbose, timeout=args.timeout)
    else:
        raise NotImplementedError()

    if args.write:
        solver.write(args.write)
    else:
        solver.solve()
