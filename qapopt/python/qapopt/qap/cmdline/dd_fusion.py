#!/usr/bin/env python3

import argparse
import sys
import json
from collections import deque
import time
import numpy as np

from qapopt import qap, utils


def parse_arguments():
    available_solvers = ['ilp', 'ilp-py', 'lsatr']

    for weak_persistency in (False, True):
        for probe in (False, True):
            for improve in (False, True):
                s = 'qpbo-'
                if weak_persistency:
                    s += 'w'
                if probe:
                    s += 'p'
                if improve:
                    s += 'i'
                available_solvers.append(s.rstrip('-'))

    parser = argparse.ArgumentParser(prog='qap_dd_fusion', description='Optimizer for *.dd quadratic assignment models.')
    #parser.add_argument('--merged-solutions', default=None, help='Specifies *.txt to save all solutions in merging process.')
    parser.add_argument('--output', default=None, help='Specifies *.txt to save the final merged solution to.')
    parser.add_argument('--solver', choices=available_solvers, default='ilp')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.dd input file.')
    parser.add_argument('solutions_filename', metavar="SOLUTIONS", help='File that contains solutions for fusing.')
    return parser.parse_args()


def parse_solutions(model, f):
    def set_captured_energy(primals, energy):
        primals.evaluate = lambda: energy
    result = []
    for line in f:
        obj = json.loads(line)
        labeling, energy = obj['labeling'], obj['energy']
        primals = qap.Primals(model, [x if x != -1 else None for x in labeling])
        assert(abs(primals.evaluate() - energy) < 1e-8)
        # We need to use a separate function here, because the lambda captures
        # the variable binding and not the value. The function creates a new
        # scope.
        set_captured_energy(primals, energy)
        result.append(primals)
    return result


def fusion_move_factory(deco, solver, args):
    if args.solver == 'ilp-py':
        def runner(current, candidate):
            fusion_solver = qap.GurobiFusionModel(qap.FusionMovesModel(model, current, candidate), ilp_mode=True)
            fusion_solver.run(output=0)
            fused = fusion_solver.get_primals()
            fused_ub = fused.evaluate()
            return fused, fused_ub
    elif args.solver == 'ilp':
        def runner(current, candidate):
            solver.execute_fusion_move(
                qap.prepare_import_primal(deco, current),
                qap.prepare_import_primal(deco, candidate))
            fused = qap.extract_primals(deco, solver)
            fused_ub = solver.evaluate_primal()
            return fused, fused_ub
    elif args.solver.startswith('qpbo'):
        options = args.solver[5:]
        w = 'w' in options
        p = 'p' in options
        i = 'i' in options
        def runner(current, candidate):
            solver.execute_qpbo(
                qap.prepare_import_primal(deco, current),
                qap.prepare_import_primal(deco, candidate),
                w, p, i)
            fused = qap.extract_primals(deco, solver)
            fused_ub = solver.evaluate_primal()
            return fused, fused_ub
    elif args.solver == 'qpbo-pi':
        raise NotImplementedError
    elif args.solver == 'lsatr':
        def runner(current, candidate):
            solver.execute_lsatr(
                qap.prepare_import_primal(deco, current),
                qap.prepare_import_primal(deco, candidate))
            fused = qap.extract_primals(deco, solver)
            fused_ub = solver.evaluate_primal()
            return fused, fused_ub
    else:
        raise NotImplementedError

    return runner


def main():
    args = parse_arguments()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = qap.parse_dd_model(f)

    with open(args.solutions_filename) as f:
        solutions = parse_solutions(model, f)

    deco = qap.ModelDecomposition(model, with_uniqueness=True, add_infinity_edge_arcs=False)
    solver = qap.construct_solver(deco)

    # Reverse list so that we can pop individual elements.
    solutions.reverse()
    current = solutions.pop()
    current_ub = current.evaluate()

    runner = fusion_move_factory(deco, solver, args)
    iterations = 0
    time_start = time.monotonic()

    while solutions:
        candidate = solutions.pop()

        candidate_ub = candidate.evaluate()

        fused, fused_ub = runner(current, candidate)
        iterations += 1
        runtime = time.monotonic() - time_start
        print(f'it={iterations} t={runtime} current={current_ub} candidate={candidate_ub} fused={fused_ub}', flush=True)

        if candidate_ub < current_ub:
            current, current_ub = candidate, candidate_ub

        if fused_ub < current_ub:
            current, current_ub = fused, fused_ub

    print('fusing result:', current_ub)

    if args.output:
        with open(args.output, 'wt') as f:
            energy = current.evaluate()
            obj = {'labeling': current.labeling,
                   'energy': energy}
            json.dump(obj, f)
            f.write('\n')


if __name__ == '__main__':
    main()
