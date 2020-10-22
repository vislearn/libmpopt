#!/usr/bin/env python3

import sys

from mpopt import gm
from mpopt import utils


ENABLE_BREADTH_FIRST_SEARCH = True


if __name__ == '__main__':
    input_filename, = sys.argv[1:]
    with utils.smart_open(input_filename, 'rt') as f:
        model = gm.parse_uai_file(f)

    if ENABLE_BREADTH_FIRST_SEARCH:
        ordered = utils.breadth_first_search(
            len(model.unaries),
            list((u, v) for u, v, c in model.pairwise))
        model.reorder_unaries(ordered)

    solver = gm.construct_solver(model)
    print('initial lower bound: {}'.format(solver.lower_bound()))
    solver.run()
