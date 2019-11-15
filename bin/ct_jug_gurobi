#!/usr/bin/env python3

import sys

from mpopt import ct
from mpopt import utils


if __name__ == '__main__':
    input_filename, = sys.argv[1:]
    with utils.smart_open(input_filename, 'rt') as f:
        model, bimap = ct.convert_jug_to_ct(ct.parse_jug_model(f))

    gurobi = ct.Gurobi(model)
    gurobi.construct()
    gurobi.run()

    primals = ct.extract_primals_from_gurobi(gurobi)
    with open('tracking.sol', 'w') as f:
        ct.format_jug_primals(primals, bimap, f)
