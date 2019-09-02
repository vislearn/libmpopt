#!/usr/bin/env python3

import sys

from mpopt.gm.uai import write_uai_file
from mpopt.qap.dd import parse_dd_model
from mpopt.qap.solver import construct_gm_model
from mpopt.utils import smart_open


if __name__ == '__main__':
    input_filename, output_filename = sys.argv[1:]
    with smart_open(input_filename, 'rt') as f_in:
        model = parse_dd_model(f_in)
    model = construct_gm_model(model)
    with smart_open(output_filename, 'wt') as f_out:
        write_uai_file(model, f_out)
