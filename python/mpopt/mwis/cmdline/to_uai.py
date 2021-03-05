#!/usr/bin/env python3

import argparse

from mpopt import mwis, gm, utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_to_uai', description='Converter for Maximum Weighted Independent Set Problems.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.mwis input file.')
    parser.add_argument('output_filename', metavar='OUTPUT', help='Specifies the *.uai output file.')
    args = parser.parse_args()

    print('Loading file...', end=' ', flush=True)
    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)
    print('ok', flush=True)

    #model.constant = round(model.constant * 1000)
    #model.nodes = [round(c * 1000) for c in model.nodes]

    print('Converting model...', end=' ', flush=True)
    gm_model = mwis.convert_to_gm(model)
    print('ok', flush=True)

    with utils.smart_open(args.output_filename, 'wt') as f:
        gm.write_uai_file(gm_model, f)
