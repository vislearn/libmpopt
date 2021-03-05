#!/usr/bin/env python3

import argparse

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_jug_to_mwis', description='Converter for *.jug MetaSeg files.')
    parser.add_argument('-i', '--model-index', type=int, default=0, help='Specifies model index to load.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    parser.add_argument('output_filename', metavar='OUTPUT', help='Specifies the *.mwis output file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.convert_jug_to_mwis(mwis.parse_jug_model(f))[args.model_index]

    with open(args.output_filename, 'wt') as f:
        mwis.serialize(model, f)
