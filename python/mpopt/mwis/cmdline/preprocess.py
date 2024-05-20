#!/usr/bin/env python3

import argparse
import sys

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_preprocess', description='TODO')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.json input file.')
    parser.add_argument('output_filename', metavar='OUTPUT', help='Specifies the *.json output file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)

    model, mapping = mwis.persistency(model)

    original = len(mapping)
    reduced = sum(1 for x in mapping if x is None)
    print(f'Preprocessing: {original} -> {reduced} nodes ({reduced/original*100:.2f}%)')

    with utils.smart_open(args.output_filename, 'wt') as f:
        mwis.serialize(model, f)
