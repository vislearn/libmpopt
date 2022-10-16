#!/usr/bin/env python3

import argparse

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_json_to_dimacs', description='Converter for *.json MWIS files.')
    parser.add_argument('--scaling-factor', '-f', metavar='FLOAT', help='Scales all cost by fixed factor.')
    parser.add_argument('input_filename', metavar='INPUT', help='Specifies the *.jug input file.')
    parser.add_argument('output_filename', metavar='OUTPUT', help='Specifies the *.dimacs output file.')
    args = parser.parse_args()

    with utils.smart_open(args.input_filename, 'rt') as f:
        model = mwis.deserialize(f)
    
    with utils.smart_open(args.output_filename, 'wt') as f:
        mwis.write_dimacs(model, f, scaling_factor=args.scaling_factor)
