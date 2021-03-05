#!/usr/bin/env python3

import argparse
import os
import os.path

from mpopt import utils, mwis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='mwis_batch_convert', description='Converter for *.jug MetaSeg files.')
    parser.add_argument('input_filenames', metavar='INPUT', nargs='+', help='Specifies the *.jug input file.')
    args = parser.parse_args()

    for input_filename in args.input_filenames:
        with utils.smart_open(input_filename, 'rt') as f:
            models = mwis.convert_jug_to_mwis(mwis.parse_jug_model(f))

        print(input_filename, end=': ', flush=True)

        for i, model in enumerate(models):
            output_filename = os.path.join('dimacs', input_filename, f'{i:03d}.dimacs')
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, 'wt') as f:
                mwis.write_dimacs(model, f)
            print('.', end='', flush=True)

            output_filename = os.path.join('mwis', input_filename, f'{i:03d}.mwis')
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            with open(output_filename, 'wt') as f:
                mwis.serialize(model, f)
            print('.', end='', flush=True)

        print()
