#!/usr/bin/python3

import sys

from mpopt import utils, mwis


if __name__ == '__main__':
    input_filename, output_filename = sys.argv[1:]

    with utils.smart_open(input_filename, 'rt') as f:
        model = mwis.parse_snap_file(f)

    with utils.smart_open(output_filename, 'wt') as f:
        mwis.serialize(model, f, store_edges=False)
