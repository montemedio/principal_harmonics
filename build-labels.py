import sys
import argparse
from pathlib import Path
import warnings

import principal_harmonics as ph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help='directory to read files from')
    parser.add_argument("--glob", help='glob expression to filter input files.', default='*.mp3')
    parser.add_argument("--filename-parser", help='if the filenames have a known syntax, additional information can '
                                                  'be included in labels.csv by parsing the filename',
                        choices=ph.dataset.FILENAME_PARSERS.keys(), default='dummy')
    args_namespace = parser.parse_args()
    input_dir = args_namespace.input_dir
    parser = args_namespace.filename_parser
    glob = args_namespace.glob

    try:
        warnings.simplefilter('error', RuntimeWarning)
        ph.dataset.build_labels(input_dir, glob, parser)
    except ph.PhException as e:
        print(f"{e.message}: {e}")
        sys.exit(1)
