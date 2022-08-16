import sys
import pathlib
import argparse
import warnings

import numpy as np

import principal_harmonics as ph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a principal harmonics dataset')
    parser.add_argument('base_dir', type=pathlib.Path)
    parser.add_argument('--pitch-mode', choices=['constant', 'variable'], default='constant')
    parser.add_argument('--clip-strategy', choices=ph.pvoc.CLIP_STRATEGIES.keys(), default='dont-clip')
    parser.add_argument('--peak-matching', choices=ph.pvoc.PEAK_MATCHERS.keys(), default='simple')

    arg_namespace = parser.parse_args()

    try:
        warnings.simplefilter('error', RuntimeWarning)
        ph.dataset.build_dataset(base_dir=arg_namespace.base_dir,
                                fill_value=np.nan, 
                                pitch_mode=arg_namespace.pitch_mode,
                                interpolate_hole_limit=10,
                                clip_strategy=arg_namespace.clip_strategy,
                                n_periods=6)
    except ph.PhException as e:
        print(f"{e.message}: {e}")
        sys.exit(1) 