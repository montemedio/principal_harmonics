from pathlib import Path
from tqdm import tqdm
from typing import Union

import numpy as np
import pandas as pd
import librosa
import pya

from ..exceptions import *
from .filename_parsers import *


class WriteLabelsException(ph.PhException):
    message = "Could not write labels.csv file"

class ReadLabelsException(ph.PhException):
    message = "Could not read labels.csv file"

class ReadDirectoryException(ph.PhException):
    message = "Could not read from directory"

class WriteAnalysisResultException(ph.PhException):
    message = "Could not write analysis results"


### ----------------------------------------------------------------------


def build_labels(base_dir: Union[str, Path], glob: str,
                 parser: Union[str, 'ph.dataset.FilenameParser']) -> pd.DataFrame:
    """Build a labels.csv file inside a dataset directory"""

    if glob == 'labels.csv':
        raise ParameterException("glob cannot be labels.csv")

    if isinstance(base_dir, str):
        base_dir = Path(base_dir)
    parser = get_filename_parser(parser)

    _ensure_valid_directory(base_dir)
    df = parse_files(base_dir, glob, parser)
    _save_labels(base_dir, df)


def _ensure_valid_directory(base_dir: Path):
    if not base_dir.exists():
        raise ReadDirectoryException(FileNotFoundError(base_dir))
    if not base_dir.is_dir():
        raise ReadDirectoryException(NotADirectoryError(base_dir))

    output_file = base_dir / 'labels.csv'
    if output_file.exists():
        raise WriteLabelsException(FileExistsError(output_file))


def parse_files(base_dir: Path, glob: str, parser: 'ph.dataset.FilenameParser'):
    """Parse all filenames matching `glob` inside `base_dir`, using a given `parser`.
    Returns a pd.DataFrame containing the labels for all files to be analyzed."""
    parser = get_filename_parser(parser)

    files = _list_dir(base_dir, glob)
    default_fields = {'filename', 'midi', 'exclude'}
    additional_fields = parser.FIELDS - default_fields
    df = pd.DataFrame({'filename': [file.name for file in files],
                       'midi': np.nan,
                       'exclude': False} |
                      {field: None for field in additional_fields}) 

    for i, p in tqdm(enumerate(files)):
        filename_fields = parser.parse(p.stem)
        df.loc[i, filename_fields.keys()] = filename_fields.values()

    df['note'] = df['midi'].apply(librosa.midi_to_note)
    return df


def _list_dir(base_dir: Path, glob: str) -> list[Path]:
    try:
        files = list(filter(lambda p: p.name != 'labels.csv',
                            base_dir.glob(glob)))
        print(f"Got {len(files)} files: ")
        for f in files:
            print(f.name)
        print(f"({len(files)} files)")
        return files
    except OSError as e:
        raise ReadDirectoryException(str(e))

    
def _save_labels(base_dir: Path, df: pd.DataFrame):
    try:
        df.to_csv(base_dir / 'labels.csv')
    except Exception as e:
        raise WriteLabelsException(e)


### -----------------------------------------------------------------------


def build_dataset(base_dir: Path,
                  pitch_mode='constant',
                  clip_strategy=None,
                  stride=256, **analysis_args):
    """Run the timbre analysis for all files mentioned in the labels.csv file
    inside `base_dir`. Requires that a labels.csv file exists within `base_dir`.
    Writes the analysis results into `base_dir`.
    """

    labels = _read_labels(base_dir)
    for i, row in tqdm(labels.iterrows()):
        asig_path  = base_dir / row.filename

        asig = pya.Asig(str(asig_path)).norm()
        midi_note = row.midi
        results, metrics = ph.pvoc.analyze(asig, midi_note, pitch_mode, stride, 
                                           clip_strategy, **analysis_args) 

        _report_metrics(asig_path.name, metrics)
        _save_analysis(base_dir, asig_path.stem, results, metrics)


def _report_metrics(filename: str, metrics: dict):
    print(f"Analyzed {filename}:\t{metrics}")


def _save_analysis(target_dir: Path, stem: str, 
                   results: ph.pvoc.TimbreAnalysisResult,
                   metrics: dict):

    try:
        clipped_asig_path = target_dir / (stem + '_clipped.wav')
        results.asig.save_wavfile(str(clipped_asig_path))

        npz_path = target_dir / (stem + '.npz')
        np.savez(npz_path, freqs=results.freqs, coefs=results.coefs, **metrics)

        harmonic_path = target_dir / (stem + '_harmonic-analysis.wav')
        results.harmonic_asig.save_wavfile(str(harmonic_path))

        noise_path = target_dir / (stem + '_noise-residue.wav')
        results.noise_asig.save_wavfile(str(noise_path))

    except OSError as e:
        raise WriteAnalysisResultException(str(e))


### ------------------------------------------------------------


def open_dataset(base_dir: Union[str, Path], expand=None) -> pd.DataFrame:
    """Load a timbre dataset from a directory. 

    Args:
        base_dir (Union[str, Path]): The directory to load the analyses from. Needs to contain
                                     a labels.csv file.
        expand (_type_, optional): _description_. Whether to "expand" some columns using
                                                  `expand_ndarrays`

    Returns:
        pd.DataFrame: A pd.DataFrame containing sample labels and all analysis 
                      results.
    """
    if isinstance(base_dir, str):
        base_dir = Path(base_dir)

    df1 = _read_labels(base_dir)
    # remove rows that have 'exclude' set
    df1 = df1.loc[~df1.exclude]

    # build paths to analysis results, read analysis results
    records = []
    for fname in df1.filename:
        p = Path(fname)
        stem = p.stem

        record = {}
        record['filename']    = fname
        record['clipped_wav'] = base_dir / (stem + '_clipped.wav')
        record['resynth_wav'] = base_dir / (stem + '_harmonic-analysis.wav')
        record['noise_wav']   = base_dir / (stem + '_noise-residue.wav')

        with np.load(base_dir / (stem + '.npz')) as analysis_results:
            assert 'freqs' in analysis_results.keys() and 'coefs' in analysis_results.keys()
            for key in analysis_results:
                val = analysis_results[key]
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    # downcast 0d arrays so they can be properly handled by pandas
                    val = val[()]
                record[key] = val

        records.append(record)
    df2 = pd.DataFrame(records)
    df = df1.merge(df2, on='filename')
    df.set_index('filename', inplace=True)
    
    if expand:
        return expand_ndarrays(df, expand)
    else:
        return df


def expand_ndarrays(df: pd.DataFrame, arr_cols: Union[str, list[str]]) -> pd.DataFrame:
    """Unwrap dataframe columns into the dataframe. 

    When loading a dataset from disk, it will contain a `freqs` column and a
    `coefs` column. Each value of the column is a 2d array. To get easier
    access to the data within the arrays, this function "expands" the
    arrays: The resulting dataframe will be Multi-indexed, with the slow
    index being the index of the input dataframe. 

    See 1-serras-analysis.ipynb for an illustration.

    Args:
        df (pd.DataFrame): The input dataframe
        arr_cols (Union[str, list[str]]): The names of the columns to expand. All
                                          2d arrays in the columns must have the same number of columns,
                                          but do not need to have the same number of rows.

    Returns:
        pd.DataFrame: The expanded dataframe.
    """
    if isinstance(arr_cols, str):
        arr_cols = [arr_cols]
    elif not isinstance(arr_cols, list):
        raise TypeError("Cols need to be a string or a list")

    # Input Validation
    if not set(arr_cols) < set(df.columns):
        raise KeyError("Unknown columns")
    is_ndarray = df[arr_cols].applymap(lambda arr: isinstance(arr, np.ndarray) 
                                                   and arr.ndim == 2)
    if not is_ndarray.all(axis=None):
        raise TypeError()
    nrows = df[arr_cols].applymap(lambda arr: arr.shape[0])
    ncols = df[arr_cols].applymap(lambda arr: arr.shape[1])
    if (not nrows.nunique(axis=1).eq(1).all() or 
        not ncols.nunique(axis=0).eq(1).all):
        raise TypeError("Got uneven array shapes")
    nrows = nrows.iloc[:, 0]
    ncols = ncols.iloc[0]

    remaining_cols = list(set(df.columns) - set(arr_cols))
    remaining_df = df.loc[:, remaining_cols]

    # Make index
    if remaining_df.index.name is None:
        remaining_df.index.name = 'sample_ix'
    slow_ix_name = remaining_df.index.name
    slow_ixs = remaining_df.index.repeat(nrows)
    fast_ixs = np.concatenate([np.arange(n) for n in nrows])
    multi_ix = pd.MultiIndex.from_arrays([slow_ixs, fast_ixs],
                                         names=(slow_ix_name, 'n'))

    # Merge
    dfs_tmp = []
    for i, arr_col in enumerate(arr_cols):
        column_names = [f'{arr_col}-{j}' for j in range(ncols.iloc[i])]
        df_tmp = pd.DataFrame(np.vstack(df[arr_col]), columns=column_names)
        dfs_tmp.append(df_tmp)
    expanded = pd.concat(dfs_tmp, axis='columns')
    expanded.set_index(multi_ix, inplace=True)
    
    return expanded.join(remaining_df, 
                         on=slow_ix_name)


### ---------------------------------------

def _read_labels(base_dir: Path) -> pd.DataFrame:
    label_path = base_dir / 'labels.csv'
    try:
        df = pd.read_csv(label_path)
        df.index.name = 'file_ix'
        return df
    except (OSError, ValueError) as e:
        raise ReadLabelsException(str(e))