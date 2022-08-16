from abc import ABC, abstractmethod, abstractproperty
from typing import Union

import librosa

import principal_harmonics as ph


class FilenameParseException(RuntimeError):
    """An Error occurred when parsing a filename"""


class FilenameParser(ABC):
    """Abstract class fore filename parsers.

    A filename parser extracts labels for a sample from its filename.
    """
    @abstractproperty
    def FIELDS(self) -> set[str]:
        """The set of names of the fields that are extracted from the filename"""
        raise NotImplementedError()

    @abstractmethod
    def parse(self, spec_str: str) -> dict:
        """Parses a filename `spec_str`, returning a dict of labels.
        The keys of the returned dict should be exactly those returned
        by self.FIELDS.
        """
        pass


class DummyFilenameParser(FilenameParser):
    @property
    def FIELDS(self) -> set[str]:
        return set()
    
    def parse(self, spec_str: str) -> dict:
        return {}


class PhilharmoniaFilenameParser(FilenameParser):
    @property
    def FIELDS(self) -> set[str]:
        return {'instrument', 'midi', 'duration', 'dynamic', 'other'}

    def parse(self, spec_str: str) -> dict:
        fields = spec_str.split('_')
        if not len(fields) == len(self.FIELDS):
            raise FilenameParseException(f"Parsing {spec_str}: "
                                         f"Expected {len(self.FIELDS)} fields, "
                                         f"got {len(fields)}")

        res = {}
        (
            res['instrument'], 
            note,
            res['duration'],   
            res['dynamic'], 
            res['other']
        ) = fields
        note = note.replace('s', '#')

        try:
            res['midi'] = librosa.note_to_midi(note)
        except librosa.ParameterError as e:
            raise FilenameParseException(e)

        return res


FILENAME_PARSERS = {
    'dummy':        DummyFilenameParser,
    'philharmonia': PhilharmoniaFilenameParser
}

def get_filename_parser(parser: Union[str, FilenameParser]):
    """Gets a filename parser, or returns `parser` if it is already 
    a FilenameParser. (For use in parameter sanitation)

    Args:
        strategy (Union[str, FilenameParser]): 

    Raises:
        ph.StrategyException: If the filename parser string is not known.

    Returns:
        ClipStrategy: A filename parser of the speficied type, or `parser` if it was
                      already a FilenameParser
    """
    if isinstance(parser, FilenameParser):
        return parser
    elif parser in FILENAME_PARSERS:
        return FILENAME_PARSERS[parser]()
    else:
        raise ph.StrategyException(f'Filename parser {parser}')