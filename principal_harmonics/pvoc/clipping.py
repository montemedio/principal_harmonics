from abc import ABC, abstractmethod
from typing import Union

import pya
import numpy as np

from ..exceptions import *

class ClipStrategy(ABC):
    """Abstract class for clip strategies.

    A clip strategy is responsible for trimming timbre vectors
    to remove silence before and after the note that was analyzed.

    Args:
        ABC (): _description_
    """
    @abstractmethod
    def clip(self, coefs: np.ndarray) -> tuple[int, int]:
        """Given a set of timbre vectors or coefficients, return the frame
        indices at which the vectors should be trimmed.
        """
        pass 


class DontClip(ClipStrategy):
    def clip(self, coefs):
        return 0, coefs.shape[0]


class ClipTransient(ClipStrategy):
    """A strategy for clipping transient instrument sounds. 
    It looks at the first partial, and clips from the
    globally maximal amplitude of the first partial to the first
    time that it contains a hole.
    """
    def clip(self, coefs):
        fst_partial = pya.ampdb(np.abs(coefs[:, 1]) + 1e-12)
        start   = np.nanargmax(fst_partial)

        # find all nans starting from `start`
        nans    = np.isnan(fst_partial[start:])
        nan_ixs = np.flatnonzero(nans)
        if len(nan_ixs) == 0:
            stop = len(coefs)
        else:
            stop = start + nan_ixs[0]
        return start, stop


class ClipStationary(ClipStrategy):
    """A strategy for clipping stationary-timbre sounds.
    Calculates the maximum total amplitude `max_amplitude`. Clips from the
    first time the total amplitude is greater than `threshold * max_amplitude`,
    to the last time the total amplitude is greater than `threshold * max_amplitude`.
    Args:
        threshold (float): The share of the maximal amplitude from which to consider
                           the instrument active. I.e.: If threshold==0.5, 
                           consider the instrument active iff its current total
                           amplitude is greater than 50% of the maximum amplitude
                           found over the course of the sample.
    """
    def __init__(self, threshold=0.3):
        self.threshold = threshold

    def clip(self, coefs):
        total_ampl = np.nansum(np.abs(coefs), axis=1)
        total_ampl /= total_ampl.max()
        exceeded = np.flatnonzero(total_ampl > self.threshold)
        assert len(exceeded) > 0
        return exceeded[0], exceeded[-1]


# Use this to register new clip strategies so they can be found
# by the commandline entry points and `get_clip_strategy()`
CLIP_STRATEGIES = {
    'dont_clip': DontClip,
    'transient': ClipTransient,
    'stationary': ClipStationary
}

def get_clip_strategy(strategy: Union[str, ClipStrategy]) -> ClipStrategy:
    """Gets a clip strategy, or returns `strategy` if it is already 
    a ClipStrategy. (For use in parameter sanitation)

    Args:
        strategy (Union[str, ClipStrategy]): 

    Raises:
        StrategyException: If the strategy string is not known.

    Returns:
        ClipStrategy: A strategy of the speficied type, or `strategy` if it was
                      already a ClipStrategy.
    """
    if isinstance(strategy, ClipStrategy):
        return strategy
    elif strategy in CLIP_STRATEGIES:
        return CLIP_STRATEGIES[strategy]()
    else:
        raise StrategyException(f'Clip strategy {strategy}')
