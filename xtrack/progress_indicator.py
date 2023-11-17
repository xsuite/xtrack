from math import ceil
from typing import cast, Iterable, Collection

from .general import _print


class DefaultProgressIndicator:
    """Display progress of a task.

    This is a simple progress indicator, with an API that is a subset of the
    one of `tqdm`. It will provide simple progress information if tqdm is
    missing on the system.

    Parameters
    ----------
    iterable: Iterable
        The iterable to iterate over.
    desc: str
        Description of the task.
    total: int, optional
        Total number of iterations, if unspecified `len(iterable)` is used.
    miniters: int, optional
        Minimum number of iterations between updates, by default 1.
    unit_scale: int, optional
        Unused, kept for compatibility with tqdm. To be interpreted as the
        scale by which the iteration number and `total` are multiplied: i.e.,
        one iteration corresponds to `unit_scale` events.
    """
    def __init__(
            self,
            iterable: Iterable,
            desc: str,
            total: int = None,
            miniters: int = None,
            unit_scale: int = None,
    ):
        self.iterable = iterable
        self.desc = desc
        self._iterator = None
        self._iteration = 0
        self._total = total or len(cast(Collection, iterable))
        self._update_interval = miniters or ceil(self._total / 100)
        self._unit_scale = unit_scale or 1
        print(f'Init: interval {self._update_interval}, total {self._total}')

    def __iter__(self):
        self._iterator = iter(self.iterable)
        return self

    def __next__(self):
        if self._iteration % self._update_interval == 0:
            self._print_progress()

        try:
            return next(self._iterator)
        except StopIteration:
            self._print_progress()
            _print()  # finish with a newline
            raise
        finally:
            self._iteration += 1

    def _print_progress(self):
        percent = round(self._iteration * 100 / self._total)
        index = self._iteration * self._unit_scale
        scaled_total = self._total * self._unit_scale
        message = f'{self.desc}: {percent:3}% ({index}/{scaled_total} iterations)'
        _print(message, end='\r', flush=True)


class _ProgressIndicatorConfig:
    default_indicator_cls = DefaultProgressIndicator
    default_options = {}


_config = _ProgressIndicatorConfig()


def set_default_indicator(indicator_cls, **options):
    _config.default_indicator_cls = indicator_cls
    _config.default_options = options


def progress(iterable: Iterable, **options):
    indicator = _config.default_indicator_cls
    return indicator(iterable, **_config.default_options, **options)


try:
    from tqdm.autonotebook import tqdm
    set_default_indicator(tqdm)
except ModuleNotFoundError:
    pass
