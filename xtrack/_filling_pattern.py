import numpy as np


class _FillingPattern:
    """Immutable, host-side representation of a bunch filling pattern.

    This is intentionally an internal helper. Public APIs accept either a
    dense ``filling_pattern`` or sparse ``filled_slots`` and use this class to
    validate and retain the information needed to move safely between the two
    representations.
    """

    __slots__ = ('_filled_slots', '_num_slots')

    def __init__(self, filled_slots, num_slots):
        object.__setattr__(
            self, '_filled_slots', tuple(int(slot) for slot in filled_slots))
        object.__setattr__(self, '_num_slots', int(num_slots))

    def __setattr__(self, name, value):
        raise AttributeError(f'{type(self).__name__} is immutable')

    @classmethod
    def from_inputs(cls, *, filling_pattern=None, filled_slots=None,
                    filling_scheme=None, num_slots=None, allow_none=True):
        provided = [
            name for name, value in (
                ('filling_pattern', filling_pattern),
                ('filled_slots', filled_slots),
                ('filling_scheme', filling_scheme),
            ) if value is not None
        ]
        if len(provided) > 1:
            raise ValueError(
                'Only one of `filling_pattern`, `filled_slots`, and '
                '`filling_scheme` can be provided')
        if not provided:
            if num_slots is not None:
                raise ValueError(
                    '`num_slots` can be provided only with `filled_slots` or '
                    '`filling_pattern`')
            if allow_none:
                return None
            raise ValueError(
                'One of `filling_pattern` or `filled_slots` must be provided')

        normalized_num_slots = _validate_num_slots(num_slots)
        if filling_pattern is None:
            filling_pattern = filling_scheme

        if filling_pattern is not None:
            pattern = np.asarray(filling_pattern)
            if pattern.ndim != 1:
                raise ValueError('`filling_pattern` must be one-dimensional')
            if not np.all((pattern == 0) | (pattern == 1)):
                raise ValueError(
                    '`filling_pattern` can contain only zero and one')
            if (normalized_num_slots is not None
                    and len(pattern) != normalized_num_slots):
                raise ValueError(
                    f'`filling_pattern` has length {len(pattern)}, expected '
                    f'{normalized_num_slots}')
            slots = np.flatnonzero(pattern).astype(np.int64)
            return cls(slots, len(pattern))

        slots = _validate_filled_slots(filled_slots)
        if normalized_num_slots is None:
            normalized_num_slots = 0 if len(slots) == 0 else int(slots[-1]) + 1
        if len(slots) and slots[-1] >= normalized_num_slots:
            raise ValueError(
                f'`filled_slots` contains slot {int(slots[-1])}, which is '
                f'outside `num_slots={normalized_num_slots}`')
        return cls(slots, normalized_num_slots)

    @property
    def num_slots(self):
        return self._num_slots

    @property
    def num_filled_slots(self):
        return len(self._filled_slots)

    @property
    def filled_slots(self):
        return np.asarray(self._filled_slots, dtype=np.int64).copy()

    @property
    def filling_pattern(self):
        pattern = np.zeros(self._num_slots, dtype=np.int64)
        if self._filled_slots:
            pattern[np.asarray(self._filled_slots, dtype=np.int64)] = 1
        return pattern

    def with_num_slots(self, num_slots):
        """Return the same sparse pattern with an explicit total slot count."""
        return type(self).from_inputs(
            filled_slots=self._filled_slots, num_slots=num_slots,
            allow_none=False)

    def normalize_bunch_selection(self, bunch_selection=None):
        """Return validated indices into the ordered list of filled slots."""
        if bunch_selection is None:
            return np.arange(self.num_filled_slots, dtype=np.int64)
        selection = _validate_integer_array(
            bunch_selection, argument_name='bunch_selection')
        if len(np.unique(selection)) != len(selection):
            raise ValueError('`bunch_selection` cannot contain duplicates')
        if np.any(selection < 0) or np.any(selection >= self.num_filled_slots):
            raise ValueError(
                '`bunch_selection` contains an index outside the filled '
                'bunches')
        return selection


def _validate_num_slots(num_slots):
    if num_slots is None:
        return None
    if isinstance(num_slots, (bool, np.bool_)):
        raise ValueError('`num_slots` must be a non-negative integer')
    try:
        normalized = int(num_slots)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('`num_slots` must be a non-negative integer') from None
    if normalized != num_slots or normalized < 0:
        raise ValueError('`num_slots` must be a non-negative integer')
    return normalized


def _validate_filled_slots(filled_slots):
    slots = _validate_integer_array(
        filled_slots, argument_name='filled_slots')
    if np.any(slots < 0):
        raise ValueError('Slot numbers must be non-negative')
    if len(np.unique(slots)) != len(slots):
        raise ValueError('`filled_slots` cannot contain duplicates')
    slots.sort()
    return slots


def _validate_integer_array(values, argument_name):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError(f'`{argument_name}` must be one-dimensional')
    if len(values) == 0:
        return np.array([], dtype=np.int64)
    if (np.issubdtype(values.dtype, np.bool_)
            or not np.issubdtype(values.dtype, np.integer)):
        raise ValueError(f'`{argument_name}` must contain integers')
    return values.astype(np.int64, copy=True)
