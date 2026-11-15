"""Compensated longitudinal arithmetic used while composing a line."""

from dataclasses import dataclass
from math import fsum


def _two_sum(a, b):
    """Return a rounded sum and its rounding residual (binary64)."""
    total = a + b
    b_rounded = total - a
    error = (a - (total - b_rounded)) + (b - b_rounded)
    return total, error


@dataclass(frozen=True)
class CompensatedFloat:
    """An immutable high/low pair for compensated scalar arithmetic.

    ``hi`` and ``lo`` are binary64 values representing ``hi + lo``. Addition
    and subtraction retain the rounding residual instead of discarding it.
    ``float(value)`` rounds to binary64; ``difference_from`` cancels the common
    part before rounding a local difference. Division by two is supported for
    center anchors; this is not a general-purpose arbitrary-precision type.

    A value can be supplied directly as :class:`xtrack.Place`'s ``at`` argument,
    for example ``xt.CompensatedFloat(20000.) + 1e-12``.
    """

    hi: float = 0.
    lo: float = 0.

    def __add__(self, other):
        if not isinstance(other, CompensatedFloat):
            other = CompensatedFloat(float(other))
        hi, error = _two_sum(self.hi, other.hi)
        hi, lo = _two_sum(hi, fsum((error, self.lo, other.lo)))
        return CompensatedFloat(hi, lo)

    def __sub__(self, other):
        if not isinstance(other, CompensatedFloat):
            other = CompensatedFloat(float(other))
        return self + CompensatedFloat(-other.hi, -other.lo)

    def __truediv__(self, divisor):
        # Only division by two is needed for center anchors; it is exact here.
        assert divisor == 2
        return CompensatedFloat(self.hi / 2, self.lo / 2)

    def __float__(self):
        return float(self.hi + self.lo)

    def difference_from(self, other, offset=0.):
        """Return ``self - other - offset``, rounded once to binary64.

        ``other`` can be a CompensatedFloat or an ordinary scalar. ``offset``
        is an ordinary scalar subtracted before rounding the result.
        """
        if not isinstance(other, CompensatedFloat):
            other = CompensatedFloat(float(other))
        return fsum((self.hi, self.lo, -other.hi, -other.lo, -offset))

    def to_dict(self):
        """Return a JSON-compatible representation of this scalar."""
        return {'__class__': 'CompensatedFloat',
                'hi': float(self.hi), 'lo': float(self.lo)}

    @classmethod
    def from_dict(cls, data):
        """Restore a scalar serialized by :meth:`to_dict`."""
        if data.get('__class__') != 'CompensatedFloat':
            raise ValueError('Expected a serialized CompensatedFloat.')
        return cls(float(data['hi']), float(data['lo']))


def _position_from_table(table, index):
    error = table['_s_start_error', index] if '_s_start_error' in table else 0.
    return CompensatedFloat(table['s_start', index], error)


def _remaining_length(table, length):
    return CompensatedFloat(float(length)).difference_from(
        _position_from_table(table, -1), table.length[-1])


def _line_length(line):
    """Sum retained transport lengths, including the residual of the total."""
    table = line.get_table()
    lengths = table.length * table.isthick
    hi = fsum(lengths)
    lo = fsum([*lengths, -hi])
    return CompensatedFloat(hi, lo)
