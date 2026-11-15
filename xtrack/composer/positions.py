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
class _Position:
    """A normalized high/low pair; neither part is discarded during resolution."""

    hi: float = 0.
    lo: float = 0.

    def __add__(self, other):
        if not isinstance(other, _Position):
            other = _Position(float(other))
        hi, error = _two_sum(self.hi, other.hi)
        hi, lo = _two_sum(hi, fsum((error, self.lo, other.lo)))
        return _Position(hi, lo)

    def __sub__(self, other):
        if not isinstance(other, _Position):
            other = _Position(float(other))
        return self + _Position(-other.hi, -other.lo)

    def __truediv__(self, divisor):
        # Only division by two is needed for center anchors; it is exact here.
        assert divisor == 2
        return _Position(self.hi / 2, self.lo / 2)

    def __float__(self):
        return float(self.hi + self.lo)

    def gap_from(self, upstream, length=0.):
        """Round a local gap once, after canceling the large common position."""
        return fsum((self.hi, self.lo, -upstream.hi, -upstream.lo, -length))


def _position_from_table(table, index):
    error = table['_s_start_error', index] if '_s_start_error' in table else 0.
    return _Position(table['s_start', index], error)


def _remaining_length(table, length):
    return _Position(float(length)).gap_from(
        _position_from_table(table, -1), table.length[-1])


def _line_length(line):
    """Sum retained transport lengths, including the residual of the total."""
    table = line.get_table()
    lengths = table.length * table.isthick
    hi = fsum(lengths)
    lo = fsum([*lengths, -hi])
    return _Position(hi, lo)
