"""Internal immutable records passed between Composer pipeline stages."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PlacementSpec:
    """Caller-owned ``Place`` data copied at the pipeline boundary."""

    source_index: int
    name: Any
    at: Any
    from_: str | None
    anchor: str | None
    from_anchor: str | None


@dataclass(frozen=True)
class ResolvedPlacement:
    """Absolute placement used by coordinate and ordering stages."""

    source_index: int
    name: Any
    table_name: str
    env_name: str
    length: Any
    isthick: bool
    s_start: Any
    from_: str | None
    from_anchor: str | None

    @property
    def s_center(self):
        return self.s_start + self.length / 2

    @property
    def s_end(self):
        return self.s_start + self.length
