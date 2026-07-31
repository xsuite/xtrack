"""TpsaOptics: read optical functions (and their knob derivatives) off a map.

For a ``ParticlesTpsa`` whose Jacobian is the propagated normalizing matrix ``A``, the uncoupled optical
functions are algebra on ``A``'s columns: ``betx = A00²+A01²``, ``alfx = -(A00·A10+A01·A11)``, ``mux = atan2(A01,A00)/2pi``, and the
dispersion is the delta column.

Because a *parametric* map carries knob dependence in every ``A(i,j)``, the derivative of any
optical function with respect to a knob is a chain rule on the map's mixed coefficients
``d A(i,j)/d knob_k = coeff(coord_i, [var_j=1, knob_k=1])``. No new C: values come from
``jacobian()``, gradients from ``coefficient()``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .particles import ParticlesTpsa

_TWO_PI = 2.0 * np.pi
# (name -> the plane's 2x2 block origin (i0, j0) in A: x at (0,0), y at (2,2)).
_PLANE = {"x": (0, 0), "y": (2, 2)}


class TpsaOptics:
    """Uncoupled optics of a map, with per-knob first-order derivatives.

    Values (``.betx``, ``.alfx``, ``.mux``, ``.dx``, ... floats) come from the map's Jacobian.
    ``.gradient(name)`` returns ``d name / d knob`` (length ``num_params``); needs a knobbed
    map of order >= 2 (the mixed derivative is an order-2 term).
    """

    _NAMES = ("betx", "bety", "alfx", "alfy", "mux", "muy", "dx", "dpx", "dy", "dpy")

    def __init__(self, m: ParticlesTpsa) -> None:
        self._np = m.num_params
        self.knob_names = list(m.knob_names)
        self._J = np.asarray(m.jacobian(), dtype=float)  # 6x6 A-matrix (const parts)
        self._has_order2 = m.order >= 2
        self._m = m
        self._nv = m.num_vars
        # dJ[i, j] = d A(i,j) / d knob (length-np). Built lazily per (i,j) on first use
        # (a value read touches no dJ; a gradient builds only the coefficients it needs).
        self._dJ: dict[tuple[int, int], np.ndarray] = {}

    def _dJij(self, i: int, j: int) -> np.ndarray:
        """``d A(i,j) / d knob`` (length-np), read from the map's mixed coefficients once."""
        v = self._dJ.get((i, j))
        if v is None:
            monos = np.zeros((self._np, self._nv + self._np), dtype=int)
            monos[:, j] = 1
            for k in range(self._np):
                monos[k, self._nv + k] += 1
            v = np.atleast_1d(self._m.coefficient(i, monos))
            self._dJ[(i, j)] = v
        return v

    # --- values -------------------------------------------------------------- #

    def _bet(self, plane: str) -> float:
        i0, j0 = _PLANE[plane]
        return self._J[i0, j0] ** 2 + self._J[i0, j0 + 1] ** 2

    def _alf(self, plane: str) -> float:
        i0, j0 = _PLANE[plane]
        return -(self._J[i0, j0] * self._J[i0 + 1, j0]
                 + self._J[i0, j0 + 1] * self._J[i0 + 1, j0 + 1])

    def _mu(self, plane: str) -> float:
        i0, j0 = _PLANE[plane]
        return np.arctan2(self._J[i0, j0 + 1], self._J[i0, j0]) / _TWO_PI

    @property
    def betx(self) -> float: return self._bet("x")
    @property
    def bety(self) -> float: return self._bet("y")
    @property
    def alfx(self) -> float: return self._alf("x")
    @property
    def alfy(self) -> float: return self._alf("y")
    @property
    def mux(self) -> float: return self._mu("x")
    @property
    def muy(self) -> float: return self._mu("y")
    @property
    def dx(self) -> float: return self._J[0, 5]
    @property
    def dpx(self) -> float: return self._J[1, 5]
    @property
    def dy(self) -> float: return self._J[2, 5]
    @property
    def dpy(self) -> float: return self._J[3, 5]

    def to_dict(self) -> dict[str, float]:
        """All optical function values as ``{name: value}``."""
        return {n: getattr(self, n) for n in self._NAMES}

    # --- knob gradients ------------------------------------------------------ #

    def _need_knobs(self) -> None:
        if self._np == 0:
            raise ValueError("no knobs: build ParticlesTpsa(..., knobs=Knobs(...))")
        if not self._has_order2:
            raise ValueError("knob gradient needs a map of order >= 2 "
                             "(the mixed d A/d knob is an order-2 term)")

    def gradient(self, name: str) -> np.ndarray:
        """``d name / d knob`` as a length-``num_params`` array (knobbed order-2 map)."""
        self._need_knobs()
        if name not in self._NAMES:
            raise KeyError(f"unknown optical function {name!r}; one of {self._NAMES}")
        return getattr(self, "_grad_" + name)()

    def gradients(self) -> dict[str, np.ndarray]:
        """All knob gradients as ``{name: array}``."""
        return {n: self.gradient(n) for n in self._NAMES}

    def _grad_bet(self, plane: str) -> np.ndarray:
        i0, j0 = _PLANE[plane]
        return (2 * self._J[i0, j0] * self._dJij(i0, j0)
                + 2 * self._J[i0, j0 + 1] * self._dJij(i0, j0 + 1))

    def _grad_alf(self, plane: str) -> np.ndarray:
        i0, j0 = _PLANE[plane]
        J = self._J
        return -(self._dJij(i0, j0) * J[i0 + 1, j0] + J[i0, j0] * self._dJij(i0 + 1, j0)
                 + self._dJij(i0, j0 + 1) * J[i0 + 1, j0 + 1]
                 + J[i0, j0 + 1] * self._dJij(i0 + 1, j0 + 1))

    def _grad_mu(self, plane: str) -> np.ndarray:
        i0, j0 = _PLANE[plane]
        a11, a12 = self._J[i0, j0], self._J[i0, j0 + 1]
        return ((a11 * self._dJij(i0, j0 + 1) - a12 * self._dJij(i0, j0))
                / (a11 ** 2 + a12 ** 2) / _TWO_PI)

    def _grad_betx(self): return self._grad_bet("x")
    def _grad_bety(self): return self._grad_bet("y")
    def _grad_alfx(self): return self._grad_alf("x")
    def _grad_alfy(self): return self._grad_alf("y")
    def _grad_mux(self): return self._grad_mu("x")
    def _grad_muy(self): return self._grad_mu("y")
    def _grad_dx(self): return self._dJij(0, 5)
    def _grad_dpx(self): return self._dJij(1, 5)
    def _grad_dy(self): return self._dJij(2, 5)
    def _grad_dpy(self): return self._dJij(3, 5)

    def __repr__(self) -> str:
        return (f"TpsaOptics(betx={self.betx:.6g}, bety={self.bety:.6g}, "
                f"knobs={self.knob_names})")
