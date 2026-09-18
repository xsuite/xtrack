from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import xtrack as xt
from xtrack.particles.particles import dptau2ddelta, ptau2delta
from xtrack.survey import SurveyTable

from .match import Action

from .general import _print

# Xsuite name -> MAD-NG name. Not injective: ``delta`` and ``ptau`` are both
# ``pt`` in MAD-NG, which is why the reverse map is spelled out separately
# rather than derived from this one.
NG_XS_MAP = {
    'beta11': 'betx',
    'beta22': 'bety',
    'alfa11': 'alfx',
    'alfa22': 'alfy',
    'mu1': 'mux',
    'mu2': 'muy',
}

XS_NG_MAP = {
    'betx': 'beta11',
    'bety': 'beta22',
    'alfx': 'alfa11',
    'alfy': 'alfa22',
    'mux': 'mu1',
    'muy': 'mu2',
    'dx': 'dx',
    'dpx': 'dpx',
    'x': 'x',
    'px': 'px',
    'y': 'y',
    'py': 'py',
    'zeta': 't',
    'delta': 'pt',
    'ptau': 'pt',
}

# fmt: off
BETA0_COLUMNS = [
    'x', 'px', 'y', 'py', 't', 'pt',
    'dx', 'dy', 'dpx', 'dpy', 'ddx', 'ddpx', 'ddy', 'ddpy',
    'wx', 'wxp', 'wy', 'wyp',
    'mu1', 'mu2', 'mu3', 'dmu1', 'dmu2', 'dmu3',
    'r11', 'r12', 'r21', 'r22',
    'alfa11', 'alfa12', 'alfa13', 'alfa21', 'alfa22', 'alfa23', 'alfa31', 'alfa32', 'alfa33',
    'beta11', 'beta12', 'beta13', 'beta21', 'beta22', 'beta23', 'beta31', 'beta32', 'beta33',
    'gama11', 'gama12', 'gama13', 'gama21', 'gama22', 'gama23', 'gama31', 'gama32', 'gama33',
]

TW_BASE_COLUMNS = [
    's',
    'beta11', 'beta22', 'beta33',
    'alfa11', 'alfa22', 'alfa33',
    'gama11', 'gama22', 'gama33',
    'x', 'px', 'y', 'py', 't', 'pt',
    'dx', 'dy', 'dpx', 'dpy',
    'mu1', 'mu2', 'mu3',
]

OPTFUN_QUANTITIES = [
    'beta11', 'beta22', 'alfa11', 'alfa22', 'gama11', 'gama22', 'mu1', 'mu2',
    'dx', 'dy', 'dpx', 'dpy',
]

CHROM_COLUMNS = [
    'dmu1', 'dmu2', 'dmu3',
    'Dx', 'Dpx', 'Dy', 'Dpy',
    'ddx', 'ddpx', 'ddy', 'ddpy',
    'wx', 'wy', 'wxp', 'wyp',
]

COUPLING_COLUMNS = [
    'alfa12', 'alfa13', 'alfa21', 'alfa23', 'alfa31', 'alfa32',
    'beta12', 'beta13', 'beta21', 'beta23', 'beta31', 'beta32',
    'gama12', 'gama13', 'gama21', 'gama23', 'gama31', 'gama32',
    'f1001', 'f1010', 'r11', 'r12', 'r21', 'r22',
]

PART_COORDS = ['x', 'px', 'y', 'py', 't', 'pt']

BETA0_QUANTITIES = [
    'beta11', 'beta22', 'alfa11', 'alfa22', 'dx', 'dpx', 'dy', 'dpy',
]

TPSA_ALLOWED_TARGETS = {
    'beta11', 'beta22', 'alfa11', 'alfa22', 'mu1', 'mu2',
    'dx', 'dpx', 'dy', 'dpy',
    'x', 'px', 'y', 'py', 't', 'pt',
}
# fmt: on

# No leading underscore: pymadng refuses to retrieve names it deems private
XSUITE_MADNG_ENV_NAME = 'xsuite_matching_env'

# Prelude for MAD-NG commands, giving access to the table in which xsuite keeps
# its data (the sequence, the damaps, ...) under the local name ``env``.
MNG_ENV_PRELUDE = f'local env = {XSUITE_MADNG_ENV_NAME}\n'


def _ng_run(mng: Any, script: str, *payload: Any) -> Any:
    """Run ``script`` in MAD-NG with the xsuite environment bound to ``env``.

    The values in ``payload`` are sent on pymadng's data channel in the order
    given, to be read by the ``py:recv()`` calls of the script. Passing them
    this way rather than interpolating them keeps names and floating point
    values off the MAD-NG source, where they would have to be quoted and
    rounded.
    """
    mng.send(MNG_ENV_PRELUDE + script)
    for value in payload:
        mng.send(value)
    return mng


def _ensure_madng_model(line: xt.Line) -> Any:
    """Return the MAD-NG model attached to ``line``, creating it if needed."""
    if not hasattr(line.tracker, '_madng'):
        line.build_madng_model()
    return line.tracker._madng


def _normal_form_columns(values: Sequence[Any]) -> dict[str, Any]:
    """Map the normal-form response array to its public column names."""
    #fmt: off
    names = (
        'q1','q2', 'dq1', 'dq2',
        'd2q1', 'd2q2', 'd3q1', 'd3q2', 'd4q1', 'd4q2', 'd5q1', 'd5q2',
        'dqxdjx', 'dqydjy', 'dqxdjy', 'dqydjx',
    )
    #fmt: on
    result = dict(zip(names, values))
    for name in ('dqxdjx', 'dqydjy', 'dqxdjy', 'dqydjx'):
        result[name] *= 2.0
    return result

# A variable being matched has been turned into a (c)tpsa, of which only the
# constant part may be updated, or the TPSA would be destroyed.
_LUA_SET_VAR = """
    local name, value = py:recv(), py:recv()
    local var = MADX[name]
    if MAD.typeid.is_tpsa(var) or MAD.typeid.is_ctpsa(var) then
        var:set0(value)
    else
        MADX[name] = value
    end
    """


class MadngVars:
    """Expose MAD-NG variables through Xsuite's variable-update mechanism."""

    def __init__(self, mad: Any) -> None:
        self.mad = mad

    def __setitem__(self, key: str, value: Any) -> None:
        # Only values are propagated; deferred expressions would need MADX's
        # own environment, opened with ``MADX:open_env()``.
        _ng_run(self.mad, _LUA_SET_VAR, key.replace('.', '_'), value)


def build_madng_model(line: xt.Line, sequence_name: str = 'seq', **kwargs: Any) -> Any:
    """
    Build and attach the MAD-NG model associated with this line.

    Parameters
    ----------
    sequence_name : str, optional
        Name of the MAD-NG sequence to be created.
    **kwargs
        Additional keyword arguments forwarded to MAD-NG model creation.

    Returns
    -------
    model : object
        Built MAD-NG model.
    """
    _print(f'Building MAD-NG model for line {line.name} with sequence name {sequence_name}')
    if line.tracker is None:
        line.build_tracker()
    mng = line.to_madng(sequence_name=sequence_name, **kwargs)
    mng._sequence_name = sequence_name
    line.tracker._madng = mng
    line.tracker._madng_vars = MadngVars(mng)
    line.vars.vars_to_update.add(line.tracker._madng_vars)
    return mng


def discard_madng_model(line: xt.Line) -> None:
    """Remove the MAD-NG model and variable hook attached to ``line``."""
    line.tracker._madng = None
    line.vars.vars_to_update.remove(line.tracker._madng_vars)


def regen_madng_model(line: xt.Line) -> None:
    """Discard and rebuild the MAD-NG model associated with ``line``."""
    discard_madng_model(line)
    build_madng_model(line)


def to_ng_name(name: str) -> str:
    """Return the MAD-NG name of the quantity ``name``.

    A trailing ``_ng`` marks a name that is a MAD-NG one already.
    """
    return name[:-3] if name.endswith('_ng') else XS_NG_MAP[name]


def to_ng_target(xs_qty: str) -> str:
    """Return the MAD-NG name of a target quantity, rejecting unsupported ones."""
    qty = to_ng_name(xs_qty)
    if qty not in TPSA_ALLOWED_TARGETS:
        raise ValueError(
            f"Target quantity '{xs_qty}' not allowed with TPSA matching."
        )
    return qty


@dataclass
class _TwissRange:
    """The element range of a MAD-NG Twiss, and how to trim its result.

    MAD-NG brackets the requested range with marker rows, and how many it adds
    depends on how the range is expressed, so the trimming belongs with the
    range itself rather than at the call site.
    """

    start: str | None
    end: str | None
    i_start: int
    i_end: int
    element_names: tuple[str, ...]
    # Index at which a wrap-around range folds back onto the start of the line
    wrap_idx: int | None

    @classmethod
    def from_line(
        cls, line: xt.Line, start: str | None, end: str | None
    ) -> _TwissRange:
        names = line._element_names_unique
        i_start = names.index(start) if start is not None else 0
        i_end = names.index(end) if end is not None else len(names) - 1
        wrap_idx = None
        if i_start > i_end > 1:
            wrap_idx = len(line.element_names) - list(line.element_names).index(start)
        return cls(start, end, i_start, i_end, names, wrap_idx)

    @property
    def is_partial(self) -> bool:
        """Whether a sub-range of the line was requested."""
        return self.start is not None and self.end is not None

    @property
    def marker_nums(self) -> int:
        """Number of extra marker rows MAD-NG adds for a wrap-around range."""
        return 2 if self.i_start > self.i_end else 0

    def selected_names(self) -> np.ndarray:
        """Element names covered by the range, with Xsuite's end-point marker."""
        names = self.element_names
        if self.i_start > self.i_end:
            selected = names[self.i_start :] + names[: self.i_end + 1]
        else:
            selected = names[self.i_start : self.i_end + 1]
        return np.array(selected + ('_end_point',))

    def trim(self, data: Any) -> np.ndarray:
        """Drop the MAD-NG marker rows from one returned column."""
        data = np.atleast_1d(np.squeeze(data))
        if not self.is_partial:
            return data[:-1]
        if self.wrap_idx is not None:
            return np.concatenate(
                (data[0:1], data[0 : self.wrap_idx], data[self.wrap_idx + 2 :])
            )
        if self.marker_nums:
            return np.concatenate((data[0:1], data[: -self.marker_nums]))
        return np.concatenate((data[0:1], data))


_LUA_TWISS = """
    local config, columns = py:recv(), py:recv()
    config.sequence = env.sequence

    -- Initial conditions given as beta0 values rather than as a map. The key
    -- is cleared again, as twiss only accepts the options it knows.
    if config.beta0 then
        config.X0 = MAD.beta0(config.beta0)
        config.beta0 = nil
    end
    if config.trkrdt then
        -- The map below fixes the order, so any map definition must give way
        config.mapdef = nil
        config.X0 = MAD.damap {nv=6, mo=4}
        config.info = 2
        config.saverdt = true
        config.coupling = true
        config.chrom = true
    end

    local mtbl = twiss(config)
    for _, column in ipairs(columns) do py:send(mtbl[column], true) end
    """

_LUA_NORMAL_FORM = """
    local config = py:recv()
    config.sequence = env.sequence
    local _, mytrkflow = MAD.track(config)

    local normal in MAD.gphys  -- like "from MAD.gphys import normal"
    -- anh stands for anharmonicity
    local nf = normal(mytrkflow[1]):analyse('anh')

    last_nf = nf
    normal_forms_to_send = {
            nf:q1{1}, -- qx from the normal form (fractional part)
            nf:q2{1}, -- qy
            nf:dq1{1}, -- dqx / d delta
            nf:dq2{1}, -- dqy / d delta
            nf:dq1{2}, -- d2 qx / d delta2
            nf:dq2{2}, -- d2 qy / d delta2
            nf:dq1{3}, -- d3 qx / d delta3
            nf:dq2{3}, -- d3 qy / d delta3
            nf:dq1{4}, -- d4 qx / d delta4
            nf:dq2{4}, -- d4 qy / d delta4
            nf:dq1{5}, -- d5 qx / d delta5
            nf:dq2{5}, -- d5 qy / d delta5
            nf:anhx{1, 0}, -- dqx / d(2 jx)
            nf:anhy{0, 1}, -- dqy / d(2 jy)
            nf:anhx{0, 1}, -- dqx / d(2 jy)
            nf:anhy{1, 0}, -- dqy / d(2 jx)
            }
    py:send(normal_forms_to_send)
    """


def _twiss_config(
    method: int,
    nslice: int,
    mapdef: int,
    coupling: bool,
    chromatic: bool,
    rng: _TwissRange,
    X0: Any,
    beta0_data: Mapping[str, Any] | None,
    rdts: Sequence[str],
) -> dict[str, Any]:
    """Build the option table for the MAD-NG ``twiss`` command."""
    config: dict[str, Any] = {
        'method': method,
        'implicit': True,
        'nslice': nslice,
        'misalign': True,
        'coupling': coupling,
        'chrom': chromatic,
    }
    if X0 is not None:
        # A reference to a map that already lives in MAD-NG, whose order is
        # fixed, so the map definition would have nothing left to say
        config['X0'] = X0
    else:
        # MAD-NG builds the map itself here, and the order is ours to choose
        config['mapdef'] = mapdef
        if beta0_data is not None:
            # Turned into a map by MAD-NG, which knows the beta0 constructor
            config['beta0'] = beta0_data
    if rdts:
        # An empty table would be truthy in MAD-NG, so only set it when needed
        config['trkrdt'] = list(rdts)
    if rng.is_partial:
        config['range'] = f'{rng.start}/{rng.end}'
    return config


def _add_chromatic_columns(tw: xt.TwissTable) -> None:
    """Replace MAD-NG's chromatic amplitude and phase by their components."""
    for plane in ('x', 'y'):
        wave = tw[f'w{plane}_ng'] * np.exp(1j * 2 * np.pi * tw[f'w{plane}p_ng'])
        tw[f'a{plane}_ng'] = np.imag(wave)
        tw[f'b{plane}_ng'] = np.real(wave)
        del tw[f'w{plane}p_ng']


def _add_normal_form_columns(
    mng: Any, tw: xt.TwissTable, method: int, mapdef: int, nslice: int
) -> None:
    """Compute the normal-form quantities and add them to ``tw``."""
    _ng_run(
        mng, _LUA_NORMAL_FORM,
        {'method': method, 'mapdef': mapdef, 'nslice': nslice},
    )
    for nn, val in _normal_form_columns(mng.recv('normal_forms_to_send')).items():
        tw[f'{nn}_nf_ng'] = val


def _tw_ng(
    line: xt.Line,
    rdts: Sequence[str] = (),
    normal_form: bool = False,
    mapdef_twiss: int = 2,
    mapdef_normal_form: int = 4,
    nslice: int = 3,
    xsuite_tw: bool = True,
    X0: Any = None,
    compute_chromatic_properties: bool = False,
    coupling_edw_teng: bool = False,
    method: int = 4,
    **tw_kwargs: Any,
) -> xt.TwissTable:
    """
    Run a Twiss calculation using the MAD-NG model.

    If the MAD-NG model is not available, it is created automatically.

    Parameters
    ----------
    rdts : tuple, optional
        Resonance driving terms to compute.
    normal_form : bool, optional
        If ``True``, also compute normal-form quantities.
    mapdef_twiss : int, optional
        Map order used for the MAD-NG Twiss computation. Only has an effect
        when ``X0`` is not given, as a map carries its own order.
    mapdef_normal_form : int, optional
        Map order used for the MAD-NG normal-form computation.
    nslice : int, optional
        Number of slices used in MAD-NG tracking/Twiss internals.
    xsuite_tw : bool, optional
        If ``True``, use Xsuite Twiss output structure and enrich it with MAD-NG data.
    X0 : object, optional
        Initial condition object for open Twiss calculations.
    method : int, optional
        MAD-NG method identifier for Twiss/tracking calls.
    **tw_kwargs
        Additional keyword arguments forwarded to Twiss setup.

    Returns
    -------
    twiss : xtrack.TwissTable
        Twiss table with MAD-NG columns.
    """

    _action = ActionTwissMadng(
        line,
        {
            'rdts': rdts,
            'normal_form': normal_form,
            'mapdef_twiss': mapdef_twiss,
            'mapdef_normal_form': mapdef_normal_form,
            'nslice': nslice,
            **tw_kwargs,
        },
    )

    mng = _ensure_madng_model(line)

    start = tw_kwargs.get('start', None)
    end = tw_kwargs.get('end', None)
    init = tw_kwargs.get('init', None)

    beta0_data = None
    if X0 is None:
        if init is not None and isinstance(init, xt.TwissTable):
            raise NotImplementedError('TwissTable as init not implemented.')
        beta0_data = {
            ng_key: value
            for key, value in tw_kwargs.items()
            if (ng_key := XS_NG_MAP.get(key, key)) in BETA0_COLUMNS
        } or None

    if (start is None) != (end is None):
        raise ValueError('Start and end must be specified together.')

    rng = _TwissRange.from_line(line, start, end)

    if rng.is_partial and X0 is None and beta0_data is None:
        raise ValueError(
            'Initial conditions must be specified when start and end are given.'
        )

    tw_columns = TW_BASE_COLUMNS.copy()
    if coupling_edw_teng:
        tw_columns += COUPLING_COLUMNS
    if compute_chromatic_properties:
        tw_columns += CHROM_COLUMNS
    columns = tw_columns + list(rdts)

    if rng.is_partial and not rdts:
        normal_form = False

    config = _twiss_config(
        method=method,
        nslice=nslice,
        mapdef=mapdef_twiss,
        coupling=coupling_edw_teng,
        chromatic=compute_chromatic_properties,
        rng=rng,
        X0=X0,
        beta0_data=beta0_data,
        rdts=rdts,
    )

    _ng_run(mng, _LUA_TWISS, config, list(columns))
    out_dct = {c: mng.recv() for c in columns}

    if xsuite_tw:
        xs_tw_kwargs = {NG_XS_MAP.get(k, k): v for k, v in tw_kwargs.items()}
        tw = line.twiss(method='4d', reverse=False, **xs_tw_kwargs)
    else:
        tw = xt.TwissTable({'name': rng.selected_names()})
    tw._action = _action

    first_col = np.atleast_1d(np.squeeze(out_dct[columns[0]]))
    if not rng.is_partial:
        assert len(first_col) == len(tw) + 1
    else:
        assert len(first_col) == len(tw) + rng.marker_nums - 1

    for nn in tw_columns:
        tw[f'{nn}_ng'] = rng.trim(out_dct[nn])

    for nn in rdts:
        tw[nn] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]

    if compute_chromatic_properties:
        _add_chromatic_columns(tw)

    if normal_form:
        _add_normal_form_columns(mng, tw, method, mapdef_normal_form, nslice)

    return tw


_LUA_GET_INIT = """
    local at = py:recv()
    env.sequence:select(MAD.element.flags.observed, {list = {at}})
    local twpart = twiss {
        sequence = env.sequence, observe = 1, savemap = true, info = 2
    }
    env.X0 = twpart[at].__map
    """


def madng_get_init(line: xt.Line, at: Any) -> Any:
    """Return a reference to the MAD-NG map at ``at``, computed with a Twiss.

    ``at`` is an element name, or ``xt.START`` for the start of the line.
    """
    mng = _ensure_madng_model(line)

    # The location travels on the data channel: an element name needs no
    # quoting there, and the start of the line is simply the index 1.
    _ng_run(mng, _LUA_GET_INIT, 1 if at == xt.START else at)
    return mng._env.X0


def _survey_ng(line: xt.Line) -> SurveyTable:
    """
    Run a survey using the MAD-NG model.

    If the MAD-NG model is not available, it is created automatically.

    Returns
    -------
    survey : SurveyTable
        Survey result produced by MAD-NG.
    """
    mng = _ensure_madng_model(line)
    mng['srv'] = mng.survey(sequence=mng._sequence)

    survey_tab_keys = {
        'x': 'X',
        'y': 'Y',
        'z': 'Z',
        'l': 'length',
        'kind': 'element_type',
    }

    element_types = {
        'drift': 'Drift',
        'sbend': 'Bend',
        'rbend': 'RBend',
        'quadrupole': 'Quadrupole',
        'sextupole': 'Sextupole',
        'octupole': 'Octupole',
        'multipole': 'Multipole',
        'kicker': 'Kicker',  # no coloring in survey plot
        'rfcavity': 'Cavity',
        'marker': 'Marker',
    }

    survey_df = mng['srv'][0].to_df()
    survey_dict = survey_df.to_dict(orient='list')
    survey_dict = {k: np.array(v) for k, v in survey_dict.items()}
    for k, v in survey_tab_keys.items():
        if k in survey_dict:
            survey_dict[v] = survey_dict[k]
            del survey_dict[k]

    survey_dict['element_type'] = np.array(
        [element_types.get(et, et) for et in survey_dict['element_type']]
    )

    for i in survey_dict:
        # Interpretation of survey is shifted by 1 in MAD-NG vs. Xsuite
        if i in ['name', 'length', 'kind', 'element_type', 'angle', 'tilt']:
            survey_dict[i] = survey_dict[i][1:]
        else:
            survey_dict[i] = survey_dict[i][:-1]

    return SurveyTable(survey_dict)


class ActionTwissMadng(Action):
    """Matching action that evaluates Twiss parameters with MAD-NG."""

    def __init__(
        self,
        line: xt.Line,
        tw_kwargs: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.line = line
        self.tw_kwargs = {} if tw_kwargs is None else dict(tw_kwargs)
        self.tw_kwargs.update(kwargs)
        self._already_prepared = False
        self.X0 = None

    def prepare(self, force: bool = False) -> None:
        """Prepare the initial MAD-NG map used by the action."""
        if self._already_prepared and not force:
            return

        init = self.tw_kwargs.get('init', None)
        start = self.tw_kwargs.get('start', None)
        end = self.tw_kwargs.get('end', None)

        if init is not None and start is not None and end is not None:
            assert isinstance(init, xt.TwissTable)
            self.X0 = madng_get_init(self.line, at=start)
        elif init is not None:
            assert isinstance(init, xt.TwissTable)
            self.X0 = madng_get_init(self.line, at=xt.START)

        self._already_prepared = True

    def run(self, allow_failure: bool = False) -> xt.TwissTable:
        """Evaluate and return the requested MAD-NG Twiss table."""
        return self.line.madng_twiss(xsuite_tw=False, X0=self.X0, **self.tw_kwargs)


# The matrices are sent one by one: a Lua table would arrive as a reference
# rather than as its contents.
_LUA_RMATRICES = """
    for i, options in ipairs(py:recv()) do
        options.sequence = env.sequence
        options.X0 = env.empty_X0
        local _, flow = MAD.track(options)
        env.rmat_map_arr[i] = flow[1]
        py:send(flow[1]:get1())
    end
    """

# One row per target, one column per matching variable. Both counts come from
# the environment: the initial map carries one parameter per matching variable.
_LUA_JACOBIAN = """
    local varylen = env.init_X0_map:np()
    local nv = env.init_X0_map:nv()
    env.jac = MAD.matrix(#env.targets_arr, varylen)

    for i, target in ipairs(env.targets_arr) do
        local map = nil
        if target.optfun or target.orbit then
            map = env.target_loc_map[target.loc]
        elseif target.rmat then
            map = env.rmat_map_arr[target.tag]
        end

        local monom = MAD.monomial(nv + varylen) -- BUILD MONOMIAL
        for j = 1, map:np(), 1 do
            local jac_idx = (i-1)*varylen + j

            -- Quantity which can be calculated with optfun
            if target.optfun then
                -- If loc_start (phase advance) is defined, we provide initial map
                local a0 = target.loc_start
                    and env.target_loc_map[target.loc_start]
                env.jac[jac_idx] = MAD.gphys.optfun(
                    map, target.qty .. "_", j, 1, a0)

            -- Orbit Quantity
            elseif target.orbit then
                monom[nv + j] = 1
                env.jac[jac_idx] = map[target.orbit]:get(monom)
                monom[nv + j] = 0

            elseif target.rmat then
                -- rmatrix terms are extracted directly from the damap
                local ind_1 = tonumber(target.qty:sub(2,2))
                local ind_2 = tonumber(target.qty:sub(3,3))
                monom[nv + j] = 1
                monom[ind_2] = 1
                env.jac[jac_idx] = map[ind_1]:get(monom)
                monom[nv + j] = 0
                monom[ind_2] = 0
            end
        end
    end

    py:send(env.jac)
    """

_LUA_CLEANUP = """
    for _, var_name in ipairs(py:recv()) do
        MADX[var_name] = MADX[var_name]:get0()
    end
    """

_LUA_TRACK = """
    local operation, options = py:recv(), py:recv()
    options.sequence = env.sequence
    options.X0 = env.init_X0_map
    env.trk = MAD[operation](options)
    """

# After a Track calculation the optical functions are not present in the table,
# so they are added as derived columns, named as the user (Xsuite) defined the
# targets. A Twiss already carries every quantity TPSA matching allows, so the
# test below skips them all and no flag is needed to tell the two apart.
_LUA_RESULT = """
    local trk = env.trk

    for _, tar in ipairs(env.tar_optics_qtys) do
        if not trk[env.xs_ng_target_map[tar]] then
            trk:addcol(tar, \\ri -> MAD.gphys.optfun(
                trk[ri].__map, env.xs_ng_target_map[tar] .. '_'))
        end
    end

    -- Save damaps
    for _, location in ipairs(py:recv()) do
        env.target_loc_map[location] = trk[location].__map
    end
    py:send(trk)
    """

# The values are received in the order in which they are sent, and are only
# kept in a local when used more than once.
_LUA_PREPARE_TPSA = """
    local locations = py:recv() -- target locations
    env.sequence:select(MAD.element.flags.observed, {list = locations})

    local params = py:recv() -- names of the matching variables

    local X0 = MAD.damap {
        nv=6, -- number of variables
        mo=2, -- max order of variables
        np=#params, -- number of parameters
        po=1, -- max order of parameters
        pn=params, -- parameter names
    }

    -- Converting to TPSA (mutating type)
    for _, v in ipairs(params) do
        MADX[v] = MADX[v] + X0[v]
    end

    local map1 = MAD.gphys.bet2map(MAD.beta0(py:recv()), X0:copy())

    -- Initial orbit, as {name, value} pairs
    for _, coordinate in ipairs(py:recv()) do
        map1[coordinate[1]]:set0(coordinate[2])
    end

    -- Maps target locations to damaps
    -- e.g. { 'BPM1' = damap1, 'BPM2' = damap2, ... }
    env.target_loc_map = table.new(0, #locations)

    -- Maps rmat tags to rmat damaps
    -- e.g. {rmat_damap1, rmat_damap2, ... }
    env.rmat_map_arr = table.new(py:recv(), 0)

    -- Initial map for tracking/twiss
    env.init_X0_map = map1

    -- Identity map
    env.empty_X0 = X0
    """


@dataclass
class _Target:
    """One matching target, in the form the MAD-NG environment expects.

    Exactly one of ``optfun``, ``orbit`` and ``rmat`` says how the quantity is
    evaluated: from an optical function, from a coordinate of the damap, or
    from a transfer matrix term.
    """

    loc: str
    qty: str  # MAD-NG name of the quantity
    xs_qty: str  # Name the user asked for, which may be an Xsuite one
    loc_start: str | None = None
    optfun: bool = False
    orbit: int | None = None  # 1-based index of the coordinate in the damap
    rmat: bool = False
    tag: int | None = None
    rtag: str | None = None  # Xsuite's '<tag>_r<ij>' label for the term

    def as_ng(self) -> dict[str, Any]:
        """Return the Lua table for this target; omitted keys read as nil."""
        data: dict[str, Any] = {'loc': self.loc, 'qty': self.qty}
        if self.loc_start is not None:
            data['loc_start'] = self.loc_start
        if self.optfun:
            data['optfun'] = True
        if self.orbit is not None:
            data['orbit'] = self.orbit
        if self.rmat:
            data['rmat'] = True
            # Xsuite numbers the transfer maps from zero, MAD-NG arrays from one
            data['tag'] = self.tag + 1 if self.tag is not None else None
        return data


def _target_range(
    target: Any, init: xt.TwissTable, start: str | None, end: str | None
) -> tuple[str, str]:
    """Resolve the endpoints of a target, expanding Xsuite's placeholders."""
    if target.start != '__ele_start__':
        loc_start = target.start
    elif start is not None:
        loc_start = start
    else:
        loc_start = init.name[0]

    if target.end != '__ele_stop__':
        loc_end = target.end
    elif end is not None:
        loc_end = end
    else:
        loc_end = init.name[-2]

    return loc_start, loc_end


def _build_targets(
    targets: Sequence[Any],
    init: xt.TwissTable,
    start: str | None,
    end: str | None,
) -> list[_Target]:
    """Translate the Xsuite targets into their MAD-NG counterparts."""
    built = []
    for target in targets:
        if isinstance(target.tar, tuple):
            xs_qty, loc = target.tar
            qty = to_ng_target(xs_qty)
            built.append(_Target(
                loc=loc,
                qty=qty,
                xs_qty=xs_qty,
                optfun=qty in OPTFUN_QUANTITIES,
                orbit=(
                    PART_COORDS.index(qty) + 1
                    if qty not in OPTFUN_QUANTITIES and qty in PART_COORDS
                    else None
                ),
            ))

        elif hasattr(target, 'start') and hasattr(target, 'end'):
            loc_start, loc_end = _target_range(target, init, start, end)

            if isinstance(target, xt.TargetRelPhaseAdvance):
                built.append(_Target(
                    loc=loc_end,
                    qty=to_ng_target(target.var),
                    xs_qty=target.var,
                    loc_start=loc_start,
                    optfun=True,
                ))

            elif isinstance(target, xt.TargetRmatrixTerm):
                built.append(_Target(
                    loc=loc_end,
                    qty=target.term,
                    xs_qty=target.term,
                    loc_start=loc_start,
                    rmat=True,
                    tag=int(target.rtag.split('_')[0]),
                    rtag=target.rtag,
                ))

            else:
                raise NotImplementedError(
                    f'Target of type {type(target)} not implemented for '
                    'MAD-NG TPSA matching.'
                )

        else:
            raise NotImplementedError(
                f'Target of type {type(target)} not implemented for '
                'MAD-NG TPSA matching.'
            )

    return built


class ActionTwissMadngTPSA(Action):
    """Matching action using MAD-NG TPSA maps for optics sensitivities."""

    def __init__(
        self,
        line: xt.Line,
        vary_names: Sequence[str],
        targets: Sequence[Any] | None = None,
        tw_kwargs: Mapping[str, Any] | None = None,
        sum_rmat_tar: int = 0,
        **kwargs: Any,
    ) -> None:
        self.line = line
        self.vary_names = vary_names
        self.targets = [] if targets is None else targets
        self.mng: Any = None
        self.optics_target_locations: list[str] = []
        self.optics_target_quantities: set[str] = set()
        self.tw_kwargs = {} if tw_kwargs is None else dict(tw_kwargs)
        self.tw_kwargs.update(kwargs)
        self._already_prepared = False
        self.sum_rmat_tar = sum_rmat_tar
        self.rmat_start_end_list: list[tuple[str, str]] = []
        self.rmat_tags: list[str] = []
        self._last_res: Any = None
        self._needs_zeta_scale: list[int] = []
        self._needs_delta_scale: list[int] = []

    @property
    def twiss_flag(self) -> bool:
        """Whether a Twiss is needed, as a Track gives no phase advance."""
        return any(
            isinstance(tar, xt.TargetRelPhaseAdvance) for tar in self.targets
        )

    @property
    def match_rmat(self) -> bool:
        """Whether any target is a transfer matrix term."""
        return any(
            isinstance(tar, xt.TargetRmatrixTerm) for tar in self.targets
        )

    def prepare(self, force: bool = False) -> None:
        """
        Prepare the MAD-NG TPSA matching environment.
        This method sets up the MAD-NG environment for TPSA matching by
        configuring the initial conditions, setting target locations, and quantities
        based on the provided targets.
        To achieve that, arrays and maps are created within MAD-NG to keep track of
        the target locations, quantities and differential algebraic maps.

        Parameters
        ----------
        force : bool, optional
            If True, forces re-preparation even if already prepared. Default is False.

        Raises
        ------
        ValueError
            If the target quantity is not allowed with TPSA matching
            or if start and end are provided without initial conditions.
        """

        if self._already_prepared and not force:
            return

        init = self.tw_kwargs.get('init', None)

        if init is None:
            init = self.line.madng_twiss(**self.tw_kwargs)
            self.tw_kwargs.update({'init': init})

        assert isinstance(init, xt.TwissTable)
        self.mng = _ensure_madng_model(self.line)

        # Keep dynamic values in pymadng's native data channel. The MAD-NG
        # command below is deliberately independent of target names and values.
        targets = self._process_targets(init)
        beta0_data, coordinates = self._initial_conditions(init)
        self.mng._env.xs_ng_target_map = {t.xs_qty: t.qty for t in targets}
        self.mng._env.targets_arr = [t.as_ng() for t in targets]
        self.mng._env.tar_optics_qtys = list(self.optics_target_quantities)

        _ng_run(
            self.mng,
            _LUA_PREPARE_TPSA,
            self.optics_target_locations,
            list(self.vary_names),
            beta0_data,
            coordinates,
            self.sum_rmat_tar,
        )

        self._already_prepared = True

    def _process_targets(self, init: xt.TwissTable) -> list[_Target]:
        """Translate the Xsuite targets and derive the state they imply."""
        targets = _build_targets(
            self.targets,
            init,
            start=self.tw_kwargs.get('start', None),
            end=self.tw_kwargs.get('end', None),
        )

        locations: set[str] = set()
        for target in targets:
            if target.rmat:
                # Transfer matrices are tracked separately, over their own range
                continue
            locations.add(target.loc)
            if target.loc_start is not None:
                locations.add(target.loc_start)
        self.optics_target_locations = list(locations)

        self.optics_target_quantities = {
            target.xs_qty for target in targets if not target.rmat
        }
        self.rmat_tags = [t.rtag for t in targets if t.rtag is not None]

        # Ordered by tag, as MAD-NG indexes the transfer maps by their position.
        # The start is never absent, ``_target_range`` having resolved it.
        rmat_ranges = {
            t.tag: (t.loc_start, t.loc)
            for t in targets
            if t.rmat and t.loc_start is not None
        }
        self.rmat_start_end_list = [
            rmat_ranges[tag] for tag in range(self.sum_rmat_tar)
        ]

        # Only orbit targets are expressed in Xsuite units that MAD-NG does not
        # share, so only their Jacobian rows need rescaling.
        self._needs_zeta_scale = [
            i for i, t in enumerate(targets)
            if t.orbit is not None and t.xs_qty == 'zeta'
        ]
        self._needs_delta_scale = [
            i for i, t in enumerate(targets)
            if t.orbit is not None and t.xs_qty == 'delta'
        ]

        return targets

    def _initial_conditions(
        self, init: xt.TwissTable
    ) -> tuple[dict[str, Any], list[list[Any]]]:
        """Extract the beta0 values and the closed orbit at the start of the range.

        ``init`` may come either from a MAD-NG Twiss (columns suffixed with
        ``_ng``) or from an Xsuite Twiss (columns with Xsuite names). Both are
        read here with MAD-NG naming conventions.
        """
        loc = self.tw_kwargs.get('start', None) or 0

        if 'x_ng' in init.cols:
            values = {nn: init[f'{nn}_ng', loc] for nn in BETA0_QUANTITIES + PART_COORDS}
        else:
            beta0 = self.line.particle_ref.beta0[0]
            values = {
                nn: init[NG_XS_MAP.get(nn, nn), loc]
                for nn in BETA0_QUANTITIES + PART_COORDS[:4]
            }
            values['t'] = init['zeta', loc] / beta0
            values['pt'] = init['ptau', loc]  # ptau corresponds to pt

        beta0_data = {nn: values[nn] for nn in BETA0_QUANTITIES}

        # The orbit is applied in MAD-NG with ``:set0`` (set the 0th-order /
        # constant part) and NOT ``map1.x = val``: to preserve TPSA.
        # Otherwise TPSA is replaced which corrupts the A-matrix row and
        # the optical functions for non-zero orbit.
        coordinates = [
            [nn, values[nn]] for nn in PART_COORDS if abs(values[nn]) > 1e-12
        ]

        return beta0_data, coordinates

    def _track_options(self, start: str | None, end: str | None) -> dict[str, Any]:
        """Return the MAD-NG track/twiss options for the given range.

        The sequence is passed by reference, so that its name never has to be
        interpolated into a MAD-NG command.
        """
        options = {
            'savemap': True,
            'observe': 1,
        }
        if start is not None and end is not None:
            options['range'] = f'{start}/{end}'
        return options

    def run(self, allow_failure: bool = False) -> xt.TwissTable:
        """
        Execute the MAD-NG TPSA matching action.
        This method performs either a Twiss or Track operation in MAD-NG
        depending if quantities can be calculated through tracking or not.
        It retrieves the results and constructs a TwissTable with the requested
        target quantities at the specified target locations.

        Returns
        -------
        xt.TwissTable
            A TwissTable containing the results of the Twiss or Track operation
            with the requested target quantities at the specified target locations.
        """

        if self._already_prepared is False:
            self.prepare()

        _ng_run(
            self.mng,
            _LUA_TRACK,
            'twiss' if self.twiss_flag else 'track',
            self._track_options(
                self.tw_kwargs.get('start'), self.tw_kwargs.get('end')
            ),
        )
        _ng_run(self.mng, _LUA_RESULT, self.optics_target_locations)
        res = xt.TwissTable(
            self.mng.recv(f'{XSUITE_MADNG_ENV_NAME}.trk').to_df()
        )

        if self.twiss_flag:
            # Alias the target quantities to the names used by the user (Xsuite)
            for qty in self.optics_target_quantities:
                if qty not in res.cols:
                    res[qty] = res[to_ng_name(qty)]

        if 'zeta' in self.optics_target_quantities:
            res._data.loc[:, 'zeta'] = res['t'] * self.line.particle_ref.beta0[0]
        if 'delta' in self.optics_target_quantities:
            res._data.loc[:, 'delta'] = ptau2delta(
                res['pt'], self.line.particle_ref.beta0[0]
            )

        if self.match_rmat:
            res = self.handle_rmatrices(res)

        self._last_res = res
        return res

    def handle_rmatrices(self, res: xt.TwissTable) -> xt.TwissTable:
        """Evaluate requested transfer-matrix terms and attach them to ``res``."""
        options_arr = []
        for start_rmat, end_rmat in self.rmat_start_end_list:
            if start_rmat == '__ele_start__':
                start_rmat = self.tw_kwargs.get('start', None)
            if end_rmat == '__ele_stop__':
                end_rmat = self.tw_kwargs.get('end', None)
            options_arr.append(self._track_options(start_rmat, end_rmat))

        _ng_run(self.mng, _LUA_RMATRICES, options_arr)
        rmatrices = [self.mng.recv() for _ in options_arr]

        for tag in self.rmat_tags:
            t0, term = tag.split('_')
            ii = int(term[1]) - 1
            jj = int(term[2]) - 1
            res._data.attrs[tag] = rmatrices[int(t0)][ii, jj]

        return res

    def acquire_jacobian(self) -> np.ndarray:
        """
        Acquire the Jacobian matrix for the TPSA matching targets and variables.
        This method computes the Jacobian matrix for the specified targets and
        variables using MAD-NG's TPSA capabilities. It constructs
        the Jacobian matrix by evaluating the sensitivity of each target quantity
        with respect to each variable using MAD-NG's optfun function (optical functions)
        or by direct extraction from the TPSA (orbit).

        Returns
        -------
        np.ndarray
            A 2D NumPy array representing the Jacobian matrix, where each row
            corresponds to a target and each column corresponds to a variable.
        """

        _ng_run(self.mng, _LUA_JACOBIAN)
        jac = np.array(self.mng.recv())

        for i in self._needs_zeta_scale:
            jac[i, :] *= self.line.particle_ref.beta0[0]
        for i in self._needs_delta_scale:
            jac[i, :] *= dptau2ddelta(
                self._last_res['delta', self.targets[i].tar[1]],
                self.line.particle_ref.beta0[0],
            )
        return jac

    def cleanup(self) -> None:
        """Restore matched MAD-NG variables and release the TPSA environment."""
        if self._already_prepared is True:
            _ng_run(self.mng, _LUA_CLEANUP, list(self.vary_names))
            self.mng._env.X0 = None
            self._already_prepared = False


def line_to_madng(
    line: xt.Line,
    sequence_name: str = 'seq',
    temp_fname: str | None = None,
    keep_files: bool = False,
    **kwargs: Any,
) -> Any:
    """Serialize an Xsuite line and load it into a new MAD-NG session.

    Parameters
    ----------
    line : xtrack.Line
        Line to convert into a MAD-NG sequence.
    sequence_name : str, optional
        Name assigned to the generated MAD-NG sequence.
    temp_fname : str, optional
        Prefix for the temporary MAD-NG input file.
    keep_files : bool, optional
        Keep the generated input file after loading it.
    **kwargs
        Additional options passed to :class:`pymadng.MAD`.

    Returns
    -------
    object
        The initialized ``pymadng.MAD`` session.
    """
    try:
        if temp_fname is None:
            temp_fname = f'temp_madng_{uuid.uuid4()}'

        from .mad_writer import to_madng_sequence

        madx_seq = to_madng_sequence(line, name=sequence_name)
        Path(f'{temp_fname}.mad').write_text(madx_seq)

        from pymadng import MAD

        nocharge = kwargs.pop('nocharge', True)

        # Typed as Any: xtrack stashes its own state on the MAD-NG session
        mng: Any = MAD(**kwargs)
        mng.MAD.option.nocharge = nocharge
        mng.MADX.option.rbarc = True
        mng.send('assert(loadfile(py:recv(), nil, MADX))()').send(
            str(Path(f'{temp_fname}.mad').resolve())
        )
        mng[XSUITE_MADNG_ENV_NAME] = [] # Create an empty table in MAD-NG to store xsuite data
        mng._env = mng[XSUITE_MADNG_ENV_NAME]
        mng._init_madx_data = madx_seq

        # A variable that does not exist in the MAD-X environment defaults to 0
        sequence = mng.MADX[sequence_name]
        if sequence == 0:
            raise ValueError(
                f"Sequence '{sequence_name}' not found in MAD-NG model. "
                "Check the generated MAD-NG input file for errors."
            )
        # The sequence is kept in the xsuite environment, so that commands can
        # refer to it without its name ever being interpolated into them.
        mng._sequence = sequence
        mng._env.sequence = sequence
        mng[sequence_name] = sequence
        mng[sequence_name].beam = mng.beam(
            particle="'custom'",
            mass=line.particle_ref.mass0 / 1e9,  # xsuite mass eV -> ng mass GeV.
            charge=line.particle_ref.q0,
            betgam=line.particle_ref.beta0[0] * line.particle_ref.gamma0[0],
        )

    finally:
        if not keep_files:
            for nn in [f'{temp_fname}.madx', f'{temp_fname}.mad']:
                Path(nn).unlink(missing_ok=True)

    return mng
