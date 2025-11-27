import numpy as np

from .match import Action
import os
import uuid

import xtrack as xt

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
}

BETA0_COLUMNS = ['x', 'px', 'y', 'py', 't', 'pt',
                 'dx', 'dy', 'dpx', 'dpy', 'ddx', 'ddpx', 'ddy', 'ddpy', 'wx', 'phix',
                 'wy', 'phiy', 'mu1', 'mu2', 'mu3', 'dmu1', 'dmu2', 'dmu3', 'r11',
                 'r12', 'r21', 'r22', 'alfa11', 'alfa12', 'alfa13', 'alfa21',
                 'alfa22', 'alfa23', 'alfa31', 'alfa32', 'alfa33', 'beta11',
                 'beta12', 'beta13', 'beta21', 'beta22', 'beta23', 'beta31',
                 'beta32', 'beta33', 'gama11', 'gama12', 'gama13', 'gama21',
                 'gama22', 'gama23', 'gama31', 'gama32', 'gama33']

TW_BASE_COLUMNS = ['s', 'beta11', 'beta22', 'beta33', 'alfa11', 'alfa22', 'alfa33',
                   'gama11', 'gama22', 'gama33', 'x', 'px', 'y', 'py', 't', 'pt',
                   'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2', 'mu3']

OPTFUN_QUANTITIES = ['beta11', 'beta22', 'alfa11', 'alfa22', 'gama11', 'gama22',
                     'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2']

CHROM_COLUMNS = ['dmu1', 'dmu2', 'dmu3', 'Dx', 'Dpx', 'Dy',
                 'Dpy', 'ddx', 'ddpx', 'ddy', 'ddpy', 'wx', 'wy', 'phix', 'phiy']

COUPLING_COLUMNS = ['alfa12', 'alfa13', 'alfa21', 'alfa23', 'alfa31', 'alfa32',
                    'beta12', 'beta13', 'beta21', 'beta23', 'beta31', 'beta32',
                    'gama12', 'gama13', 'gama21', 'gama23', 'gama31', 'gama32',
                    'f1001', 'f1010', 'r11', 'r12', 'r21', 'r22']

TPSA_ALLOWED_TARGETS = { 'beta11', 'beta22', 'alfa11', 'alfa22', 'dx', 'dpx', 'dy', 'dpy',
                         'mu1', 'mu2', 'x', 'px', 'y', 'py', 't', 'pt' }

XSUITE_MADNG_ENV_NAME = "_xsuite_matching_env"

def dp2pt(dp, beta0):
    """Convert relative momentum deviation dp/p to transverse momentum pt/p.

    Parameters
    ----------
    dp : float
        Relative momentum deviation (dp/p, dimensionless).
    beta0 : float
        Particle relativistic beta (v/c).

    Returns
    -------
    float
        Transverse momentum relative to total momentum (pt/p, dimensionless).
    """

    _beta0 = 1 / beta0
    return np.sqrt((1 + dp) ** 2 + (_beta0**2 - 1)) - _beta0

class MadngVars:

    def __init__(self, mad):
        self.mad = mad

    def __setitem__(self, key, value):
        #setattr(self.mad.MADX, key.replace('.', '_'), value)
        # Check for key if it's a ctpsa or tpsa

        var = f"MADX['{key.replace('.', '_')}']"
        is_tpsa = self.mad.send(f"py:send(MAD.typeid.is_tpsa({var}) or MAD.typeid.is_ctpsa({var}))").recv()
        if is_tpsa:
            self.mad.send(f"{var}:set0(py:recv())").send(value)
        else:
            self.mad[var] = value


        #Expressions still to be handled, could use the following:
        # mng.send(
        #     MADX:open_env()
        #     a = 3
        #     b =\ 3 * a
        #     c =\ 4 * a
        #     MADX:close_env()
        #     ''')

def build_madng_model(line, sequence_name='seq', **kwargs):
    print('Building MAD-NG model for line', line.name, 'with sequence name', sequence_name)
    if line.tracker is None:
        line.build_tracker()
    mng = line.to_madng(sequence_name=sequence_name, **kwargs)
    mng._sequence_name = sequence_name
    line.tracker._madng = mng
    line.tracker._madng_vars = MadngVars(mng)
    line.vars.vars_to_update.add(line.tracker._madng_vars)
    return mng

def discard_madng_model(line):
    line.tracker._madng = None
    line.tracker.vars_to_update.remove(line.tracker._madng_vars)
    return

def regen_madng_model(line):
    discard_madng_model(line)
    build_madng_model(line)
    return

def _build_column_send_script(columns):
    assert len(columns) > 0
    mng_columns_to_send = ["mtbl." + col for col in columns]
    send_cmd = f'''
        -- send columns to Python
        columns = {{{", ".join(mng_columns_to_send)}}}
        py:send(columns, true)
    '''
    return send_cmd

def _build_rdt_script(mng_sequence_name, rdts, columns):
    assert len(rdts) > 0
    rdt_cmd = 'local rdts = {"' + '", "'.join(rdts) + '"}'
    send_cmd = _build_column_send_script(columns)
    # Create damap and X0, then twiss with rdts
    script = f'''
        local damap in MAD
        {rdt_cmd}

        -- create phase-space damap at 4th order
        local X0 = damap {{nv=6, mo=4}}

        -- twiss with RDTs
        local mtbl = twiss {{ sequence={mng_sequence_name}, X0=X0, trkrdt=rdts, info=2, saverdt=true, coupling=true, chrom=true }}

        {send_cmd}
    '''
    return script

def _build_beta0_block_string(tw_kwargs):
    flag_init = False
    beta0_keys = []
    beta0_vals = []
    for k in tw_kwargs.keys():
        if k in BETA0_COLUMNS:
            beta0_keys.append(k)
            beta0_vals.append(tw_kwargs[k])
            flag_init = True
        elif k in XS_NG_MAP:
            beta0_keys.append(XS_NG_MAP[k])
            beta0_vals.append(tw_kwargs[k])
            flag_init = True

    if flag_init:
        # Construct beta0 string
        beta0_str = 'X0 = beta0 {'
        for k, v in zip(beta0_keys, beta0_vals):
            beta0_str += f'{k} = {v}, '
        beta0_str = beta0_str[:-2] + '}, '
    else:
        beta0_str = ''
    return beta0_str

def _tw_ng(line, rdts=(), normal_form=False,
           mapdef_twiss=2, mapdef_normal_form=4,
           nslice=3, xsuite_tw=True, X0=None, compute_chromatic_properties=False,
           coupling_edw_teng=False, **tw_kwargs):

    _action = ActionTwissMadng(line, {
        "rdts": rdts,
        "normal_form": normal_form,
        "mapdef_twiss": mapdef_twiss,
        "mapdef_normal_form": mapdef_normal_form,
        "nslice": nslice,
        **tw_kwargs
    })

    if not hasattr(line.tracker, '_madng'):
        line.build_madng_model()
    mng = line.tracker._madng

    start = tw_kwargs.get('start', None)
    end = tw_kwargs.get('end', None)
    init = tw_kwargs.get('init', None)

    if X0 is None:
        if init is not None and isinstance(init, xt.TwissTable):
            raise NotImplementedError('TwissTable as init not implemented.')
        X0_str = _build_beta0_block_string(tw_kwargs)
    else:
        X0_str = f'X0 = {X0}, '

    if start is not None and end is None or start is None and end is not None:
        raise ValueError('Start and end must be specified together.')

    if start is not None and end is not None and X0_str == '':
        raise ValueError('Initial conditions must be specified when start and end are given.')

    # if not (start is None and end is None and init is None) \
    #     and not (start is not None and end is not None and X0_str != ''):
    #     raise ValueError('Start and end must be specified together, as well as initial conditions, if open twiss is used.')

    full_twiss_str = ''

    tw_columns = TW_BASE_COLUMNS.copy()

    full_twiss_str = f"implicit=true, nslice={nslice}, misalign=true, coupling={str(coupling_edw_teng).lower()}, chrom={str(compute_chromatic_properties).lower()}"

    if coupling_edw_teng:
        tw_columns += COUPLING_COLUMNS
    if compute_chromatic_properties:
        tw_columns += CHROM_COLUMNS
    columns = tw_columns + list(rdts)

    send_cmd = _build_column_send_script(columns)

    if len(rdts) > 0:
        mng_script = _build_rdt_script(mng._sequence_name, rdts, columns)
    else:
        range_str = ''

        if start is not None and end is not None:
            normal_form = False
            # Range Twiss
            range_str = f"range = '{start}/{end}', "

        mng_script = ('''
        -- twiss without RDTs
        local mtbl = twiss {sequence=''' f'{mng._sequence_name}, method=4,\
        {X0_str} {range_str} {full_twiss_str}'
        '''}
        '''
        + send_cmd)

    mng.send(mng_script)

    out = mng.recv('columns')
    out_dct = {k: v for k, v in zip(columns, out)}

    # Add to table
    names = line._element_names_unique
    i_start = names.index(start) if start is not None else 0
    i_end = names.index(end) if end is not None else len(names) - 1
    marker_nums = 2 if i_start > i_end else 0 # MAD-NG wrap-around markers

    if xsuite_tw:
        xs_tw_kwargs = {
            NG_XS_MAP.get(k, k): v for k, v in tw_kwargs.items()
        }
        try:
            tw = line.twiss(method='4d', reverse=False, **xs_tw_kwargs)
        except Exception as e:
            print(f"Error occurred while getting twiss: {e}\nContinuing without Xsuite Twiss")
            xsuite_tw = False

    if not xsuite_tw:
        # Handle wrap-around range
        if i_start > i_end:
            name_co = np.array(names[i_start:] + names[:i_end + 1] + ('_end_point',))
        else:
            name_co = np.array(names[i_start:i_end + 1] + ('_end_point',))

        tw = xt.TwissTable({"name": name_co})
    tw._action = _action

    # Consistency check
    if start is None and end is None:
        assert len(out[0]) == len(tw) + 1
    else:
        assert len(out[0]) == len(tw) + marker_nums - 1

    if start is None and end is None:
        mode = "full"
    elif i_start > i_end and i_end > 1:
        mode = "wrap"
        end_idx = len(line.element_names) - list(line.element_names).index(start)
    elif marker_nums > 0:
        mode = "marker"
    else:
        mode = "range"

    def _process_data(data):
        data = np.atleast_1d(np.squeeze(data))
        if mode == "full":
            return data[:-1]
        elif mode == "wrap":
            return np.concatenate((data[0:1], data[0:end_idx], data[end_idx + 2:]))
        elif mode == "marker":
            return np.concatenate((data[0:1], data[:-marker_nums]))
        elif mode == "range":
            return np.concatenate((data[0:1], data))
        else:
            raise ValueError(f"Unexpected mode: {mode}")

    # enforce marker
    for nn in tw_columns:
        tw[f"{nn}_ng"] = _process_data(out_dct[nn])

    for nn in rdts:
        tw[nn] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]

    if compute_chromatic_properties:
        temp_x = tw.wx_ng * np.exp(1j*2*np.pi*tw.phix_ng) # potentially multiply phix_ng by 2 (MAD-NG 1.1.8)
        tw['ax_ng'] = np.imag(temp_x)
        tw['bx_ng'] = np.real(temp_x)
        temp_y = tw.wy_ng * np.exp(1j*2*np.pi*tw.phiy_ng) # potentially multiply phiy_ng by 2 (MAD-NG 1.1.8)
        tw['ay_ng'] = np.imag(temp_y)
        tw['by_ng'] = np.real(temp_y)
        del tw['phix_ng']
        del tw['phiy_ng']

    if normal_form:
        mng_script_nf = (
            '''
            local track in MAD  -- like "from MAD import track"
            local mytrktable, mytrkflow = MAD.track{sequence='''
            f'{mng._sequence_name}, method=4, mapdef={mapdef_normal_form}, nslice={nslice}'
            '''}

            local normal in MAD.gphys  -- like "from MAD.gphys import normal"
            local my_norm_for = normal(mytrkflow[1]):analyse('anh') -- anh stands for anharmonicity

            local nf = my_norm_for
            last_nf = my_norm_for
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
        ''')
        mng.send(mng_script_nf)
        out_nf = mng.recv('normal_forms_to_send')

        dct_nf = dict(
            q1 =   out_nf[0],
            q2 =   out_nf[1],
            dq1 =  out_nf[2],
            dq2 =  out_nf[3],
            d2q1 = out_nf[4],
            d2q2 = out_nf[5],
            d3q1 = out_nf[6],
            d3q2 = out_nf[7],
            d4q1 = out_nf[8],
            d4q2 = out_nf[9],
            d5q1 = out_nf[10],
            d5q2 = out_nf[11],
            dqxdjx = out_nf[6]*2.,
            dqydjy = out_nf[7]*2.,
            dqxdjy = out_nf[8]*2.,
            dqydjx = out_nf[9]*2.,
        )
        for nn in dct_nf:
            tw[nn+'_nf_ng'] = dct_nf[nn]

    return tw

def madng_get_init(line, at):
    if not hasattr(line.tracker, '_madng'):
        line.build_madng_model()
    mng = line.tracker._madng
    if at == xt.START:
        at = "1"
    else:
        at = f"'{at}'"

    mng.send(f"""
        local observed in MAD.element.flags
        {mng._sequence_name}:select(observed, {{list = {{{at}}}}})
        twpart, mf = twiss {{sequence = {mng._sequence_name}, observe = 1, savemap = true, info = 2}}
        {XSUITE_MADNG_ENV_NAME}.X0 = twpart[{at}].__map
    """)
    return f"{XSUITE_MADNG_ENV_NAME}.X0"

def _survey_ng(line):
    if not hasattr(line.tracker, '_madng'):
        line.build_madng_model()
    mng = line.tracker._madng
    mng['srv'] = mng.survey(sequence=mng._sequence_name)

    survey_tab_keys = {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "l": "length",
        "kind": "element_type"
    }

    element_types = {
        "drift": "Drift",
        "sbend": "Bend",
        "rbend": "RBend",
        "quadrupole": "Quadrupole",
        "sextupole": "Sextupole",
        "octupole": "Octupole",
        "multipole": "Multipole",
        "kicker": "Kicker", # no coloring in survey plot
        "rfcavity": "Cavity",
        "marker": "Marker"
    }

    # create SurveyTable from DataFrame
    survey_df = mng['srv'][0].to_df()
    survey_dict = survey_df.to_dict(orient='list')
    survey_dict = {k: np.array(v) for k, v in survey_dict.items()}
    for k in survey_tab_keys.keys():
        if k in survey_dict:
            # Rename keys to match SurveyTable
            survey_dict[survey_tab_keys[k]] = survey_dict[k]
            del survey_dict[k]

    survey_dict['element_type'] = np.array([element_types.get(et, et) for et in survey_dict['element_type']])

    for i in survey_dict.keys():
        # Interpretation of survey is shifted by 1 in MAD-NG vs. Xsuite
        if i in ['name', 'length', 'kind', 'element_type', 'angle', 'tilt']:
            survey_dict[i] = survey_dict[i][1:]
        else:
            survey_dict[i] = survey_dict[i][:-1]

    survey_tab = xt.survey.SurveyTable(survey_dict)
    return survey_tab


class ActionTwissMadng(Action):
    def __init__(self, line, tw_kwargs={}, **kwargs):
        self.line = line
        self.tw_kwargs = tw_kwargs
        self.tw_kwargs.update(kwargs)
        self._alredy_prepared = False
        self.X0 = None

    def prepare(self, force=False):

        if self._alredy_prepared and not force:
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

        self._alredy_prepared = True

    def run(self):
        return self.line.madng_twiss(xsuite_tw = False, X0=self.X0, **self.tw_kwargs)

class ActionTwissMadngTPSA(Action):
    def __init__(self, line, vary_names, targets = [], tw_kwargs={}, twiss_flag=True, **kwargs):
        self.line = line
        self.vary_names = vary_names
        self.targets = targets
        self.target_locations = None # set in prepare
        self.tw_kwargs = tw_kwargs
        self.tw_kwargs.update(kwargs)
        self.twiss_flag = twiss_flag
        self._already_prepared = False

    def prepare(self, force=False):
        '''
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
        '''

        if self._already_prepared and not force:
            return

        init = self.tw_kwargs.get('init', None)
        start = self.tw_kwargs.get('start', None)
        end = self.tw_kwargs.get('end', None)

        if init is None:
            if start is not None and end is not None:
                raise ValueError('If start and end are specified, init must be provided as TwissTable.')
            else:
                init = self.line.madng_twiss(**self.tw_kwargs)
            self.tw_kwargs.update({'init': init})

        assert isinstance(init, xt.TwissTable)
        madng_init_flag = "x_ng" in init.cols
        quantity_appendix = "_ng" if madng_init_flag else ""

        if not hasattr(self.line.tracker, '_madng'):
            self.line.build_madng_model()
        mng = self.line.tracker._madng
        self.mng = mng

        self.target_locations = set()
        targets_map_str = ''
        self.target_quantities = set()

        xs_ng_target_map = XSUITE_MADNG_ENV_NAME + '.xs_ng_target_map = {}\n'

        for i, target in enumerate(self.targets):
            if isinstance(target.tar, tuple):
                qty = target.tar[0][:-3] if target.tar[0].endswith('_ng') else XS_NG_MAP[target.tar[0]]
                assert qty in TPSA_ALLOWED_TARGETS, f"Target quantity '{target.tar[0]}' not allowed with TPSA matching."
                self.target_locations.add(target.tar[1])

                aux_str = ''
                if qty in OPTFUN_QUANTITIES:
                    aux_str = 'optfun = true'
                elif qty in ['x', 'px', 'y', 'py', 't', 'pt']:
                    aux_str = f'orbit = {['x', 'px', 'y', 'py', 't', 'pt'].index(qty) + 1}'

                # set string for quantity mapping + loc to save in madng
                targets_map_str += f"{XSUITE_MADNG_ENV_NAME}.targets_map[{i+1}] = {{ loc = '{target.tar[1]}', qty = '{qty}', {aux_str} }}\n"
                self.target_quantities.add(target.tar[0])
                xs_ng_target_map += f"{XSUITE_MADNG_ENV_NAME}.xs_ng_target_map['{target.tar[0]}'] = '{qty}'\n"

            elif hasattr(target, "start") and hasattr(target, "end"):
                start_loc_str = 'nil'
                end_loc_str = end
                if target.start != '__ele_start__':
                    self.target_locations.add(target.start)
                    start_loc_str = target.start
                if target.end != '__ele_stop__':
                    self.target_locations.add(target.end)
                    end_loc_str = target.end

                qty = target.var[:-3] if target.var.endswith('_ng') else XS_NG_MAP[target.var]
                assert qty in TPSA_ALLOWED_TARGETS, f"Target quantity '{target.var}' not allowed with TPSA matching."
                targets_map_str += f"{XSUITE_MADNG_ENV_NAME}.targets_map[{i+1}] = {{ loc = '{end_loc_str}', qty = '{qty}', loc_start = '{start_loc_str}', optfun = true }}\n"
                self.target_quantities.add(target.var)
                xs_ng_target_map += f"{XSUITE_MADNG_ENV_NAME}.xs_ng_target_map['{target.var}'] = '{qty}'\n"
        self.target_locations = list(self.target_locations)

        # set coords (TODO: delta)
        beta0 = self.line.particle_ref.beta0[0]

        start_loc = 0 if start is None else start
        init_coord = np.array([init['x' + quantity_appendix, start_loc],
                                init['px' + quantity_appendix, start_loc],
                                init['y' + quantity_appendix, start_loc],
                                init['py' + quantity_appendix, start_loc],
                                0,
                                0])
        init_coord[4] = init['t' + quantity_appendix, start_loc] if madng_init_flag else init['zeta', start_loc] * beta0
        init_coord[5] = init['pt' + quantity_appendix, start_loc] if madng_init_flag else dp2pt(init['delta', start_loc], beta0)

        coord_str = ''
        part_order = ['x', 'px', 'y', 'py', 't', 'pt']
        for part, val in zip(part_order, init_coord):
            if np.abs(val) > 1e-12:
                coord_str += f'X0.{part} = {val} '

        param_assignment_str = ''
        param_list_str = '{'
        for name in self.vary_names:
            param_assignment_str += f"MADX['{name}'] = MADX['{name}'] + X0['{name}'] \n"
            param_list_str += f"'{name}', "
        param_list_str = param_list_str[:-2] + '}'

        observables_str = '{'
        if start is not None and end is not None:
            observables_str += f"'{start}', '{end}', "

        if self.target_locations is not None:
            for loc in self.target_locations:
                if loc != start and loc != end:
                    observables_str += f"'{loc}', "
        observables_str += '}'

        qty_str = '{'
        for qty in self.target_quantities:
            qty_str += f"'{qty}', "
        qty_str += '}'

        if madng_init_flag:
            init_cond_str = f"local B0 = MAD.beta0 {{ beta11 = {init['beta11' + quantity_appendix, start_loc]},\n" + f"beta22 = {init['beta22' + quantity_appendix, start_loc]},\n"\
            + f"alfa11 = {init['alfa11' + quantity_appendix, start_loc]},\n" + f"alfa22 = {init['alfa22' + quantity_appendix, start_loc]},\n"\
            + f"dx = {init['dx' + quantity_appendix, start_loc]},\n" + f"dpx = {init['dpx' + quantity_appendix, start_loc]},\n"\
            + f"dy = {init['dy' + quantity_appendix, start_loc]},\n" + f"dpy = {init['dpy' + quantity_appendix, start_loc]}\n }}"
        else:
            init_cond_str = f"local B0 = MAD.beta0 {{ beta11 = {init['betx', start_loc]},\n" + f"beta22 = {init['bety', start_loc]},\n"\
            + f"alfa11 = {init['alfx', start_loc]},\n" + f"alfa22 = {init['alfy', start_loc]},\n"\
            + f"dx = {init['dx', start_loc]},\n" + f"dpx = {init['dpx', start_loc]},\n"\
            + f"dy = {init['dy', start_loc]},\n" + f"dpy = {init['dpy', start_loc]}\n }}"

        mng_init_str = r'''
            ''' + XSUITE_MADNG_ENV_NAME + r''' = {} -- to avoid variable name clashes
            local obs_flag = MAD.element.flags.observed

            local pts=''' + observables_str + r'''

            ''' + mng._sequence_name + r''':select(obs_flag, {list=pts})

            local params = ''' + param_list_str + r'''

            local X0 = MAD.damap {
                nv=6, -- number of variables
                mo=2, -- max order of variables
                np=#params, -- number of parameters
                po=1, -- max order of parameters
                pn=params, -- parameter names
            }

            ''' + coord_str + r'''

            -- Converting to TPSA (mutating type)
            for _, v in ipairs(params) do
                MADX[v] = MADX[v] + X0[v]
            end

            ''' + init_cond_str + r'''

            local map1 = MAD.gphys.bet2map(B0, X0)

            -- Maps target locations to damaps
            ''' + XSUITE_MADNG_ENV_NAME + r'''.target_loc_map = table.new(0, ''' + str(len(self.target_locations)) + r''')
            -- Array of targets with additional info (location, quantity, orbit/optical function)
            ''' + XSUITE_MADNG_ENV_NAME + r'''.targets_map = table.new(''' + str(len(self.targets)) + r''', 0)
            -- List of target quantities (suitable for MAD-NG)
            ''' + XSUITE_MADNG_ENV_NAME + r'''.tar_qtys = ''' + qty_str + r'''
            -- Initial map for tracking/twiss
            ''' + XSUITE_MADNG_ENV_NAME + r'''.init_X0_map = map1
            -- Defining targets array
            ''' + targets_map_str + r'''
            -- Mapping from xsuite quantity names to madng quantity names
            ''' + xs_ng_target_map + r'''
            '''

        mng.send(mng_init_str)

        self._already_prepared = True

    def run(self):
        '''
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
        '''

        if self._already_prepared is False:
            self.prepare()

        start = self.tw_kwargs.get('start', None)
        end = self.tw_kwargs.get('end', None)

        operation = 'twiss' if self.twiss_flag else 'track'

        range_str = ''
        if start is not None and end is not None:
            range_str = f"range = '{start}/{end}', "
        mng_track_str = (
            f"local trk, mflw = MAD.{operation}{{\n"
            f"    sequence={self.mng._sequence_name},\n"
            f"    X0={XSUITE_MADNG_ENV_NAME}.init_X0_map,\n"
            f"    savemap=true,\n"
            f"    observe=1,\n"
            f"    {range_str}\n"
            f"}}\n"
            f"{XSUITE_MADNG_ENV_NAME}.trk = trk\n"
        )

        self.mng.send(mng_track_str)

        loc_map_str = ''
        for i, loc in enumerate(self.target_locations):
            loc_map_str += f"{XSUITE_MADNG_ENV_NAME}.target_loc_map['{loc}'] = {XSUITE_MADNG_ENV_NAME}.trk['{loc}'].__map\n"


        if self.twiss_flag:
            mng_table_str = r'''
            local trk = ''' + XSUITE_MADNG_ENV_NAME + r'''.trk
            ''' + loc_map_str + r'''
            py:send(trk)
            '''

            res = self.mng.send(mng_table_str).recv(XSUITE_MADNG_ENV_NAME + '.trk').to_df()
            res = xt.TwissTable(res)

            # Add quantities which are not present yet with the name corresponding to the target quantity
            for qty in self.target_quantities:
                if qty not in res.cols:
                    res[qty] = res[qty[:-3] if qty.endswith('_ng') else XS_NG_MAP[qty]]
        else:
            loc_map_str = ''
            for loc in self.target_locations:
                loc_map_str += f"{XSUITE_MADNG_ENV_NAME}.target_loc_map['{loc}'] = {XSUITE_MADNG_ENV_NAME}.trk['{loc}'].__map\n"

            mng_table_str = r'''
            local trk = ''' + XSUITE_MADNG_ENV_NAME + r'''.trk
            -- Add derived columns which are not present due to Track calculation
            -- and use target names as defined from the user (Xsuite)
            for _, tar in ipairs( ''' + XSUITE_MADNG_ENV_NAME + r'''.tar_qtys ) do
                if not trk[''' + XSUITE_MADNG_ENV_NAME + r'''.xs_ng_target_map[tar]] then
                    trk:addcol(tar, \ri -> MAD.gphys.optfun(trk[ri].__map, ''' + XSUITE_MADNG_ENV_NAME + r'''.xs_ng_target_map[tar] .. '_'))
                end
            end

            -- Save damaps
            ''' + loc_map_str + r'''
            py:send(trk)
            '''

            res = self.mng.send(mng_table_str).recv(XSUITE_MADNG_ENV_NAME + '.trk').to_df()
            res = xt.TwissTable(res)

        return res

    def acquire_jacobian(self):
        '''
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
        '''

        tar_len_str = f"local tarlen = {len(self.targets)}\n"
        vary_len_str = f"local varylen = {len(self.vary_names)}\n"
        jac_decl_str = f"{XSUITE_MADNG_ENV_NAME}.jac = MAD.matrix(tarlen, varylen)\n"

        mng_str = tar_len_str + vary_len_str + jac_decl_str + r'''
        -- Compute Jacobian
        for i, target in ipairs(  ''' + XSUITE_MADNG_ENV_NAME + r'''.targets_map ) do
            local map = ''' + XSUITE_MADNG_ENV_NAME + r'''.target_loc_map[target.loc]
            for j = 1, map.np(map), 1 do
                -- Quantity which can be calculated with optfun
                if target.optfun then
                    -- If loc_start (phase advance) is defined, we provide initial map
                    if target.loc_start then
                        local a0 = ''' + XSUITE_MADNG_ENV_NAME + r'''.target_loc_map[target.loc_start]
                        ''' + XSUITE_MADNG_ENV_NAME + r'''.jac[(i-1)*varylen + j] = MAD.gphys.optfun(map, target.qty .. "_", j, 1, a0)
                    else
                        ''' + XSUITE_MADNG_ENV_NAME + r'''.jac[(i-1)*varylen + j] = MAD.gphys.optfun(map, target.qty .. "_", j, 1)
                    end

                -- Orbit Quantity
                elseif target.orbit then
                    -- Index 7 + j corresponds to one index of order 0 plus six (for x,px,y,py,t,pt) of order 1 plus parameter index
                    ''' + XSUITE_MADNG_ENV_NAME + r'''.jac[(i-1)*varylen + j] = map[target.orbit]:get(7 + j)
                end
            end
        end

        py:send(''' + XSUITE_MADNG_ENV_NAME + r'''.jac)
        '''

        self.mng.send(mng_str)

        jac = np.array(self.mng.recv())

        return jac

    def cleanup(self):
        # Need to reconvert TPSAs to normal values
        if self._already_prepared is True:
            mng_str = ''
            for var_name in self.vary_names:
                mng_str += f"MADX['{var_name}'] = MADX['{var_name}']:get0()\n"
            mng_str += f"{XSUITE_MADNG_ENV_NAME}.X0 = nil\n"
            self.mng.send(mng_str)
            self._already_prepared = False

def line_to_madng(line, sequence_name='seq', temp_fname=None, keep_files=False,
                  **kwargs):
    try:
        _ge = xt.elements._get_expr
        if temp_fname is None:
            temp_fname = 'temp_madng_' + str(uuid.uuid4())

        from .mad_writer import to_madng_sequence
        madx_seq = to_madng_sequence(line, name=sequence_name)
        with open(f'{temp_fname}.mad', 'w') as fid:
            fid.write(madx_seq)

        from pymadng import MAD

        nocharge = str(kwargs.pop('nocharge', True)).lower()

        mng = MAD(mad_path="/home/babreufi/git/MAD-NG-1.1.5/bin/mad", **kwargs)
        mng.send(f"""
                 local mad_func = loadfile('{temp_fname}.mad', nil, MADX)
                 assert(mad_func)
                 mad_func()
                 MAD.option.nocharge = {nocharge}
                 MADX.option.rbarc = true
                 {XSUITE_MADNG_ENV_NAME} = {{}} -- to avoid variable name clashes
                 """)
        mng._init_madx_data = madx_seq

        mng[sequence_name] = mng.MADX[sequence_name] # this ensures that the file has been read
        mng[sequence_name].beam = mng.beam(particle="'custom'",
                        mass=line.particle_ref.mass0 * 1e9,
                        charge=line.particle_ref.q0,
                        betgam=line.particle_ref.beta0[0] * line.particle_ref.gamma0[0])

    finally:
        if not keep_files:
            for nn in [temp_fname + '.madx', temp_fname + '.mad']:
                if os.path.isfile(nn):
                    os.remove(nn)

    return mng
