import numpy as np
from .match import Action
import os
import uuid

from .mad_writer import mad_str_or_value
import xtrack as xt

class MadngVars:

    def __init__(self, mad):
        self.mad = mad

    def __setitem__(self, key, value):
        setattr(self.mad.MADX, key.replace('.', '_'), value)
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

def _tw_ng(line, rdts=(), normal_form=True,
           mapdef_twiss=2, mapdef_normal_form=4,
           nslice=3):

    tw_kwargs = locals()
    del tw_kwargs['line']
    _action = ActionTwissMadng(line, tw_kwargs)

    if not hasattr(line.tracker, '_madng'):
        line.build_madng_model()
    mng = line.tracker._madng

    tw = line.twiss(method='4d', reverse=False)
    tw._action = _action

    tw_columns = ['s', 'beta11', 'beta22', 'alfa11', 'alfa22',
                'x', 'px', 'y', 'py', 't', 'pt',
                'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2',
                'beta12', 'beta21', 'alfa12', 'alfa21',
                'wx', 'wy', 'phix', 'phiy', 'dmu1', 'dmu2',
                'f1001', 'f1010', 'r11', 'r12', 'r21', 'r22',
    ]

    columns = tw_columns + list(rdts)
    rdt_cmd = 'local rdts = {"' + '", "'.join(rdts) + '"}'
    mng_columns_to_send = ["mtbl." + col for col in columns]
    send_cmd = f'''
        columns = {{{", ".join(mng_columns_to_send)}}}
        py:send(columns, true)
    '''

    if len(rdts) > 0:
        mng_script = (
            f'''
            local damap in MAD
            local seq = MADX.{mng._sequence_name}
            -- list of RDTs
            '''
            + rdt_cmd +
            '''
            -- create phase-space damap at 4th order
            local X0 = damap {nv=6, mo=4}

            -- twiss with RDTs
            local mtbl = twiss {sequence=seq, X0=X0, trkrdt=rdts, info=2, saverdt=true, coupling=true, chrom=true}

            -- send columns to Python
            '''
            + send_cmd
        )
    else:
        mng_script = ('''
        local damap in MAD
        '''
        f'local seq = MADX.{mng._sequence_name}'
        '''

        -- twiss with RDTs
        local mtbl = twiss {sequence=seq, method=4,'''
        f'mapdef={mapdef_twiss}'
        ''', implicit=true, '''
        f'nslice={nslice}, misalgn=true, coupling=true, chrom=true'
        '''}

        -- send columns to Python
        '''
        + send_cmd)

    mng.send(mng_script)

    out = mng.recv('columns')
    out_dct = {k: v for k, v in zip(columns, out)}

    # Add to table
    assert len(out[0]) == len(tw) + 1
    for nn in tw_columns:
        tw[nn+'_ng'] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]
    for nn in rdts:
        tw[nn] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]

    temp_x = tw.wx_ng * np.exp(1j*2*np.pi*tw.phix_ng)
    tw['ax_ng'] = np.imag(temp_x)
    tw['bx_ng'] = np.real(temp_x)
    temp_y = tw.wy_ng * np.exp(1j*2*np.pi*tw.phiy_ng)
    tw['ay_ng'] = np.imag(temp_y)
    tw['by_ng'] = np.real(temp_y)
    del tw['phix_ng']
    del tw['phiy_ng']

    if normal_form:
        mng_script_nf = (
            f'local seq = MADX.{mng._sequence_name}'
            '''
            local track in MAD  -- like "from MAD import track"
            local mytrktable, mytrkflow = MAD.track{sequence=seq, method=4,'''
            f'mapdef={mapdef_normal_form}, nslice={nslice}'
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
    def __init__(self, line, tw_kwargs):
        self.line = line
        self.tw_kwargs = tw_kwargs

    def run(self):
        return self.line.madng_twiss(**self.tw_kwargs)


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

        mng = MAD(**kwargs)
        mng.send(f"""
                 mad_func = loadfile('{temp_fname}.mad', nil, MADX)
                 assert(mad_func)
                 mad_func()
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