"""
This is a **very limited** xtrack to bmad converter.
It only works the specific cases used in the tests.
"""


def bmad_run(line, track=None):

    tt = line.get_table(attr=True)

    out_lines = []
    if track is not None:
        x = track['x']
        px = track['px']
        y = track['y']
        py = track['py']
        delta = track['delta']
        spin_x = track['spin_x']
        spin_y = track['spin_y']
        spin_z = track['spin_z']
        out_lines += [
            f'beam, energy  = {line.particle_ref.energy0[0]/1e9}',
            'parameter[particle] = positron',
            'parameter[geometry] = open',
            'bmad_com[spin_tracking_on]=T',
            'bmad_com[radiation_damping_on]=F',
            'bmad_com[radiation_fluctuations_on]=F',
            ''
            f'particle_start[spin_x] = {spin_x}',
            f'particle_start[spin_y] = {spin_y}',
            f'particle_start[spin_z] = {spin_z}',
            f'particle_start[x] = {x}',
            f'particle_start[px] = {px}',
            f'particle_start[y] = {y}',
            f'particle_start[py] = {py}',
            f'particle_start[pz] = {delta}',
            ''
            'beginning[beta_a]  =  1',
            'beginning[alpha_a]=  0',
            'beginning[beta_b] =   1',
            'beginning[alpha_b] =   0',
            'beginning[eta_x] =  0',
            'beginning[etap_x] =0',
        ]
    else:
        out_lines += [
            f'beam, energy  = {line.particle_ref.energy0[0]/1e9}',
            'parameter[particle] = electron',
            'parameter[geometry] = closed',
            'bmad_com[spin_tracking_on]=T',
            'bmad_com[radiation_damping_on]=F',
            'bmad_com[radiation_fluctuations_on]=F',
            ''
        ]

    for nn in line.element_names:
        if '$' in nn:
            continue
        ee = line[nn]
        clssname = ee.__class__.__name__

        if clssname == 'Marker':
            out_lines.append(f'{nn}: marker')
        elif clssname == 'Drift':
            out_lines.append(f'{nn}: drift, l = {ee.length}')
        elif clssname == 'Quadrupole':
            if ee.k1s == 0:
                out_lines.append(f'{nn}: quadrupole, l = {ee.length}, k1 = {ee.k1}')
            else:
                assert ee.k1 == 0
                out_lines.append(f'{nn}: quadrupole, l = {ee.length}, k1 = {-ee.k1s}, tilt')
        elif clssname == 'Multipole':
            raise ValueError('Multipole not supported')
        elif clssname == 'Magnet':
            out_lines.append(f'{nn}: kicker, l = {ee.length}, hkick={-ee.knl[0]},'
                            f' vkick={ee.ksl[0]}')
            # if np.linalg.norm(ee.knl) == 0 and np.linalg.norm(ee.ksl) == 0:
            #     out_lines.append(f'{nn}: marker') # Temporary
            # else:
            #     assert len(ee.knl) == 1
            #     assert len(ee.ksl) == 1
            #     out_lines.append(f'{nn}: ab_multipole, l = {ee.length},'
            #                      f'a0 = {ee.ksl[0]}, b0 = {ee.knl[0]}')
        elif clssname == 'Sextupole':
            out_lines.append(f'{nn}: sextupole, l = {ee.length}, k2 = {ee.k2}')
        elif clssname == 'Bend':
            out_lines.append(f'{nn}: sbend, l = {ee.length}, angle = {ee.angle}')
        elif clssname == 'RBend':
            out_lines.append(f'{nn}: rbend, l = {ee.length_straight}, angle = {ee.angle}')
        elif clssname == 'Cavity':
            out_lines.append(f'{nn}: marker') # Patch!!!!
            # out_lines.append(f'{nn}: rfcavity, voltage = {ee.voltage},'
            #                  f'rf_frequency = {ee.frequency}, phi0 = 0.') # Lag hardcoded for now!
        elif clssname == 'Octupole':
            out_lines.append(f'{nn}: octupole, l = {ee.length}, k3 = {ee.k3}')
        elif clssname == 'DriftSlice':
            ll = tt['length', nn]
            out_lines.append(f'{nn}: drift, l = {ll}')
        elif clssname == 'Solenoid':
            out_lines.append(f'{nn}: solenoid, l = {ee.length}, ks = {ee.ks}')
        else:
            raise ValueError(f'Unknown element type {clssname} for {nn}')

    out_lines += [
        '',
        'ring: line = ('
    ]

    for nn in line.element_names:
        if '$' in nn:
            continue
        out_lines.append(f'    {nn},')
    # Strip last comma
    out_lines[-1] = out_lines[-1][:-1]
    out_lines += [
        ')',
        'use, ring',
    ]

    with open('test.bmad', 'w') as fid:
        fid.write('\n'.join(out_lines))

    from pytao import Tao
    if track is not None:
        tao = Tao(' -lat test.bmad -noplot ')
        tao.cmd('show -write orbit.txt lat -all -att orbit.x@es20.10 '
                '-att orbit.y@es20.10 -att beta.a@es20.10 -att beta.b@es20.10')
        tao.cmd('show -write vvv.txt lat -spin -all')
    else:
        tao = Tao(' -lat test.bmad -noplot ')
        tao.cmd('show -write spin.txt spin')
        tao.cmd('show -write orbit.txt lat -all -att orbit.x@es20.10 '
                '-att orbit.y@es20.10 -att beta.a@es20.10 -att beta.b@es20.10')
        tao.cmd('show -write vvv.txt lat -spin -all')


    import pandas as pd
    import io
    def parse_file_pandas(filename, ftype='spin'):
        # Read the whole file first
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Filter out comment lines
        data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]

        # Join the data into a single string
        data_str = ''.join(data_lines)

        if ftype == 'spin':
            col_names = [
                'index', 'name', 'key', 's',
                'spin_x', 'spin_y', 'spin_z',
                'spin_dn_dpz_x', 'spin_dn_dpz_y', 'spin_dn_dpz_z', 'spin_dn_dpz_amp'
            ]
        elif ftype == 'orbit':
            col_names = [
                'name', 's', 'l', 'x', 'y', 'betx', 'bety'
            ]

        # Now read into pandas
        df = pd.read_csv(
            io.StringIO(data_str),
            sep=r'\s+',
            header=None,
            names=col_names
        )

        return df

    def parse_twiss_file_pandas(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        # Remove comment and empty lines
        data_lines = [line for line in lines if not line.strip().startswith('#') and line.strip()]

        # Join into a single string buffer
        data_str = ''.join(data_lines)

        # Define column names manually based on the comment headers
        column_names = col_names = [
                'name', 's', 'l', 'x', 'y', 'betx', 'bety'
            ]

        # Read into pandas
        df = pd.read_csv(
            io.StringIO(data_str),
            sep='\s+',
            header=None,
            names=column_names
        )

        # Optionally: convert '---' to NaN in numeric columns
        df.replace('---', pd.NA, inplace=True)
        df[['l']] = df[['l']].apply(pd.to_numeric, errors='coerce')

        return df


    df = parse_file_pandas('vvv.txt')
    df_orb = parse_twiss_file_pandas('orbit.txt')

    if track is None:
        spin_summary_bmad = {}
        with open('spin.txt', 'r') as fid:
            spsumm_lines = fid.readlines()
        for ll in spsumm_lines:
            if ':' in ll:
                key, val = ll.split(':')
                val = val.strip()
                if ' ' in val:
                    val = [float(v) for v in val.split(' ') if v]
                else:
                    val = float(val.strip())
                spin_summary_bmad[key.strip()] = val

    out= {
        'optics': df_orb,
        'spin': df
    }
    if track is None:
        out['spin_summary'] = spin_summary_bmad

    return out