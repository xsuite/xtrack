import xtrack as xt

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq')
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

ll = env.lhcb2
llr = ll.copy()

def _reverse_element(env, name):
        """Return a reversed element without modifying the original."""

        SUPPORTED={'RBend', 'Bend', 'Quadrupole', 'Sextupole', 'Octupole',
                   'Multipole', 'Cavity', 'Solenoid', 'RFMultipole',
                   'Marker', 'Drift'}

        ee = env.get(name)
        ee_ref = env.ref[name]

        if ee.__class__.__name__ not in SUPPORTED:
            raise NotImplementedError(
                f'Cannot reverse the element `{name}`, as reversing elements '
                f'of type `{element.__class__.__name__}` is not supported!'
            )

        def _reverse_field(key):
            if hasattr(ee, key):
                key_ref = getattr(ee_ref, key)
                if key_ref._expr is not None:
                    setattr(ee_ref, key,  -(key_ref._expr))
                else:
                    setattr(ee_ref, key,  -(key_ref._value))

        def _exchange_fields(key1, key2):
            value1 = None
            if hasattr(ee, key1):
               key1_ref = getattr(ee_ref, key1)
               value1 = key1_ref._expr or key1_ref._value

            value2 = None
            if hasattr(ee, key2):
               key2_ref = getattr(ee_ref, key2)
               value2 = key2_ref._expr or key2_ref._value


            if value1 is not None:
                setattr(ee_ref, key2, value1)

            if value2 is not None:
                setattr(ee_ref, key1, value2)

        _reverse_field('k0s')
        _reverse_field('k1')
        _reverse_field('k2s')
        _reverse_field('k3')
        _reverse_field('ks')
        _reverse_field('ksi')
        _reverse_field('vkick')
        _reverse_field('tilt')

        if hasattr(ee, 'lag'):
            ee_ref.lag = 180 - (ee_ref.lag._expr or ee_ref.lag._value)
            element['lag']['expr'] = f'180 - ({element["lag"]["expr"]})'

        if hasattr(ee, 'knl'):
            for i in range(1, len(ee.knl), 2):
                ee_ref.knl[i] = -(ee_ref.knl[i]._expr or ee_ref.knl[i]._value)

        if hasattr(ee, 'ksl'):
            for i in range(0, len(ee.ksl), 2):
                ee_ref.ksl[i] = -(ee_ref.ksl[i]._expr or ee_ref.ksl[i]._value)

        _exchange_fields('edge_entry_model', 'edge_exit_model')
        _exchange_fields('edge_entry_angle', 'edge_exit_angle')
        _exchange_fields('edge_entry_angle_fdown', 'edge_exit_angle_fdown')
        _exchange_fields('edge_entry_fint', 'edge_exit_fint')
        _exchange_fields('edge_entry_hgap', 'edge_exit_hgap')

for nn in llr.element_names:
    _reverse_element(llr, nn)

llr.discard_tracker()
llr.element_names = llr.element_names[::-1]

llr.particle_ref = xt.Particles(p0c=7e12)