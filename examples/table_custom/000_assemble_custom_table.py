import xtrack as xt
import numpy as np
tt = xt.Table.from_tfs('./h4-ht-postLS3-350GeV_0wobbling_xsuite_twiss.tfs')

# Assemble keyword column (you can implement you own logic)
keyword = []
for nn in tt.element_type:
    new_nn = nn.upper() # make it uppercase
    if nn == 'DriftSlice':
        new_nn = 'DRIFT'
    elif nn.startswith('Limit'):
        new_nn = 'LIMIT'
    keyword.append(new_nn)

# Assemble a custom table with selected arrays
out_cols = {
    'NAME': tt.name,
    'KEYWORD': np.array(keyword),
    'S': tt.s,
    'L': tt.length,
    'BETX': tt.betx,
    'BETY': tt.bety,
    'ALFX': tt.alfx,
    'ALFY': tt.alfy,
    'MUX': tt.mux,
    'MUY': tt.muy,
    'DX': tt.dx,
    'DPX': tt.dpx,
    'DY': tt.dy,
    'DPY': tt.dpy,
    'X': tt.x,
    'Y': tt.y,
    'PX': tt.px,
    'PY': tt.py,
    'MUX': tt.mux,
    'MUY': tt.muy,
    'K0L': tt.k0l,
    'K1L': tt.k1l,
    'K2L': tt.k2l,
    'HKICK': tt.hkick,
    'VKICK': tt.vkick,
    'ISTHICK': tt.isthick
}

# Content for the header
header = {
    'TITLE':            "Matched Optics Table",
    'ORIGIN':           "Base file: h4-ht-postLS3-350GeV_0wobbling.str",
    'DATE':             "23/07/26",
    'TIME':             "17.46.18",
    'TYPE':             "TWISS",
    'PARTICLE':         "PROTON",
    'MASS':             tt.particle_on_co.mass0 * 1e-9,
    'CHARGE':           tt.particle_on_co.q0,
    'ENERGY':           tt.particle_on_co.energy0[0] * 1e-9,
    'PC':               tt.particle_on_co.p0c[0] * 1e-9,
}

out_table = xt.Table(out_cols, index='NAME')
out_table._data.update(header)

# Save as TFS
out_table.to_tfs('./custom_table.tfs', float_precision=9,
                 exclude=['__class__', 'xtrack_version'])