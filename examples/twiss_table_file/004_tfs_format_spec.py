"""Versioned TFS output from Python, built-in presets, JSON and TOML."""

# start-tfs-presets
import json
from pathlib import Path

import xtrack as xt


line = xt.Line(elements=[
    xt.Drift(length=1), xt.Multipole(knl=[0, 0.1]),
    xt.Drift(length=1), xt.Multipole(knl=[0, -0.1]),
] * 4)
line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
tw = line.twiss4d(strengths=True)

# Stable built-in layouts. The strengths preset requires strengths=True.
tw.to_tfs('optics_orbit.tfs', format_spec=xt.format_spec.optics_orbit_v1)
tw.to_tfs('optics_orbit_strengths.tfs',
          format_spec=xt.format_spec.optics_orbit_strengths_v1)
tw.to_tfs('madx_layout.tfs', format_spec=xt.format_spec.madx_twiss_v1)
# end-tfs-presets

# start-tfs-dictionary
# A dictionary is the complete output contract. Only listed fields are emitted.
# name is mandatory; source_name defaults to name. Names are looked up before
# case conversion. The input table's rows and row order are preserved.
spec = {
    'version': '1',
    'case': 'upper',
    'float_format': '.16e',
    'columns': [
        {'name': 'name', 'type': 'string'},
        {'name': 's', 'type': 'float', 'format': '.9f'},
        {'name': 'betx', 'type': 'float'},
        {'name': 'bety', 'type': 'float'},
    ],
    'scalars': [
        {'name': 'TYPE', 'value': 'TWISS'},
        {'name': 'Q1', 'source_name': 'qx', 'type': 'float', 'format': '.12f'},
        {'name': 'Q2', 'source_name': 'qy', 'type': 'float', 'format': '.12f'},
    ],
}
tw.to_tfs('custom.tfs', format_spec=spec)
# end-tfs-dictionary

# start-tfs-json
# JSON is convenient for saving a specification made in Python.
Path('custom_format.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
tw.to_tfs('custom_from_json.tfs', format_spec='custom_format.json')
# end-tfs-json

# start-tfs-toml
tw.to_tfs('custom_from_toml.tfs', format_spec='tfs_format_spec.toml')
# end-tfs-toml

# start-tfs-customize
# Built-ins are immutable; load returns a fully independent mutable copy.
modified = xt.format_spec.load(xt.format_spec.optics_orbit_v1)
modified['float_format'] = '.12g'
tw.to_tfs('modified.tfs', format_spec=modified)
# end-tfs-customize

#!end-doc-part

assert Path('custom.tfs').read_bytes() == Path('custom_from_json.tfs').read_bytes()
assert Path('custom.tfs').read_bytes() == Path('custom_from_toml.tfs').read_bytes()
