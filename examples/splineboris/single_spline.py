import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from spline_fitter.field_fitter import FieldFitter
from xtrack.beam_elements.spline_param_schema import SplineParameterSchema

"""
This example shows how to use a single spline to represent a field.
The field and its transverse derivatives are zero at the edges.
In the polynomial basis of FieldFitter, this corresponds to c_1 = c_2 = c_3 = c_4 = 0 and c_5 = \int_s_0^s_1 g_i(s) ds.
where g_i(s) is the i-th order gradient of the field.

The field is then represented as a single spline, with the coefficients of the spline being the coefficients of the polynomial.
The spline is then used to calculate the field at any point in space.

The example shows how to use a single spline to represent a field, and how to use the spline to calculate the field at any point in space.

The coefficients need to be ordered correctly in order to pass them to the SplineBoris element.

The example also shows how to use the SplineParameterSchema to create a parameter table for the spline.
"""

# Splines for each field component:
# Field:
# ks represent coefficients of Bx
# kn represent coefficients of By
# bs represents coefficients of Bz
ks_0 = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
kn_0 = np.array([0.0, 0.0, 0.0, 0.0, 2.0])
bs   = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

# Derivatives w.r.t. x:
# ks represent coefficients of Bx'
# kn represent coefficients of By'
ks_1 = np.array([0.0, 0.0, 0.0, 0.0, 0.5])
kn_1 = np.array([0.0, 0.0, 0.0, 0.0, 1.5])

# Position along the line:
s = np.linspace(0, 1, 100)

# Calculate the field and its derivatives:
Bx_0 = FieldFitter._poly(s[0], s[-1], ks_0)(s)
By_0 = FieldFitter._poly(s[0], s[-1], kn_0)(s)
Bz_0 = FieldFitter._poly(s[0], s[-1], bs)(s)
Bx_1 = FieldFitter._poly(s[0], s[-1], ks_1)(s)
By_1 = FieldFitter._poly(s[0], s[-1], kn_1)(s)

# Plot the field and its derivatives:
fig, ax = plt.subplots()
ax.plot(s, Bx_0, label="Bx (on-axis)")
ax.plot(s, By_0, label="By (on-axis)")
ax.plot(s, Bz_0, label="Bz (on-axis)")
ax.plot(s, Bx_1, label="dBx/dx (on-axis)")
ax.plot(s, By_1, label="dBy/dx (on-axis)")
ax.legend()
plt.show()

# Manually construct parameter table for the splines in correct order
# MultiIndex fields: (field_component, derivative_x, region_name, s_start, s_end, idx_start, idx_end, param_index)
# For each, the "derivative_x" is 0 (on axis) or 1 (d/dx), only one "region" for the test ("slice_0"):
param_table = []

# Build a canonical parameter table following the SplineParameterSchema
multipole_order = 1  # order 0 for on-axis, order 1 for d/dx
poly_order = 4       # order of the spline, currently the only order supported is 4.

# The naming convention is defined in the SplineParameterSchema.
expected_param_names = SplineParameterSchema.get_param_names(
    multipole_order=multipole_order,
    poly_order=poly_order,
)

# Map our simple coefficients to the canonical parameter names
param_dict = {}
for k, value in enumerate(ks_0):
    param_dict[f"ks_0_{k}"] = float(value)
for k, value in enumerate(kn_0):
    param_dict[f"kn_0_{k}"] = float(value)
for k, value in enumerate(bs):
    param_dict[f"bs_{k}"] = float(value)
for k, value in enumerate(ks_1):
    param_dict[f"ks_1_{k}"] = float(value)
for k, value in enumerate(kn_1):
    param_dict[f"kn_1_{k}"] = float(value)

# Print the parameter names.
# This is the "canonical" parameter naming convention for the SplineBoris element.
print("Parameters:")
print(param_dict)

# {'ks_0_0': 0.0, 'ks_0_1': 0.0, 'ks_0_2': 0.0, 'ks_0_3': 0.0, 'ks_0_4': 1.0,
# 'kn_0_0': 0.0, 'kn_0_1': 0.0, 'kn_0_2': 0.0, 'kn_0_3': 0.0, 'kn_0_4': 2.0,
# 'bs_0': 0.0, 'bs_1': 0.0, 'bs_2': 0.0, 'bs_3': 0.0, 'bs_4': 0.0,
# 'ks_1_0': 0.0, 'ks_1_1': 0.0, 'ks_1_2': 0.0, 'ks_1_3': 0.0, 'ks_1_4': 0.5,
# 'kn_1_0': 0.0, 'kn_1_1': 0.0, 'kn_1_2': 0.0, 'kn_1_3': 0.0, 'kn_1_4': 1.5}

# Single canonical row, ordered according to the schema
param_row = [param_dict.get(name, 0.0) for name in expected_param_names]

# For this toy example the field is constant along s, so we reuse the same row at all steps
n_steps = 100
param_table = np.tile(param_row, (n_steps, 1))

print("Parameter table:")
print(param_table)

# Initialize the field calculator using the schema-compliant parameter table
spline_element = xt.SplineBoris(
    params=param_table.tolist(),
    multipole_order=multipole_order,
    s_start=s[0],
    s_end=s[-1],
    length=s[-1] - s[0],
    n_steps=n_steps,
)

line = xt.Line(elements=[spline_element])

line.particle_ref = xt.Particles(energy0=1e9, mass0=xt.ELECTRON_MASS_EV)

# Open Twiss:
tw = line.twiss4d(betx=1, bety=1, include_collective=True)

# Plot some Twiss:
tw.plot('x y')
tw.plot('betx bety', 'dx dy')
plt.show()