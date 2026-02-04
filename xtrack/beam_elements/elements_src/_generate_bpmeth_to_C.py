import os
import sys
from pathlib import Path

import sympy as sp
import bpmeth as bp

THIS_DIR = os.path.dirname(__file__)
ELEMENTS_SRC_DIR = THIS_DIR   # since the script is already in elements_src

# SplineBoris.ParamFormat defines the canonical param order; use repo root on PYTHONPATH.
from xtrack.beam_elements.elements import SplineBoris

# This script generates C code for evaluating magnetic field components
# based on symbolic expressions derived from the bpmeth formalism.
# The generated C code supports multipole expansions up to a specified order
# and polynomial dependencies on the longitudinal coordinate 's'.

multipole_order = 7 # 14-pole.

# To get the power in the format '*s*s*...*s' for a given power.
def s_power(power):
    str = ''
    for _ in range(power):
        str += '*s'
    return str

# Generates the symbols for the ks, kn and bs coefficients up to the given multipole and polynomial order.
def make_symbols(multipole_order=multipole_order, poly_order=4):
    ks_symbols = ()
    kn_symbols = ()
    bs_symbols = ()
    for i in range(multipole_order):
        for k in range(poly_order+1):
            ks_symbol = sp.symbols(f'ks_{i}_{k}')
            kn_symbol = sp.symbols(f'kn_{i}_{k}')
            ks_symbols += (ks_symbol,)
            kn_symbols += (kn_symbol,)

            if i ==0:
                bs_symbol = sp.symbols(f'bs_{k}')
                bs_symbols += (bs_symbol,)

    return ks_symbols, kn_symbols, bs_symbols


# Sets strings for the expressions of the ks, kn and bs coefficients up to the given multipole and polynomial order.
def set_exprs(multipole_order=multipole_order, poly_order=4):
    ks_symbols, kn_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)

    ks_exprs = ()
    kn_exprs = ()
    bs_expr = 0

    poly_order = 4
    for i in range(multipole_order):
        ks_expr = 0
        kn_expr = 0
        for k in range(poly_order+1):
            ks_symbol = ks_symbols[i * (poly_order + 1) + k]
            kn_symbol = kn_symbols[i * (poly_order + 1) + k]

            ks_expr += ks_symbol * sp.Pow(sp.symbols('s'), k)
            kn_expr += kn_symbol * sp.Pow(sp.symbols('s'), k)

            if i == 0:
                bs_symbol = bs_symbols[k]
                bs_expr += bs_symbol * sp.Pow(sp.symbols('s'), k)

        ks_exprs += (ks_expr,)
        kn_exprs += (kn_expr,)
    bs_expr  = bs_expr

    return ks_exprs, kn_exprs, bs_expr

ks_symbols, kn_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=4)

#ks_exprs, kn_exprs, bs_expr = set_exprs(ks_symbols, kn_symbols, bs_symbols)
#print(ks_exprs)
#print(kn_exprs)
#print(bs_expr)

# Sets up the generic field expressions for given curvature, multipole and polynomial order.
# Note: bpmeth uses the naming convention where a_ij corresponds to j-th polynomial coefficient of the (i-1)-th derivative w.r.t x of B_x,
# and b_ij corresponds to j-th polynomial coefficient of the (i-1)-th derivative w.r.t y of B_y.
# We use the usual beam-dynamics naming convention where ks_ij corresponds to j-th polynomial coefficient of the i-th derivative w.r.t x of B_x,
# and kn_ij corresponds to j-th polynomial coefficient of the i-th derivative w.r.t y of B_y.
def generic_field_exprs(curv, multipole_order=multipole_order, poly_order=4):
    ks_symbols, kn_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)
    s = sp.symbols('s')
    ks_exprs, kn_exprs, bs_exprs = set_exprs(multipole_order=multipole_order, poly_order=poly_order)

    generic_B = bp.GeneralVectorPotential(hs=curv, a=ks_exprs, b=kn_exprs, bs=bs_exprs, nphi=multipole_order)
    symbolic_Bx, symbolic_By, symbolic_Bs = generic_B.get_Bfield(lambdify=False)
    symbolic_Ax, symbolic_Ay, symbolic_As = generic_B.get_A()

    return symbolic_Bx, symbolic_By, symbolic_Bs, symbolic_Ax, symbolic_Ay, symbolic_As

#symbolic_Bx, symbolic_By, symbolic_Bs, symbolic_Ax, symbolic_Ay, symbolic_As = generic_field_exprs(curv='0', multipole_order=multipole_order, poly_order=4)

# print(f"Bx = {symbolic_Bx}")
# print(f"By = {symbolic_By}")
# print(f"Bs = {symbolic_Bs}")
# print(f"Ax = {symbolic_Ax}")
# print(f"Ay = {symbolic_Ay}")
# print(f"As = {symbolic_As}")

# Reduces the expressions using common sub-expression elimination.
def _get_reduced_expressions(exprs_list):
    from sympy import cse, symbols

    # Perform common sub-expression elimination
    cse_subs, reduced_exprs = cse(exprs_list)

    return cse_subs, reduced_exprs

# B_exprs = [symbolic_Bx, symbolic_By, symbolic_Bs]
# A_exprs = [symbolic_Ax, symbolic_Ay, symbolic_As]
#
# B_cse_subs, B_reduced_exprs = _get_reduced_expressions(B_exprs)
# A_cse_subs, A_reduced_exprs = _get_reduced_expressions(A_exprs)

# print("B field common sub-expressions:")
# for lhs, rhs in B_cse_subs:
#     print(f"{lhs} = {rhs}")
# print("B field reduced expressions:")
# for i, expr in enumerate(B_reduced_exprs):
#     print(f"Expr {i}: {expr}")

# Currently, curvature is set to '0' for straight sections, but can be set to a non-zero value for curved sections.
# However, the Boris Integrator does not support curved reference frames yet, so we leave the curvature zero here.
def start_to_finish(multipole_order=multipole_order, poly_order=4, field='B', curvature='0'):
    #ks_symbols, kn_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)
    param_names = SplineBoris.ParamFormat.get_param_names(multipole_order=multipole_order, poly_order=poly_order)
    ks_exprs, kn_exprs, bs_expr = set_exprs(multipole_order=multipole_order, poly_order=poly_order)
    symbolic_Bx, symbolic_By, symbolic_Bs, symbolic_Ax, symbolic_Ay, symbolic_As = generic_field_exprs(curv=curvature, multipole_order=multipole_order, poly_order=poly_order)
    if field == 'B':
        exprs = [symbolic_Bx, symbolic_By, symbolic_Bs]
    else:
        exprs = [symbolic_Ax, symbolic_Ay, symbolic_As]
    cse_subs, reduced_exprs = _get_reduced_expressions(exprs)

    return param_names, cse_subs, reduced_exprs


# Writes the C code for field evaluation to a file.
# Currently, max_order is set to correspond to a 14-pole (multipole_order=7).
# Multipole orders follow 1 for dipole, 2 for quadrupole, 3 for sextupole, etc., just like bpmeth.
# There used to be an array version of this function, but it was not necessary, so we only support scalar evaluation.
def write_to_C(max_order=multipole_order, poly_order=4, field='B', curvature='0'):
    from sympy.printing.c import C99CodePrinter

    class MulPowerPrinter(C99CodePrinter):
        def _print_Pow(self, expr):
            base, exp = expr.as_base_exp()

            # Only rewrite integer positive powers
            if exp.is_integer and exp.is_positive:
                n = int(exp)
                return "*".join([self._print(base)] * n)

            # Fallback to default handling
            return super()._print_Pow(expr)

    printer = MulPowerPrinter()

    if field == 'A':
        filename = os.path.join(ELEMENTS_SRC_DIR, '_spline_A_field_eval.h')
    else:
        filename = os.path.join(ELEMENTS_SRC_DIR, '_spline_B_field_eval.h')

    with open(filename, 'w') as f:
        f.write(f"#include <stddef.h>\n")
        f.write(f"#include <stdio.h>\n\n")

        # Generate header guard from filename only (not full path)
        # Remove path, extension, and convert to valid macro name
        basename = os.path.basename(filename)  # Get just the filename
        guard_name = basename[:-2].upper().replace('.', '_').replace('-', '_')  # Remove .h, convert to macro
        f.write(f"#ifndef {guard_name}_H\n")
        f.write(f"#define {guard_name}_H\n\n")

        f.write(f"// Auto-generated symbolic field expressions for {field}\n")
        f.write(
            f"void evaluate_{field}(const double x, const double y, const double s, const double *params, const int multipole_order, double *Bx_out, double *By_out, double *Bs_out){{\n\n")
        names = [f'{field}x_out', f'{field}y_out', f'{field}s_out']

        f.write(f"\tswitch (multipole_order) {{\n")

        for order in range(1, multipole_order+1):

            param_names, cse_subs, reduced_exprs = start_to_finish(multipole_order=order, poly_order=poly_order, field=field, curvature=curvature)

            f.write(f"\tcase {order}:{{\n")

            f.write("\t\t// Parameter List\n")
            for j, name in enumerate(param_names):
                f.write(f"\t\tconst double {name} = params[{j}];\n")
            f.write("\n")

            f.write("\t\t// Common sub-expressions\n")
            for lhs, rhs in cse_subs:
                f.write(f"\t\tconst double {lhs} = {printer.doprint(rhs)};\n")
            f.write("\n")
            f.write("\t\t// Reduced expressions\n")
            for order, expr in enumerate(reduced_exprs):
                f.write(f"\t\t*{names[order]} = {printer.doprint(expr)};\n")
            f.write(f"\t\treturn;\n\n")
            f.write(f"\t}}\n")
        f.write(f"\tdefault:{{\n")
        f.write("\t\tprintf(\"Error: Unsupported multipole order %d\\n\", multipole_order);\n")
        f.write(f"\t\tprintf(\"Supported orders are 0 to {multipole_order}\\n\");\n")
        f.write(f"\t\tprintf(\"Setting field values to zero.\\n\");\n")
        f.write("\t\t// Reduced expressions\n")
        for order in range(3):
            f.write(f"\t\t*{names[order]} = 0;\n")
        f.write(f"\t\treturn;\n")
        f.write(f"\t}}\n")
        f.write(f"\t}}\n")
        f.write(f"}}\n\n")
        f.write(f"#endif // XSUITE{filename[:-2].upper()}_H\n")


def write_to_python(max_order=multipole_order, poly_order=4, field='B', curvature='0'):
    """
    Generate a pure-Python implementation of the field evaluation routine.

    The generated file is named ``_spline_{field}_field_eval.py`` and contains a function
    ``evaluate_{field}(x, y, s, params, multipole_order)`` that mirrors the logic of the
    C implementation produced by ``write_to_C``.
    """
    from sympy.printing.pycode import PythonCodePrinter

    printer = PythonCodePrinter()

    if field == 'A':
        filename = os.path.join(ELEMENTS_SRC_DIR, '_spline_A_field_eval_python.py')
    else:
        filename = os.path.join(ELEMENTS_SRC_DIR, '_spline_B_field_eval_python.py')

    with open(filename, 'w') as f:
        f.write("# Auto-generated symbolic field expressions for {field}\n".format(field=field))
        f.write("# This file is generated by _generate_bpmeth_to_C.py\n")
        f.write("# Do not edit it directly.\n\n")

        f.write("import math\n\n")

        f.write(
            "def evaluate_{field}(x, y, s, params, multipole_order):\n".format(
                field=field
            )
        )
        f.write('    """\n')
        f.write("    Auto-generated symbolic field evaluation for {field}.\n".format(field=field))
        f.write("    Parameters are expected as a flat sequence in ``params``.\n")
        f.write("    The meaning and ordering of the parameters match the C implementation.\n")
        f.write('    """\n')

        # Generate per-order branches, analogous to the C switch-case
        for order in range(1, max_order + 1):
            param_names, cse_subs, reduced_exprs = start_to_finish(
                multipole_order=order,
                poly_order=poly_order,
                field=field,
                curvature=curvature,
            )

            if order == 1:
                f.write(f"    if multipole_order == {order}:\n")
            else:
                f.write(f"    elif multipole_order == {order}:\n")

            # Parameters
            f.write("        # Parameter list\n")
            for j, name in enumerate(param_names):
                f.write(f"        {name} = params[{j}]\n")
            f.write("\n")

            # Common sub-expressions
            if cse_subs:
                f.write("        # Common sub-expressions\n")
                for lhs, rhs in cse_subs:
                    f.write(f"        {lhs} = {printer.doprint(rhs)}\n")
                f.write("\n")

            # Reduced expressions
            names = [f'{field}x', f'{field}y', f'{field}s']
            f.write("        # Reduced expressions\n")
            for idx, expr in enumerate(reduced_exprs):
                f.write(f"        {names[idx]} = {printer.doprint(expr)}\n")
            f.write(f"        return {names[0]}, {names[1]}, {names[2]}\n\n")

        f.write(
            "    raise ValueError("
            f"f'Unsupported multipole order {{multipole_order}}; supported orders are 1 to {max_order}'"
            ")\n"
        )

# Currently, we generate the code for a 14-pole (multipole_order=7), but can be easily extended to higher orders.
# We also generate the code for a straight section (curvature='0'), but can be easily extended to curved sections.
# However, the Boris Integrator does not support curved reference frames yet, so we leave the curvature zero here.
multipole_order = 7
poly_order = 4
field = 'B'
curvature = '0'
write_to_C(max_order=multipole_order, poly_order=poly_order, field=field, curvature=curvature)
write_to_python(max_order=multipole_order, poly_order=poly_order, field=field, curvature=curvature)