import os
import sys
from pathlib import Path

import sympy as sp

# Uses bpmeth 0.0.0
# Commit: 95e8f0e on main
import bpmeth as bp

THIS_DIR = os.path.dirname(__file__)
ELEMENTS_SRC_DIR = THIS_DIR   # since the script is already in elements_src

# SplineBoris module defines the canonical param order; use repo root on PYTHONPATH.
from xtrack.beam_elements.splineboris import _get_param_names

# This script generates C code for evaluating magnetic field components
# based on symbolic expressions derived from the bpmeth formalism.
# The generated C code supports multipole expansions up to a specified order
# and polynomial dependencies on the longitudinal coordinate 's'.

multipole_order = 7  # 14-pole.

# Default poly_order=4 below implies (poly_order + 1) coefficients per block; must match
# SplineBoris._POLY_ORDER / _NUM_COEFFS and XTRACK_SPLINE_B_N_POLY_COEFFS in splineboris.h.

# To get the power in the format '*s*s*...*s' for a given power.
def s_power(power):
    str = ''
    for _ in range(power):
        str += '*s'
    return str

# Sympy symbols for SplineBoris wire format: bs_k, by_i_k, bx_i_k (names match SplineBoris._get_param_names).
def make_symbols(multipole_order=multipole_order, poly_order=4):
    bx_symbols = ()
    by_symbols = ()
    bs_symbols = ()
    for i in range(multipole_order):
        for k in range(poly_order + 1):
            bx_symbols += (sp.symbols(f'bx_{i}_{k}'),)
            by_symbols += (sp.symbols(f'by_{i}_{k}'),)
            if i == 0:
                bs_symbols += (sp.symbols(f'bs_{k}'),)
    return bx_symbols, by_symbols, bs_symbols


def set_exprs(multipole_order=multipole_order, poly_order=4):
    bx_symbols, by_symbols, bs_symbols = make_symbols(
        multipole_order=multipole_order, poly_order=poly_order
    )

    bx_exprs = ()
    by_exprs = ()
    bs_expr = 0

    for i in range(multipole_order):
        bx_expr = 0
        by_expr = 0
        for k in range(poly_order + 1):
            bx_sym = bx_symbols[i * (poly_order + 1) + k]
            by_sym = by_symbols[i * (poly_order + 1) + k]

            bx_expr += bx_sym * sp.Pow(sp.symbols('s'), k)
            by_expr += by_sym * sp.Pow(sp.symbols('s'), k)

            if i == 0:
                bs_expr += bs_symbols[k] * sp.Pow(sp.symbols('s'), k)

        bx_exprs += (bx_expr,)
        by_exprs += (by_expr,)

    return bx_exprs, by_exprs, bs_expr


# Sets up the generic field expressions for given curvature, multipole and polynomial order.
# bpmeth GeneralVectorPotential: a = B_x multipole coeffs (bx), b = B_y (by), bs = longitudinal field (bs).
def generic_field_exprs(curv, multipole_order=multipole_order, poly_order=4):
    bx_exprs, by_exprs, bs_exprs = set_exprs(
        multipole_order=multipole_order, poly_order=poly_order
    )

    # nphi must be > multipole_order to include y-dependent terms from dBs/ds
    # The recursion phi_{n+2} = f(phi_n) means we need at least nphi = multipole_order + 2
    # to capture the solenoid focusing terms (-y*(d(bs)/ds + bx_1)) in By
    generic_B = bp.GeneralVectorPotential(
        hs=curv, a=bx_exprs, b=by_exprs, bs=bs_exprs, nphi=multipole_order + 2
    )
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
    param_names = _get_param_names(multipole_order=multipole_order)
    symbolic_Bx, symbolic_By, symbolic_Bs, symbolic_Ax, symbolic_Ay, symbolic_As = generic_field_exprs(
        curv=curvature, multipole_order=multipole_order, poly_order=poly_order
    )
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
        filename = os.path.join(ELEMENTS_SRC_DIR, 'spline_A_field_eval.h')
    else:
        filename = os.path.join(ELEMENTS_SRC_DIR, 'spline_B_field_eval.h')

    with open(filename, 'w') as f:
        # Generate header guard from filename only (not full path)
        basename = os.path.basename(filename)
        guard_name = basename[:-2].upper().replace('.', '_').replace('-', '_')

        if field == 'A':
            # Original params-based interface for the vector potential.
            f.write("#include <stddef.h>\n")
            f.write("#include <stdio.h>\n\n")
            f.write(f"#ifndef {guard_name}_H\n")
            f.write(f"#define {guard_name}_H\n\n")

            f.write("// Auto-generated symbolic field expressions for A\n")
            f.write("// NOTE: 's' is the local coordinate within the element (s_local = s - s_start),\n")
            f.write("//        not the global s-coordinate along the beamline.\n")
            f.write("GPUFUN\n")
            f.write(
                "void evaluate_A(const double x, const double y, const double s, "
                "const double *params, const int multipole_order, "
                "double *Ax_out, double *Ay_out, double *As_out){\n\n"
            )
            names = [f'{field}x_out', f'{field}y_out', f'{field}s_out']

            f.write("\tswitch (multipole_order) {\n")
            for order in range(1, max_order + 1):
                param_names, cse_subs, reduced_exprs = start_to_finish(
                    multipole_order=order,
                    poly_order=poly_order,
                    field=field,
                    curvature=curvature,
                )

                f.write(f"\tcase {order}: {{\n")
                f.write("\t\t// Parameter List\n")
                for j, name in enumerate(param_names):
                    f.write(f"\t\tconst double {name} = params[{j}];\n")
                f.write("\n")

                f.write("\t\t// Common sub-expressions\n")
                for lhs, rhs in cse_subs:
                    f.write(f"\t\tconst double {lhs} = {printer.doprint(rhs)};\n")
                f.write("\n")

                f.write("\t\t// Reduced expressions\n")
                for idx, expr in enumerate(reduced_exprs):
                    f.write(f"\t\t*{names[idx]} = {printer.doprint(expr)};\n")
                f.write("\t\treturn;\n\n")
                f.write("\t}\n")

            f.write("\tdefault: {\n")
            f.write("\t\tprintf(\"Error: Unsupported multipole order %d\\n\", multipole_order);\n")
            f.write(f"\t\tprintf(\"Supported orders are 1 to {max_order}\\n\");\n")
            f.write("\t\tprintf(\"Setting field values to zero.\\n\");\n")
            f.write("\t\t// Reduced expressions\n")
            for idx in range(3):
                f.write(f"\t\t*{names[idx]} = 0;\n")
            f.write("\t\treturn;\n")
            f.write("\t}\n")
            f.write("\t}\n")
            f.write("}\n\n")
            f.write(f"#endif // {guard_name}_H\n")

        else:
            # Hermite-based interface for the magnetic field B.
            f.write("#include <stddef.h>\n")
            f.write("#include <stdio.h>\n")
            f.write("#include <string.h>\n")
            f.write("#define MAX_DEGREE 4\n\n")
            f.write(f"#ifndef {guard_name}_H\n")
            f.write(f"#define {guard_name}_H\n\n")

            f.write("// Auto-generated symbolic field expressions for B\n")
            f.write("// NOTE:\n")
            f.write("//   - 's' is the local coordinate within the element: s_local ∈ [0, L].\n")
            f.write("//   - Hermite coefficients are defined on s_local ∈ [0, L] and are converted\n")
            f.write("//     internally to polynomials in s_local via hermite_to_polynomial(0, L, ...).\n")
            f.write("//\n")
            f.write("// Hermite input layout\n")
            f.write("// --------------------\n")
            f.write("//   - bs        : one scalar Hermite polynomial (5 coeffs) for bs(s_local)\n")
            f.write("//   - by[i]     : Hermite coeffs (5) for polynomial group by_i_*(s_local)\n")
            f.write("//   - bx[i]     : Hermite coeffs (5) for polynomial group bx_i_*(s_local)\n")
            f.write("//\n")
            f.write("// For multipole_order = n (1 ≤ n ≤ 7):\n")
            f.write("//   - bs:       1 polynomial      → bs_0..bs_4 from bs\n")
            f.write("//   - by:       n polynomials     → by_i_0..by_i_4 from by[i], i=0..n-1\n")
            f.write("//   - bx:       n polynomials     → bx_i_0..bx_i_4 from bx[i], i=0..n-1\n")
            f.write("//\n")
            f.write("// The symbolic expressions below are unchanged; only the way the bs_*, by_*_*,\n")
            f.write("// and bx_*_* scalars are populated has been refactored to use Hermite data.\n")

            # Poly helpers are emitted at file scope because C does not support nested functions.
            f.write("typedef struct {\n")
            f.write("\tdouble coeffs[MAX_DEGREE + 1]; /* coeffs[i] = coefficient of x^i */\n")
            f.write("\tint degree;\n")
            f.write("} Poly;\n\n")

            f.write("static inline Poly poly_scale(Poly p, double s) {\n")
            f.write("\tfor (int i = 0; i <= p.degree; i++) p.coeffs[i] *= s;\n")
            f.write("\treturn p;\n")
            f.write("}\n\n")

            f.write("static inline Poly poly_add(Poly a, Poly b) {\n")
            f.write("\tPoly result = {0};\n")
            f.write("\tresult.degree = a.degree > b.degree ? a.degree : b.degree;\n")
            f.write("\tfor (int i = 0; i <= a.degree; i++) result.coeffs[i] += a.coeffs[i];\n")
            f.write("\tfor (int i = 0; i <= b.degree; i++) result.coeffs[i] += b.coeffs[i];\n")
            f.write("\treturn result;\n")
            f.write("}\n\n")

            f.write("static inline Poly poly_mul(Poly a, Poly b) {\n")
            f.write("\tPoly result = {0};\n")
            f.write("\tint deg = a.degree + b.degree;\n")
            f.write("\tif (deg > MAX_DEGREE)\n")
            f.write("\t\tdeg = MAX_DEGREE;\n")
            f.write("\tresult.degree = deg;\n")
            f.write("\tfor (int i = 0; i <= a.degree; i++) {\n")
            f.write("\t\tfor (int j = 0; j <= b.degree; j++) {\n")
            f.write("\t\t\tint k = i + j;\n")
            f.write("\t\t\tif (k <= MAX_DEGREE)\n")
            f.write("\t\t\t\tresult.coeffs[k] += a.coeffs[i] * b.coeffs[j];\n")
            f.write("\t\t}\n")
            f.write("\t}\n")
            f.write("\treturn result;\n")
            f.write("}\n\n")

            f.write("/* Compose f(g(x)) via Horner's method:\n")
            f.write("   result = f[n] * g^n + ... + f[0]\n")
            f.write("\t\t  = f[0] + g*(f[1] + g*(f[2] + ... + g*f[n]))  */\n")
            f.write("static inline Poly poly_compose(Poly f, Poly g) {\n")
            f.write("\tPoly result = {0};\n")
            f.write("\tresult.coeffs[0] = f.coeffs[f.degree]; /* start with leading coeff */\n")
            f.write("\tresult.degree = 0;\n")
            f.write("\tfor (int i = f.degree - 1; i >= 0; i--) {\n")
            f.write("\t\tresult = poly_mul(result, g);       /* result = result * g      */\n")
            f.write("\t\tif (result.degree < MAX_DEGREE) {\n")
            f.write("\t\t\tresult.degree++;\n")
            f.write("\t\t}\n")
            f.write("\t\tresult.coeffs[0] += f.coeffs[i];   /* result = result * g + f[i] */\n")
            f.write("\t}\n")
            f.write("\treturn result;\n")
            f.write("}\n\n")

            f.write("static inline Poly hermite_to_polynomial(double s_start, double s_end, const double coeffs[5]) {\n")
            f.write("\tdouble c1 = coeffs[0], c2 = coeffs[1], c3 = coeffs[2];\n")
            f.write("\tdouble c4 = coeffs[3], c5 = coeffs[4];\n")
            f.write("\tdouble L = s_end - s_start;\n\n")
            f.write("\t/* t(s_local) = s_local / L */\n")
            f.write("\tPoly t = { .coeffs = {0.0, 1.0/L}, .degree = 1 };\n\n")
            f.write("\t/* Hermite basis polynomials in t on [0,1] */\n")
            f.write("\tPoly b1 = { .coeffs = { 1,  0,  -18,   32,  -15}, .degree = 4 };\n")
            f.write("\tPoly b2 = { .coeffs = { 0,  1, -4.5,    6, -2.5}, .degree = 4 };\n")
            f.write("\tPoly b3 = { .coeffs = { 0,  0,  -12,   28,  -15}, .degree = 4 };\n")
            f.write("\tPoly b4 = { .coeffs = { 0,  0,  1.5,   -4,  2.5}, .degree = 4 };\n")
            f.write("\tPoly b5 = { .coeffs = { 0,  0,   30,  -60,   30}, .degree = 4 };\n\n")
            f.write("\t/* poly_t = c1*b1 + L*c2*b2 + c3*b3 + L*c4*b4 + c5*b5 */\n")
            f.write("\tPoly poly_t = {0};\n")
            f.write("\tpoly_t = poly_add(poly_t, poly_scale(b1, c1));\n")
            f.write("\tpoly_t = poly_add(poly_t, poly_scale(b2, L * c2));\n")
            f.write("\tpoly_t = poly_add(poly_t, poly_scale(b3, c3));\n")
            f.write("\tpoly_t = poly_add(poly_t, poly_scale(b4, L * c4));\n")
            f.write("\tpoly_t = poly_add(poly_t, poly_scale(b5, c5));\n\n")
            f.write("\t/* poly_s(s_local) = poly_t(t(s_local)) */\n")
            f.write("\treturn poly_compose(poly_t, t);\n")
            f.write("}\n\n")

            f.write("GPUFUN\n")
            f.write(
                "void evaluate_B(const double x, const double y, const double s,\n"
                "                const double *bs,\n"
                "                const double *const *by,\n"
                "                const double *const *bx,\n"
                "                const double L,\n"
                "                const int multipole_order,\n"
                "                double *Bx_out, double *By_out, double *Bs_out){\n\n"
            )

            names = ['Bx_out', 'By_out', 'Bs_out']

            f.write("\tswitch (multipole_order) {\n")

            for order in range(1, max_order + 1):
                param_names, cse_subs, reduced_exprs = start_to_finish(
                    multipole_order=order,
                    poly_order=poly_order,
                    field=field,
                    curvature=curvature,
                )

                f.write(f"\tcase {order}: {{\n")
                f.write(f"\t\t// Hermite → polynomial coefficients (order {order})\n")
                f.write("\t\tconst Poly bs_poly = hermite_to_polynomial(0.0, L, bs);\n")
                f.write("\t\tconst double bs_0   = bs_poly.coeffs[0];\n")
                f.write("\t\tconst double bs_1   = bs_poly.coeffs[1];\n")
                f.write("\t\tconst double bs_2   = bs_poly.coeffs[2];\n")
                f.write("\t\tconst double bs_3   = bs_poly.coeffs[3];\n")
                f.write("\t\tconst double bs_4   = bs_poly.coeffs[4];\n\n")

                # by groups
                for i in range(order):
                    f.write(
                        f"\t\tconst Poly by{i}_poly = hermite_to_polynomial(0.0, L, by[{i}]);\n"
                    )
                    for k in range(poly_order + 1):
                        f.write(
                            f"\t\tconst double by_{i}_{k} = by{i}_poly.coeffs[{k}];\n"
                        )
                    f.write("\n")

                # bx groups
                for i in range(order):
                    f.write(
                        f"\t\tconst Poly bx{i}_poly = hermite_to_polynomial(0.0, L, bx[{i}]);\n"
                    )
                    for k in range(poly_order + 1):
                        f.write(
                            f"\t\tconst double bx_{i}_{k} = bx{i}_poly.coeffs[{k}];\n"
                        )
                    f.write("\n")

                f.write("\t\t// Common sub-expressions\n")
                for lhs, rhs in cse_subs:
                    f.write(f"\t\tconst double {lhs} = {printer.doprint(rhs)};\n")
                f.write("\n")

                f.write("\t\t// Reduced expressions\n")
                for idx, expr in enumerate(reduced_exprs):
                    f.write(f"\t\t*{names[idx]} = {printer.doprint(expr)};\n")
                f.write("\t\treturn;\n\n")
                f.write("\t}\n")

            f.write("\tdefault: {\n")
            f.write("\t\tprintf(\"Error: Unsupported multipole order %d\\n\", multipole_order);\n")
            f.write(f"\t\tprintf(\"Supported orders are 1 to {max_order}\\n\");\n")
            f.write("\t\tprintf(\"Setting field values to zero.\\n\");\n")
            f.write("\t\t// Reduced expressions\n")
            for idx in range(3):
                f.write(f"\t\t*{names[idx]} = 0;\n")
            f.write("\t\treturn;\n")
            f.write("\t}\n")
            f.write("\t}\n")
            f.write("}\n\n")
            f.write(f"#endif // {guard_name}_H\n")


def write_to_python(max_order=multipole_order, poly_order=4, field='B', curvature='0'):
    """
    Generate a pure-Python implementation of the field evaluation routine.

    The generated file is named ``spline_{field}_field_eval_python.py`` and contains a function
    ``evaluate_{field}(x, y, s, params, multipole_order)`` that mirrors the logic of the
    C implementation produced by ``write_to_C``.
    """
    from sympy.printing.pycode import PythonCodePrinter

    printer = PythonCodePrinter()

    if field == 'A':
        filename = os.path.join(ELEMENTS_SRC_DIR, 'spline_A_field_eval_python.py')
    else:
        filename = os.path.join(ELEMENTS_SRC_DIR, 'spline_B_field_eval_python.py')

    with open(filename, 'w') as f:
        f.write("# Auto-generated symbolic field expressions for {field}\n".format(field=field))
        f.write("# This file is generated by _generate_bpmeth_to_C.py\n")
        f.write("# Do not edit it directly.\n")
        f.write("# NOTE: 's' is the local coordinate within the element (s_local = s - s_start),\n")
        f.write("#        not the global s-coordinate along the beamline.\n\n")

        if field == 'A':
            # Keep params-based Python interface for the vector potential.
            f.write("import math\n\n")
            f.write(
                "def evaluate_A(x, y, s, params, multipole_order):\n"
            )
            f.write('    """\n')
            f.write("    Auto-generated symbolic field evaluation for A.\n")
            f.write("    Parameters are expected as a flat sequence in ``params``.\n")
            f.write("    The meaning and ordering of the parameters match the C implementation.\n")
            f.write('    """\n')

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

                f.write("        # Parameter list\n")
                for j, name in enumerate(param_names):
                    f.write(f"        {name} = params[{j}]\n")
                f.write("\n")

                if cse_subs:
                    f.write("        # Common sub-expressions\n")
                    for lhs, rhs in cse_subs:
                        f.write(f"        {lhs} = {printer.doprint(rhs)}\n")
                    f.write("\n")

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

        else:
            # Hermite-based Python interface for the magnetic field B.
            f.write("import math\n")
            f.write("import numpy as np\n\n")

            # Emit hermite_to_polynomial helper (Python) matching the runtime implementation.
            f.write("def hermite_to_polynomial(s_start, s_end, coeffs):\n")
            f.write("    \"\"\"Build a fourth-order polynomial over [s_start, s_end] from Hermite data.\n\n")
            f.write("    Mirrors ``SplineBoris.hermite_to_polynomial`` in ``elements.py`` and the\n")
            f.write("    Hermite → polynomial mapping used in the C backend.\n")
            f.write("    \"\"\"\n")
            f.write("    if len(coeffs) != 5:\n")
            f.write("        raise ValueError('coeffs must be a 5-element array')\n\n")
            f.write("    c1, c2, c3, c4, c5 = coeffs\n")
            f.write("    L = s_end - s_start\n")
            f.write("    t = np.polynomial.Polynomial([0, 1 / L])  # t = s_local / L\n\n")
            f.write("    # Basis functions on [0, 1]\n")
            f.write("    b1 = np.polynomial.Polynomial([1, 0, -18, 32, -15])\n")
            f.write("    b2 = np.polynomial.Polynomial([0, 1, -4.5, 6, -2.5])\n")
            f.write("    b3 = np.polynomial.Polynomial([0, 0, -12, 28, -15])\n")
            f.write("    b4 = np.polynomial.Polynomial([0, 0, 1.5, -4, 2.5])\n")
            f.write("    b5 = np.polynomial.Polynomial([0, 0, 30, -60, 30])\n\n")
            f.write("    poly_t = (c1 * b1 + L * c2 * b2 + c3 * b3 + L * c4 * b4 + c5 * b5)\n")
            f.write("    poly_s = poly_t(t)\n\n")
            f.write("    # Ensure we always have a length-5 coefficient array (degree 4 polynomial),\n")
            f.write("    # even for the degenerate all-zero Hermite input where numpy compresses to degree 0.\n")
            f.write("    if poly_s.coef.size < 5:\n")
            f.write("        coeffs_padded = np.zeros(5, dtype=float)\n")
            f.write("        coeffs_padded[: poly_s.coef.size] = poly_s.coef\n")
            f.write("        poly_s = np.polynomial.Polynomial(coeffs_padded)\n\n")
            f.write("    return poly_s\n\n")

            f.write(
                "def evaluate_B(x, y, s, bs, by, bx, L, multipole_order):\n"
            )
            f.write('    """\n')
            f.write("    Auto-generated symbolic field evaluation for B.\n")
            f.write("    Hermite coefficients are provided as:\n")
            f.write("      - bs : array-like length 5\n")
            f.write("      - by : sequence of length n (order), each a length-5 array\n")
            f.write("      - bx : sequence of length n (order), each a length-5 array\n")
            f.write('    """\n')

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

                # Hermite → polynomial mapping
                f.write(f"        # Hermite → polynomial coefficients (order {order})\n")
                f.write("        bs_poly = hermite_to_polynomial(0.0, L, bs)\n")
                f.write("        bs_0, bs_1, bs_2, bs_3, bs_4 = bs_poly.coef[0:5]\n\n")

                # by groups
                for i in range(order):
                    f.write(
                        f"        by{i}_poly = hermite_to_polynomial(0.0, L, by[{i}])\n"
                    )
                    f.write(
                        f"        by_{i}_0, by_{i}_1, by_{i}_2, by_{i}_3, by_{i}_4 = "
                        f"by{i}_poly.coef[0:5]\n\n"
                    )

                # bx groups
                for i in range(order):
                    f.write(
                        f"        bx{i}_poly = hermite_to_polynomial(0.0, L, bx[{i}])\n"
                    )
                    f.write(
                        f"        bx_{i}_0, bx_{i}_1, bx_{i}_2, bx_{i}_3, bx_{i}_4 = "
                        f"bx{i}_poly.coef[0:5]\n\n"
                    )

                # Common sub-expressions
                if cse_subs:
                    f.write("        # Common sub-expressions\n")
                    for lhs, rhs in cse_subs:
                        f.write(f"        {lhs} = {printer.doprint(rhs)}\n")
                    f.write("\n")

                # Reduced expressions
                names = ['Bx', 'By', 'Bs']
                f.write("        # Reduced expressions\n")
                for idx, expr in enumerate(reduced_exprs):
                    f.write(f"        {names[idx]} = {printer.doprint(expr)}\n")
                f.write("        return Bx, By, Bs\n\n")

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