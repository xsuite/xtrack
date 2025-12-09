import sympy as sp
import bpmeth as bp

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

# Generates the symbols for the a, b and bs coefficients up to the given multipole and polynomial order.
def make_symbols(multipole_order=multipole_order, poly_order=4):
    a_symbols = ()
    b_symbols = ()
    bs_symbols = ()
    for i in range(multipole_order):
        for k in range(poly_order+1):
            a_symbol = sp.symbols(f'a_{i + 1}_{k}')
            b_symbol = sp.symbols(f'b_{i + 1}_{k}')
            a_symbols += (a_symbol,)
            b_symbols += (b_symbol,)

            if i ==0:
                bs_symbol = sp.symbols(f'bs_{k}')
                bs_symbols += (bs_symbol,)

    return a_symbols, b_symbols, bs_symbols

def param_names_list(multipole_order=multipole_order, poly_order=4):
    param_names = []
    for i in range(multipole_order):
        for k in range(poly_order+1):
            a_name = f'a_{i + 1}_{k}'
            b_name = f'b_{i + 1}_{k}'
            param_names.append(a_name)
            param_names.append(b_name)

            if i ==0:
                bs_name = f'bs_{k}'
                param_names.append(bs_name)

    param_names.sort()

    return param_names

# Sets strings for the expressions of the a, b and bs coefficients up to the given multipole and polynomial order.
def set_exprs(multipole_order=multipole_order, poly_order=4):
    a_symbols, b_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)

    a_exprs = ()
    b_exprs = ()
    bs_expr = 0

    poly_order = 4
    for i in range(multipole_order):
        a_expr = 0
        b_expr = 0
        for k in range(poly_order+1):
            a_symbol = a_symbols[i * (poly_order + 1) + k]
            b_symbol = b_symbols[i * (poly_order + 1) + k]

            a_expr += a_symbol * sp.Pow(sp.symbols('s'), k)
            b_expr += b_symbol * sp.Pow(sp.symbols('s'), k)

            if i == 0:
                bs_symbol = bs_symbols[k]
                bs_expr += bs_symbol * sp.Pow(sp.symbols('s'), k)

        a_exprs += (a_expr,)
        b_exprs += (b_expr,)
    bs_expr  = bs_expr

    return a_exprs, b_exprs, bs_expr

a_symbols, b_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=4)

#a_exprs, b_exprs, bs_expr = set_exprs(a_symbols, b_symbols, bs_symbols)
#print(a_exprs)
#print(b_exprs)
#print(bs_expr)

# Sets up the generic field expressions for given curvature, multipole and polynomial order.
def generic_field_exprs(curv, multipole_order=multipole_order, poly_order=4):
    a_symbols, b_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)
    s = sp.symbols('s')
    a_exprs, b_exprs, bs_exprs = set_exprs(multipole_order=multipole_order, poly_order=poly_order)

    generic_B = bp.GeneralVectorPotential(hs=curv, a=a_exprs, b=b_exprs, bs=bs_exprs, nphi=multipole_order)
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

def start_to_finish(multipole_order=multipole_order, poly_order=4, field='B'):
    #a_symbols, b_symbols, bs_symbols = make_symbols(multipole_order=multipole_order, poly_order=poly_order)
    param_names = param_names_list(multipole_order=multipole_order, poly_order=poly_order)
    a_exprs, b_exprs, bs_expr = set_exprs(multipole_order=multipole_order, poly_order=poly_order)
    symbolic_Bx, symbolic_By, symbolic_Bs, symbolic_Ax, symbolic_Ay, symbolic_As = generic_field_exprs(curv='0', multipole_order=multipole_order, poly_order=poly_order)
    if field == 'B':
        exprs = [symbolic_Bx, symbolic_By, symbolic_Bs]
    else:
        exprs = [symbolic_Ax, symbolic_Ay, symbolic_As]
    cse_subs, reduced_exprs = _get_reduced_expressions(exprs)

    return param_names, cse_subs, reduced_exprs

# Writes the C code for field evaluation to a file.
# Currently, max_order is set to correspond to a 14-pole (multipole_order=7).
# Multipole orders follow 1 for dipole, 2 for quadrupole, 3 for sextupole, etc., just like bpmeth.
def write_to_c(max_order=multipole_order, poly_order=4, field='B'):
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
        filename = '_bpmeth_A_field_eval.h'
    else:
        filename = '_bpmeth_B_field_eval.h'

    with open(filename, 'w') as f:
        f.write(f"#include <stddef.h>\n")
        f.write(f"#include <stdio.h>\n\n")

        f.write(f"#ifndef XSUITE{filename[:-2].upper()}_H\n")
        f.write(f"#define XSUITE{filename[:-2].upper()}_H\n\n")

        f.write(f"// Auto-generated symbolic field expressions for {field}\n")
        f.write(
            f"void evaluate_{field}(const double x_array[], const double y_array[], const double s_array[], size_t n, const double *params_flat, const int multipole_order, double {field}x_out[], double {field}y_out[], double {field}s_out[]){{\n\n")
        names = [f'{field}x_out', f'{field}y_out', f'{field}s_out']

        f.write(f"\tswitch (multipole_order) {{\n")

        for order in range(1, multipole_order+1):

            param_names, cse_subs, reduced_exprs = start_to_finish(multipole_order=order, poly_order=poly_order, field=field)

            f.write(f"\tcase {order}:\n")

            f.write("\t\tfor (size_t ii = 0; ii < n; ++ii, ++x_array, ++y_array, ++s_array, ++Bx_out, ++By_out, ++Bs_out) {\n")
            f.write("\t\t\tconst double x = *x_array;\n")
            f.write("\t\t\tconst double y = *y_array;\n")
            f.write("\t\t\tconst double s = *s_array;\n\n")

            f.write("\t\t\t// Parameter List\n")
            f.write(f"\t\t\tconst double *p = params_flat + ii * {(2*order+1)*(poly_order+1)};\n") # Adjusted for number of parameters
            for j, name in enumerate(param_names):
                f.write(f"\t\t\tconst double {name} = p[{j}];\n")
            f.write("\n")

            f.write("\t\t\t// Common sub-expressions\n")
            for lhs, rhs in cse_subs:
                f.write(f"\t\t\tconst double {lhs} = {printer.doprint(rhs)};\n")
            f.write("\n")

            f.write("\t\t\t// Reduced expressions\n")
            for order, expr in enumerate(reduced_exprs):
                f.write(f"\t\t\t*{names[order]} = {printer.doprint(expr)};\n")

            f.write(f"\t\t}}\n")
            f.write(f"\t\treturn;\n\n")
        f.write(f"\tdefault:\n")
        f.write("\t\tprintf(\"Error: Unsupported multipole order %d\\n\", multipole_order);\n")
        f.write(f"\t\tprintf(\"Supported orders are 0 to {multipole_order}\\n\");\n")
        f.write(f"\t\tprintf(\"Setting field values to zero.\\n\");\n")
        f.write("\t\tfor (size_t ii = 0; ii < n; ++ii, ++Bx_out, ++By_out, ++Bs_out) {\n")
        f.write("\t\t\t// Reduced expressions\n")
        for order in range(3):
            f.write(f"\t\t\t*{names[order]} = 0;\n")
        f.write(f"\t\t}}\n")
        f.write(f"\t\treturn;\n")

        f.write(f"\t}}\n")
        f.write(f"}}\n\n")

        f.write(f"#endif // XSUITE{filename[:-2].upper()}_H\n")

multipole_order = 7
poly_order = 4
field = 'B'
write_to_c(max_order=multipole_order, poly_order=poly_order, field=field)