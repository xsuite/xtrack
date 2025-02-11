"""

Structure of the code:

MadLoader takes a sequence and several options
MadLooder.make_line(buffer=None) returns a line with elements installed in one buffer
MadLooder.iter_elements() iterates over the elements of the sequence,
                          yielding a MadElement and applies some simplifications

Developers:

- MadElem encapsulate a mad element, it behaves like an elemenent from the expanded sequence
but returns as attributes a value, or an expression if present.

- Use `if MadElem(mad).l: to check for no zero value and NOT `if MadElem(mad).l!=0:` because if l is an expression it will create the expression l!=0 and return True


- ElementBuilder, is a class that builds an xtrack element from a definition. If a values is expression, the value calculated from the expression, the expression if present is attached to the line.


Developer should write
Loader.convert_<name>(mad_elem)->List[ElementBuilder] to convert new element in a list

or in alternative

Loader.add_<name>(mad_elem,line,buffer) to add a new element to line

if the want to control how the xobject is created
"""
from typing import List, Union

import numpy as np

import xobjects
import xtrack
from .general import _print
from .progress_indicator import progress

# Generic functions

clight = 299792458

DEFAULT_BEND_N_MULT_KICKS = 5
DEFAULT_FIELD_ERR_NUM_KICKS = 1


def iterable(obj):
    return hasattr(obj, "__iter__")


def set_if_not_none(dct, key, val):
    if val is not None:
        dct[key] = val


def rad2deg(rad):
    return rad * 180 / np.pi


def get_value(x):
    if is_expr(x):
        return x._get_value()
    elif isinstance(x, list) or isinstance(x, tuple):
        return [get_value(xx) for xx in x]
    elif isinstance(x, np.ndarray):
        arr = np.zeros_like(x, dtype=float)
        for ii in np.ndindex(*x.shape):
            arr[ii] = get_value(x[ii])
    elif isinstance(x, dict):
        return {k: get_value(v) for k, v in x.items()}
    else:
        return x


def set_expr(target, key, xx):
    """
    Assumes target is either a struct supporting attr assignment or an array supporint item assignment.

    """
    if isinstance(xx, list):
        out = getattr(target, key)
        for ii, ex in enumerate(xx):
            set_expr(out, ii, ex)
    elif isinstance(xx, np.ndarray):
        out = getattr(target, key)
        for ii in np.ndindex(*xx.shape):
            set_expr(out, ii, xx[ii])
    elif isinstance(xx, dict):
        for kk, ex in xx.items():
            set_expr(target[key], kk, ex)
    elif xx is not None:
        if isinstance(key, int) or isinstance(key, tuple):
            target[key] = xx
        else:
            setattr(target, key, xx)  # issue if target is not a structure


# needed because cannot used += with numpy arrays of expressions
def add_lists(a, b, length):
    out = []
    for ii in range(length):
        if ii < len(a) and ii < len(b):
            c = a[ii] + b[ii]
        elif ii < len(a):
            c = a[ii]
        elif ii < len(b):
            c = b[ii]
        else:
            c = 0
        out.append(c)
    return out


def non_zero_len(lst):
    for ii, x in enumerate(lst[::-1]):
        if x:  # could be expression
            return len(lst) - ii
    return 0


def trim_trailing_zeros(lst):
    for ii in range(len(lst) - 1, 0, -1):
        if lst[ii] != 0:
            return lst[: ii + 1]
    return []


def is_expr(x):
    return hasattr(x, "_get_value")


def nonzero_or_expr(x):
    if is_expr(x):
        return True
    else:
        return x != 0


def value_if_expr(x):
    if is_expr(x):
        return x._value
    else:
        return x


def eval_list(par, madeval):
    if madeval is None:
        return par.value
    else:
        return [
            madeval(expr) if expr else value for value, expr in zip(par.value, par.expr)
        ]


def generate_repeated_name(line, name):
    if name in line.element_dict:
        ii = 0
        while f"{name}:{ii}" in line.element_dict:
            ii += 1
        return f"{name}:{ii}"
    else:
        return name


class FieldErrors:
    def __init__(self, field_errors):
        self.dkn = np.array(field_errors.dkn)
        self.dks = np.array(field_errors.dks)


class PhaseErrors:
    def __init__(self, phase_errors):
        self.dpn = np.array(phase_errors.dpn)
        self.dps = np.array(phase_errors.dps)


class MadElem:
    def __init__(self, name, elem, sequence, madeval=None, name_prefix=None):
        if name_prefix is None:
            self.name = name
        else:
            self.name = name_prefix + name
        self.elem = elem
        self.sequence = sequence
        self.madeval = madeval
        ### needed for merge multipoles
        if hasattr(elem, "field_errors") and elem.field_errors is not None:
            self.field_errors = FieldErrors(elem.field_errors)
        else:
            self.field_errors = None
        if elem.base_type.name != 'translation' and (
                elem.dphi or elem.dtheta or elem.dpsi
                or elem.dx or elem.dy or elem.ds):
            raise NotImplementedError

    # @property
    # def field_errors(self):
    #    elem=self.elem
    #    if hasattr(elem, "field_errors") and elem.field_errors is not None:
    #        return FieldErrors(elem.field_errors)

    def get_type_hierarchy(self, cpymad_elem=None):
        if cpymad_elem is None:
            cpymad_elem = self.elem

        if cpymad_elem.name == cpymad_elem.parent.name:
            return [cpymad_elem.name]

        parent_types = self.get_type_hierarchy(cpymad_elem.parent)
        return [cpymad_elem.name] + parent_types

    @property
    def phase_errors(self):
        elem = self.elem
        if hasattr(elem, "phase_errors") and elem.phase_errors is not None:
            return PhaseErrors(elem.phase_errors)

    @property
    def align_errors(self):
        elem = self.elem
        if hasattr(elem, "align_errors") and elem.align_errors is not None:
            return elem.align_errors

    def __repr__(self):
        return f"<{self.name}: {self.type}>"

    @property
    def type(self):
        return self.elem.base_type.name

    @property
    def slot_id(self):
        return self.elem.slot_id

    def __getattr__(self, k):
        par = self.elem.cmdpar.get(k)
        if par is None:
            raise AttributeError(
                f"Element `{self.name}: {self.type}` has no attribute `{k}`"
            )
        if isinstance(par.value, list):
            # return ParList(eval_list(par, self.madeval))
            return eval_list(par, self.madeval)
        elif isinstance(par.value, str):
            return par.value  # no need to make a Par for strings
        elif self.madeval is not None and par.expr is not None:
            return self.madeval(par.expr)
        else:
            return par.value

    def get(self, key, default=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return default

    def has_aperture(self):
        el = self.elem
        has_aper = hasattr(el, "aperture") and (
            el.aperture[0] != 0.0 or len(el.aperture) > 1
        )
        has_aper = has_aper or (hasattr(el, "aper_vx") and len(el.aper_vx) > 2)
        return has_aper

    def is_empty_marker(self):
        return self.type == "marker" and not self.has_aperture()

    def same_aperture(self, other):
        return (
            self.aperture == other.aperture
            and self.aper_offset == other.aper_offset
            and self.aper_tilt == other.aper_tilt
            and self.aper_vx == other.aper_vx
            and self.aper_vy == other.aper_vy
            and self.apertype == other.apertype
        )

    def merge_multipole(self, other):
        if (
            self.same_aperture(other)
            and self.align_errors == other.align_errors
            and self.tilt == other.tilt
            and self.angle == other.angle
        ):
            self.knl += other.knl
            self.ksl += other.ksl
            if self.field_errors is not None and other.field_errors is not None:
                for ii in range(len(self.field_errors.dkn)):
                    self.field_errors.dkn[ii] += other.field_errors.dkn[ii]
                    self.field_errors.dks[ii] += other.field_errors.dks[ii]
            self.name = self.name + "_" + other.name
            return True
        else:
            return False


class ElementBuilder:
    """
    init  is a dictionary of element data passed to the __init__ function of the element class
    attrs is a dictionary of extra data to be added to the element data after creation
    """

    def __init__(self, name, type, **attrs):
        self.name = name
        self.type = type
        self.attrs = {} if attrs is None else attrs

    def __repr__(self):
        return "Element(%s, %s, %s)" % (self.name, self.type, self.attrs)

    def __setattr__(self, k, v):
        if hasattr(self, "attrs") and k not in ('name', 'type', 'attrs'):
            self.attrs[k] = v
        else:
            super().__setattr__(k, v)

    def add_to_line(self, line, buffer):
        if self.type is xtrack.Drift:
            self.attrs.pop("rot_s_rad", None)
            self.attrs.pop("shift_x", None)
            self.attrs.pop("shift_y", None)
        name_associated_aperture = self.attrs.pop("name_associated_aperture", None)
        xtel = self.type(**self.attrs, _buffer=buffer)
        if name_associated_aperture:
            xtel.name_associated_aperture = name_associated_aperture
        name = generate_repeated_name(line, self.name)
        line.append_element(xtel, name)


class ElementBuilderWithExpr(ElementBuilder):
    def add_to_line(self, line, buffer):

        if self.type is xtrack.Drift:
            self.attrs.pop("rot_s_rad", None)
            self.attrs.pop("shift_x", None)
            self.attrs.pop("shift_y", None)

        attr_values = {k: get_value(v) for k, v in self.attrs.items()}
        name_associated_aperture = attr_values.pop("name_associated_aperture", None)
        xtel = self.type(**attr_values, _buffer=buffer)
        name = generate_repeated_name(line, self.name)
        if name_associated_aperture:
            xtel.name_associated_aperture = name_associated_aperture
        line.append_element(xtel, name)
        elref = line.element_refs[name]
        for k, p in self.attrs.items():
            set_expr(elref, k, p)
        return xtel


class Aperture:
    def __init__(self, mad_el, enable_errors, loader):
        self.mad_el = mad_el
        self.aper_tilt = mad_el.aper_tilt
        self.aper_offset = mad_el.aper_offset
        self.name = self.mad_el.name
        self.dx = self.aper_offset[0]
        if len(self.aper_offset) > 1:
            self.dy = self.aper_offset[1]
        else:
            self.dy = 0
        if enable_errors and self.mad_el.align_errors is not None:
            self.dx += mad_el.align_errors.arex
            self.dy += mad_el.align_errors.arey
        self.apertype = self.mad_el.apertype
        self.loader = loader
        self.classes = loader.classes
        self.Builder = loader.Builder

    def aperture(self):
        if len(self.mad_el.aper_vx) > 2:
            builder = self.Builder(
                    self.name + "_aper",
                    self.classes.LimitPolygon,
                    x_vertices=self.mad_el.aper_vx,
                    y_vertices=self.mad_el.aper_vy,
                )
            if self.dx or self.dy or self.aper_tilt:
                builder.shift_x = self.dx
                builder.shift_y = self.dy
                builder.rot_s_rad = self.aper_tilt
            return [builder]
        else:
            conveter = getattr(self.loader, "convert_" + self.apertype, None)
            if conveter is None:
                raise ValueError(f"Aperture type `{self.apertype}` not supported")
            out = conveter(self.mad_el)
            assert len(out) == 1
            if self.dx or self.dy or self.aper_tilt:
                out[0].shift_x = self.dx
                out[0].shift_y = self.dy
                out[0].rot_s_rad = self.aper_tilt
            return out


class Alignment:
    def __init__(self, mad_el, enable_errors, classes, Builder, bv, custom_tilt=None):
        self.mad_el = mad_el
        self.bv = bv
        self.tilt = bv * mad_el.get("tilt", 0)  # some elements do not have tilt
        if custom_tilt is not None:
            self.tilt += custom_tilt
        self.name = mad_el.name
        self.dx = 0
        self.dy = 0
        if (
            enable_errors
            and hasattr(mad_el, "align_errors")
            and mad_el.align_errors is not None
        ):
            if bv != 1:
                raise NotImplementedError("Alignment errors not supported for bv=-1")
            self.align_errors = mad_el.align_errors
            self.dx = self.align_errors.dx
            self.dy = self.align_errors.dy
            self.tilt += self.align_errors.dpsi
        self.classes = classes
        self.Builder = Builder

class Dummy:
    type = "None"

def _default_factory():
    return 0.

class MadLoader:
    @staticmethod
    def init_line_expressions(line, mad, replace_in_expr):  # to be added to Line....
        """Enable expressions"""
        if line._var_management is None:
            line._init_var_management()

        from xdeps.madxutils import MadxEval

        _var_values = line._var_management["data"]["var_values"]
        _var_values.default_factory = _default_factory
        for name, par in mad.globals.cmdpar.items():
            if replace_in_expr is not None:
                for k, v in replace_in_expr.items():
                    name = name.replace(k, v)
            _var_values[name] = par.value
        _ref_manager = line._var_management["manager"]
        _vref = line._var_management["vref"]
        _fref = line._var_management["fref"]
        _lref = line._var_management["lref"]

        madeval_no_repl = MadxEval(_vref, _fref, mad.elements).eval

        if replace_in_expr is not None:
            def madeval(expr):
                for k, v in replace_in_expr.items():
                    expr = expr.replace(k, v)
                return madeval_no_repl(expr)
        else:
            madeval = madeval_no_repl

        # Extract expressions from madx globals
        for name, par in mad.globals.cmdpar.items():
            ee = par.expr
            if ee is not None:
                if "table(" in ee:  # Cannot import expressions involving tables
                    continue
                _vref[name] = madeval(ee)
        return madeval

    def __init__(
        self,
        sequence,
        enable_expressions=False,
        enable_errors=None,
        enable_field_errors=None,
        enable_align_errors=None,
        enable_apertures=False,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
        error_table=None,
        ignore_madtypes=(),
        expressions_for_element_types=None,
        classes=xtrack,
        replace_in_expr=None,
        allow_thick=False,
        name_prefix=None,
        enable_layout_data=False,
    ):


        if enable_errors is not None:
            if enable_field_errors is None:
                enable_field_errors = enable_errors
            if enable_align_errors is None:
                enable_align_errors = enable_errors

        if enable_field_errors is None:
            enable_field_errors = False
        if enable_align_errors is None:
            enable_align_errors = False

        if allow_thick is None:
            allow_thick = True

        if expressions_for_element_types is not None:
            assert enable_expressions, ("Expressions must be enabled if "
                                "`expressions_for_element_types` is not None")

        self.sequence = sequence
        self.enable_expressions = enable_expressions
        self.enable_field_errors = enable_field_errors
        self.enable_align_errors = enable_align_errors
        self.error_table = error_table
        self.skip_markers = skip_markers
        self.merge_drifts = merge_drifts
        self.merge_multipoles = merge_multipoles
        self.enable_apertures = enable_apertures
        self.expressions_for_element_types = expressions_for_element_types
        self.classes = classes
        self.replace_in_expr = replace_in_expr
        self._drift = self.classes.Drift
        self.ignore_madtypes = ignore_madtypes
        self.name_prefix = name_prefix
        self.enable_layout_data = enable_layout_data

        self.allow_thick = allow_thick
        self.bv = 1

    def iter_elements(self, madeval=None):
        """Yield element data for each known element"""
        if len(self.sequence.expanded_elements)==0:
            raise ValueError(f"{self.sequence} has no elements, please do {self.sequence}.use()")
        last_element = Dummy
        if self.bv == -1:
            expanded_elements = list(self.sequence.expanded_elements)[::-1]
        elif self.bv == 1:
            expanded_elements = self.sequence.expanded_elements
        else:
            raise ValueError(f"bv should be 1 or -1, not {self.bv}")
        for el in expanded_elements:
            madelem = MadElem(el.name, el, self.sequence, madeval,
                              name_prefix=self.name_prefix)
            if self.skip_markers and madelem.is_empty_marker():
                pass
            elif (
                self.merge_drifts
                and last_element.type == "drift"
                and madelem.type == "drift"
            ):
                last_element.l += el.l
            elif (
                self.merge_multipoles
                and last_element.type == "multipole"
                and madelem.type == "multipole"
            ):
                merged = last_element.merge_multipole(madelem)
                if not merged:
                    yield last_element
                    last_element = madelem
            elif madelem.type in self.ignore_madtypes:
                pass
            else:
                if last_element is not Dummy:
                    yield last_element
                last_element = madelem
        yield last_element

    def make_line(self, buffer=None):
        """Create a new line in buffer"""

        mad = self.sequence._madx

        if buffer is None:
            buffer = xobjects.context_default.new_buffer()

        line = self.classes.Line()
        self.line = line

        if self.enable_expressions:
            madeval = MadLoader.init_line_expressions(line, mad,
                                                      self.replace_in_expr)
            self.Builder = ElementBuilderWithExpr
        else:
            madeval = None
            self.Builder = ElementBuilder

        bv = self.sequence.beam.bv
        assert bv==1 or bv==-1, f"bv should be 1 or -1, not {bv}"
        self.bv = bv

        # Avoid progress bar if there are few elements
        if len(self.sequence.expanded_elements) > 10:
            _prog = progress(
                self.iter_elements(madeval=madeval),
                desc=f'Converting sequence "{self.sequence.name}"',
                total=len(self.sequence.expanded_elements))
        else:
            _prog = self.iter_elements(madeval=madeval)

        for ii, el in enumerate(_prog):
            # for each mad element create xtract elements in a buffer and add to a line
            converter = getattr(self, "convert_" + el.type, None)
            adder = getattr(self, "add_" + el.type, None)
            if self.expressions_for_element_types is not None:
               if el.type in self.expressions_for_element_types:
                   self.Builder = ElementBuilderWithExpr
                   el.madeval = madeval
               else:
                    self.Builder = ElementBuilder
                    el.madeval = None
            if adder:
                adder(el, line, buffer)
            elif converter:
                converted_el = converter(el)
                self.add_elements(converted_el, line, buffer)
            else:
                raise ValueError(
                    f'Element {el.type} not supported,\nimplement "add_{el.type}"'
                    f" or convert_{el.type} in function in MadLoader"
                )

        # copy layout data
        if self.enable_layout_data:
            layout_data = {}
            for nn in line.element_names:
                if nn in mad.elements:
                    madel = mad.elements[nn]
                    # offset represent the offset of the assembly with respect to mid-beam
                    eldata = {}
                    eldata["offset"] = [madel.mech_sep / 2 * self.bv, madel.v_pos]
                    eldata["assembly_id"] = madel.assembly_id
                    eldata["slot_id"] = madel.slot_id
                    eldata["aperture"] = [
                        madel.apertype,
                        list(madel.aperture),
                        list(madel.aper_tol),
                    ]
                    layout_data[nn] = eldata

            line.metadata["layout_data"] = layout_data

        return line

    def add_elements(
        self,
        elements: List[Union[ElementBuilder]],
        line,
        buffer,
    ):
        out = {}  # tbc
        for el in elements:
            xt_element = el.add_to_line(line, buffer)
            out[el.name] = xt_element  # tbc
        return out  # tbc

    @property
    def math(self):
        if issubclass(self.Builder, ElementBuilderWithExpr):
            return self.line._var_management['fref']

        return np

    def _assert_element_is_thin(self, mad_el):
        if value_if_expr(mad_el.l) != 0:
            if self.allow_thick:
                raise NotImplementedError(
                    f'Cannot load element {mad_el.name}, as thick elements of '
                    f'type {"/".join(mad_el.get_type_hierarchy())} are not '
                    f'yet supported.'
                )
            else:
                raise ValueError(
                    f'Element {mad_el.name} is thick, but importing thick '
                    f'elements is disabled. Did you forget to set '
                    f'`allow_thick=True`?'
                )

    def _make_drift_slice(self, mad_el, weight, name_pattern):
        return self.Builder(
            name_pattern.format(mad_el.name),
            self.classes.Drift,
            length=mad_el.l * weight,
        )

    def make_composite_element(
            self,
            xtrack_el,
            mad_el,
            custom_tilt=None,
    ):
        """Add aperture and transformations to a thin element:
        tilt, offset, aperture, offset, tilt, tilt, offset, kick, offset, tilt

        Parameters
        ----------
        xtrack_el: list
            List of xtrack elements to which the aperture and transformations
            should be added.
        mad_el: MadElement
            The element for which the aperture and transformations should be
            added.
        custom_tilt: float, optional
            If not None, the element will be additionally tilted by this
            amount.
        """
        # TODO: Implement permanent alignment

        align = Alignment(
            mad_el, self.enable_align_errors, self.classes, self.Builder,
            self.bv, custom_tilt)

        aperture_seq = []
        if self.enable_apertures and mad_el.has_aperture():
            if self.bv == -1:
                raise NotImplementedError("Apertures for bv=-1 are not yet supported.")
            aper = Aperture(mad_el, self.enable_align_errors, self)
            aperture_seq = aper.aperture()

        # using directly tilt and shift in the element
        for xtee in xtrack_el:
            if align.tilt or align.dx or align.dy:
                xtee.rot_s_rad = align.tilt
                if align.dx or align.dy:
                    xtee.shift_x = align.dx
                    xtee.shift_y = align.dy
        align.tilt = 0
        align.dx = 0
        align.dy = 0

        # Attach aperture to main element
        if aperture_seq:
            assert len(aperture_seq) <= 1, (
                "Only one aperture per mad element is supported")
            main_element=None
            for ee in xtrack_el:
                if ee.name== mad_el.name:
                    main_element=ee
                    break
            assert main_element is not None
            xtrack_el[0].name_associated_aperture = aperture_seq[0].name

        elem_list = aperture_seq + xtrack_el

        return elem_list

    def convert_quadrupole(self, mad_el):
        if self.allow_thick:
            if not mad_el.l:
                raise ValueError(
                    "Thick quadrupole with length zero are not supported.")
            return self._convert_quadrupole_thick(mad_el)
        else:
            raise NotImplementedError(
                "Quadrupole are not supported in thin mode."
            )

    def _convert_quadrupole_thick(self, mad_el): # bv done
        kwargs = {}
        if self.enable_field_errors:
            kwargs = _prepare_field_errors_thick_elem(mad_el)
            kwargs['num_multipole_kicks'] = 1

        return self.make_composite_element(
            [
                self.Builder(
                    mad_el.name,
                    self.classes.Quadrupole,
                    k1=self.bv * mad_el.k1,
                    k1s=mad_el.k1s,
                    length=mad_el.l,
                    **kwargs,
                ),
            ],
            mad_el,
        )

    def convert_rbend(self, mad_el): # bv done
        return self._convert_bend(mad_el)

    def convert_sbend(self, mad_el): # bv done
        return self._convert_bend(mad_el)

    def _convert_bend( # bv done
        self,
        mad_el,
    ):
        assert self.allow_thick, "Bends are not supported in thin mode."

        if mad_el.type == 'sbend':
            element_type = self.classes.Bend
        elif mad_el.type == 'rbend':
            element_type = self.classes.RBend
        else:
            raise ValueError(f'Unknown bend type {mad_el.type}.')

        bend_kwargs = {}

        if mad_el.type == 'rbend' and self.sequence._madx.options.rbarc:
            l_curv = mad_el.l / self.math.sinc(0.5 * mad_el.angle)
            bend_kwargs['length_straight'] = mad_el.l
        else:
            l_curv = mad_el.l
            bend_kwargs['length'] = l_curv

        bend_kwargs['angle'] = mad_el.angle

        # Edge angles
        e1 = mad_el.e1
        e2 = mad_el.e2
        if self.bv == -1:
            e1, e2 = e2, e1

        # Edge angles for field errors
        if mad_el.k0:
            h = mad_el.angle / l_curv
            angle_fdown = (mad_el.k0 - h) * l_curv / 2
        else:
            angle_fdown = 0

        if self.enable_field_errors:
            kwargs = _prepare_field_errors_thick_elem(mad_el)
            knl = kwargs['knl']
            ksl = kwargs['ksl']
            num_multipole_kicks = 1
        else:
            knl = [0] * 3
            ksl = []
            num_multipole_kicks = 0

        knl[2] += mad_el.k2 * l_curv

        if mad_el.k0:
            k0_from_h = False
            bend_kwargs['k0'] = mad_el.k0
        else:
            k0_from_h = True

        # Convert bend core
        bend_core = self.Builder(
            mad_el.name,
            element_type,
            k0_from_h=k0_from_h,
            k1=self.bv * mad_el.k1,
            edge_entry_angle=e1,
            edge_exit_angle=e2,
            edge_entry_angle_fdown=angle_fdown,
            edge_exit_angle_fdown=angle_fdown,
            edge_entry_fint=mad_el.fint,
            edge_exit_fint=(
                mad_el.fintx if value_if_expr(mad_el.fintx) >= 0 else mad_el.fint),
            edge_entry_hgap=mad_el.hgap,
            edge_exit_hgap=mad_el.hgap,
            knl=knl,
            ksl=ksl,
            num_multipole_kicks=num_multipole_kicks,
            **bend_kwargs,
        )

        sequence = [bend_core]

        return self.make_composite_element(sequence, mad_el)

    def convert_sextupole(self, mad_el): # bv done
        kwargs = {}

        if self.enable_field_errors:
            kwargs = _prepare_field_errors_thick_elem(mad_el)

        return self.make_composite_element(
            [
                self.Builder(
                    mad_el.name,
                    self.classes.Sextupole,
                    k2=mad_el.k2,
                    k2s=self.bv * mad_el.k2s,
                    length=mad_el.l,
                    **kwargs,
                ),
            ],
            mad_el,
        )

    def convert_octupole(self, mad_el): # bv done
        kwargs = {}

        if self.enable_field_errors:
            kwargs = _prepare_field_errors_thick_elem(mad_el)

        return self.make_composite_element(
            [
                self.Builder(
                    mad_el.name,
                    self.classes.Octupole,
                    k3=self.bv*mad_el.k3,
                    k3s=mad_el.k3s,
                    length=mad_el.l,
                    **kwargs,
                ),
            ],
            mad_el,
        )


    def convert_rectangle(self, mad_el):
        h, v = mad_el.aperture[:2]
        return [
            self.Builder(
                mad_el.name + "_aper",
                self.classes.LimitRect,
                min_x=-h,
                max_x=h,
                min_y=-v,
                max_y=v,
            )
        ]

    def convert_racetrack(self, mad_el):
        h, v, a, b = mad_el.aperture[:4]
        return [
            self.Builder(
                mad_el.name + "_aper",
                self.classes.LimitRacetrack,
                min_x=-h,
                max_x=h,
                min_y=-v,
                max_y=v,
                a=a,
                b=b,
            )
        ]

    def convert_ellipse(self, mad_el):
        a, b = mad_el.aperture[:2]
        return [
            self.Builder(mad_el.name + "_aper", self.classes.LimitEllipse, a=a, b=b)
        ]

    def convert_circle(self, mad_el):
        a = mad_el.aperture[0]
        return [
            self.Builder(mad_el.name + "_aper", self.classes.LimitEllipse, a=a, b=a)
        ]

    def convert_rectellipse(self, mad_el):
        h, v, a, b = mad_el.aperture[:4]
        return [
            self.Builder(
                mad_el.name + "_aper",
                self.classes.LimitRectEllipse,
                max_x=h,
                max_y=v,
                a=a,
                b=b,
            )
        ]

    def convert_octagon(self, ee):
        # MAD-X assumes X and Y symmetry, defines 2 points per quadrant
        a0 = ee.aperture[0]  # half-width
        a1 = ee.aperture[1]  # half-height
        a2 = ee.aperture[2]  # angle between the lower point and the X axis
        a3 = ee.aperture[3]  # angle between the other point and the X axis
        V1 = (a0, a0 * self.math.tan(a2))
        V2 = (a1 / self.math.tan(a3), a1)
        el = self.Builder(
            ee.name + "_aper",
            self.classes.LimitPolygon,
            x_vertices=[V1[0], V2[0], -V2[0], -V1[0], -V1[0], -V2[0], V2[0], V1[0]],
            y_vertices=[V1[1], V2[1], V2[1], V1[1], -V1[1], -V2[1], -V2[1], -V1[1]],
        )
        return [el]

    def convert_polygon(self, ee):
        x_vertices = ee.aper_vx[0::2]
        y_vertices = ee.aper_vy[1::2]
        el = self.Builder(
            ee.name + "_aper",
            self.classes.LimitPolygon,
            x_vertices=x_vertices,
            y_vertices=y_vertices,
        )
        return [el]

    def convert_drift(self, mad_elem):
        return [self.Builder(mad_elem.name, self._drift, length=mad_elem.l)]

    def convert_marker(self, mad_elem):
        el = self.Builder(mad_elem.name, self.classes.Marker)
        return self.make_composite_element([el], mad_elem)

    def convert_drift_like(self, mad_elem):
        el = self.Builder(mad_elem.name, self._drift, length=mad_elem.l)
        return self.make_composite_element([el], mad_elem)

    convert_monitor = convert_drift_like
    convert_hmonitor = convert_drift_like
    convert_vmonitor = convert_drift_like
    convert_collimator = convert_drift_like
    convert_rcollimator = convert_drift_like
    convert_elseparator = convert_drift_like
    convert_instrument = convert_drift_like

    def convert_solenoid(self, mad_elem): # bv done
        if get_value(mad_elem.l) == 0:
            _print(f'Warning: Thin solenoids are not yet implemented, '
                   f'reverting to importing `{mad_elem.name}` as a drift.')
            return self.convert_drift_like(mad_elem)

        kwargs = {}

        if self.enable_field_errors:
            kwargs = _prepare_field_errors_thick_elem(mad_elem)

        el = self.Builder(
            mad_elem.name,
            self.classes.Solenoid,
            length=mad_elem.l,
            ks=self.bv * mad_elem.ks,
            ksi=self.bv * mad_elem.ksi,
            **kwargs,
        )
        return self.make_composite_element([el], mad_elem)

    def convert_multipole(self, mad_elem): # bv done
        self._assert_element_is_thin(mad_elem)
        # getting max length of knl and ksl
        if self.bv == -1:
            knl = [(-1)**ii * x for ii, x in enumerate(mad_elem.knl)]
            ksl = [(-1)**(ii+1) * x for ii, x in enumerate(mad_elem.ksl)]
        else:
            knl = mad_elem.knl
            ksl = mad_elem.ksl
        lmax = max(non_zero_len(knl), non_zero_len(ksl), 1)
        if mad_elem.field_errors is not None and self.enable_field_errors:
            dkn = mad_elem.field_errors.dkn
            dks = mad_elem.field_errors.dks
            lmax = max(lmax, non_zero_len(dkn), non_zero_len(dks))
            knl = add_lists(knl, dkn, lmax)
            ksl = add_lists(ksl, dks, lmax)
        el = self.Builder(mad_elem.name, self.classes.Multipole, order=lmax - 1)
        el.knl = knl[:lmax]
        el.ksl = ksl[:lmax]

        if hasattr(mad_elem, 'ksl') and mad_elem.ksl[0]:
            raise NotImplementedError("Multipole with ksl[0] is not supported.")

        if hasattr(el, 'hyl') and el.hyl:
            raise NotImplementedError("Multipole with hyl is not supported.")

        if (
            mad_elem.angle
        ):  # testing for non-zero (cannot use !=0 as it creates an expression)
            el.hxl = mad_elem.angle
        else:
            el.hxl = mad_elem.knl[0]  # in madx angle=0 -> dipole
        el.length = mad_elem.lrad
        return self.make_composite_element([el], mad_elem)

    def convert_kicker(self, mad_el): # bv done
        hkick = [-mad_el.hkick] if mad_el.hkick else []
        vkick = [self.bv * mad_el.vkick] if mad_el.vkick else []
        thin_kicker = self.Builder(
            mad_el.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=(mad_el.l or mad_el.lrad),
            hxl=0,
        )

        if value_if_expr(mad_el.l) != 0:
            if not self.allow_thick:
                self._assert_element_is_thin(mad_el)

            sequence = [
                self._make_drift_slice(mad_el, 0.5, "drift_{}..1"),
                thin_kicker,
                self._make_drift_slice(mad_el, 0.5, "drift_{}..2"),
            ]
        else:
            sequence = [thin_kicker]

        return self.make_composite_element(sequence, mad_el)

    convert_tkicker = convert_kicker

    def convert_hkicker(self, mad_el): # bv done
        if mad_el.hkick:
            raise ValueError(
                "hkicker with hkick is not supported, please use kick instead")

        hkick = [-mad_el.kick] if mad_el.kick else []
        vkick = []
        thin_hkicker = self.Builder(
            mad_el.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=(mad_el.l or mad_el.lrad),
            hxl=0,
        )

        if value_if_expr(mad_el.l) != 0:
            if not self.allow_thick:
                self._assert_element_is_thin(mad_el)

            sequence = [
                self._make_drift_slice(mad_el, 0.5, "drift_{}..1"),
                thin_hkicker,
                self._make_drift_slice(mad_el, 0.5, "drift_{}..2"),
            ]
        else:
            sequence = [thin_hkicker]

        return self.make_composite_element(sequence, mad_el)

    def convert_vkicker(self, mad_el): # bv done
        if mad_el.vkick:
            raise ValueError(
                "vkicker with vkick is not supported, please use kick instead")

        hkick = []
        vkick = [self.bv * mad_el.kick] if mad_el.kick else []
        thin_vkicker = self.Builder(
            mad_el.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=(mad_el.l or mad_el.lrad),
            hxl=0,
        )

        if value_if_expr(mad_el.l) != 0:
            if not self.allow_thick:
                self._assert_element_is_thin(mad_el)

            sequence = [
                self._make_drift_slice(mad_el, 0.5, "drift_{}..1"),
                thin_vkicker,
                self._make_drift_slice(mad_el, 0.5, "drift_{}..2"),
            ]
        else:
            sequence = [thin_vkicker]

        return self.make_composite_element(sequence, mad_el)

    def convert_dipedge(self, mad_elem):
        if self.bv == -1:
            raise NotImplementedError("Dipole edges for bv=-1 are not yet supported.")
        # TODO LRAD
        el = self.Builder(
            mad_elem.name,
            self.classes.DipoleEdge,
            h=mad_elem.h,
            e1=mad_elem.e1,
            hgap=mad_elem.hgap,
            fint=mad_elem.fint,
        )
        return self.make_composite_element([el], mad_elem)

    def convert_rfcavity(self, ee): # bv done
        # TODO LRAD
        if ee.freq == 0 and ee.harmon:
            frequency = (
                ee.harmon * self.sequence.beam.beta * clight / self.sequence.length
            )
        else:
            frequency = ee.freq * 1e6
        if (hasattr(self.sequence, 'beam')
                and self.sequence.beam.particle == 'ion'):
            scale_voltage = 1./self.sequence.beam.charge
        else:
            scale_voltage = 1.
        if self.bv == -1:
            lag_deg = -ee.lag * 360 + 180
        elif self.bv == 1:
            lag_deg = ee.lag * 360
        else:
            raise ValueError(f"bv should be 1 or -1, not {self.bv}")
        el = self.Builder(
            ee.name,
            self.classes.Cavity,
            voltage=scale_voltage * ee.volt * 1e6,
            frequency=frequency,
            lag=lag_deg,
        )

        if value_if_expr(ee.l) != 0:
            sequence = [
                self._make_drift_slice(ee, 0.5, f"drift_{{}}..1"),
                el,
                self._make_drift_slice(ee, 0.5, f"drift_{{}}..2"),
            ]
        else:
            sequence = [el]

        return self.make_composite_element(sequence, ee)

    def convert_rfmultipole(self, ee):
        if self.bv == -1:
            raise NotImplementedError("RF multipole for bv=-1 are not yet supported.")
        self._assert_element_is_thin(ee)
        # TODO LRAD
        if ee.harmon:
            raise NotImplementedError
        if ee.l:
            raise NotImplementedError
        el = self.Builder(
            ee.name,
            self.classes.RFMultipole,
            voltage=ee.volt * 1e6,
            frequency=ee.freq * 1e6,
            lag=ee.lag * 360,
            knl=ee.knl,
            ksl=ee.ksl,
            pn=[v * 360 for v in ee.pnl],
            ps=[v * 360 for v in ee.psl],
        )
        return self.make_composite_element([el], ee)

    def convert_wire(self, ee):
        if self.bv == -1:
            raise NotImplementedError("Wire for bv=-1 are not yet supported.")
        self._assert_element_is_thin(ee)
        if len(ee.L_phy) == 1:
            # the index [0] is present because in MAD-X multiple wires can
            # be defined within the same element
            el = self.Builder(
                ee.name,
                self.classes.Wire,
                L_phy=ee.L_phy[0],
                L_int=ee.L_int[0],
                current=ee.current[0],
                xma=ee.xma[0],
                yma=ee.yma[0],
            )
            return self.make_composite_element([el], ee)
        else:
            # TODO: add multiple elements for multiwire configuration
            raise ValueError("Multiwire configuration not supported")

    def convert_crabcavity(self, ee):
        self._assert_element_is_thin(ee)
        # This has to be disabled, as it raises an error when l is assigned to an
        # expression:
        # for nn in ["l", "harmon", "lagf", "rv1", "rv2", "rph1", "rph2"]:
        #     if getattr(ee, nn):
        #         raise NotImplementedError(f"Invalid value {nn}={getattr(ee, nn)}")

        # ee.volt in MV, sequence.beam.pc in GeV
        if abs(ee.tilt - np.pi / 2) < 1e-9:
            el = self.Builder(
                ee.name,
                self.classes.RFMultipole,
                frequency=ee.freq * 1e6,
                ksl=[-ee.volt / self.sequence.beam.pc * 1e-3],
                ps=[ee.lag * 360 + 90],
            )
            ee.tilt = 0
        else:
            el = self.Builder(
                ee.name,
                self.classes.RFMultipole,
                frequency=ee.freq * 1e6,
                knl=[ee.volt / self.sequence.beam.pc * 1e-3 * self.bv],
                pn=[ee.lag * self.bv * 360 + 90],  # TODO: Changed sign to match sixtrack
                # To be checked!!!!
            )
        return self.make_composite_element([el], ee)

    def convert_beambeam(self, ee):
        if self.bv == -1:
            raise NotImplementedError("BeamBeam for bv=-1 are not yet supported.")
        self._assert_element_is_thin(ee)
        import xfields as xf

        if ee.slot_id == 6 or ee.slot_id == 60:
            # force no expression by using ElementBuilder and not self.Builder
            el = ElementBuilder(
                ee.name,
                xf.BeamBeamBiGaussian3D,
                old_interface={
                    "phi": 0.0,
                    "alpha": 0.0,
                    "x_bb_co": 0.0,
                    "y_bb_co": 0.0,
                    "charge_slices": [0.0],
                    "zeta_slices": [0.0],
                    "sigma_11": 1.0,
                    "sigma_12": 0.0,
                    "sigma_13": 0.0,
                    "sigma_14": 0.0,
                    "sigma_22": 1.0,
                    "sigma_23": 0.0,
                    "sigma_24": 0.0,
                    "sigma_33": 0.0,
                    "sigma_34": 0.0,
                    "sigma_44": 0.0,
                    "x_co": 0.0,
                    "px_co": 0.0,
                    "y_co": 0.0,
                    "py_co": 0.0,
                    "zeta_co": 0.0,
                    "delta_co": 0.0,
                    "d_x": 0.0,
                    "d_px": 0.0,
                    "d_y": 0.0,
                    "d_py": 0.0,
                    "d_zeta": 0.0,
                    "d_delta": 0.0,
                },
            )
        else:
            # BB interaction is 4D
            # force no expression by using ElementBuilder and not self.Builder
            el = ElementBuilder(
                ee.name,
                xf.BeamBeamBiGaussian2D,
                n_particles=0.0,
                q0=0.0,
                beta0=1.0,
                mean_x=0.0,
                mean_y=0.0,
                sigma_x=1.0,
                sigma_y=1.0,
                d_px=0,
                d_py=0,
            )
        return self.make_composite_element([el], ee)

    def convert_placeholder(self, ee):
        # assert not is_expr(ee.slot_id) can be done only after release MADX 5.09
        if ee.slot_id == 1:
            raise ValueError("This feature is discontinued!")
            # newele = classes.SCCoasting()
        elif ee.slot_id == 2:
            # TODO Abstraction through `classes` to be introduced
            raise ValueError("This feature is discontinued!")
            # import xfields as xf
            # lprofile = xf.LongitudinalProfileQGaussian(
            #         number_of_particles=0.,
            #         sigma_z=1.,
            #         z0=0.,
            #         q_parameter=1.)
            # newele = xf.SpaceChargeBiGaussian(
            #     length=0,
            #     apply_z_kick=False,
            #     longitudinal_profile=lprofile,
            #     mean_x=0.,
            #     mean_y=0.,
            #     sigma_x=1.,
            #     sigma_y=1.)

        elif ee.slot_id == 3:
            el = self.Builder(ee.name, self.classes.SCInterpolatedProfile)
        else:
            el = self.Builder(ee.name, self._drift, length=ee.l)
        return self.make_composite_element([el], ee)

    def convert_matrix(self, ee):
        if self.bv == -1:
            raise NotImplementedError("Matrix for bv=-1 are not yet supported.")
        length = ee.l
        m0 = np.zeros(6, dtype=object)
        for m0_i in range(6):
            att_name = f"kick{m0_i+1}"
            if hasattr(ee, att_name):
                m0[m0_i] = getattr(ee, att_name)
        m1 = np.zeros((6, 6), dtype=object)
        for m1_i in range(6):
            for m1_j in range(6):
                att_name = f"rm{m1_i+1}{m1_j+1}"
                if hasattr(ee, att_name):
                    m1[m1_i, m1_j] = getattr(ee, att_name)
        el = self.Builder(
            ee.name, self.classes.FirstOrderTaylorMap, length=length, m0=m0, m1=m1
        )
        return self.make_composite_element([el], ee)

    def convert_srotation(self, ee):
        if self.bv == -1:
            raise NotImplementedError("SRotation for bv=-1 are not yet supported.")
        angle = ee.angle*180/np.pi
        el = self.Builder(
            ee.name, self.classes.SRotation, angle=angle
        )
        return self.make_composite_element([el], ee)

    def convert_xrotation(self, ee):
        if self.bv == -1:
            raise NotImplementedError("XRotation for bv=-1 are not yet supported.")
        angle = ee.angle*180/np.pi
        el = self.Builder(
            ee.name, self.classes.XRotation, angle=angle
        )
        return self.make_composite_element([el], ee)

    def convert_yrotation(self, ee):
        if self.bv == -1:
            raise NotImplementedError("YRotation for bv=-1 are not yet supported.")
        angle = ee.angle*180/np.pi
        el = self.Builder(
            ee.name, self.classes.YRotation, angle=angle
        )
        return self.make_composite_element([el], ee)

    def convert_translation(self, ee):
        if self.bv == -1:
            raise NotImplementedError("Translation for bv=-1 are not yet supported.")
        el_transverse = self.Builder(
            ee.name, self.classes.XYShift, dx=ee.dx, dy=ee.dy
        )
        if ee.ds:
            raise NotImplementedError # Need to implement ShiftS element
        ee.dx = 0
        ee.dy = 0
        ee.ds = 0
        return self.make_composite_element([el_transverse], ee)

    def convert_nllens(self, mad_elem):
        if self.bv == -1:
            raise NotImplementedError("Non-linear lens for bv=-1 are not yet supported.")
        el = self.Builder(
            mad_elem.name,
            self.classes.NonLinearLens,
            knll=mad_elem.knll,
            cnll=mad_elem.cnll,
        )
        return self.make_composite_element([el], mad_elem)


def _prepare_field_errors_thick_elem(mad_el):
    if mad_el.field_errors is None:
        return {}

    dkn = mad_el.field_errors.dkn
    dks = mad_el.field_errors.dks
    lmax = max(non_zero_len(dkn), non_zero_len(dks))
    if lmax > 6:
        raise ValueError(
            "Only up to dodecapoles are supported for field errors"
            " of thick magnets for now."
        )
    if len(dkn) > lmax:
        dkn = dkn[:lmax]
    if len(dks) > lmax:
        dks = dks[:lmax]

    kwargs_to_add = {}
    if np.any(np.abs(dkn)) or np.any(np.abs(dks)):
        kwargs_to_add['knl'] = dkn
        kwargs_to_add['ksl'] = dks

    return kwargs_to_add
