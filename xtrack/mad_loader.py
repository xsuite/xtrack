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

from typing import List

import numpy as np

import xtrack, xobjects


# Generic functions

clight = 299792458


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

def not_zero(x):
    if is_expr(x):
        return True
    else:
        return x != 0


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
    def __init__(self, name, elem, sequence, madeval=None):
        self.name = name
        self.elem = elem
        self.sequence = sequence
        self.madeval = madeval
        ### needed for merge multipoles
        if hasattr(elem, "field_errors") and elem.field_errors is not None:
            self.field_errors = FieldErrors(elem.field_errors)
        else:
            self.field_errors = None
        if elem.dphi or elem.dtheta or elem.dpsi or elem.dx or elem.dy or elem.ds:
            raise NotImplementedError

    # @property
    # def field_errors(self):
    #    elem=self.elem
    #    if hasattr(elem, "field_errors") and elem.field_errors is not None:
    #        return FieldErrors(elem.field_errors)

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
        return "Element(%s, %s, %s, %s)" % (self.name, self.type, self.init, self.attrs)

    def __setattr__(self, k, v):
        if hasattr(self, "attrs"):
            self.attrs[k] = v
        else:
            super().__setattr__(k, v)

    def add_to_line(self, line, buffer):
        xtel = self.type(**self.attrs, _buffer=buffer)
        name = generate_repeated_name(line, self.name)
        line.append_element(xtel, name)


class ElementBuilderWithExpr(ElementBuilder):
    def add_to_line(self, line, buffer):
        attr_values = {k: get_value(v) for k, v in self.attrs.items()}
        xtel = self.type(**attr_values, _buffer=buffer)
        name = generate_repeated_name(line, self.name)
        line.append_element(xtel, name)
        elref = line.element_refs[name]
        for k, p in self.attrs.items():
            set_expr(elref, k, p)
        return xtel


class Aperture:
    def __init__(self, mad_el, enable_errors, loader):
        self.mad_el = mad_el
        self.aper_tilt = rad2deg(mad_el.aper_tilt)
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

    def entry(self):
        out = []
        if self.aper_tilt:
            out.append(
                self.Builder(
                    self.name + "_aper_tilt_entry",
                    self.classes.SRotation,
                    angle=self.aper_tilt,
                )
            )
        if self.dx or self.dy:
            out.append(
                self.Builder(
                    self.name + "_aper_offset_entry",
                    self.classes.XYShift,
                    dx=self.dx,
                    dy=self.dy,
                )
            )
        return out

    def exit(self):
        out = []
        if self.dx or self.dy:
            out.append(
                self.Builder(
                    self.name + "_aper_offset_exit",
                    self.classes.XYShift,
                    dx=-self.dx,
                    dy=-self.dy,
                )
            )
        if not_zero(self.aper_tilt):
            out.append(
                self.Builder(
                    self.name + "_aper_tilt_exit",
                    self.classes.SRotation,
                    angle=-self.aper_tilt,
                )
            )
        return out

    def aperture(self):
        if len(self.mad_el.aper_vx) > 2:
            return [
                self.Builder(
                    self.name + "_aper",
                    self.classes.LimitPolygon,
                    x_vertices=self.mad_el.aper_vx,
                    y_vertices=self.mad_el.aper_vy,
                )
            ]
        else:
            conveter = getattr(self.loader, "convert_" + self.apertype, None)
            if conveter is None:
                raise ValueError(f"Aperture type `{self.apertype}` not supported")
            return conveter(self.mad_el)


class Alignment:
    def __init__(self, mad_el, enable_errors, classes, Builder):
        self.mad_el = mad_el
        self.tilt = mad_el.get("tilt", 0)  # some elements do not have tilt
        if self.tilt:
            self.tilt = rad2deg(self.tilt)
        self.name = mad_el.name
        self.dx = 0
        self.dy = 0
        if (
            enable_errors
            and hasattr(mad_el, "align_errors")
            and mad_el.align_errors is not None
        ):
            self.align_errors = mad_el.align_errors
            self.dx = self.align_errors.dx
            self.dy = self.align_errors.dy
            self.tilt += rad2deg(self.align_errors.dpsi)
        self.classes = classes
        self.Builder = Builder

    def entry(self):
        out = []
        if self.tilt:
            out.append(
                self.Builder(
                    self.name + "_tilt_entry",
                    self.classes.SRotation,
                    angle=self.tilt,
                )
            )
        if self.dx or self.dy:
            out.append(
                self.Builder(
                    self.name + "_offset_entry",
                    self.classes.XYShift,
                    dx=self.dx,
                    dy=self.dy,
                )
            )
        return out

    def exit(self):
        out = []
        if self.dx or self.dy:
            out.append(
                self.Builder(
                    self.name + "_offset_exit",
                    self.classes.XYShift,
                    dx=-self.dx,
                    dy=-self.dy,
                )
            )
        if self.tilt:
            out.append(
                self.Builder(
                    self.name + "_tilt_exit",
                    self.classes.SRotation,
                    angle=-self.tilt,
                )
            )
        return out


class Dummy:
    type = "None"


class MadLoader:
    @staticmethod
    def init_line_expressions(line, mad):  # to be added to Line....
        """Enable expressions"""
        line._init_var_management()

        from xdeps.madxutils import MadxEval

        _var_values = line._var_management["data"]["var_values"]
        _var_values.default_factory = lambda: 0
        for name, par in mad.globals.cmdpar.items():
            _var_values[name] = par.value
        _ref_manager = line._var_management["manager"]
        _vref = line._var_management["vref"]
        _fref = line._var_management["fref"]
        _lref = line._var_management["lref"]

        madeval = MadxEval(_vref, _fref, None).eval

        # Extract expressions from madx globals
        for name, par in mad.globals.cmdpar.items():
            if par.expr is not None:
                if "table(" in par.expr:  # Cannot import expressions involving tables
                    continue
                _vref[name] = madeval(par.expr)
        return madeval

    def __init__(
        self,
        sequence,
        enable_expressions=False,
        enable_errors=False,
        enable_apertures=False,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
        error_table=None,
        exact_drift=False,
        ignore_madtypes=(),
        classes=xtrack,
    ):
        self.sequence = sequence
        self.enable_expressions = enable_expressions
        self.enable_errors = enable_errors
        self.error_table = error_table
        self.skip_markers = skip_markers
        self.merge_drifts = merge_drifts
        self.merge_multipoles = merge_multipoles
        self.enable_apertures = enable_apertures
        self.classes = classes
        if exact_drift:
            self._drift = self.classes.DriftExact # will probably be removed
                                                  # DriftExact is implemented
                                                  # with compile flag
        else:
            self._drift = self.classes.Drift
        self.ignore_madtypes = ignore_madtypes

    def iter_elements(self, madeval=None):
        """Yield element data for each known element"""
        last_element = Dummy
        for el in self.sequence.expanded_elements:
            madelem = MadElem(el.name, el, self.sequence, madeval)
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

        if self.enable_expressions:
            madeval = MadLoader.init_line_expressions(line, mad)
            self.Builder = ElementBuilderWithExpr
        else:
            madeval = None
            self.Builder = ElementBuilder

        nelem = len(self.sequence.expanded_elements)

        for ii, el in enumerate(self.iter_elements(madeval=madeval)):
            # for each mad element create xtract elements in a buffer and add to a line
            converter = getattr(self, "convert_" + el.type, None)
            adder = getattr(self, "add_" + el.type, None)
            if adder:
                adder(el, line, buffer)
            elif converter:
                self.add_elements(converter(el), line, buffer)
            else:
                raise ValueError(
                    f"Element {el.type} not supported,\n implement add_{el.type} or convert_{el.type} in function in MadLoader"
                )
            if ii % 100 == 0:
                print(
                    f'Converting sequence "{self.sequence.name}": {round(ii/nelem*100):2d}%     ',
                    end="\r",
                    flush=True,
                )
        print()
        return line

    def add_elements(
        self,
        elements: List[ElementBuilder],
        line,
        buffer,
    ):
        out = {}  # tbc
        for el in elements:
            xtel = el.add_to_line(line, buffer)
            out[el.name] = xtel  # tbc
        return out  # tbc

    def convert_thin_element(self, xtrack_el, mad_el):
        """add aperture and transformations to a thin element
        tilt, offset, aperture, offset, tilt, tilt, offset, kick, offset, tilt
        """
        align = Alignment(mad_el, self.enable_errors, self.classes, self.Builder)
        # perm=self.permanent_alignement(mad_el) #to be implemented
        elem_list = []
        # elem_list.extend(perm.entry())
        if self.enable_apertures and mad_el.has_aperture():
            aper = Aperture(mad_el, self.enable_errors, self)
            elem_list.extend(aper.entry())
            elem_list.extend(aper.aperture())
            elem_list.extend(aper.exit())
        elem_list.extend(align.entry())
        elem_list.extend(xtrack_el)
        elem_list.extend(align.exit())
        # elem_list.extend(perm.exit())

        return elem_list

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
        a0 = ee.aperture[0]
        a1 = ee.aperture[1]
        a2 = ee.aperture[2]
        a3 = ee.aperture[3]
        V1 = (a0, a0 * np.tan(a2))  # expression will fail
        V2 = (a1 / np.tan(a3), a1)  # expression will fail
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
        return self.convert_thin_element([el], mad_elem)

    def convert_drift_like(self, mad_elem):
        el = self.Builder(mad_elem.name, self._drift, length=mad_elem.l)
        return self.convert_thin_element([el], mad_elem)

    convert_monitor = convert_drift_like
    convert_hmonitor = convert_drift_like
    convert_vmonitor = convert_drift_like
    convert_collimator = convert_drift_like
    convert_rcollimator = convert_drift_like
    convert_collimator = convert_drift_like
    convert_elseparator = convert_drift_like
    convert_instrument = convert_drift_like
    convert_solenoid = convert_drift_like

    def convert_multipole(self, mad_elem):
        # getting max length of knl and ksl
        knl = mad_elem.knl
        ksl = mad_elem.ksl
        lmax = max(non_zero_len(knl), non_zero_len(ksl), 1)
        if mad_elem.field_errors is not None and self.enable_errors:
            dkn = mad_elem.field_errors.dkn
            dks = mad_elem.field_errors.dks
            lmax = max(lmax, non_zero_len(dkn), non_zero_len(dks))
            knl = add_lists(knl, dkn, lmax)
            ksl = add_lists(ksl, dks, lmax)
        el = self.Builder(mad_elem.name, self.classes.Multipole, order=lmax - 1)
        el.knl = knl[:lmax]
        el.ksl = ksl[:lmax]
        if (
            mad_elem.angle
        ):  # testing for non-zero (cannot use !=0 as it creates an expression)
            el.hxl = mad_elem.angle
        else:
            el.hxl = mad_elem.knl[0]  # in madx angle=0 -> dipole
            el.hyl = mad_elem.ksl[0]  # in madx angle=0 -> dipole
        el.length = mad_elem.lrad
        return self.convert_thin_element([el], mad_elem)

    def convert_kicker(self, mad_elem):
        hkick = [-mad_elem.hkick] if mad_elem.hkick else []
        vkick = [mad_elem.vkick] if mad_elem.vkick else []
        el = self.Builder(
            mad_elem.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=mad_elem.lrad,
            hxl=0,
            hyl=0,
        )
        return self.convert_thin_element([el], mad_elem)

    convert_tkicker = convert_kicker

    def convert_hkicker(self, mad_elem):
        hkick = [-mad_elem.kick] if mad_elem.kick else []
        vkick = []
        el = self.Builder(
            mad_elem.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=mad_elem.lrad,
            hxl=0,
            hyl=0,
        )
        return self.convert_thin_element([el], mad_elem)

    def convert_vkicker(self, mad_elem):
        hkick = []
        vkick = [mad_elem.kick] if mad_elem.kick else []
        el = self.Builder(
            mad_elem.name,
            self.classes.Multipole,
            knl=hkick,
            ksl=vkick,
            length=mad_elem.lrad,
            hxl=0,
            hyl=0,
        )
        return self.convert_thin_element([el], mad_elem)

    def convert_dipedge(self, mad_elem):
        # TODO LRAD
        el = self.Builder(
            mad_elem.name,
            self.classes.DipoleEdge,
            h=mad_elem.h,
            e1=mad_elem.e1,
            hgap=mad_elem.hgap,
            fint=mad_elem.fint,
        )
        return self.convert_thin_element([el], mad_elem)

    def convert_rfcavity(self, ee):
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
        el = self.Builder(
            ee.name,
            self.classes.Cavity,
            voltage=scale_voltage * ee.volt * 1e6,
            frequency=frequency,
            lag=ee.lag * 360,
        )
        return self.convert_thin_element([el], ee)

    def convert_rfmultipole(self, ee):
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
        return self.convert_thin_element([el], ee)

    def convert_wire(self, ee):
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
            return self.convert_thin_element([el], ee)
        else:
            # TODO: add multiple elements for multiwire configuration
            raise ValueError("Multiwire configuration not supported")

    def convert_crabcavity(self, ee):
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
                knl=[ee.volt / self.sequence.beam.pc * 1e-3],
                pn=[ee.lag * 360 + 90],  # TODO: Changed sign to match sixtrack
                # To be checked!!!!
            )
        return self.convert_thin_element([el], ee)

    def convert_beambeam(self, ee):
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
        return self.convert_thin_element([el], ee)

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
        return self.convert_thin_element([el], ee)

    def convert_matrix(self, ee):
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
        return self.convert_thin_element([el], ee)

    def convert_srotation(self, ee):
        angle = getattr(ee, 'angle')*180/np.pi
        el = self.Builder(
            ee.name, self.classes.SRotation, angle=angle
        )
        return self.convert_thin_element([el], ee)

    def convert_xrotation(self, ee):
        angle = getattr(ee, 'angle')*180/np.pi
        el = self.Builder(
            ee.name, self.classes.XRotation, angle=angle
        )
        return self.convert_thin_element([el], ee)

    def convert_yrotation(self, ee):
        angle = getattr(ee, 'angle')*180/np.pi
        el = self.Builder(
            ee.name, self.classes.YRotation, angle=angle
        )
        return self.convert_thin_element([el], ee)

