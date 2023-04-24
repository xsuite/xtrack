"""
mad -> iterator of element data mad_data (dict of key, Par or ParList, or substructures)
Par,Parlist take an optional madeval function to transform string into expressions

Some structures contains only values and not paramenters (e.g. field_errors or align_errors)

for each mad element the converter creates a list of xtrack elements in form of:
    name, type, data for initialization (does not contain expressions), data for after initialization (that can contain expressions)

Par and Parlist support arithmetic operations
"""

from collections import namedtuple
from typing import Iterable, List, Tuple, Dict, Any

import numpy as np

import xtrack, xobjects


# Generic functions

clight=299792458

def iterable(obj):
    return hasattr(obj, "__iter__")


def set_if_not_none(dct, key, val):
    if val is not None:
        dct[key] = val

def rad2deg(rad):
    return rad * 180 / np.pi


# Mad class helpers

class Par:
    def __init__(self, value, expr=None, madeval=None):
        self.value = value
        self.madeval = madeval
        if madeval is not None and isinstance(expr,str):
            expr = madeval(expr)
        self.expr = expr

    def __repr__(self):
        ss = f"{self.value}"
        if self.has_expr():
            ss = ss + "<-" + repr(self.expr)
        return ss

    def has_expr(self):
        return self.expr is not None

    def no_expr(self):
        return not self.has_expr()

    def not_zero(self):
        return self.value != 0 or self.has_expr()

    def __getitem__(self, k):
        return Par(self.value[k], self.expr[k], self.madeval)

    def apply_new(self, func, inplace=False):
        return Par(func(self.value), func(self.expr) if self.expr is not None else None, self.madeval)

    def apply_inplace(self, func):
        self.value = func(self.value)
        if self.expr is not None:
            self.expr = func(self.expr)

    def __neg__(self):
        return Par(-self.value, -self.expr if self.expr is not None else None, self.madeval)

    def __pos__(self):
        return Par(self.value, self.expr if self.expr is not None else None, self.madeval)

    def __abs__(self):
        return Par(abs(self.value), abs(self.expr) if self.expr is not None else None, self.madeval)

    def __add__(self, other):
        return self.apply_new(lambda x: x + other)

    def __sub__(self, other):
        return self.apply_new(lambda x: x - other)

    def __mul__(self, other):
        return self.apply_new(lambda x: x * other)

    def __truediv__(self, other):
        return self.apply_new(lambda x: x / other)

    def __pow__(self, other):
        return self.apply_new(lambda x: x**other)

    def __radd__(self, other):
        return self.apply_new(lambda x: other + x)

    def __rsub__(self, other):
        return self.apply_new(lambda x: other - x)

    def __rmul__(self, other):
        return self.apply_new(lambda x: other * x)

    def __rtruediv__(self, other):
        return self.apply_new(lambda x: other / x)

    def __rpow__(self, other):
        return self.apply_new(lambda x: other**x)

    def __isub__(self, other):
        return self.apply_inplace(lambda x: x - other)

    def __imul__(self, other):
        return self.apply_inplace(lambda x: x * other)

    def __iadd__(self, other):
        return self.apply_inplace(lambda x: x + other)

    def __ipow__(self, other):
        return self.apply_inplace(lambda x: x**other)

    def __itruediv__(self, other):
        return self.apply_inplace(lambda x: x / other)

    def __ifloordiv__(self, other):
        return self.apply_inplace(lambda x: x // other)
    
    def __eq__(self, other):
        return self.value == other.value and self.expr == other.expr

    def __ne__(self, other):
        return self.value != other.value or self.expr != other.expr

    def __lt__(self, other):
        if self.expr is not None:
            raise ValueError("Cannot compare Par with expression")
        return self.value < other.value

    def __le__(self, other):
        if self.expr is not None:
            raise ValueError("Cannot compare Par with expression")
        return self.value <= other.value

    def __gt__(self, other):
        if self.expr is not None:
            raise ValueError("Cannot compare Par with expression")
        return self.value > other.value or self.expr

    def __ge__(self, other):
        if self.expr is not None:
            raise ValueError("Cannot compare Par with expression")
        return self.value >= other.value or self.expr

    def __hash__(self):
        return hash((self.value, self.expr))

    def __nonzero__(self):
        return self.value != 0 or self.expr is not None


class ParList:
    def __init__(self, value, expr=(), madeval=None):
        self.value = value
        self.expr = expr
        self.madeval = madeval
        if madeval is not None:
            self.expr = [madeval(x) for x in expr if x is not None]


    def __repr__(self):
        ss = f"{self.value}"
        if self.has_expr():
            ss = ss + "<-" + repr(self.expr)
        return ss

    def has_expr(self):
        for ex in self.expr:
            if ex is not None:
                return True
        else:
            return False

    def no_expr(self):
        return not self.has_expr()

    def add_zeros(self, length):
        for ii in range(length - len(self.value)):
            self.value.append(0)
            self.expr.append(None)

    def __getitem__(self, k):
        return Par(self.value[k], self.expr[k], self.has_real_expr)

    def apply_new(self, func):
        new_val = [func(x) for x in self.value]
        new_expr = [func(x) for x in self.expr if x is not None]
        return Par(new_val, new_expr, self.madeval)

    def apply_inplace(self, func, inplace=False):
        self.value = [func(x) for x in self.value]
        self.expr = [func(x) for x in self.expr if x is not None]


    def __iter__(self):
        for ii in range(len(self.value)):
            yield Par(self.value[ii],self.expr[ii], self.madeval)

    def __len__(self):
        return len(self.value)

    def __iadd__(self, other):
        if iterable(other):
            if isinstance(other, ParList):
              for ii in range(min(len(other),len(self))):
                self.value[ii] += other.value[ii]
                if other.expr[ii] is not None:
                    if self.expr[ii] is None:
                        self.expr[ii] = other.expr[ii]+self.value[ii]
                    else:
                        self.expr[ii] += other.expr[ii]
                self.expr[ii]+= other[ii]
              for ii in range(len(self),len(other)):
                self.value.append(other.value[ii])
                self.expr.append(other.expr[ii])
            elif isinstance(other, Par):
               for ii in range(min(len(other),len(self))):
                self.value[ii] += other.value
                if other.expr is not None:
                    if self.expr[ii] is None:
                        self.expr[ii] = other.expr+self.value[ii]
                    else:
                        self.expr[ii] += other.expr
               self.value.extend(other.value[len(self):])
               self.expr.extend(other.expr[len(self):])
            else:
                self.value[ii] += other
                if self.expr[ii] is not None:
                    self.expr[ii] += other

class FieldErrors:
    def __init__(self, field_errors):
        self.dkn=np.array(self.dkn)
        self.dks=np.array(self.dks)

class PhaseErrors:
    def __init__(self, phase_errors):
        self.dpn=np.array(self.dpn)
        self.dps=np.array(self.dps)


def not_zero(x):
    if hasattr(x, '_get_value'):
        return True
    else:
        return x != 0


class MadElem:
    def __init_(self, name, elem, sequence, madeval=None):
        self.name = name
        self.elem = elem
        self.sequence = sequence
        self.madeval = madeval
        self.field_errors=FieldErrors(self.elem.field_errors) if self.elem.field_errors is not None else None
        self.phase_errors=PhaseErrors(self.elem.phase_errors) if self.elem.phase_errors is not None else None
        self.align_errors=self.elem.align_errors
 
    @property
    def type(self):
        return self.elem.base_type.name

    def __getattr__(self,k):
        par = self.elem.cmdpar[k]
        if isinstance(par.value, list):
            return ParList(par.value, par.expr, self.madeval)
        elif isinstance(par.value, str):
            return par.value  # no need to make a Par for strings
        elif self.madeval is not None and par.expr is not None:
            return self.madeval(par.expr)
        else:
            return par.value 

    def has_aperture(self):
        el=self.elem
        return hasattr(el, "aperture") and (el.aperture[0] != 0.0 or len(el.aperture) > 1)

    def is_empty_marker(self):
        return self.type == "marker" and not self.has_aperture()

    def same_aperture(self, other):
        return (self.aperture == other.aperture and
                self.aperture_offset == other.aperture_offset and 
                self.aper_tilt == other.aper_tilt and
                self.aper_vx == other.aper_vx and
                self.aper_vy == other.aper_vy and
                self.apertype==other.apertype)

    def merge_multipole(self,other):
        if self.same_aperture(other) and self.align_errors==other.align_errors:
            self.knl += other.knl
            self.ksl += other.ksl
            if self.field_errors is not None and other.field_errors is not None:
                for ii in range(len(self.field_errors['knl'])):
                    self.field_errors.knl[ii] += other.field_errors[ii]
            self.name=self.name+"_"+other.name


def get_value(x):
    if hasattr(x, '_get_value'):
        return x._get_value()
    else:
        return x


class XtElementBuilder:
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
        if hasattr(self,'attrs'):
            self.attrs[k] = v
        else:
            super().__setattr__(k,v)

    def add_to_line(self, line, buffer):
          attr_values = {k: get_value(v) for k, v in self.attrs.items()}
          xtel = self.type(**attr_values,_buffer=buffer)
          line.append_element(xtel, self.name)
          if self.enable_expressions:
              elref = line.element_refs[self.name]
              for k, p in self.attrs.items():
                  if isinstance(p,ParList):
                    for pi in p:
                        if pi is not None:
                            setattr(elref,k,pi)
                  if p.expr is not None:
                      setattr(elref, k, p.expr)
          return xtel

def init_line_expressions(line, mad): # to be added to Line....
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

class MadLoader:
    def __init__(self, sequence, enable_expressions=False,  enable_errors=True, error_table=None,
        skip_markers=False, merge_drifts=False, merge_multipoles=False, enable_apertures=True):
        self.sequence = sequence
        self.enable_expressions = enable_expressions
        self.enable_errors = enable_errors
        self.error_table = error_table
        self.skip_markers = skip_markers
        self.merge_drifts = merge_drifts
        self.merge_multipoles = merge_multipoles
        self.enable_apertures = enable_apertures

    
    def iter_elements(self, madeval=None):
        """Yield element data for each known element
        """
        last_element = MadElem(None, None, None)
        last_element.type="dummy"
        for el in self.sequence.expanded_elements:
            madelem=MadElem(el.name, el, self.sequence, madeval)
            if self.skip_markers and el.is_empty_marker():
                pass
            elif self.merge_drifts and last_element.type == "drift" and el.type == "drift":
                last_element.l += el.l
            elif self.merge_multipoles and last_element.type == "multipole" and el.type == "multipole":
                self.merge_multipole(last_element, el)
            else:
               yield last_element
               last_element=MadElem(el,self.sequence,madeval)
        yield last_element


    def make_line(self,sequence, buffer=None):
        """Create a new line in buffer
        """
        mad = sequence._madx

        if buffer is None:
            buffer = xobjects.context_default.new_buffer()

        line = xtrack.Line()

        if self.enable_expressions:
            madeval = init_line_expressions(line, mad)
        else:
            madeval = None

        for el in self.iter.elements(madeval=madeval):
            # for each mad element create xtract elements in a buffer and add to a line
            fn="convert_"+el.type
            if hasattr(self,fn):
                getattr(self,fn)(el,line,buffer)
            else:
                raise ValueError(f"Element {el.type} not supported,\n implement convert_{el.type} function in MadLoader")
        return line

    def add_elements(
        self, elements: List[XtElementBuilder], line, buffer, 
    ):
        out={} # tbc
        for el in elements:
            xtel = el.add_to_line(line, buffer)
            out[el.name] = xtel # tbc
        return out # tbc


    def add_thin_element_to_line(self, xtrack_el, mad_el, line, buffer):
        """add aperture and transformations to a thin element
           tilt, offset, aperture, offset, tilt, tilt, offset, kick, offset, tilt
        """
        elem_list = [xtrack_el]
        # very brittle, order matters here!!! 
        self.add_aperture_to_thin_element(mad_el,elem_list) # insert aperture before element
        self.add_alignment_to_thin_element(mad_el,elem_list) # enclose element with transformations
        self.add_permanent_misalignement_to_thin_element(mad_el,elem_list) # enclose everything with transformations
        xtrack_elements=self.add_elements_to_line(elem_list, line, buffer)


    def add_aperture_to_thin_element(self, mad_el, elem_list):
        if self.enable_apertures and mad_el.has_aperture():
            if len(mad_el.aper_vx) > 2:
                et=xtrack.LimitPolygon
                init={'x_vertices':mad_el.aper_vx,'y_vertices':mad_el.aper_vy}
            else:
                apertype = mad_el.aperture_type
                aperture = mad_el.aperture
                if apertype == "rectangle":
                    et = xtrack.LimitRect
                    init = {
                        "x_min": -aperture[0],
                        "x_max": aperture[0],
                        "y_min": -aperture[1],
                        "y_max": aperture[1],
                    }
                elif apertype == "racetrack":
                    et = xtrack.LimitRacetrack
                    init={
                        "min_x":-aperture[0],
                        "max_x":aperture[0],
                        "min_y":-aperture[1],
                        "max_y":aperture[1],
                        "a":aperture[2],
                        "b":aperture[3],
                    }
                ## TODO: add other aperture types
                else:
                   raise ValueError(f"Aperture type {apertype} not supported")
                out=[XtElementBuilder(mad_el.name + "_aperture", et, init)]
                if abs(mad_el.aper_offset[0])!=0:
                    out.insert(0,XtElementBuilder(mad_el.name + "_aper_offset_entry", xtrack.XYShift, {"dx":mad_el.aper_offset[0],"dy":mad_el.aper_offset[1]}))
                    out.append(XtElementBuilder(mad_el.name + "_aper_offset_exit", xtrack.XYShift, {"dx":-mad_el.aper_offset[0],"dy":-mad_el.aper_offset[1]}))
                if abs(mad_el.aper_tilt)!=0:
                    out.insert(0,XtElementBuilder(mad_el.name + "_aper_roll_entry", xtrack.SRotation, {"angle":mad_el.aper_tilt}))
                    out.append(XtElementBuilder(mad_el.name + "_aper_roll_exit", xtrack.SRotation, {"angle":-mad_el.aper_tilt}))
                elem_list.extend(out)




    def add_alignment_to_thin_element(self,mad_el, elem_list):
        tilt = rad2deg(mad_el.tilt)
        st=len(elem_list)
        if mad_el.align_errors is not None:
            tilt+=rad2deg(mad_el.align_errors.dpsi)
            dx = mad_el.align_errors.dx
            dy = mad_el.align_errors.dy
            if dx.not_zero() or dy.not_zero():
                elem_list.insert(st,XtElementBuilder(mad_el.name + "_offset_entry", xtrack.XYShift, {"dx":dx,"dy":dy}))
                elem_list.append(XtElementBuilder(mad_el.name + "_offset_entry", xtrack.XYShift, {"dx":-dx,"dy":-dy}))
        if tilt.not_zero():
            elem_list.insert(st, XtElementBuilder(mad_el.name +  + "_roll_entry", "SRotation", dict(angle=tilt)))
            elem_list.append(XtElementBuilder(mad_el.name +  + "_roll_exit", "SRotation", dict(angle=-tilt)))


    def add_permanent_misalignement_to_thin_element(self, elname, mad_data, elem_list):
        pass


    def convert_multipole(self,mad_elem,line,buffer):
        """Return Transform or None"""
        # getting max length of knl and ksl
        lmax=max(len(mad_elem.knl),len(mad_elem.ksl))
        if mad_elem.field_errors is not None and self.enable_errors:
            lmax=max(lmax,len(mad_elem.field_errors.knl),len(mad_elem.field_errors.ksl))
            mad_elem.knl.extend([0]*(lmax-len(mad_elem.knl)))
            mad_elem.ksl.extend([0]*(lmax-len(mad_elem.ksl)))

        el=XtElementBuilder(mad_elem.name,xtrack.Multipole,init={'order':lmax-1})
        el.knl=mad_elem.knl
        el.ksl=mad_elem.ksl
        if mad_elem.angle.not_zero():
            el.hxl=mad_elem.angle 
        else:
            el.hxl=mad_elem.knl[0]
        el.length=mad_elem.lrad
        self.add_thin_element(el, mad_elem, line, buffer)

    def convert_drift(self,mad_elem,line,buffer):
        el=XtElementBuilder(mad_elem.name,xtrack.Drift,init={'length':mad_elem.l})
        self.add_elements_to_line([el],line,buffer)

    def convert_marker(self,mad_elem,line,buffer):
        el = XtElementBuilder(  mad_elem.name, xtrack.Drift, init={'length':0}) 
        self.add_elements_to_line([el],line,buffer)





