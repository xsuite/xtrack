# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
import xobjects as xo
from ._common import ElectronCoolerRecord

class ElectronCooler(BeamElement):
    """
    Beam element modeling an electron cooler. In particular, this beam element uses the Parkhomchuk model for electron cooling.
    Every turn each particle receives transverse and longitudinal kicks based on the cooling force provided by the Parkhomchuk model.


    Parameters
        ----------
        current : float, optional
            The current in the electron beam, in amperes.
        length  : float, optional
            The length of the electron cooler, in meters.
        radius_e_beam : float, optional
            The radius of the electron beam, in meters.
        temp_perp : float, optional
            The transverse temperature of the electron beam, in electron volts.
        temp_long : float, optional
            The longitudinal temperature of the electron beam, in electron volts.
        magnetic_field : float, optional
            The magnetic field strength, in tesla.
        offset_x : float, optional
            The horizontal offset of the electron cooler, in meters.
        offset_px : float, optional
            The horizontal angle of the electron cooler, in rad.
        offset_y : float, optional
            The horizontal offset of the electron cooler, in meters.
        offset_py : float, optional
            The vertical angle of the electron cooler, in rad.
        offset_energy : float, optional
            The energy offset of the electrons, in eV.
        magnetic_field_ratio : float, optional
            The ratio of perpendicular component of magnetic field with the
            longitudinal component of the magnetic field. This is a measure
            of the magnetic field quality. With the ideal magnetic field quality
            being 0.
        space_charge : float, optional
            Whether space charge of electron beam is enabled. 0 is off and 1 is on.

    """

    _xofields = {
        'current'       :  xo.Float64,
        'length'        :  xo.Float64,
        'radius_e_beam' :  xo.Float64,
        'temp_perp'     :  xo.Float64,
        'temp_long'     :  xo.Float64,
        'magnetic_field':  xo.Float64,

        'offset_x'      :  xo.Float64,
        'offset_px'     :  xo.Float64,
        'offset_y'      :  xo.Float64,
        'offset_py'     :  xo.Float64,
        'offset_energy' :  xo.Float64,

        'magnetic_field_ratio' :  xo.Float64,
        'space_charge_factor'  : xo.Float64,
        'record_flag': xo.Int64,
        }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/electroncooler.h"',
    ]

    _internal_record_class = ElectronCoolerRecord

    def __init__(self,  current        = 0,
                        length         = 0,
                        radius_e_beam  = 0,
                        temp_perp      = 0,
                        temp_long      = 0,
                        magnetic_field = 0,

                        offset_x       = 0,
                        offset_px      = 0,
                        offset_y       = 0,
                        offset_py      = 0,
                        offset_energy  = 0,

                        magnetic_field_ratio = 0,
                        space_charge_factor  = 0,
                        record_flag          =0,
                        **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        super().__init__(**kwargs)
        self.current        = current
        self.length         = length
        self.radius_e_beam  = radius_e_beam
        self.temp_perp      = temp_perp
        self.temp_long      = temp_long
        self.magnetic_field = magnetic_field

        self.offset_x       = offset_x
        self.offset_px      = offset_px
        self.offset_y       = offset_y
        self.offset_py      = offset_py
        self.offset_energy  = offset_energy

        self.magnetic_field_ratio = magnetic_field_ratio
        self.space_charge_factor  = space_charge_factor
        self.record_flag          =  record_flag

    def get_backtrack_element(self, _context=None, _buffer=None, _offset=None):
        raise NotImplementedError
