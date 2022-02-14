import numpy as np
import xtrack as xt

def madx_sequence_to_xtrack_line(
    sequence,
    classes,
    ignored_madtypes=[],
    exact_drift=False,
    drift_threshold=1e-6,
    install_apertures=False,
    deferred_expressions=False
):

    if exact_drift:
        myDrift = classes.DriftExact
    else:
        myDrift = classes.Drift
    seq = sequence


    line = xt.Line(elements=[], element_names=[])

    if deferred_expressions:
        line._init_var_management()
        mad = sequence._madx

        from xdeps.madxutils import MadxEval

        # Extract globals values from madx
        _var_values = line._var_management['data']['var_values']
        for name,par in mad.globals.cmdpar.items():
            _var_values[name]=par.value

        # Extract element values from madx
        _mad_elements_dct= line._var_management['data']['mad_elements_dct']
        for name,elem in mad.elements.items():
            elemdata={}
            for parname, par in elem.cmdpar.items():
                elemdata[parname]=par.value
            _mad_elements_dct[name]=elemdata
            _mad_elements_dct[name]['__basetype__'] = elem.base_type.name

        _ref_manager = line._var_management['manager']
        _vref = line._var_management['vref']
        _fref = line._var_management['fref']
        _lref = line._var_management['lref']
        _eref = line._var_management['eref']
        madeval=MadxEval(_vref,_fref,_eref).eval

        # Extract expressions from madx globals
        for name,par in mad.globals.cmdpar.items():
            if par.expr is not None:
                _vref[name]=madeval(par.expr)

        # Extract expressions from madx elements
        for name,elem in mad.elements.items():
            for parname, par in elem.cmdpar.items():
                if par.expr is not None:
                    if par.dtype==12: # handle lists
                        for ii,ee in enumerate(par.expr):
                            if ee is not None:
                                _eref[name][parname][ii]=madeval(ee)
                    else:
                        _eref[name][parname]=madeval(par.expr)

    elements = seq.elements
    ele_pos = seq.element_positions()

    old_pp = 0.0
    i_drift = 0
    for pp, ee in sorted(zip(ele_pos,elements),key=lambda x:x[0]):
        skiptilt=False

        if pp > old_pp + drift_threshold:
            line.append_element(myDrift(length=(pp - old_pp)), f"drift_{i_drift}")
            old_pp = pp
            i_drift += 1

        eename = ee.name
        mad_etype = ee.base_type.name

        if mad_etype in [
            "marker",
            "monitor",
            "hmonitor",
            "vmonitor",
            "collimator",
            "rcollimator",
            "elseparator",
            "instrument",
            "solenoid",
            "drift",
        ]:
            newele = myDrift(length=ee.l)
            old_pp += ee.l
            line.element_dict[eename] = newele

        elif mad_etype in ignored_madtypes:
            pass

        elif mad_etype == "multipole":
            knl = ee.knl if hasattr(ee, "knl") else [0]
            ksl = ee.ksl if hasattr(ee, "ksl") else [0]
            newele = classes.Multipole(
                knl=list(knl),
                ksl=list(ksl),
                hxl=knl[0],
                hyl=ksl[0],
                length=ee.lrad,
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                for ii, _ in enumerate(knl):
                    _lref[eename].knl[ii] = _eref[eename]['knl'][ii]
                for ii, _ in enumerate(ksl):
                    _lref[eename].ksl[ii] = _eref[eename]['ksl'][ii]

        elif mad_etype == "tkicker" or mad_etype == "kicker":
            hkick = [-ee.hkick] if hasattr(ee, "hkick") else []
            vkick = [ee.vkick] if hasattr(ee, "vkick") else []
            newele = classes.Multipole(
                knl=hkick, ksl=vkick, length=ee.lrad, hxl=0, hyl=0
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                if hasattr(ee, 'hkick'):
                    _lref[eename].knl[0] = -_eref[eename]['hkick']
                if hasattr(ee, 'vkick'):
                    _lref[eename].ksl[0] = _eref[eename]['vkick']

        elif mad_etype == "vkicker":
            newele = classes.Multipole(
                knl=[], ksl=[ee.kick], length=ee.lrad, hxl=0, hyl=0
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                _lref[eename].ksl[0] = _eref[eename]['kick']

        elif mad_etype == "hkicker":
            newele = classes.Multipole(
                knl=[-ee.kick], ksl=[], length=ee.lrad, hxl=0, hyl=0
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                _lref[eename].knl[0] = -_eref[eename]['kick']

        elif mad_etype == "dipedge":
            newele = classes.DipoleEdge(
                h=ee.h, e1=ee.e1, hgap=ee.hgap, fint=ee.fint
            )
            line.element_dict[eename] = newele

        elif mad_etype == "rfcavity":
            newele = classes.Cavity(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                _lref[eename].voltage = _eref[eename]['volt'] * 1e6
                _lref[eename].frequency = _eref[eename]['freq'] * 1e6
                _lref[eename].lag = _eref[eename]['lag'] * 360

        elif mad_etype == "rfmultipole":
            newele = classes.RFMultipole(
                voltage=ee.volt * 1e6,
                frequency=ee.freq * 1e6,
                lag=ee.lag * 360,
                knl=ee.knl[:],
                ksl=ee.ksl[:],
                pn=[v * 360 for v in ee.pnl],
                ps=[v * 360 for v in ee.psl],
            )
            line.element_dict[eename] = newele
            if deferred_expressions:
                _lref[eename].voltage = _eref[eename]['volt'] * 1e6
                _lref[eename].frequency = _eref[eename]['freq'] * 1e6
                _lref[eename].lag = _eref[eename]['lag'] * 360
                for ii, _ in enumerate(ee.knl):
                    _lref[eename].knl[ii] = _eref[eename]['knl'][ii]
                for ii, _ in enumerate(ee.ksl):
                    _lref[eename].ksl[ii] = _eref[eename]['ksl'][ii]
                for ii, _ in enumerate(ee.pnl):
                    _lref[eename].pn[ii] = _eref[eename]['pnl'][ii] * 360
                for ii, _ in enumerate(ee.psl):
                    _lref[eename].ps[ii] = _eref[eename]['psl'][ii] * 360

        elif mad_etype == "crabcavity":
            #ee.volt in MV, sequence.beam.pc in GeV
            if abs(ee.tilt-np.pi/2)<1e-9:
                newele = classes.RFMultipole(
                    frequency=ee.freq * 1e6,
                    ksl=[-ee.volt / sequence.beam.pc*1e-3],
                    ps=[ee.lag * 360 + 90],
                )
                line.element_dict[eename] = newele
                skiptilt=True

                if deferred_expressions:
                    _lref[eename].frequency = _eref[eename]['freq'] * 1e6
                    _lref[eename].ksl[0] = (-_eref[eename]['volt']
                                            / sequence.beam.pc * 1e-3)
                    _lref[eename].ps[0] = _eref[eename]['lag'] * 360 + 90
            else:
                newele = classes.RFMultipole(
                    frequency=ee.freq * 1e6,
                    knl=[ee.volt / sequence.beam.pc*1e-3],
                    pn=[ee.lag * 360 + 90], # TODO: Changed sign to match sixtrack
                                            # To be checked!!!!
                )
                line.element_dict[eename] = newele

                if deferred_expressions:
                    _lref[eename].frequency = _eref[eename]['freq'] * 1e6
                    _lref[eename].knl[0] = (_eref[eename]['volt']
                                            / sequence.beam.pc * 1e-3)
                    _lref[eename].pn[0] = _eref[eename]['lag'] * 360 + 90

        elif mad_etype == "beambeam":
            if ee.slot_id == 6 or ee.slot_id == 60:
                # BB interaction is 6D

                import xfields as xf
                newele = xf.BeamBeamBiGaussian3D(old_interface={
                    'phi': 0.0,
                    'alpha': 0.0,
                    'x_bb_co': 0.0,
                    'y_bb_co': 0.0,
                    'charge_slices': [0.0],
                    'zeta_slices': [0.0],
                    'sigma_11': 1.0,
                    'sigma_12': 0.0,
                    'sigma_13': 0.0,
                    'sigma_14': 0.0,
                    'sigma_22': 1.0,
                    'sigma_23': 0.0,
                    'sigma_24': 0.0,
                    'sigma_33': 0.0,
                    'sigma_34': 0.0,
                    'sigma_44': 0.0,
                    'x_co': 0.0,
                    'px_co': 0.0,
                    'y_co': 0.0,
                    'py_co': 0.0,
                    'zeta_co': 0.0,
                    'delta_co': 0.0,
                    'd_x': 0.0,
                    'd_px': 0.0,
                    'd_y': 0.0,
                    'd_py': 0.0,
                    'd_zeta': 0.0,
                    'd_delta': 0.0,})
            else:
                # BB interaction is 4D
                import xfields as xf
                newele = xf.BeamBeamBiGaussian2D(
                    n_particles=0.,
                    q0=0.,
                    beta0=1.,
                    mean_x=0.,
                    mean_y=0.,
                    sigma_x=1.,
                    sigma_y=1.,
                    d_px=0,
                    d_py=0)

            line.element_dict[eename] = newele

        elif mad_etype == "placeholder":
            if ee.slot_id == 1:
                newele = classes.SCCoasting()
            elif ee.slot_id == 2:
                # TODO Abstraction through `classes` to be introduced
                import xfields as xf
                lprofile = xf.LongitudinalProfileQGaussian(
                        number_of_particles=0.,
                        sigma_z=1.,
                        z0=0.,
                        q_parameter=1.)
                newele = xf.SpaceChargeBiGaussian(
                    length=0,
                    apply_z_kick=False,
                    longitudinal_profile=lprofile,
                    mean_x=0.,
                    mean_y=0.,
                    sigma_x=1.,
                    sigma_y=1.)

            elif ee.slot_id == 3:
                newele = classes.SCInterpolatedProfile()
            else:
                newele = myDrift(length=ee.l)
                old_pp += ee.l
            line.element_dict[eename] = newele
        else:
            raise ValueError(f'MAD element "{mad_etype}" not recognized')


        if hasattr(ee,'tilt') and abs(ee.tilt)>0 and not skiptilt:
            tilt=np.rad2deg(ee.tilt)
        else:
            tilt=0

        if abs(tilt)>0:
            line.append_element(classes.SRotation(angle=tilt), eename+"_pretilt")

        line.element_names.append(eename)

        if abs(tilt)>0:
            line.append_element(classes.SRotation(angle=-tilt), eename+"_posttilt")

        if (
            install_apertures
            and hasattr(ee, "aperture")
            and (min(ee.aperture) > 0)
        ):
            if ee.apertype == "rectangle":
                newaperture = classes.LimitRect(
                    min_x=-ee.aperture[0],
                    max_x=ee.aperture[0],
                    min_y=-ee.aperture[1],
                    max_y=ee.aperture[1],
                )
            elif ee.apertype == "racetrack":
                newaperture = classes.LimitRacetrack(
                    min_x=-ee.aperture[0],
                    max_x=ee.aperture[0],
                    min_y=-ee.aperture[1],
                    max_y=ee.aperture[1],
                    a=ee.aperture[2],
                    b=ee.aperture[3],
                )
            elif ee.apertype == "ellipse":
                newaperture = classes.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[1]
                )
            elif ee.apertype == "circle":
                newaperture = classes.LimitEllipse(
                    a=ee.aperture[0], b=ee.aperture[0]
                )
            elif ee.apertype == "rectellipse":
                newaperture = classes.LimitRectEllipse(
                    max_x=ee.aperture[0],
                    max_y=ee.aperture[1],
                    a=ee.aperture[2],
                    b=ee.aperture[3],
                )
            elif ee.apertype == "octagon":
                a0 = ee.aperture[0]
                a1 = ee.aperture[1]
                a2 = ee.aperture[2]
                a3 = ee.aperture[3]
                V1 = (a0, a0*np.tan(a2))
                V2 = (a1/np.tan(a3), a1)
                newaperture = classes.LimitPolygon(
                    x_vertices = [V1[0],  V2[0],
                                 -V2[0], -V1[0],
                                 -V1[0], -V2[0],
                                  V2[0],  V1[0]],
                    y_vertices = [V1[1],  V2[1],
                                  V2[1],  V1[1],
                                 -V1[1], -V2[1],
                                 -V2[1], -V1[1]],
                )
            else:
                raise ValueError("Aperture type not recognized")

            line.append_element(newaperture, eename + "_aperture")

    if hasattr(seq, "length") and seq.length > old_pp:
        line.append_element(myDrift(length=(seq.length - old_pp)), f"drift_{i_drift}")

    return line


class MadPoint(object):
    @classmethod
    def from_survey(cls, name, mad):
        return cls(name, mad, use_twiss=False, use_survey=True)

    @classmethod
    def from_twiss(cls, name, mad):
        return cls(name, mad, use_twiss=True, use_survey=False)

    def __init__(self, name, mad, use_twiss=True, use_survey=True):

        self.use_twiss = use_twiss
        self.use_survey = use_survey

        if not (use_survey) and not (use_twiss):
            raise ValueError(
                "use_survey and use_twiss cannot be False at the same time"
            )

        self.name = name
        if use_twiss:
            twiss = mad.table.twiss
            names = twiss.name
        if use_survey:
            survey = mad.table.survey
            names = survey.name

        idx = np.where(names == name)[0][0]

        if use_twiss:
            self.tx = twiss.x[idx]
            self.ty = twiss.y[idx]
            self.tpx = twiss.px[idx]
            self.tpy = twiss.py[idx]
        else:
            self.tx = None
            self.ty = None
            self.tpx = None
            self.tpy = None

        if use_survey:
            self.sx = survey.x[idx]
            self.sy = survey.y[idx]
            self.sz = survey.z[idx]
            self.sp = np.array([self.sx, self.sy, self.sz])
            theta = survey.theta[idx]
            phi = survey.phi[idx]
            psi = survey.psi[idx]
        else:
            self.sx = None
            self.sy = None
            self.sz = None
            self.sp = None
            theta = 0.0
            phi = 0.0
            psi = 0.0

        thetam = np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )
        phim = np.array(
            [
                [1, 0, 0],
                [0, np.cos(phi), np.sin(phi)],
                [0, -np.sin(phi), np.cos(phi)],
            ]
        )
        psim = np.array(
            [
                [np.cos(psi), -np.sin(psi), 0],
                [np.sin(psi), np.cos(psi), 0],
                [0, 0, 1],
            ]
        )
        wm = np.dot(thetam, np.dot(phim, psim))
        self.ex = np.dot(wm, np.array([1, 0, 0]))
        self.ey = np.dot(wm, np.array([0, 1, 0]))
        self.ez = np.dot(wm, np.array([0, 0, 1]))

        self.p = np.array([0.0, 0.0, 0.0])

        if use_twiss:
            self.p += self.ex * self.tx + self.ey * self.ty

        if use_survey:
            self.p += self.sp

    def dist(self, other):
        return np.sqrt(np.sum((self.p - other.p) ** 2))

    def distxy(self, other):
        dd = self.p - other.p
        return np.dot(dd, self.ex), np.dot(dd, self.ey)


