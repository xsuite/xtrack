from math import factorial

import numpy as np

clight = 299792458
pi = np.pi


def bn_mad(bn_mad, n, sign):
    return sign * bn_mad * factorial(n - 1)


def bn_rel(bn16, bn3, r0, d0, sign):
    out = []
    for nn, (a, b) in enumerate(zip(bn16, bn3)):
        n = nn + 1
        sixval = d0 * a * b * r0 ** (1 - n) * 10 ** (3 * n - 6)
        out.append(bn_mad(sixval, n, sign))
    return out


def _expand_struct(sixinput, convert):
    elems = []
    count = {}
    icount = 0
    iconv = []
    names = []
    rest = []
    Drift = convert.Drift
    Multipole = convert.Multipole
    Cavity = convert.Cavity
    XYShift = convert.XYShift
    SRotation = convert.SRotation
    # Line = convert.Line
    BeamBeamBiGaussian2D = convert.BeamBeamBiGaussian2D
    BeamBeamBiGaussian3D = convert.BeamBeamBiGaussian3D
    RFMultipole = convert.RFMultipole
    exclude = False
    # add special elenents
    if "CAV" in sixinput.iter_struct():
        sixinput.single["CAV"] = [
            12 * sixinput.ition,
            sixinput.u0,
            sixinput.harm,
            0,
        ]
    for nnn in sixinput.iter_struct():
        exclude = False
        ccc = count.setdefault(nnn, 0)
        if len(sixinput.single[nnn]) == 7:
            etype, d1, d2, d3, _, _, _ = sixinput.single[nnn]
        else:
            etype, d1, d2, d3, = sixinput.single[nnn]
        elem = None
        if nnn in sixinput.align:
            dx, dy, tilt = sixinput.align[nnn][ccc]
            tilt = tilt * 180e-3 / pi
            dx *= 1e-3
            dy *= 1e-3
            hasshift = abs(dx) + abs(dy) > 0
            hastilt = abs(tilt) > 0
            if hasshift:
                names.append(nnn + "_preshift")
                elems.append(XYShift(dx=dx, dy=dy))
                icount += 1
            if hastilt:
                names.append(nnn + "_pretilt")
                elems.append(SRotation(angle=tilt))
                icount += 1
        if etype in [0, 25]:
            elem = Drift(length=d3)
            if d3 > 0:
                exclude = True
        elif abs(etype) in [1, 2, 3, 4, 5, 7, 8, 9, 10]:
            bn_six = d1
            nn = abs(etype)
            sign = -etype / nn
            madval = bn_mad(bn_six, nn, sign)
            knl = [0] * (nn - 1) + [madval]
            ksl = [0] * nn
            if sign == 1:
                knl, ksl = ksl, knl
            elem = Multipole(knl=knl, ksl=ksl, hxl=0, hyl=0, length=0)
        elif etype == 11:
            knl, ksl = sixinput.get_knl(nnn, ccc)
            hxl = 0
            hyl = 0
            length = 0
            # beaware of the case of thick bend
            # see beambeam example where mbw has the length
            if d3 == -1:
                hxl = -d1
                length = d2
                knl[0] = hxl
            elif d3 == -2:
                hyl = -d1  # strange sign!!!
                length = d2
                ksl[0] = hyl
            elem = Multipole(knl=knl, ksl=ksl, hxl=hxl, hyl=hyl, length=length)
        elif etype == 12:
            e0=sixinput.initialconditions[-1]
            p0c=np.sqrt(e0**2-sixinput.pma**2)
            beta0=p0c/e0
            v = d1 * 1e6
            freq = d2 * clight * beta0 / sixinput.tlen
            #print(v,freq)
            elem = Cavity(voltage=v, frequency=freq, lag=180 - d3)
        elif etype == 20:
            thisbb = sixinput.bbelements[nnn]
            dct={}
            if hasattr(thisbb, "sigma_x"):
                from scipy.constants import e as qe
                dct['n_particles'] = thisbb.charge/qe # ducktrack has it in coulumb
                dct['q0'] = qe # TODO change implementation
                dct['beta0'] = thisbb.beta_r
                dct['mean_x'] = thisbb.x_bb
                dct['mean_y'] = thisbb.y_bb
                dct['sigma_x'] = thisbb.sigma_x
                dct['sigma_y'] = thisbb.sigma_y
                dct['d_px'] = thisbb.d_px
                dct['d_py'] = thisbb.d_py
                elem = convert.BeamBeamBiGaussian2D(**dct)
            elif hasattr(thisbb, "phi"):
                elem = convert.BeamBeamBiGaussian3D(old_interface=thisbb._asdict())
            else:
                raise ValueError("What?!")
        elif etype == 23:
            #print(nnn, sixinput.single[nnn])
            voltage_V = d1 *1e6
            freq_Hz = d2 * 1e6
            phase_rad = d3
            p0c_eV = sixinput.initialconditions[12]*1e6
            elem = RFMultipole(
                frequency = freq_Hz,
                knl= [voltage_V / p0c_eV],
                pn=[90.],
                )
        elif etype == -23:
            voltage_V = d1 *1e6
            freq_Hz = d2 * 1e6
            phase_rad = d3
            p0c_eV = sixinput.initialconditions[12]*1e6
            print(nnn, sixinput.single[nnn])
            print(f'p0c_eV: {p0c_eV}')
            elem = RFMultipole(
                frequency = freq_Hz,
                ksl= [-voltage_V / p0c_eV],
                ps=[90.],
                )
        else:
            rest.append([nnn] + sixinput.single[nnn])
        if elem is not None:
            elems.append(elem)
            names.append(nnn)
        if nnn in sixinput.align:
            if hastilt:
                names.append(nnn + "_posttilt")
                elems.append(SRotation(angle=-tilt))
                icount += 1
            if hasshift:
                names.append(nnn + "_postshift")
                elems.append(XYShift(dx=-dx, dy=-dy))
                icount += 1
        if elem is not None:
            if not exclude:
                iconv.append(icount)
            icount += 1
        count[nnn] = ccc + 1
    # newelems = [dict(i._asdict()) for i in elems]
    types = [i.__class__.__name__ for i in elems]
    return list(zip(names, types, elems)), rest, iconv
