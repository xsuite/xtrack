import numpy as np
import math

def get_amplitude_detuning(line, nemitt_x=1.e-6, nemitt_y=1.e-6, num_turns=100, zeropad=100000):
    '''
    returns axx, axy, ayx, ayy that correspond to the first order amplitude-detuning,
    estimated through tracking.
    The relations between tune (Q) and action (J) in this case is:
    $$ Q_x = Q_{x,0} + axx * Jx + axy * Jy $$
    $$ Q_y = Q_{y,0} + ayx * Jx + ayy * Jy $$
    '''

    egeom_x = nemitt_x / line.particle_ref.gamma0
    egeom_y = nemitt_y / line.particle_ref.gamma0

    frequency = np.fft.fftfreq(zeropad)

    sigma = 2
    num_r = 50
    JJ = np.linspace(0.01, sigma**2/2., num_r)
    A_norm = np.sqrt(2*JJ).flatten()
    other_norm = 0.01

    particles = line.build_particles(x_norm=A_norm, y_norm=other_norm,
                                     nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
    # line.tracker._context.synchronize()

    x = line.record_last_track.x
    y =  line.record_last_track.y

    qx = [abs(frequency[np.argmax(np.abs(
        np.fft.fft(x[ii]*np.hanning(x.shape[1]),n=zeropad)))])
        for ii in range(x.shape[0])]
    qy = [abs(frequency[np.argmax(np.abs(
        np.fft.fft(y[ii]*np.hanning(x.shape[1]), n=zeropad)))])
        for ii in range(x.shape[0])]

    axx = np.polyfit(JJ*egeom_x, qx, 1)[0]
    ayx = np.polyfit(JJ*egeom_x, qy, 1)[0]

    # switch x and y
    particles = line.build_particles(x_norm=other_norm, y_norm=A_norm,
                                     nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
    # line.tracker._context.synchronize()

    x = line.record_last_track.x
    y = line.record_last_track.y

    qx = [abs(frequency[np.argmax(np.abs(np.fft.fft(x[ii]*np.hanning(x.shape[1]), n=zeropad)))]) for ii in range(x.shape[0])]
    qy = [abs(frequency[np.argmax(np.abs(np.fft.fft(y[ii]*np.hanning(x.shape[1]), n=zeropad)))]) for ii in range(x.shape[0])]

    axy = np.polyfit(JJ*egeom_y, qx, 1)[0]
    ayy = np.polyfit(JJ*egeom_y, qy, 1)[0]

    return axx, axy, ayx, ayy

class Chromaticity(object):
    def __init__(self, deltas, qx, qy, order):
        self.deltas = deltas
        self.qx = qx
        self.qy = qy
        self.order = order
        factorials = np.array([math.factorial(ii) for ii in range(order + 1)])
        self.qx_derivatives = np.polynomial.polynomial.Polynomial.fit(deltas, qx, order).convert().coef / factorials
        self.qy_derivatives = np.polynomial.polynomial.Polynomial.fit(deltas, qy, order).convert().coef / factorials


def get_nonlinear_chromaticity(line, max_delta=1.e-3, npoints=21, order=2):
    '''
    returns an object containing chromaticity information.
    It includes the derivatives of tune with respect to delta (Q', Q'', ...), estimated 
    through off-momentum twiss evaluations
    $$ Q^{(n)}_x = \frac{\partial Q}{\partial \delta} $$.
    '''
    twiss4d = line.twiss(method='4d')
    dx = twiss4d.dx[0]
    dpx = twiss4d.dpx[0]
    dy = twiss4d.dy[0]
    dpy = twiss4d.dpy[0]

    deltas = np.linspace(-max_delta, max_delta, npoints)
    qx = np.zeros(len(deltas))
    qy = np.zeros(len(deltas))

    for ii,delta in enumerate(deltas):
        part_co_guess = line.particle_ref.copy()
        part_co_guess.x = delta*dx
        part_co_guess.px = delta*dpx
        part_co_guess.y = delta*dy
        part_co_guess.py = delta*dpy
        twiss = line.twiss(method='4d', delta0=delta, co_guess=part_co_guess)
        qx[ii] = twiss.qx
        qy[ii] = twiss.qy

    chromaticity = Chromaticity(deltas, qx, qy, order)
    return chromaticity
