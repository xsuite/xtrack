import numpy as np
from scipy import integrate
from scipy.constants import c as clight
from scipy.constants import e as qe


def beta(s, beta0, alpha_s0):
    '''Beta function in drift space'''
    return beta0-2*alpha_s0*s+(1+alpha_s0**2)/beta0*s**2

def dispersion(s, d0, dp0):
    '''Dispersion in drift space'''
    return d0+s*dp0

def sigma(beta, epsilon0, betagamma):
    '''Betatronic sigma'''
    return np.sqrt(beta*epsilon0/betagamma)

def luminosity(f, nb,
      N1, N2,
      x_1, x_2,
      y_1, y_2,
      px_1, px_2,
      py_1, py_2,
      energy_tot1, energy_tot2,
      deltap_p0_1, deltap_p0_2,
      epsilon_x1, epsilon_x2,
      epsilon_y1, epsilon_y2,
      sigma_z1, sigma_z2,
      beta_x1, beta_x2,
      beta_y1, beta_y2,
      alpha_x1, alpha_x2,
      alpha_y1, alpha_y2,
      dx_1, dx_2,
      dy_1, dy_2,
      dpx_1, dpx_2,
      dpy_1, dpy_2,
      crab_crossing=None,
      verbose=False, sigma_integration=3, rest_mass_b1=0.93827231, rest_mass_b2=0.93827231):
    '''
    Returns luminosity in Hz/cm^2.
    f: revolution frequency
    nb: number of colliding bunch per beam in the specific Interaction Point (IP).
    N1,2: B1,2 number of particle per bunch
    x,y,1,2: horizontal/vertical position at the IP of B1,2, as defined in MADX [m]
    px,y,1,2: px,py at the IP of B1,2, as defined in MADX
    energy_tot1,2: total energy of the B1,2 [GeV]
    deltap_p0_1,2: rms momentum spread of B1,2 (formulas assume Gaussian off-momentum distribution)
    epsilon_x,y,1,2: horizontal/vertical normalized emittances of B1,2 [m rad]
    sigma_z1,2: rms longitudinal spread in z of B1,2 [m]
    beta_x,y,1,2: horizontal/vertical beta-function at IP of B1,2 [m]
    alpha_x,y,1,2: horizontal/vertical alpha-function at IP of B1,2
    dx,y_1,2: horizontal/vertical dispersion-function at IP of B1,2, as defined in MADX [m]
    dpx,y_1,2: horizontal/vertical differential-dispersion-function IP of B1,2, as defined in MADX
    CC_V_x,y,1,2: B1,2 H/V CC total of the cavities that the beam sees before reaching the IP [V]
    CC_f_x,y,1,2: B1,2 H/V CC frequency of cavities that the beam sees before reaching the IP [Hz]
    CC_phase_1,2: B1,2 H/V CC phase of cavities that the beam sees before reaching the IP.
        Sinusoidal function with respect to the center of the bunch is assumed.
        Therefore 0 rad means no kick for the central longitudinal slice [rad]
    RAB_1,2: B1,2 equivalent H/V linear transfer matrix coefficients between the CC
        that the beam sees before reaching the IP and IP itself [SI units]
    verbose: to have verbose output
    sigma_integration: the number of sigma consider for the integration
        (taken into account only if CC(s) is/are present)
    rest_mass_b1, rest_mass_b2: rest mass in GeV
    In MAD-X px is p_x/p_0 (p_x is the x-component of the momentum and p_0 is the design momentum).
    In our approximation we use the paraxial approximation: p_0~p_z so px is an angle.
    Similar arguments holds for py.
    In MAD-X, dx and dpx are the literature dispersion and is derivative in s divided by the relatistic beta.
    In fact, since pt=beta*deltap, where beta is the relativistic Lorentz factor,
    those functions given by MAD-X must be multiplied by beta a number of times equal to the order of
    the derivative to find the functions given in the literature.
    To note that dpx is normalized by the reference momentum (p_s) and not the design momentum (p_0),
    ps = p0(1+deltap). We assume that dpx is the z-derivative of the px.
    '''

    if deltap_p0_1 != 0 or deltap_p0_2 != 0:
        raise ValueError('effect of dispersion not included yet (untested).')

    gamma1 = energy_tot1 / rest_mass_b1
    br_1 = np.sqrt(1-1/gamma1**2)
    betagamma_1 = br_1*gamma1

    gamma2 = energy_tot2 / rest_mass_b2
    br_2 = np.sqrt(1-1/gamma2**2)
    betagamma_2 = br_2*gamma2

    # module of B1 speed
    v0_1=br_1*clight
    # paraxial hypothesis
    vx_1=v0_1*px_1
    vy_1=v0_1*py_1
    vz_1=v0_1*np.sqrt(1-px_1**2-py_1**2)
    v_1=np.array([vx_1, vy_1, vz_1])

    v0_2=br_2*clight # module of B2 speed
    # Assuming counter rotating B2 ('-' sign)
    vx_2=-v0_2*px_2
    vy_2=-v0_2*py_2
    # assuming px_2**2+py_2**2 < 1
    vz_2=-v0_2*np.sqrt(1-px_2**2-py_2**2)
    v_2=np.array([vx_2, vy_2, vz_2])

    if verbose:
        print(f'B1 velocity vector:{v_1}')
        print(f'B2 velocity vector:{v_2}')

    diff_v = v_1-v_2
    cross_v= np.cross(v_1, v_2)

    # normalized to get 1 for the ideal case 
    # NB we assume px_1 and py_1 constant along the z-slices 
    # NOT TRUE FOR CC! In any case the Moeller efficiency is almost equal to 1 in most cases...
    Moeller_efficiency=np.sqrt(clight**2*np.dot(diff_v,diff_v)-np.dot(cross_v,cross_v))/clight**2/2

    def sx1(s):
        '''The sigma_x of B1, quadratic sum of betatronic and dispersive sigma'''
        return np.sqrt(sigma(beta(s, beta_x1, alpha_x1), epsilon_x1, betagamma_1)**2 \
                       + (dispersion(s, br_1*dx_1, br_1*dpx_1)*deltap_p0_1)**2)

    def sy1(s):
        '''The sigma_y of B1, quadratic sum of betatronic and dispersive sigma'''
        return np.sqrt(sigma(beta(s, beta_y1, alpha_y1), epsilon_y1, betagamma_1)**2 \
                       + (dispersion(s, br_1*dy_1, br_1*dpy_1)*deltap_p0_1)**2)

    def sx2(s):
        '''The sigma_x of B2, quadratic sum of betatronic and dispersive sigma'''
        return np.sqrt(sigma(beta(s, beta_x2, alpha_x2), epsilon_x2, betagamma_2)**2 \
                       + (dispersion(s, br_2*dx_2, br_2*dpx_2)*deltap_p0_2)**2)

    def sy2(s):
        '''The sigma_y of B2, quadratic sum of betatronic and dispersive sigma'''
        return np.sqrt(sigma(beta(s, beta_y2, alpha_y2), epsilon_y2, betagamma_2)**2 \
                       + (dispersion(s, br_2*dy_2,  br_2*dpy_2)*deltap_p0_2)**2)

    sigma_z=np.max([sigma_z1, sigma_z2])

    if crab_crossing is not None and 'CC_V_x_1' in crab_crossing.keys():

        raise NotImplementedError('Crab crossing not tested yet')

        CC_V_x_1 = crab_crossing['CC_V_x_1']
        CC_V_y_1 = crab_crossing['CC_V_y_1']
        CC_V_x_2 = crab_crossing['CC_V_x_2']
        CC_V_y_2 = crab_crossing['CC_V_y_2']

        CC_phase_x_1 = crab_crossing['CC_phase_x_1']
        CC_phase_y_1 = crab_crossing['CC_phase_y_1']
        CC_phase_x_2 = crab_crossing['CC_phase_x_2']
        CC_phase_y_2 = crab_crossing['CC_phase_y_2']

        CC_f_x_1 = crab_crossing['CC_f_x_1']
        CC_f_y_1 = crab_crossing['CC_f_y_1']
        CC_f_x_2 = crab_crossing['CC_f_x_2']
        CC_f_y_2 = crab_crossing['CC_f_y_2']

        R12_1 = crab_crossing['R12_1']
        R22_1 = crab_crossing['R22_1']
        R12_2 = crab_crossing['R12_2']
        R22_2 = crab_crossing['R22_2']
        R34_1 = crab_crossing['R34_1']
        R44_1 = crab_crossing['R44_1']
        R34_2 = crab_crossing['R34_2']
        R44_2 = crab_crossing['R44_2']

        def theta_x_1(delta_z):
            # Eq. 3 of https://espace.cern.ch/acc-tec-sector/Chamonix/Chamx2012/papers/RC_9_04.pdf
            return CC_V_x_1/energy_tot1/1e9*np.sin(CC_phase_x_1 + 2*np.pi*CC_f_x_1/clight * delta_z)

        def theta_y_1(delta_z):
            return CC_V_y_1/energy_tot1/1e9*np.sin(CC_phase_y_1 + 2*np.pi*CC_f_y_1/clight * delta_z)

        def theta_x_2(delta_z):
            return CC_V_x_2/energy_tot2/1e9*np.sin(CC_phase_x_2 + 2*np.pi*CC_f_x_2/clight * delta_z)

        def theta_y_2(delta_z):
            return CC_V_y_2/energy_tot2/1e9*np.sin(CC_phase_y_2 + 2*np.pi*CC_f_y_2/clight * delta_z)

        def mx1(s, t):
            '''The mu_x of B1 as straight line'''
            return x_1 + R12_1*theta_x_1(s-clight * t) + (px_1+R22_1*theta_x_1(s-clight * t))*s

        def my1(s, t):
            '''The mu_y of B1 as straight line'''
            return y_1 + R34_1*theta_y_1(s-clight * t) + (py_1+R44_1*theta_y_1(s-clight * t))*s

        def mx2(s, t):
            '''The mu_x of B2 as straight line'''
            return x_2 + R12_2*theta_x_2(s+clight * t) + (px_2+R22_2*theta_x_2(s+clight * t))*s

        def my2(s, t):
            '''The mu_y of B2 as straight line'''
            return y_2 + R34_2*theta_y_2(s+clight * t) + (py_2+R44_2*theta_y_2(s+clight * t))*s

        def kernel_double_integral(t, s):
            return np.exp(0.5*(-(mx1(s, t) - mx2(s, t))**2/(sx1(s)**2 + sx2(s)**2) \
                               -(my1(s, t) - my2(s, t))**2/(sy1(s)**2 + sy2(s)**2) \
                               -(-br_1*clight * t+s)**2/(sigma_z1**2) \
                               -( br_2*clight * t+s)**2/(sigma_z2**2))) \
        /np.sqrt((sx1(s)**2 + sx2(s)**2)*(sy1(s)**2 + sy2(s)**2))/sigma_z1/sigma_z2

        integral=integrate.dblquad((lambda t, s: kernel_double_integral(t, s)),
                                   -sigma_integration*sigma_z, sigma_integration*sigma_z,-sigma_integration*sigma_z/c, sigma_integration*sigma_z/c)
        L0=f*N1*N2*nb * clight/2/np.pi**(2)*integral[0]

    elif crab_crossing is not None and 'phi_crab_x_1' in crab_crossing:

        raise ValueError('Not implemented yet') # Needs testing

        phi_crab_x_1 = crab_crossing['phi_crab_x_1']
        phi_crab_y_1 = crab_crossing['phi_crab_y_1']
        phi_crab_x_2 = crab_crossing['phi_crab_x_2']
        phi_crab_y_2 = crab_crossing['phi_crab_y_2']

        def mx1(s, t):
            '''The mu_x of B1 as straight line'''
            return x_1 + px_1 * s + phi_crab_x_1 * clight * t

        def my1(s, t):
            '''The mu_y of B1 as straight line'''
            return y_1 + py_1 * s + phi_crab_y_1 * clight * t

        # Signs of the crab terms are guessed...
        def mx2(s, t):
            '''The mu_x of B2 as straight line'''
            return x_2 + px_2 * s + phi_crab_x_2 * clight * t

        def my2(s, t):
            '''The mu_y of B2 as straight line'''
            return y_2 + py_2 * s + phi_crab_y_2 * clight * t

        def kernel_double_integral(t, s):
            return np.exp(0.5*(-(mx1(s, t) - mx2(s, t))**2/(sx1(s)**2 + sx2(s)**2) \
                               -(my1(s, t) - my2(s, t))**2/(sy1(s)**2 + sy2(s)**2) \
                               -(-br_1*clight * t+s)**2/(sigma_z1**2) \
                               -( br_2*clight * t+s)**2/(sigma_z2**2))) \
        /np.sqrt((sx1(s)**2 + sx2(s)**2)*(sy1(s)**2 + sy2(s)**2))/sigma_z1/sigma_z2

        integral=integrate.dblquad((lambda t, s: kernel_double_integral(t, s)),
                                   -sigma_integration*sigma_z,
                                   sigma_integration*sigma_z,
                                   -sigma_integration*sigma_z/clight,
                                   sigma_integration*sigma_z/clight)
        L0=f*N1*N2*nb * clight/2/np.pi**(2)*integral[0]

    else:
        def mx1(s):
            '''The mu_x of B1 as straight line'''
            return x_1 + px_1*s

        def my1(s):
            '''The mu_y of B1 as straight line'''
            return y_1 + py_1*s

        def mx2(s):
            '''The mu_x of B2 as straight line'''
            return x_2 + px_2*s

        def my2(s):
            '''The mu_y of B2 as straight line'''
            return y_2 + py_2*s

        def kernel_single_integral(s):
            return np.exp(0.5*(-(mx1(s) - mx2(s))**2/(sx1(s)**2 + sx2(s)**2) \
                               -(my1(s) - my2(s))**2/(sy1(s)**2 + sy2(s)**2) \
                               -((br_1+br_2)**2*s**2)/(br_2**2*sigma_z1**2 + br_1**2*sigma_z2**2))) \
            /np.sqrt((sx1(s)**2 + sx2(s)**2)*(sy1(s)**2 + sy2(s)**2)*(sigma_z1**2 + sigma_z2**2))

        integral=integrate.quad(lambda s: kernel_single_integral(s), -20*sigma_z, 20*sigma_z)
        L0=f*N1*N2*nb/np.sqrt(2)/np.pi**(3/2)*integral[0]
    result= L0*Moeller_efficiency/1e4
    if verbose:
        print(f'Moeller efficiency: {Moeller_efficiency}')
        print(f'Integral Relative Error: {integral[1]/integral[0]}')
        print(f'==> Luminosity [Hz/cm^2]: {result}')
    return result


def luminosity_from_twiss(
    n_colliding_bunches,
    num_particles_per_bunch,
    ip_name,
    nemitt_x,
    nemitt_y,
    sigma_z,
    twiss_b1,
    twiss_b2,
    f_rev=None,
    crab=None):

    assert crab is not None, 'crab crossing information is required'

    twiss_b2_rev = twiss_b2.reverse()

    if crab:
        crab_crossing = {
                'phi_crab_x_1': twiss_b1['dx_zeta', ip_name],
                'phi_crab_x_2': twiss_b2_rev['dx_zeta', ip_name],
                'phi_crab_y_1': twiss_b1['dy_zeta', ip_name],
                'phi_crab_y_2': twiss_b2_rev['dy_zeta', ip_name],
            }
    else:
        crab_crossing = None

    if f_rev is None:
        if 'T_rev0' not in twiss_b1.keys():
            raise ValueError('Revolution frequency cannot be retrieved from twiss, '
                             'please provide f_rev')
        f_rev = 1/twiss_b1.T_rev0

    lumi = luminosity(
        f=f_rev,
        rest_mass_b1=twiss_b1.particle_on_co.mass0 * 1e-9, # GeV
        rest_mass_b2=twiss_b2_rev.particle_on_co.mass0 * 1e-9, # GeV
        nb=n_colliding_bunches,
        N1=num_particles_per_bunch,
        N2=num_particles_per_bunch,
        x_1=twiss_b1['x', ip_name],
        x_2=twiss_b2_rev['x', ip_name],
        y_1=twiss_b1['y', ip_name],
        y_2=twiss_b2_rev['y', ip_name],
        px_1=twiss_b1['px', ip_name],
        px_2=twiss_b2_rev['px', ip_name],
        py_1=twiss_b1['py', ip_name],
        py_2=twiss_b2_rev['py', ip_name],
        energy_tot1=twiss_b1.particle_on_co.energy0[0]*1e-9, # GeV
        energy_tot2=twiss_b2_rev.particle_on_co.energy0[0]*1e-9, # GeV
        deltap_p0_1=0, # energy spread (for now we neglect effect of dispersion)
        deltap_p0_2=0, # energy spread (for now we neglect effect of dispersion)
        epsilon_x1=nemitt_x,
        epsilon_x2=nemitt_x,
        epsilon_y1=nemitt_y,
        epsilon_y2=nemitt_y,
        sigma_z1=sigma_z,
        sigma_z2=sigma_z,
        beta_x1=twiss_b1['betx', ip_name],
        beta_x2=twiss_b2_rev['betx', ip_name],
        beta_y1=twiss_b1['bety', ip_name],
        beta_y2=twiss_b2_rev['bety', ip_name],
        alpha_x1=twiss_b1['alfx', ip_name],
        alpha_x2=twiss_b2_rev['alfx', ip_name],
        alpha_y1=twiss_b1['alfy', ip_name],
        alpha_y2=twiss_b2_rev['alfy', ip_name],
        dx_1=twiss_b1['dx', ip_name],
        dx_2=twiss_b2_rev['dx', ip_name],
        dy_1=twiss_b1['dy', ip_name],
        dy_2=twiss_b2_rev['dy', ip_name],
        dpx_1=twiss_b1['dpx', ip_name],
        dpx_2=twiss_b2_rev['dpx', ip_name],
        dpy_1=twiss_b1['dpy', ip_name],
        dpy_2=twiss_b2_rev['dpy', ip_name],
        crab_crossing=crab_crossing,
        verbose=False, sigma_integration=3)

    return lumi