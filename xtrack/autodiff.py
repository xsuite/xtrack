import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple
import xtrack as xt


class TransferMatrixFactory:

    @staticmethod
    @jax.jit
    def quad(k1, l, beta0, gamma0):
        """Quadrupole transfer matrix.

        Parameters
        ----------
        k1 : float
            Quadrupole strength.
        l : float
            Length of the quadrupole.
        beta0 : float
            Reference relativistic beta.
        gamma0 : float
            Reference relativistic gamma.

        Returns
        -------
        f_matrix : jnp.ndarray
            Transfer matrix for the quadrupole.
        """

        kx = jnp.sqrt(k1.astype(complex))
        ky = jnp.sqrt(-k1.astype(complex))
        sx = l * jnp.sinc(kx * l / jnp.pi)
        cx = jnp.cos(kx * l)
        sy = l * jnp.sinc(ky * l / jnp.pi) # limit of sin(ky * l) / ky when ky -> 0
        cy = jnp.cos(ky * l)

        f_matrix = jnp.array([
            [cx, sx, 0, 0, 0, 0],
            [-kx**2 * sx, cx, 0, 0, 0, 0],
            [0, 0, cy, sy, 0, 0],
            [0, 0, -ky**2 * sy, cy, 0, 0],
            [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
            [0, 0, 0, 0, 0, 1]
        ])

        return f_matrix.real

    @staticmethod
    @jax.jit
    def drift(l, beta0, gamma0):
        """Drift transfer matrix.

        Parameters
        ----------
        l : float
            Length of the drift.
        beta0 : float
            Reference relativistic beta.
        gamma0 : float
            Reference relativistic gamma.
        Returns
        -------
        f_matrix : jnp.ndarray
            Transfer matrix for the drift.
        """

        f_matrix = jnp.array([
            [1, l, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, l, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, l/(beta0**2 * gamma0**2)],
            [0, 0, 0, 0, 0, 1]
        ])
        return f_matrix

    @staticmethod
    @jax.jit
    def bend(k0, k1, l, h, beta0, gamma0):
        """Bend transfer matrix.

        Parameters
        ----------
        k0 : float
            First order curvature (dipole strength).
        k1 : float
            Second order curvature (quadrupole strength).
        l : float
            Length of the bend.
        h : float
            Horizontal offset of the bend.
        beta0 : float
            Reference relativistic beta.
        gamma0 : float
            Reference relativistic gamma.
        Returns
        -------
        f_matrix : jnp.ndarray
            Transfer matrix for the bend.
        """

        kx = jnp.sqrt((h * k0 + k1).astype(complex))
        ky = jnp.sqrt(-k1.astype(complex)) # for dipoles usually 0
        sx = l * jnp.sinc(kx * l / jnp.pi)
        cx = jnp.cos(kx * l)
        sy = l * jnp.sinc(ky * l / jnp.pi)
        cy = jnp.cos(ky * l)
        dx = (1 - cx) / kx**2
        j1 = (l - sx) / kx**2

        f_matrix = jnp.array([
            [cx, sx, 0, 0, 0, h/beta0 * dx],
            [-kx**2 * sx, cx, 0, 0, 0, h/beta0 * sx],
            [0, 0, cy, sy, 0, 0],
            [0, 0, -ky**2 * sy, cy, 0, 0],
            [-h/beta0 * sx, -h/beta0 * dx, 0, 0, 1, l/(beta0**2 * gamma0**2) - h**2/beta0**2 * j1],
            [0, 0, 0, 0, 0, 1]
        ])

        return f_matrix.real

class EncodedElem(NamedTuple):
    etype: int
    data0: float = 0.0
    data1: float = 0.0
    data2: float = 0.0
    data3: float = 0.0
    k1_idx: int = -1

@jax.jit
def get_values_from_transfer_matrix(r_mat, param_values):
    """Compute Twiss parameters and dispersion from transfer matrix.

    Parameters
    ----------
    r_mat : jnp.ndarray
        Transfer matrix of the element.
    param_values : jnp.ndarray
        Initial Twiss parameters and dispersion values.

    Returns
    -------
    jnp.ndarray
        Updated Twiss parameters and dispersion values.
    """

    # Order: betx, bety, alfx, alfy, mux, muy, dx, dy, dpx, dpy
    bx0, by0, ax0, ay0, mux0, muy0, dx0, dy0, dpx0, dpy0 = param_values

    # --- Horizontal plane ---
    r00, r01, r10, r11 = r_mat[0,0], r_mat[0,1], r_mat[1,0], r_mat[1,1]

    tmp_x = r00 * bx0 - r01 * ax0
    betx = (tmp_x**2 + r01**2) / bx0
    alfx = -((tmp_x * (r10 * bx0 - r11 * ax0) + r01 * r11) / bx0)
    mux = mux0 + jnp.arctan2(r01, tmp_x) / (2 * jnp.pi)

    # --- Vertical plane ---
    r22, r23, r32, r33 = r_mat[2,2], r_mat[2,3], r_mat[3,2], r_mat[3,3]

    tmp_y = r22 * by0 - r23 * ay0
    bety = (tmp_y**2 + r23**2) / by0
    alfy = -((tmp_y * (r32 * by0 - r33 * ay0) + r23 * r33) / by0)
    muy = muy0 + jnp.arctan2(r23, tmp_y) / (2 * jnp.pi)

    # --- Dispersion ---
    dx = r00 * dx0 + r01 * dpx0 + r_mat[0,5]
    dy = r22 * dy0 + r23 * dpy0 + r_mat[2,5]
    dpx = r10 * dx0 + r11 * dpx0 + r_mat[1,5]
    dpy = r32 * dy0 + r33 * dpy0 + r_mat[3,5]

    return jnp.array([betx, bety, alfx, alfy, mux, muy, dx, dy, dpx, dpy])

def encode_elements(elements, elem_to_deriv):
    """Encode elements into a format suitable for JAX,
    containing the type of each element and serializing parameters.

    Parameters
    ----------
    elements : list of xtrack elements
        The elements to encode.
    elem_to_deriv : list of xtrack elements or None
        The elements for which derivatives are computed. If None, no derivatives are computed.
    Returns
    -------
    EncodedElem
        A NamedTuple containing the encoded elements.
    """

    if elem_to_deriv is not None:
        deriv_lookup = {id(elem): i for i, elem in enumerate(elem_to_deriv)}

    encoded = []

    for elem in elements:
        if elem_to_deriv is not None and elem in elem_to_deriv:
            encoded.append(EncodedElem(
                etype=0,
                data0=elem.length,
                k1_idx=deriv_lookup[id(elem)]
            ))
        elif isinstance(elem, xt.Quadrupole):
            encoded.append(EncodedElem(
                etype=1,
                data0=elem.k1,
                data1=elem.length
            ))
        elif isinstance(elem, xt.Bend) or isinstance(elem, xt.RBend):
            encoded.append(EncodedElem(
                etype=2,
                data0=elem.k0,
                data1=elem.k1,
                data2=elem.length,
                data3=elem.h
            ))
        elif isinstance(elem, xt.Multipole) and elem.isthick and elem.length > 0:
            encoded.append(EncodedElem(
                etype=3,
                data0=elem.length
            ))
        elif isinstance(elem, xt.Multipole):
            encoded.append(EncodedElem(
                etype=4
            ))
        elif isinstance(elem, xt.Drift) or hasattr(elem, 'length'):
            encoded.append(EncodedElem(
                etype=3,
                data0=elem.length
            ))
        else:
            encoded.append(EncodedElem(
                etype=4,
            ))

    # Convert list of NamedTuples to NamedTuple of arrays for JAX
    return EncodedElem(
        etype=jnp.array([e.etype for e in encoded]),
        data0=jnp.array([e.data0 for e in encoded]),
        data1=jnp.array([e.data1 for e in encoded]),
        data2=jnp.array([e.data2 for e in encoded]),
        data3=jnp.array([e.data3 for e in encoded]),
        k1_idx=jnp.array([e.k1_idx for e in encoded]),
    )

@partial(jax.jit, static_argnums=(2,3))
def get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params):
    """Compute Twiss parameters and dispersion from encoded elements.

    Parameters
    ----------
    k1_arr : jnp.ndarray
        Array of k1 values for the elements that require derivatives.
    encoded_elements : EncodedElem
        Encoded elements containing the type and parameters of each element.
    beta0 : float
        Reference relativistic beta.
    gamma0 : float
        Reference relativistic gamma.
    initial_params : jnp.ndarray
        Initial Twiss parameters and dispersion values.

    Returns
    -------
    jnp.ndarray
        Updated Twiss parameters and dispersion values.
    """

    def scan_step(params, elem):
        TMF = TransferMatrixFactory

        # Defining methods inside the switch to avoid recompilation
        tm = jax.lax.switch(elem.etype, [
            lambda: TMF.quad(k1_arr[elem.k1_idx], elem.data0, beta0, gamma0),
            lambda: TMF.quad(elem.data0, elem.data1, beta0, gamma0),
            lambda: TMF.bend(elem.data0, elem.data1, elem.data2, elem.data3, beta0, gamma0),
            lambda: TMF.drift(elem.data0, beta0, gamma0),
            lambda: jnp.eye(6)
            ]
        )
        new_params = get_values_from_transfer_matrix(tm, params)
        return new_params, None

    final_params, _ = jax.lax.scan(scan_step, initial_params, encoded_elements)
    return final_params

def compute_param_derivatives(elements, elem_to_deriv, init_cond, beta0, gamma0):
    """Compute the derivatives of the Twiss parameters with respect to k1 values.

    Parameters
    ----------
    elements : list of xtrack elements
        The elements for which to compute the derivatives.
    elem_to_deriv : list of xtrack elements
        The elements for which derivatives are computed.
    init_cond : list of float
        Initial conditions for the Twiss parameters and dispersion.
    beta0 : float
        Reference relativistic beta.
    gamma0 : float
        Reference relativistic gamma.

    Returns
    -------
    jnp.ndarray
        The Jacobian matrix of derivatives with respect to k1 values.
    """

    encoded_elements = encode_elements(elements, elem_to_deriv)
    k1_arr = jnp.array([elem.k1 for elem in elem_to_deriv])

    initial_params = jnp.array(init_cond)

    def wrapped_get_values(k1_arr):
        return get_values(k1_arr, encoded_elements, beta0, gamma0, initial_params)

    pushfwd = partial(jax.jvp, wrapped_get_values, (k1_arr,))
    basis = jnp.eye(len(k1_arr))
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return jac, y # to yield both jacobian and final values

@partial(jax.jit, static_argnums=(1,2))
def get_values_noderiv(encoded_elements, beta0, gamma0, initial_params):
    """Compute Twiss parameters and dispersion from encoded elements without derivatives.

    Parameters
    ----------
    encoded_elements : EncodedElem
        Encoded elements containing the type and parameters of each element.
    beta0 : float
        Reference relativistic beta.
    gamma0 : float
        Reference relativistic gamma.
    initial_params : jnp.ndarray
        Initial Twiss parameters and dispersion values.

    Returns
    -------
    jnp.ndarray
        Updated Twiss parameters and dispersion values.
    """

    def scan_step(params, elem):
        TMF = TransferMatrixFactory

        # Defining methods inside the switch to avoid recompilation
        tm = jax.lax.switch(elem.etype, [
            lambda: jnp.eye(6),
            lambda: TMF.quad(elem.data0, elem.data1, beta0, gamma0),
            lambda: TMF.bend(elem.data0, elem.data1, elem.data2, elem.data3, beta0, gamma0),
            lambda: TMF.drift(elem.data0, beta0, gamma0),
            lambda: jnp.eye(6)
            ]
        )
        new_params = get_values_from_transfer_matrix(tm, params)
        return new_params, None

    final_params, _ = jax.lax.scan(scan_step, initial_params, encoded_elements)
    return final_params

def compute_values(elements, tw0):
    """Compute Twiss parameters and dispersion from elements and initial conditions
    without calculating derivatives.

    Parameters
    ----------
    elements : list of xtrack elements
        The elements for which to compute the Twiss parameters and dispersion.
    tw0 : xtrack.Twiss
        Initial Twiss parameters and dispersion.

    Returns
    -------
    jnp.ndarray
        Updated Twiss parameters and dispersion values.
    """

    beta0 = tw0.particle_on_co.beta0[0]
    gamma0 = tw0.particle_on_co.gamma0[0]

    encoded_elements = encode_elements(elements, None)

    initial_params = jnp.array([
        tw0.betx[0], tw0.bety[0], tw0.alfx[0], tw0.alfy[0],
        tw0.mux[0], tw0.muy[0], tw0.dx[0], tw0.dy[0],
        tw0.dpx[0], tw0.dpy[0]
    ])

    return get_values_noderiv(encoded_elements, beta0, gamma0, initial_params)