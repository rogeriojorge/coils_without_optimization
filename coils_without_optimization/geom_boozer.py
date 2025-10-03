"""Geometry helpers for the Boozer→Coils workflow.

The Boozer driver needs to construct grids in the poloidal and
toroidal directions, accumulate Boozer harmonic contributions and
compute the cylindrical coordinates of coil centre lines.  These
helpers encapsulate those operations for clarity.
"""

import numpy as np


def make_grids(ntheta: int, nphi_plot: int, nphi_surface: int):
    """Create poloidal/toroidal grids for plotting and surface evaluation.

    Parameters
    ----------
    ntheta : int
        Number of poloidal samples.
    nphi_plot : int
        Number of toroidal samples for plotting coils.
    nphi_surface : int
        Number of toroidal samples for evaluating the surface (often a
        refinement of ``nphi_plot``).

    Returns
    -------
    tuple
        ``theta1D``, ``(varphi, theta)``, ``(varphi_s, theta_s)``, ``phi1D``, ``phi1D_surface``
    """
    theta1D = np.linspace(0, 2 * np.pi, ntheta)
    phi1D = np.linspace(0, 2 * np.pi, nphi_plot, endpoint=False)
    phi1D_surface = np.linspace(0, 2 * np.pi, nphi_surface, endpoint=True)
    varphi, theta = np.meshgrid(phi1D, theta1D)
    varphi_s, theta_s = np.meshgrid(phi1D_surface, theta1D)
    return theta1D, (varphi, theta), (varphi_s, theta_s), phi1D, phi1D_surface


def accum_RZnu_derivs(b, theta: np.ndarray, varphi: np.ndarray, theta_s: np.ndarray,
                       varphi_s: np.ndarray, js=None):
    """Accumulate R, Z, nu and their derivatives from Boozer harmonics."""
    R = np.zeros_like(theta)
    Z = np.zeros_like(theta)
    nu = np.zeros_like(theta)
    R_s = np.zeros_like(theta_s)
    Z_s = np.zeros_like(theta_s)
    dR = np.zeros_like(theta)
    dZ = np.zeros_like(theta)
    dR_s = np.zeros_like(theta_s)
    dZ_s = np.zeros_like(theta_s)
    for jmn in range(b.mnboz):
        m = b.xm_b[jmn]
        n = b.xn_b[jmn]
        ang = m * theta - n * varphi
        ang_s = m * theta_s - n * varphi_s
        sa, ca = np.sin(ang), np.cos(ang)
        sa_s, ca_s = np.sin(ang_s), np.cos(ang_s)
        R += b.rmnc_b[jmn, js] * ca
        R_s += b.rmnc_b[jmn, js] * ca_s
        Z += b.zmns_b[jmn, js] * sa
        Z_s += b.zmns_b[jmn, js] * sa_s
        nu += b.numns_b[jmn, js] * sa
        dR += -m * b.rmnc_b[jmn, js] * sa
        dR_s += -m * b.rmnc_b[jmn, js] * sa_s
        dZ += m * b.zmns_b[jmn, js] * ca
        dZ_s += m * b.zmns_b[jmn, js] * ca_s
    return (R, Z, nu, dR, dZ), (R_s, Z_s, dR_s, dZ_s)


def push_off_surface(R: np.ndarray, Z: np.ndarray, dR: np.ndarray, dZ: np.ndarray,
                     eps: float):
    """Offset a surface outward along its normal by a small distance ``eps``."""
    denom = np.sqrt(dR * dR + dZ * dZ)
    R2 = R - eps * (dZ / denom)
    Z2 = Z + eps * (dR / denom)
    return R2, Z2


def cyl_xyz_from_RphiZ(R: np.ndarray, phi: np.ndarray, Z: np.ndarray):
    """Convert cylindrical coordinates to Cartesian coordinates."""
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y, Z