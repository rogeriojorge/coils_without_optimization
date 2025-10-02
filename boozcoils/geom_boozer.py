import numpy as np

def make_grids(ntheta, nphi_plot, nphi_surface):
    theta1D = np.linspace(0, 2*np.pi, ntheta)
    phi1D = np.linspace(0, 2*np.pi, nphi_plot, endpoint=False)
    phi1D_surface = np.linspace(0, 2*np.pi, nphi_surface, endpoint=True)
    varphi, theta = np.meshgrid(phi1D, theta1D)
    varphi_s, theta_s = np.meshgrid(phi1D_surface, theta1D)
    return theta1D, (varphi, theta), (varphi_s, theta_s), phi1D, phi1D_surface

def accum_RZnu_derivs(b, theta, varphi, theta_s, varphi_s, js=None):
    R = np.zeros_like(theta); Z = np.zeros_like(theta); nu = np.zeros_like(theta)
    R_s = np.zeros_like(theta_s); Z_s = np.zeros_like(theta_s)
    dR = np.zeros_like(theta); dR_s = np.zeros_like(theta_s)
    dZ = np.zeros_like(theta); dZ_s = np.zeros_like(theta_s)

    for jmn in range(b.mnboz):
        m = b.xm_b[jmn]; n = b.xn_b[jmn]
        ang = m*theta - n*varphi
        ang_s = m*theta_s - n*varphi_s
        sa, ca = np.sin(ang), np.cos(ang)
        sa_s, ca_s = np.sin(ang_s), np.cos(ang_s)

        R   += b.rmnc_b[jmn, js] * ca
        R_s += b.rmnc_b[jmn, js] * ca_s
        Z   += b.zmns_b[jmn, js] * sa
        Z_s += b.zmns_b[jmn, js] * sa_s
        nu  += b.numns_b[jmn, js] * sa

        dR  += -m * b.rmnc_b[jmn, js] * sa
        dR_s+= -m * b.rmnc_b[jmn, js] * sa_s
        dZ  +=  m * b.zmns_b[jmn, js] * ca
        dZ_s+=  m * b.zmns_b[jmn, js] * ca_s

    return (R, Z, nu, dR, dZ), (R_s, Z_s, dR_s, dZ_s)

def push_off_surface(R, Z, dR, dZ, eps):
    denom = np.sqrt(dR*dR + dZ*dZ)
    R2 = R - eps * (dZ / denom)
    Z2 = Z + eps * (dR / denom)
    return R2, Z2

def cyl_xyz_from_RphiZ(R, phi, Z):
    X = R * np.cos(phi)
    Y = R * np.sin(phi)
    return X, Y, Z
