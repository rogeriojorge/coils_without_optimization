"""Plotting helpers for coil visualisation.

This module collects a handful of functions used by both the Boozer and
near–axis examples to produce 3D visualisations of coils and
surfaces.  All functions return Plotly traces; they do not call
``plotly.graph_objects.Figure.show()`` on their own.

The key helper is :func:`tubes_mesh3d_from_gammas` which converts a
set of centre–line curves into a triangulated tube mesh.  To control
the appearance of these tubes (radius, resolution, colour, opacity)
use the parameters documented on the function.
"""

import numpy as np
import plotly.graph_objects as go


def npf(a, dtype=np.float32):
    """Convert an arbitrary array to a NumPy array of the given dtype."""
    return np.asarray(a, dtype=dtype)


def closed_loop(P: np.ndarray) -> np.ndarray:
    """Ensure a polyline is closed by appending the first point if needed."""
    P = npf(P)
    return P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])


def unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the normalised vectors along the last axis."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)


def frames_parallel_transport(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tangent and normal frames along a polyline via parallel transport.

    Given a curve ``P`` of shape ``(npoints, 3)`` this routine computes a
    set of orthonormal frames (T,N1,N2) such that T is the tangent and
    (N1,N2) span the normal plane.  Frames are chosen to minimise
    twisting (parallel transport).  The curve may be closed or open.
    """
    P = npf(P)
    closed = np.allclose(P[0], P[-1])
    if closed:
        T = unit(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    else:
        Tmid = unit(P[2:] - P[:-2])
        T = np.vstack([Tmid[0], Tmid, Tmid[-1]])
    ref = np.array([0.0, 0.0, 1.0], dtype=P.dtype)
    if abs(np.dot(T[0], ref)) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=P.dtype)
    N1 = unit(np.cross(T[0], ref))[None, :]
    N2 = unit(np.cross(T[0], N1[0]))[None, :]
    for i in range(1, len(T)):
        t = T[i]
        n1 = unit(N1[-1] - np.dot(N1[-1], t) * t)
        n2 = unit(np.cross(t, n1))
        N1 = np.vstack([N1, n1])
        N2 = np.vstack([N2, n2])
    return T, N1, N2


def surface_trace_from_RZ_phi(R: np.ndarray, Z: np.ndarray, phi1D: np.ndarray,
                              color: str = "#C5B6A7", opacity: float = 0.28) -> go.Surface:
    """Build a Plotly surface trace from cylindrical (R,Z,phi) data.

    This helper expects the radial coordinate ``R`` and the vertical
    coordinate ``Z`` to be arrays of shape ``(ntheta, nphi)`` (or
    transposed), and ``phi1D`` to be an array of the toroidal angles.
    Colours are applied uniformly via the provided colour string.
    """
    colorscale = [[0, color], [1, color]]
    X = npf(R) * np.cos(npf(phi1D))
    Y = npf(R) * np.sin(npf(phi1D))
    return go.Surface(x=X, y=Y, z=npf(Z), colorscale=colorscale,
                      showscale=False, opacity=opacity,
                      lighting={"specular": 0.3, "diffuse": 0.9},
                      hoverinfo="skip")


def add_polyline_trajs(data: list, trajectories: list, color: str = "black",
                       width: float = 0.6, name: str | None = None,
                       opacity: float = 0.9, every: int = 3,
                       dtype=np.float32) -> None:
    """Append a set of field line trajectories to a Plotly data list.

    Parameters
    ----------
    data : list
        The list of Plotly traces to which lines should be appended.
    trajectories : list of array‐like
        Each element is an ``(npoints, 3)`` array representing a
        trajectory.
    color : str, optional
        The colour of the lines.
    width : float, optional
        The width of the lines.
    name : str or None, optional
        The legend name for the first trajectory; subsequent trajectories
        will not show legends.
    opacity : float, optional
        Opacity of the lines.
    every : int, optional
        Plot every ``every``–th point to decimate long trajectories.
    dtype : dtype, optional
        Data type for conversion to NumPy arrays.
    """
    for traj in trajectories:
        T = npf(traj, dtype=dtype)[::every]
        data.append(
            go.Scatter3d(
                x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines",
                line=dict(color=color, width=width), opacity=opacity,
                name=name or "trajectory", showlegend=False,
            )
        )


def tubes_mesh3d_from_gammas(gammas: list, radius: float = 0.015,
                             n_theta: int = 12, color: str = "#5B2222",
                             opacity: float = 1.0) -> go.Mesh3d:
    """Construct a triangulated tube mesh around multiple curves.

    Parameters
    ----------
    gammas : list of array‐like
        Each element is an ``(npoints, 3)`` array representing a coil
        centre line.  The loops need not be closed; this routine
        ensures closure before meshing.
    radius : float, optional
        Tube radius in metres.
    n_theta : int, optional
        Number of angular samples for the tube cross–section.
    color : str, optional
        Colour of the mesh.
    opacity : float, optional
        Opacity of the mesh.

    Returns
    -------
    go.Mesh3d
        A Plotly mesh trace representing the tubes.
    """
    gammas = [npf(g) for g in gammas]
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False, dtype=np.float32)
    c, s = np.cos(th), np.sin(th)
    verts = []
    faces_i: list[int] = []
    faces_j: list[int] = []
    faces_k: list[int] = []
    base = 0
    for P in gammas:
        P = closed_loop(P)
        _, N1, N2 = frames_parallel_transport(P)
        V = (
            P[:, None, :] + radius * (N1[:, None, :] * c[None, :, None]
                                     + N2[:, None, :] * s[None, :, None])
        ).astype(np.float32)
        n, m = V.shape[0], V.shape[1]
        verts.append(V.reshape(-1, 3))
        for i in range(n - 1):
            for j in range(m):
                a = base + i * m + j
                b = base + i * m + (j + 1) % m
                c1 = base + (i + 1) * m + j
                d = base + (i + 1) * m + (j + 1) % m
                faces_i += [a, b]
                faces_j += [b, d]
                faces_k += [c1, c1]
        base += n * m
    verts = np.vstack(verts)
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=np.asarray(faces_i, dtype=np.int32),
        j=np.asarray(faces_j, dtype=np.int32),
        k=np.asarray(faces_k, dtype=np.int32),
        color=color, opacity=opacity, flatshading=True, hoverinfo="skip",
    )