import numpy as np
import plotly.graph_objects as go

def npf(a, dtype=np.float32):
    return np.asarray(a, dtype=dtype)

def closed_loop(P):
    P = npf(P)
    return P if np.allclose(P[0], P[-1]) else np.vstack([P, P[0]])

def unit(v, eps=1e-12):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, eps, None)

def frames_parallel_transport(P):
    P = npf(P)
    closed = np.allclose(P[0], P[-1])
    if closed:
        T = unit(np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0))
    else:
        Tmid = unit(P[2:] - P[:-2]); T = np.vstack([Tmid[0], Tmid, Tmid[-1]])
    ref = np.array([0.0, 0.0, 1.0], dtype=P.dtype)
    if abs(np.dot(T[0], ref)) > 0.9: ref = np.array([0.0, 1.0, 0.0], dtype=P.dtype)
    N1 = unit(np.cross(T[0], ref))[None, :]; N2 = unit(np.cross(T[0], N1[0]))[None, :]
    for i in range(1, len(T)):
        t = T[i]
        n1 = unit(N1[-1] - np.dot(N1[-1], t) * t)
        n2 = unit(np.cross(t, n1))
        N1 = np.vstack([N1, n1]); N2 = np.vstack([N2, n2])
    return T, N1, N2

def surface_trace_from_RZ_phi(R, Z, phi1D, color="#C5B6A7", opacity=0.28):
    colorscale = [[0, color], [1, color]]
    X = npf(R) * np.cos(npf(phi1D)); Y = npf(R) * np.sin(npf(phi1D))
    return go.Surface(x=X, y=Y, z=npf(Z), colorscale=colorscale, showscale=False,
                      opacity=opacity, lighting={"specular": 0.3, "diffuse": 0.9},
                      hoverinfo="skip")

def add_polyline_trajs(data, trajectories, color="black", width=0.6, name=None, opacity=0.9, every=3, dtype=np.float32):
    for traj in trajectories:
        T = npf(traj, dtype=dtype)[::every]
        data.append(go.Scatter3d(x=T[:,0], y=T[:,1], z=T[:,2], mode="lines",
                                 line=dict(color=color, width=width), opacity=opacity,
                                 name=name or "line", showlegend=False))

def tubes_mesh3d_from_gammas(gammas, radius=0.015, n_theta=12, color="#5B2222", opacity=1.0):
    gammas = [npf(g) for g in gammas]
    th = np.linspace(0, 2*np.pi, n_theta, endpoint=False, dtype=np.float32)
    c, s = np.cos(th), np.sin(th)
    verts, faces_i, faces_j, faces_k = [], [], [], []
    base = 0
    for P in gammas:
        P = closed_loop(P)
        _, N1, N2 = frames_parallel_transport(P)
        V = (P[:, None, :] + radius*(N1[:,None,:]*c[None,:,None] + N2[:,None,:]*s[None,:,None])).astype(np.float32)
        n, m = V.shape[0], V.shape[1]
        verts.append(V.reshape(-1,3))
        for i in range(n-1):
            for j in range(m):
                a = base + i*m + j
                b = base + i*m + (j+1)%m
                c1= base + (i+1)*m + j
                d = base + (i+1)*m + (j+1)%m
                faces_i += [a,b]; faces_j += [b,d]; faces_k += [c1,c1]
        base += n*m
    verts = np.vstack(verts)
    return go.Mesh3d(x=verts[:,0], y=verts[:,1], z=verts[:,2],
                     i=np.asarray(faces_i, dtype=np.int32),
                     j=np.asarray(faces_j, dtype=np.int32),
                     k=np.asarray(faces_k, dtype=np.int32),
                     color=color, opacity=opacity, flatshading=True, hoverinfo="skip")
