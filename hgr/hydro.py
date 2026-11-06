from __future__ import annotations

import numpy as np
from numba import njit, prange

LIMITERS = {"minmod": 0, "vanleer": 1, "mc": 2}


@njit(cache=True, inline="always")
def _limit(dl, dr, kind):
    if dl * dr <= 0.0:
        return 0.0
    if kind == 0:
        return dl if abs(dl) < abs(dr) else dr
    if kind == 1:
        return 2.0 * dl * dr / (dl + dr)
    c = 0.5 * (dl + dr)
    m = min(abs(2.0 * dl), abs(2.0 * dr), abs(c))
    return m if c > 0 else -m


@njit(cache=True, inline="always")
def _hllc(rL, uL, vL, wL, pL, rR, uR, vR, wR, pR, g, out):
    cL = np.sqrt(g * pL / rL)
    cR = np.sqrt(g * pR / rR)
    EL = pL / (g - 1.0) + 0.5 * rL * (uL * uL + vL * vL + wL * wL)
    ER = pR / (g - 1.0) + 0.5 * rR * (uR * uR + vR * vR + wR * wR)
    SL = min(uL - cL, uR - cR)
    SR = max(uL + cL, uR + cR)
    if SL >= 0.0:
        out[0] = rL * uL
        out[1] = rL * uL * uL + pL
        out[2] = rL * uL * vL
        out[3] = rL * uL * wL
        out[4] = (EL + pL) * uL
        return
    if SR <= 0.0:
        out[0] = rR * uR
        out[1] = rR * uR * uR + pR
        out[2] = rR * uR * vR
        out[3] = rR * uR * wR
        out[4] = (ER + pR) * uR
        return
    Ss = (pR - pL + rL * uL * (SL - uL) - rR * uR * (SR - uR)) / (rL * (SL - uL) - rR * (SR - uR))
    if Ss >= 0.0:
        r, u, v, w, p, E, S = rL, uL, vL, wL, pL, EL, SL
    else:
        r, u, v, w, p, E, S = rR, uR, vR, wR, pR, ER, SR
    fac = r * (S - u) / (S - Ss)
    us0 = fac
    us1 = fac * Ss
    us2 = fac * v
    us3 = fac * w
    us4 = fac * (E / r + (Ss - u) * (Ss + p / (r * (S - u))))
    out[0] = r * u + S * (us0 - r)
    out[1] = r * u * u + p + S * (us1 - r * u)
    out[2] = r * u * v + S * (us2 - r * v)
    out[3] = r * u * w + S * (us3 - r * w)
    out[4] = (E + p) * u + S * (us4 - E)


@njit(cache=True, parallel=True)
def _sweep_rows(U, nrm, dtdx, g, lim, first_order):
    nrows = U.shape[1]
    n = U.shape[2]
    t1 = 1 if nrm != 1 else 2
    t2 = 3 if nrm != 3 else 2
    fallbacks = np.zeros(nrows, dtype=np.int64)
    for row in prange(nrows):
        W = np.empty((5, n))
        for i in range(n):
            r = U[0, row, i]
            W[0, i] = r
            W[1, i] = U[nrm, row, i] / r
            W[2, i] = U[t1, row, i] / r
            W[3, i] = U[t2, row, i] / r
            W[4, i] = (g - 1.0) * (U[4, row, i] - 0.5 * (U[1, row, i] ** 2 + U[2, row, i] ** 2 + U[3, row, i] ** 2) / r)
        Lf = np.empty((5, n))
        Rf = np.empty((5, n))
        d = np.empty(5)
        for i in range(n):
            im = i - 1 if i > 0 else n - 1
            ip = i + 1 if i < n - 1 else 0
            for q in range(5):
                d[q] = 0.0 if first_order else _limit(W[q, i] - W[q, im], W[q, ip] - W[q, i], lim)
            r, u, v, w, p = W[0, i], W[1, i], W[2, i], W[3, i], W[4, i]
            dr = u * d[0] + r * d[1]
            du = u * d[1] + d[4] / r
            dv = u * d[2]
            dw = u * d[3]
            dp = g * p * d[1] + u * d[4]
            h = 0.5 * dtdx
            Lf[0, i] = r - 0.5 * d[0] - h * dr
            Rf[0, i] = r + 0.5 * d[0] - h * dr
            Lf[1, i] = u - 0.5 * d[1] - h * du
            Rf[1, i] = u + 0.5 * d[1] - h * du
            Lf[2, i] = v - 0.5 * d[2] - h * dv
            Rf[2, i] = v + 0.5 * d[2] - h * dv
            Lf[3, i] = w - 0.5 * d[3] - h * dw
            Rf[3, i] = w + 0.5 * d[3] - h * dw
            Lf[4, i] = p - 0.5 * d[4] - h * dp
            Rf[4, i] = p + 0.5 * d[4] - h * dp
            if Lf[0, i] <= 0.0 or Rf[0, i] <= 0.0 or Lf[4, i] <= 0.0 or Rf[4, i] <= 0.0:
                for q in range(5):
                    Lf[q, i] = W[q, i]
                    Rf[q, i] = W[q, i]
                fallbacks[row] += 1
        F = np.empty((5, n))
        f = np.empty(5)
        for i in range(n):
            ip = i + 1 if i < n - 1 else 0
            _hllc(
                Rf[0, i],
                Rf[1, i],
                Rf[2, i],
                Rf[3, i],
                Rf[4, i],
                Lf[0, ip],
                Lf[1, ip],
                Lf[2, ip],
                Lf[3, ip],
                Lf[4, ip],
                g,
                f,
            )
            F[0, i] = f[0]
            F[nrm, i] = f[1]
            F[t1, i] = f[2]
            F[t2, i] = f[3]
            F[4, i] = f[4]
        for i in range(n):
            im = i - 1 if i > 0 else n - 1
            for q in range(5):
                U[q, row, i] -= dtdx * (F[q, i] - F[q, im])
    return fallbacks.sum()


def sweep(U: np.ndarray, axis: int, dt: float, dx: float, gamma: float, limiter: str, first_order: bool = False) -> int:
    if not U.flags.c_contiguous:
        raise ValueError("U must be C-contiguous")
    from .kernels import sweep_planes

    lim = LIMITERS[limiter]
    if axis == 2:
        return int(_sweep_rows(U.reshape(5, -1, U.shape[3]), 3, dt / dx, gamma, lim, first_order))
    V = U.transpose(0, 2, 1, 3) if axis == 0 else U
    return int(sweep_planes(V, axis + 1, dt / dx, gamma, lim, first_order))


def sweep_reference(
    U: np.ndarray, axis: int, dt: float, dx: float, gamma: float, limiter: str, first_order: bool = False
) -> int:
    lim = LIMITERS[limiter]
    V = np.ascontiguousarray(np.moveaxis(U, axis + 1, -1))
    sh = V.shape
    V2 = V.reshape(5, -1, sh[-1])
    nfb = _sweep_rows(V2, axis + 1, dt / dx, gamma, lim, first_order)
    U[...] = np.moveaxis(V2.reshape(sh), -1, axis + 1)
    return int(nfb)


def check_physical(U: np.ndarray, gamma: float, where: str) -> None:
    rho = U[0]
    p = pressure(U, gamma)
    if not (np.all(np.isfinite(U)) and rho.min() > 0 and p.min() > 0):
        raise FloatingPointError(
            f"{where}: unphysical state (ρmin={rho.min():.3e}, "
            f"Pmin={p.min():.3e}, NaN={not np.all(np.isfinite(U))})"
        )


def step(U: np.ndarray, dt: float, dx: float, gamma: float, limiter: str, parity: int) -> int:
    order = (0, 1, 2) if parity % 2 == 0 else (2, 1, 0)
    nfb = 0
    for ax in order:
        nfb += sweep(U, ax, dt, dx, gamma, limiter)
    check_physical(U, gamma, "hydro.step")
    return nfb


def cons_from_prim(rho, vx, vy, vz, p, gamma) -> np.ndarray:
    rho = np.asarray(rho, dtype=np.float64)
    shape = np.broadcast_shapes(*(np.shape(q) for q in (rho, vx, vy, vz, p)))
    U = np.empty((5,) + shape)
    U[0] = rho
    U[1] = rho * vx
    U[2] = rho * vy
    U[3] = rho * vz
    U[4] = np.asarray(p) / (gamma - 1.0) + 0.5 * rho * (np.asarray(vx) ** 2 + np.asarray(vy) ** 2 + np.asarray(vz) ** 2)
    return U


def velocity(U: np.ndarray) -> np.ndarray:
    return U[1:4] / U[0]


def pressure(U: np.ndarray, gamma: float) -> np.ndarray:
    return (gamma - 1.0) * (U[4] - 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0])


def sound_speed_field(U: np.ndarray, gamma: float) -> np.ndarray:
    return np.sqrt(gamma * pressure(U, gamma) / U[0])


def cfl_dt(U: np.ndarray, dx: float, gamma: float, cfl: float) -> float:
    cs = sound_speed_field(U, gamma)
    vmax = np.max(np.abs(U[1:4]) / U[0] + cs[None])
    return cfl * dx / float(vmax)


def gravity_half_kick(U: np.ndarray, accel: tuple[np.ndarray, ...], dt_half: float) -> None:
    for d in range(3):
        mom_old = U[1 + d].copy()
        U[1 + d] += U[0] * accel[d] * dt_half
        U[4] += 0.5 * (mom_old + U[1 + d]) * accel[d] * dt_half


def totals(U: np.ndarray, cell_volume: float) -> np.ndarray:
    return U.reshape(5, -1).sum(axis=1) * cell_volume


def apply_jeans_floor(U: np.ndarray, gamma: float, G: float, dx: float, NJ: float) -> tuple[float, np.ndarray]:
    if NJ <= 0:
        return 0.0, np.zeros(U.shape[1:], dtype=bool)
    p = pressure(U, gamma)
    pj = G * U[0] ** 2 * (NJ * dx) ** 2 / (np.pi * gamma)
    mask = p < pj * (1.0 - 1e-12)
    de = np.where(mask, (pj - p) / (gamma - 1.0), 0.0)
    U[4] += de
    return float(de.sum()), mask
