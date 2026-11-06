from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .hydro import _hllc, _limit


@njit(cache=True, inline="always")
def _lambda_hist(T):
    if T > 1e4:
        return 1e-22 * math.sqrt(T)
    if T > 300.0:
        return 1e-26 * math.exp(-400.0 / max(T, 1e-30)) * T**3.5
    return 0.0


@njit(cache=True, inline="always")
def _primordial(nH, T, xH2, Tfloor):
    if not T > Tfloor:
        return 0.0
    Tc = max(T, 1.0)
    g = 1.0 / (1.0 + math.sqrt(Tc / 1e5))
    gam = 5.85e-11 * math.sqrt(Tc) * math.exp(-157809.1 / Tc) * g
    alpha = 8.40e-11 / math.sqrt(Tc) * (Tc / 1e3) ** -0.2 / (1.0 + (Tc / 1e6) ** 0.7)
    x = gam / (gam + alpha)
    ne, nH0, nHp = x * nH, (1 - x) * nH, x * nH
    exc = 7.50e-19 * math.exp(-118348.0 / Tc) * g * ne * nH0
    ion = 1.27e-21 * math.sqrt(Tc) * math.exp(-157809.1 / Tc) * g * ne * nH0
    rec = 8.70e-27 * math.sqrt(Tc) * (Tc / 1e3) ** -0.2 / (1 + (Tc / 1e6) ** 0.7) * ne * nHp
    ff = 1.42e-27 * 1.3 * math.sqrt(Tc) * ne * nHp
    h2 = 0.0
    if Tc < 1e4:
        lt = math.log10(min(max(Tc, 10.0), 1e4))
        h2 = xH2 * nH * nH * 10.0 ** (-103.0 + 97.59 * lt - 48.05 * lt**2 + 10.80 * lt**3 - 0.9032 * lt**4)
    return exc + ion + rec + ff + h2


@njit(cache=True, inline="always")
def _rate(model, rho, T, n_fac, nH_fac, xH2, Tfloor, rate_to_code):
    if model == 0:
        n = rho * n_fac
        return n * n * _lambda_hist(T) * rate_to_code
    return _primordial(rho * nH_fac, T, xH2, Tfloor) * rate_to_code


@njit(cache=True, parallel=True)
def cooling_backward_euler(rho, e_old, dt, model, tfac, e_lo_fac, n_fac, nH_fac, xH2, Tfloor, rate_to_code):
    out = np.empty_like(e_old)
    for i in prange(rho.size):
        r, eo = rho[i], e_old[i]
        if _rate(model, r, tfac * eo / r, n_fac, nH_fac, xH2, Tfloor, rate_to_code) == 0.0:
            out[i] = eo
            continue
        lo = min(e_lo_fac * r, eo)
        hi = eo
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if mid == lo or mid == hi:
                break
            f = mid - eo + dt * _rate(model, r, tfac * mid / r, n_fac, nH_fac, xH2, Tfloor, rate_to_code)
            if f > 0:
                hi = mid
            else:
                lo = mid
        out[i] = hi
    return out


@njit(cache=True, parallel=True)
def kick_inplace(psi, phi, s):
    for i in prange(psi.size):
        th = phi[i] * s
        psi[i] = psi[i] * complex(math.cos(th), -math.sin(th))


@njit(cache=True, parallel=True)
def sweep_planes(V, nrm, dtdx, g, lim, first_order):
    P, S, I = V.shape[1], V.shape[2], V.shape[3]
    t1 = 1 if nrm != 1 else 2
    t2 = 3 if nrm != 3 else 2
    fallbacks = np.zeros(P, dtype=np.int64)
    for p in prange(P):
        W = np.empty((5, S, I))
        Lf = np.empty((5, S, I))
        Rf = np.empty((5, S, I))
        F = np.empty((5, S, I))
        d = np.empty(5)
        f = np.empty(5)
        for i in range(S):
            for k in range(I):
                r = V[0, p, i, k]
                W[0, i, k] = r
                W[1, i, k] = V[nrm, p, i, k] / r
                W[2, i, k] = V[t1, p, i, k] / r
                W[3, i, k] = V[t2, p, i, k] / r
                W[4, i, k] = (g - 1.0) * (
                    V[4, p, i, k] - 0.5 * (V[1, p, i, k] ** 2 + V[2, p, i, k] ** 2 + V[3, p, i, k] ** 2) / r
                )
        for i in range(S):
            im = i - 1 if i > 0 else S - 1
            ip = i + 1 if i < S - 1 else 0
            for k in range(I):
                for q in range(5):
                    d[q] = 0.0 if first_order else _limit(W[q, i, k] - W[q, im, k], W[q, ip, k] - W[q, i, k], lim)
                r, u, v, w, pp = W[0, i, k], W[1, i, k], W[2, i, k], W[3, i, k], W[4, i, k]
                dr = u * d[0] + r * d[1]
                du = u * d[1] + d[4] / r
                dv = u * d[2]
                dw = u * d[3]
                dp = g * pp * d[1] + u * d[4]
                h = 0.5 * dtdx
                Lf[0, i, k] = r - 0.5 * d[0] - h * dr
                Rf[0, i, k] = r + 0.5 * d[0] - h * dr
                Lf[1, i, k] = u - 0.5 * d[1] - h * du
                Rf[1, i, k] = u + 0.5 * d[1] - h * du
                Lf[2, i, k] = v - 0.5 * d[2] - h * dv
                Rf[2, i, k] = v + 0.5 * d[2] - h * dv
                Lf[3, i, k] = w - 0.5 * d[3] - h * dw
                Rf[3, i, k] = w + 0.5 * d[3] - h * dw
                Lf[4, i, k] = pp - 0.5 * d[4] - h * dp
                Rf[4, i, k] = pp + 0.5 * d[4] - h * dp
                if Lf[0, i, k] <= 0.0 or Rf[0, i, k] <= 0.0 or Lf[4, i, k] <= 0.0 or Rf[4, i, k] <= 0.0:
                    for q in range(5):
                        Lf[q, i, k] = W[q, i, k]
                        Rf[q, i, k] = W[q, i, k]
                    fallbacks[p] += 1
        for i in range(S):
            ip = i + 1 if i < S - 1 else 0
            for k in range(I):
                _hllc(
                    Rf[0, i, k],
                    Rf[1, i, k],
                    Rf[2, i, k],
                    Rf[3, i, k],
                    Rf[4, i, k],
                    Lf[0, ip, k],
                    Lf[1, ip, k],
                    Lf[2, ip, k],
                    Lf[3, ip, k],
                    Lf[4, ip, k],
                    g,
                    f,
                )
                F[0, i, k] = f[0]
                F[nrm, i, k] = f[1]
                F[t1, i, k] = f[2]
                F[t2, i, k] = f[3]
                F[4, i, k] = f[4]
        for i in range(S):
            im = i - 1 if i > 0 else S - 1
            for k in range(I):
                for q in range(5):
                    V[q, p, i, k] -= dtdx * (F[q, i, k] - F[q, im, k])
    return fallbacks.sum()


def center_weights_rfft(n: int) -> np.ndarray:
    ix = np.arange(n)
    iz = np.arange(n // 2 + 1)
    s = ((-1.0) ** ix)[:, None, None] * ((-1.0) ** ix)[None, :, None] * ((-1.0) ** iz)[None, None, :]
    m = np.where((iz == 0) | (iz == n // 2), 1.0, 2.0)[None, None, :]
    return s * m / n**3
