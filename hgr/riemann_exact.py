from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def _fk(p, rk, pk, g):
    ck = np.sqrt(g * pk / rk)
    if p > pk:
        A, B = 2 / ((g + 1) * rk), (g - 1) / (g + 1) * pk
        return (p - pk) * np.sqrt(A / (p + B))
    return 2 * ck / (g - 1) * ((p / pk) ** ((g - 1) / (2 * g)) - 1)


def star_state(rL, uL, pL, rR, uR, pR, g):
    f = lambda p: _fk(p, rL, pL, g) + _fk(p, rR, pR, g) + uR - uL  # noqa: E731
    ps = brentq(f, 1e-14 * min(pL, pR), 1e3 * max(pL, pR), xtol=1e-15, rtol=1e-15, maxiter=500)
    us = 0.5 * (uL + uR) + 0.5 * (_fk(ps, rR, pR, g) - _fk(ps, rL, pL, g))
    return ps, us


def sample(xi, rL, uL, pL, rR, uR, pR, g):
    ps, us = star_state(rL, uL, pL, rR, uR, pR, g)
    xi = np.asarray(xi, dtype=float)
    rho, u, p = np.empty_like(xi), np.empty_like(xi), np.empty_like(xi)
    cL, cR = np.sqrt(g * pL / rL), np.sqrt(g * pR / rR)
    for i, s in enumerate(xi):
        if s <= us:
            if ps > pL:
                SL = uL - cL * np.sqrt((g + 1) / (2 * g) * ps / pL + (g - 1) / (2 * g))
                if s <= SL:
                    rho[i], u[i], p[i] = rL, uL, pL
                else:
                    r = rL * (ps / pL + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pL + 1)
                    rho[i], u[i], p[i] = r, us, ps
            else:
                cs = cL * (ps / pL) ** ((g - 1) / (2 * g))
                if s <= uL - cL:
                    rho[i], u[i], p[i] = rL, uL, pL
                elif s >= us - cs:
                    rho[i], u[i], p[i] = rL * (ps / pL) ** (1 / g), us, ps
                else:
                    c = 2 / (g + 1) * (cL + (g - 1) / 2 * (uL - s))
                    rho[i] = rL * (c / cL) ** (2 / (g - 1))
                    u[i] = 2 / (g + 1) * (cL + (g - 1) / 2 * uL + s)
                    p[i] = pL * (c / cL) ** (2 * g / (g - 1))
        else:
            if ps > pR:
                SR = uR + cR * np.sqrt((g + 1) / (2 * g) * ps / pR + (g - 1) / (2 * g))
                if s >= SR:
                    rho[i], u[i], p[i] = rR, uR, pR
                else:
                    r = rR * (ps / pR + (g - 1) / (g + 1)) / ((g - 1) / (g + 1) * ps / pR + 1)
                    rho[i], u[i], p[i] = r, us, ps
            else:
                cs = cR * (ps / pR) ** ((g - 1) / (2 * g))
                if s >= uR + cR:
                    rho[i], u[i], p[i] = rR, uR, pR
                elif s <= us + cs:
                    rho[i], u[i], p[i] = rR * (ps / pR) ** (1 / g), us, ps
                else:
                    c = 2 / (g + 1) * (cR - (g - 1) / 2 * (uR - s))
                    rho[i] = rR * (c / cR) ** (2 / (g - 1))
                    u[i] = 2 / (g + 1) * (-cR + (g - 1) / 2 * uR + s)
                    p[i] = pR * (c / cR) ** (2 * g / (g - 1))
    return rho, u, p
