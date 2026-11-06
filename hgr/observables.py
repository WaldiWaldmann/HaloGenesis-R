from __future__ import annotations

import numpy as np

from .grid import Grid3D
from .hydro import pressure


def aperture_weights(grid: Grid3D, R: float, sub: int = 4) -> np.ndarray:
    off = (np.arange(sub) + 0.5) / sub - 0.5
    x = grid.x1d
    frac = np.zeros(grid.shape)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij", sparse=True)
    for ox in off:
        for oy in off:
            for oz in off:
                frac += ((X + ox * grid.dx) ** 2 + (Y + oy * grid.dx) ** 2 + (Z + oz * grid.dx) ** 2) < R**2
    return frac / sub**3


def grad_weights(w: np.ndarray, dx: float) -> tuple[np.ndarray, ...]:
    return tuple((np.roll(w, -1, ax) - np.roll(w, 1, ax)) / (2 * dx) for ax in range(3))


def aperture_mass(rho: np.ndarray, w: np.ndarray, dV: float) -> float:
    return float(np.sum(rho * w) * dV)


def flux_rate(U: np.ndarray, gw: tuple[np.ndarray, ...], dV: float) -> float:
    return float(sum(np.sum(U[1 + d] * gw[d]) for d in range(3)) * dV)


def jeans_Q(U: np.ndarray, w: np.ndarray, dV: float, gamma: float, G: float) -> tuple[float, float, float]:
    M = aperture_mass(U[0], w, dV)
    V = float(np.sum(w) * dV)
    cs2 = float(np.sum(w * gamma * pressure(U, gamma)) / np.sum(w * U[0]))
    rho_bar = M / V
    MJ = (np.pi**2.5 / 6.0) * cs2**1.5 / (G**1.5 * np.sqrt(rho_bar))
    return MJ / M, MJ, M


def o3_coherence(rho0: np.ndarray, dg: tuple, dv: tuple, w: np.ndarray) -> float:
    num = sum(np.sum(w * rho0 * dg[d] * dv[d]) for d in range(3))
    ng = np.sqrt(sum(np.sum(w * rho0 * dg[d] ** 2) for d in range(3)))
    nv = np.sqrt(sum(np.sum(w * rho0 * dv[d] ** 2) for d in range(3)))
    if ng == 0 or nv == 0:
        return float("nan")
    return float(num / (ng * nv))


def weighted_norm(rho0: np.ndarray, f: tuple, w: np.ndarray) -> float:
    return float(np.sqrt(sum(np.sum(w * rho0 * f[d] ** 2) for d in range(3))))


def occupancy(t: np.ndarray, q: np.ndarray, threshold: float) -> tuple[float, list[tuple[float, float]]]:
    t, lq, lth = np.asarray(t), np.log(np.asarray(q)), np.log(threshold)
    below = lq < lth
    intervals, start = [], (t[0] if below[0] else None)
    for i in range(1, len(t)):
        if below[i] != below[i - 1]:
            tc = t[i - 1] + (lth - lq[i - 1]) * (t[i] - t[i - 1]) / (lq[i] - lq[i - 1])
            if below[i]:
                start = tc
            else:
                intervals.append((start, tc))
                start = None
    if start is not None:
        intervals.append((start, t[-1]))
    T = t[-1] - t[0]
    return (sum(b - a for a, b in intervals) / T if T > 0 else float("nan")), intervals


def truelove_number(U: np.ndarray, gamma: float, G: float, dx: float) -> float:
    cs = np.sqrt(gamma * pressure(U, gamma) / U[0])
    lam = cs * np.sqrt(np.pi / (G * U[0]))
    return float(dx / lam.min())


class Aperture:

    def __init__(self, grid: Grid3D, R: float, with_gradient: bool, sub: int = 4) -> None:
        w = aperture_weights(grid, R, sub)
        idx = [np.nonzero(w.any(axis=tuple(a for a in range(3) if a != ax)))[0] for ax in range(3)]
        self.sl = tuple(slice(max(i.min() - 1, 0), min(i.max() + 2, grid.n)) for i in idx)
        self.w = np.ascontiguousarray(w[self.sl])
        self.gw = tuple(np.ascontiguousarray(gg[self.sl]) for gg in grad_weights(w, grid.dx)) if with_gradient else None
        self.dV = grid.cell_volume
        self.volume = float(w.sum() * grid.cell_volume)
        self.R = R
        c, self.n = grid.center_index, grid.n
        self.rel = tuple(np.arange(sl_.start, sl_.stop) - c for sl_ in self.sl)
        self.idx = self.sl

    def set_center(self, ci: tuple[int, int, int]) -> None:
        if all(int(a) == self.n // 2 for a in ci):
            self.idx = self.sl
        else:
            self.idx = np.ix_(*[(int(a) + r) % self.n for a, r in zip(ci, self.rel)])

    def take(self, f: np.ndarray) -> np.ndarray:
        return f[self.idx]

    def mass(self, rho: np.ndarray) -> float:
        return float(np.sum(rho[self.idx] * self.w) * self.dV)

    def jeans_Q(self, U: np.ndarray, P: np.ndarray, gamma: float, G: float) -> tuple[float, float, float]:
        rho = U[0][self.idx]
        M = float(np.sum(rho * self.w) * self.dV)
        cs2 = float(np.sum(self.w * gamma * P[self.idx]) / np.sum(self.w * rho))
        MJ = (np.pi**2.5 / 6.0) * cs2**1.5 / (G**1.5 * np.sqrt(M / self.volume))
        return MJ / M, MJ, M

    def flux(self, U: np.ndarray) -> float:
        return float(sum(np.sum(U[1 + d][self.idx] * self.gw[d]) for d in range(3)) * self.dV)


def track_center(rho_a: np.ndarray, prev: tuple[int, int, int], track: "Aperture", max_iter: int = 8):
    ci = tuple(int(a) for a in prev)
    for _ in range(max_iter):
        track.set_center(ci)
        w = track.take(rho_a) * track.w
        tot = w.sum()
        if tot <= 0:
            return ci
        off = [
            float(np.sum(w * r.reshape([-1 if k == ax else 1 for k in range(3)])) / tot)
            for ax, r in enumerate(track.rel)
        ]
        new = tuple((ci[k] + int(round(off[k]))) % track.n for k in range(3))
        if new == ci:
            return ci
        ci = new
    return ci


def truelove_from_pressure(rho: np.ndarray, P: np.ndarray, gamma: float, G: float, dx: float) -> float:
    return float(dx / np.sqrt(gamma * np.pi / G * np.min(P / rho**2)))
