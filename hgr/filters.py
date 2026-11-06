from __future__ import annotations

from collections import deque

import numpy as np

from .coupled import TrailingBoxcar

KERNELS = ("trailing_boxcar", "zero_lag_boxcar", "delay_half_tau", "ema")


def transfer_function(kind: str, omega: np.ndarray, tau: float) -> np.ndarray:
    w = np.asarray(omega, dtype=float)
    if tau == 0:
        return np.ones_like(w, dtype=complex)
    x = w * tau
    with np.errstate(invalid="ignore", divide="ignore"):
        hb = np.where(np.abs(x) < 1e-8, 1.0 - 0.5j * x, (1 - np.exp(-1j * x)) / (1j * x))
    if kind == "trailing_boxcar":
        return hb
    if kind == "zero_lag_boxcar":
        return 2 * hb - hb**2
    if kind == "delay_half_tau":
        return np.exp(-0.5j * x)
    if kind == "ema":
        return 1.0 / (1.0 + 1j * x)
    raise ValueError(f"unbekannter filter {kind}")


def mean_lag(kind: str, tau: float) -> float:
    return {"trailing_boxcar": tau / 2, "zero_lag_boxcar": 0.0, "delay_half_tau": tau / 2, "ema": tau}[kind]


class TemporalFilter:

    def __init__(self, tau: float) -> None:
        if tau < 0:
            raise ValueError("tau < 0")
        self.tau = tau

    def reset(self, t0: float, phi0: np.ndarray) -> None:
        raise NotImplementedError

    def push(self, t0: float, t1: float, integral: np.ndarray, phi_end: np.ndarray) -> None:
        raise NotImplementedError

    def value(self, t: float, phi_now: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class BoxcarFilter(TemporalFilter):

    def __init__(self, tau: float) -> None:
        super().__init__(tau)
        self.box = TrailingBoxcar(tau)

    def reset(self, t0, phi0):
        self.box = TrailingBoxcar(self.tau)

    def push(self, t0, t1, integral, phi_end):
        self.box.push(t0, t1, integral)

    def value(self, t, phi_now):
        return self.box.mean(t, phi_now)


class ZeroLagBoxcarFilter(TemporalFilter):

    def __init__(self, tau: float) -> None:
        super().__init__(tau)
        self.inner = TrailingBoxcar(tau)
        self.outer = TrailingBoxcar(tau)
        self._last = None

    def reset(self, t0, phi0):
        self.inner = TrailingBoxcar(self.tau)
        self.outer = TrailingBoxcar(self.tau)
        self._last = (t0, phi0)

    def push(self, t0, t1, integral, phi_end):
        self.inner.push(t0, t1, integral)
        m1 = self.inner.mean(t1, phi_end)
        t_prev, m0 = self._last
        self.outer.push(t_prev, t1, 0.5 * (m0 + m1) * (t1 - t_prev))
        self._last = (t1, m1)

    def value(self, t, phi_now):
        if self.tau == 0:
            return phi_now
        m = self.inner.mean(t, phi_now)
        return 2.0 * m - self.outer.mean(t, m)


class DelayFilter(TemporalFilter):

    def __init__(self, tau: float) -> None:
        super().__init__(tau)
        self.d = tau / 2
        self.pts: deque = deque()

    def reset(self, t0, phi0):
        self.pts = deque([(t0, phi0)])

    def push(self, t0, t1, integral, phi_end):
        self.pts.append((t1, phi_end))
        target = t1 - self.d
        while len(self.pts) > 2 and self.pts[1][0] <= target:
            self.pts.popleft()

    def value(self, t, phi_now):
        if self.tau == 0 or not self.pts:
            return phi_now
        target = t - self.d
        if target <= self.pts[0][0]:
            return self.pts[0][1]
        prev = self.pts[0]
        for tp, ph in list(self.pts)[1:] + [(t, phi_now)]:
            if tp >= target:
                w = (target - prev[0]) / (tp - prev[0]) if tp > prev[0] else 1.0
                return (1 - w) * prev[1] + w * ph
            prev = (tp, ph)
        return phi_now


class EMAFilter(TemporalFilter):

    def __init__(self, tau: float) -> None:
        super().__init__(tau)
        self.y = None
        self._x_prev = None

    def reset(self, t0, phi0):
        self.y = phi0
        self._x_prev = phi0

    def push(self, t0, t1, integral, phi_end):
        h = t1 - t0
        if self.tau == 0 or h <= 0:
            self.y, self._x_prev = phi_end, phi_end
            return
        e = np.exp(-h / self.tau)
        s = (phi_end - self._x_prev) / h
        self.y = phi_end - s * self.tau + (self.y - self._x_prev + s * self.tau) * e
        self._x_prev = phi_end

    def value(self, t, phi_now):
        return phi_now if self.tau == 0 else self.y


def make_filter(kind: str, tau: float) -> TemporalFilter:
    cls = {
        "trailing_boxcar": BoxcarFilter,
        "zero_lag_boxcar": ZeroLagBoxcarFilter,
        "delay_half_tau": DelayFilter,
        "ema": EMAFilter,
    }.get(kind)
    if cls is None:
        raise ValueError(f"unbekannter filter {kind!r}; erlaubt: {KERNELS}")
    return cls(tau)
