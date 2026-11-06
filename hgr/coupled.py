"""P4 – Kopplung Axion ↔ Gas mit zeitgefiltertem Potential, Lockstep A=1 / A=0.

Makroschritt der Länge Δt (für BEIDE Zweige identisch, Research Plan §15: gleiche Zeiten):
  1. Gas-Halbkick mit ḡ(t) = −∇Φ̄_τ(t)
  2. Hydro-Schritt Δt (MUSCL-Hancock/HLLC), danach Kühlung über Δt
  3. A=1: n SP-KDK-Teilschritte; ρ_b wird linear zwischen alt/neu interpoliert (2. Ordnung);
     ∫Φ dt per Trapez über die Teilschritte -> Filterblock [t, t+Δt]
     A=0: Φ aus ρ_b neu; ∫Φ dt per Trapez über Δt
  4. Φ̄_τ(t+Δt) aus Filterblöcken (Gl. 42, Überlappungsgewichte, Renormierung für t<τ)
  5. Gas-Halbkick mit ḡ(t+Δt)
Axionen spüren das INSTANTANE Φ, das Gas das GEFILTERTE Φ̄ (Gl. 2, 8). Für τ = 0 gilt Φ̄ ≡ Φ.
Beide Zweige nutzen denselben Code; der einzige Unterschied ist ψ ≡ 0 für A=0.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import scipy.fft as sfft

from . import cooling as co
from . import hydro as hy
from .grid import Grid3D
from .kernels import center_weights_rfft
from .poisson import gradient_from_hat, solve_poisson_periodic_hat
from .schroedinger import drift, kick_inplace, stable_dt


class TrailingBoxcar:
    """Φ̄_τ(t) = (1/|W|) ∫_W Φ dt,  W = [max(0, t−τ), t]  (Research Plan Gl. 6/42).
    Speichert Blöcke (t0, t1, ∫Φ dt); ein teilweise überlappender Block geht mit dem
    Überlappungsanteil ein (Φ innerhalb des Blocks als konstant = Blockmittel angenommen)."""

    def __init__(self, tau: float) -> None:
        if tau < 0:
            raise ValueError("tau < 0")
        self.tau = tau
        self.blocks: deque = deque()

    def push(self, t0: float, t1: float, integral: np.ndarray) -> None:
        if self.tau > 0:
            self.blocks.append((t0, t1, integral))

    def mean(self, t: float, phi_now: np.ndarray) -> np.ndarray:
        if self.tau == 0 or not self.blocks:
            return phi_now
        lo = max(0.0, t - self.tau)
        while self.blocks and self.blocks[0][1] <= lo:
            self.blocks.popleft()
        acc = np.zeros_like(phi_now)
        width = 0.0
        for t0, t1, integ in self.blocks:
            ov = min(t1, t) - max(t0, lo)
            if ov > 0:
                acc += integ * (ov / (t1 - t0))
                width += ov
        if width <= 0:
            return phi_now
        return acc / width

    def n_blocks(self) -> int:
        return len(self.blocks)


@dataclass
class Branch:
    """Zustand eines Zweigs. Φ̄ wird im Fourierraum gehalten (phibar_hat); der Ortsraum-Wert
    wird nur bei Bedarf berechnet (Property phibar) -> spart FFTs im Hauptpfad."""
    A: int
    U: np.ndarray
    psi: np.ndarray | None
    phi: np.ndarray
    phi_hat: np.ndarray
    filt: TrailingBoxcar
    grid: Grid3D
    workers: int
    phibar_hat: np.ndarray = field(init=False)
    acc: tuple = field(init=False)
    acc_inst: tuple = field(init=False)
    _phibar: np.ndarray | None = field(init=False, default=None)

    @property
    def phibar(self) -> np.ndarray:
        if self._phibar is None:
            self._phibar = sfft.irfftn(self.phibar_hat, s=self.grid.shape, workers=self.workers)
        return self._phibar

    def phibar_at(self, ci) -> float:
        """Φ̄ an Gitterzelle ci aus dem Spektrum: Phasenverschiebung relativ zum Zentrum (ohne inverse FFT)."""
        n, c = self.grid.n, self.grid.n // 2
        m_full = np.fft.fftfreq(n, 1.0 / n)
        m_half = np.fft.rfftfreq(n, 1.0 / n)
        ph = [np.exp(2j * np.pi * m * (int(p) - c) / n) for m, p in ((m_full, ci[0]), (m_full, ci[1]), (m_half, ci[2]))]
        w = _center_w(n) * ph[0][:, None, None] * ph[1][None, :, None] * ph[2][None, None, :]
        return float(np.real(np.sum(self.phibar_hat * w)))

    def phibar_center(self) -> float:
        """Φ̄ am Gitterzentrum direkt aus dem Spektrum (ohne inverse FFT)."""
        return float(np.real(np.sum(self.phibar_hat * _center_w(self.grid.n))))


_CW: dict = {}


def _center_w(n: int) -> np.ndarray:
    if n not in _CW:
        _CW.clear()
        _CW[n] = center_weights_rfft(n)
    return _CW[n]


@dataclass(frozen=True)
class CouplingParams:
    a: float                # ħ/m [kpc²/Myr]
    G: float
    gamma: float
    mu: float
    T0: float               # nur für cooling_model == "isothermal"
    cooling_model: str
    limiter: str
    cfl: float
    sp_kin_factor: float
    sp_phase_max: float
    tau: float
    workers: int
    grav_cfl: float          # Δt ≤ C·√(dx/max|g|)
    ff_fraction: float       # Δt ≤ f·t_ff(max ρ_ges)
    dt_max: float            # Obergrenze (zeitliche Auflösung der Diagnostik)
    jeans_NJ: float          # B5: 0 = aus, sonst λ_J ≥ N_J·dx erzwungen
    cooling_params: dict     # B6: Parameter für primordial_h2 (leer sonst)


class CoupledPair:
    def __init__(self, grid: Grid3D, p: CouplingParams, psi1: np.ndarray, U_gas: np.ndarray) -> None:
        self.g, self.p = grid, p
        self.t = 0.0
        self.steps = 0
        self.fallbacks = 0
        self.br = []
        for A, psi in ((1, psi1), (0, None)):
            U = U_gas.copy()  # identische Gas-Anfangsbedingung in beiden Zweigen
            phi, phi_hat = self._phi(U[0], psi)
            b = Branch(A, U, None if psi is None else psi.copy(), phi, phi_hat, TrailingBoxcar(p.tau), grid, p.workers)
            self._update_filter(b)
            self.br.append(b)
        assert np.array_equal(self.br[0].U, self.br[1].U)
        self.filter_work = [0.0, 0.0]  # ∫P_filter dt je Zweig (Gl. 44/45)
        self.floor_energy = [0.0, 0.0]  # B5: vom Jeans-Boden injizierte Energie
        self.floor_mask = [np.zeros(grid.shape, bool), np.zeros(grid.shape, bool)]

    # ------------------------------------------------------------------
    def _phi(self, rho_b: np.ndarray, psi: np.ndarray | None):
        rho = rho_b if psi is None else rho_b + (psi.real**2 + psi.imag**2)
        return solve_poisson_periodic_hat(rho, self.g, self.p.G, self.p.workers)

    def _update_filter(self, b: Branch) -> None:
        """Φ̄̂ = Mittel der Blockintegrale im Fourierraum (linear ⇒ identisch zu Ortsraum-Mittel)."""
        b.phibar_hat = b.filt.mean(self.t, b.phi_hat)
        b._phibar = None
        b.acc = tuple(-d for d in gradient_from_hat(b.phibar_hat, self.g, self.p.workers))
        if self.p.tau > 0:
            b.acc_inst = tuple(-d for d in gradient_from_hat(b.phi_hat, self.g, self.p.workers))
        else:
            b.acc_inst = b.acc

    def _filter_power(self, b: Branch) -> float:
        """P_filter = ∫ ρ v·(ḡ − g) d³x (Research Plan Gl. 44)."""
        if self.p.tau == 0:
            return 0.0
        s = 0.0
        for d in range(3):
            s += float(np.sum(b.U[1 + d] * (b.acc[d] - b.acc_inst[d])))
        return s * self.g.cell_volume

    def max_dt(self) -> float:
        """Minimum aus Hydro-CFL, Gravitations-CFL, Freifallkriterium und dt_max (beide Zweige)."""
        p, out = self.p, [self.p.dt_max]
        for b in self.br:
            out.append(hy.cfl_dt(b.U, self.g.dx, p.gamma, p.cfl))
            gmax = float(np.sqrt(max(np.max(a**2) for a in b.acc) * 3))
            if gmax > 0:
                out.append(p.grav_cfl * np.sqrt(self.g.dx / gmax))
            rho_max = float(np.max(b.U[0] if b.psi is None else b.U[0] + np.abs(b.psi) ** 2))
            out.append(p.ff_fraction * float(np.sqrt(3 * np.pi / (32 * p.G * rho_max))))
        return min(out)

    # ------------------------------------------------------------------
    def step(self, dt: float) -> dict:
        p, g = self.p, self.g
        pf0 = [self._filter_power(b) for b in self.br]
        info = {"n_sub": 0, "radiated": [0.0, 0.0]}
        for i, b in enumerate(self.br):
            hy.gravity_half_kick(b.U, b.acc, 0.5 * dt)
            rho_old = b.U[0].copy()
            self.fallbacks += hy.step(b.U, dt, g.dx, p.gamma, p.limiter, self.steps)
            info["radiated"][i] = co.apply_cooling(b.U, dt, p.cooling_model, p.gamma, p.mu, p.T0,
                                                   cparams=p.cooling_params)
            e_fl, mask = hy.apply_jeans_floor(b.U, p.gamma, p.G, g.dx, p.jeans_NJ)
            self.floor_energy[i] += e_fl * g.cell_volume
            self.floor_mask[i] = mask
            hy.check_physical(b.U, p.gamma, f"Zweig A={b.A} nach Hydro/Kühlung")
            rho_new = b.U[0]
            if b.psi is not None:
                dt_sp = stable_dt(g, b.phi, p.a, p.sp_kin_factor, p.sp_phase_max)
                n = max(1, int(np.ceil(dt / dt_sp)))
                h = dt / n
                integ = np.zeros_like(b.phi_hat)
                psi, phi, phi_hat = b.psi, b.phi, b.phi_hat
                kick_inplace(psi, phi, 0.5 * h, p.a)
                for s in range(1, n + 1):
                    # Leapfrog-Fusion: Schluss-Halbkick von s und Anfangs-Halbkick von s+1 = ein Vollkick
                    psi = drift(psi, g, h, p.a, p.workers)
                    phi_n, phi_hat_n = self._phi(rho_old + (s / n) * (rho_new - rho_old), psi)
                    kick_inplace(psi, phi_n, h if s < n else 0.5 * h, p.a)
                    integ += 0.5 * (phi_hat + phi_hat_n) * h
                    phi, phi_hat = phi_n, phi_hat_n
                b.psi, b.phi, b.phi_hat = psi, phi, phi_hat
                info["n_sub"] = n
            else:
                phi_n, phi_hat_n = self._phi(rho_new, None)
                integ = 0.5 * (b.phi_hat + phi_hat_n) * dt
                b.phi, b.phi_hat = phi_n, phi_hat_n
            b.filt.push(self.t, self.t + dt, integ)
        self.t += dt
        self.steps += 1
        for i, b in enumerate(self.br):
            self._update_filter(b)
            hy.gravity_half_kick(b.U, b.acc, 0.5 * dt)
            hy.check_physical(b.U, p.gamma, f"Zweig A={b.A} nach Kick")
            self.filter_work[i] += 0.5 * (pf0[i] + self._filter_power(b)) * dt
        return info

    # ----------------------------------------------------------- Diagnostik
    def momentum(self, b: Branch) -> np.ndarray:
        """Gesamtimpuls Gas + Axion; Axion: ρ_a v = a·Im(ψ*∇ψ)."""
        P = hy.totals(b.U, self.g.cell_volume)[1:4].copy()
        if b.psi is not None:
            pk = sfft.fftn(b.psi, workers=self.p.workers)
            for d, k in enumerate(self.g.k_full):
                dpsi = sfft.ifftn(1j * k * pk, workers=self.p.workers)
                P[d] += self.p.a * float(np.sum(np.imag(np.conj(b.psi) * dpsi))) * self.g.cell_volume
        return P
