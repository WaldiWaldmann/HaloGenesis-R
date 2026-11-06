"""P4X – Validierung der Modellerweiterungen B4–B7.

X1 (B4) Granularer Halo erzeugt echte Potentialfluktuationen (sonst ist der Filter-Test leer)
X2 (B5) Jeans-Boden erzwingt λ_J ≥ N_J dx exakt und ist idempotent
X3 (B6) Primordiale Kühlung: Λ = 0 unter T_floor, implizite Integration Ordnung 1, Vergleich mit historisch
X4 (B7) Seed: gleicher Seed bitgleich, anderer Seed verschieden

Aufruf: python -m scripts.p4x_extensions_validation --config configs/physical_fb.yaml
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
from scipy.integrate import solve_ivp

from hgr import cooling as co
from hgr import hydro as hy
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.coupled import TrailingBoxcar
from hgr.grid import Grid3D
from hgr.observables import truelove_number
from hgr.report import Recorder
from hgr.schroedinger import stable_dt, step_kdk
from hgr.setup import build_axion_field, gas_initial_condition
from hgr.units import G_CODE, AxionUnits, gas_mass_density

R = Recorder()


def x1_halo(cfg, figs, figs3, dpi, w):
    V = cfg["validation_p4x"]
    g = Grid3D(cfg["domain"]["box_kpc"], V["N"])
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    c = g.center_index
    curves, sig = {}, {}
    for kind in ("isolated", "soliton_halo"):
        cc = copy.deepcopy(cfg)
        cc["setup"]["kind"] = kind
        psi, info = build_axion_field(cc, g, w)
        from hgr.schroedinger import potential
        phi = potential(psi, None, g, G_CODE, w)
        t, T, P = 0.0, [0.0], [phi[c, c, c]]
        f = TrailingBoxcar(3.0)
        PB = [phi[c, c, c]]
        while t < V["t_myr"] - 1e-12:
            dt = min(stable_dt(g, phi, a, cfg["time"]["sp_kinetic_dt_factor"], cfg["time"]["sp_potential_phase_max_rad"]),
                     V["t_myr"] - t)
            p0 = phi[c, c, c]
            psi, phi = step_kdk(psi, phi, None, g, dt, a, G_CODE, w)
            f.push(t, t + dt, np.array([0.5 * (p0 + phi[c, c, c]) * dt]))
            t += dt
            T.append(t), P.append(phi[c, c, c]), PB.append(f.mean(t, np.array([phi[c, c, c]]))[0])
        P, PB = np.array(P), np.array(PB)
        sig[kind] = float(np.std(P) / abs(np.mean(P)))
        R.add("X1", f"{kind}: σ(Φ_c)/|⟨Φ_c⟩| über {V['t_myr']} Myr", sig[kind])
        R.add("X1", f"{kind}: max|Φ̄_τ=3 − Φ|/|⟨Φ⟩| im Zentrum", float(np.max(np.abs(PB - P)) / abs(np.mean(P))))
        curves[f"Φ_c {kind}"] = P / abs(np.mean(P))
        curves[f"Φ̄_c (τ=3) {kind}"] = PB / abs(np.mean(P))
        if kind == "soliton_halo":
            R.add("X1", "Halo: Kohärenzzeit t_dB = a/σ² [Myr]", info["t_dB_myr"])
            R.add("X1", "Halo: Gesamtmasse Axion [M_sun]", info["M_total"])
            d = P - P.mean()
            ac = np.correlate(d, d, "full")[len(d) - 1:] / (d @ d)
            tc = float(np.interp(1 / np.e, ac[::-1], np.array(T)[: len(ac)][::-1])) if np.any(ac < 1 / np.e) else np.nan
            R.add("X1", "Halo: Autokorrelationszeit von Φ_c (1/e) [Myr]", tc, note="vergleiche t_dB")
            viz.slices_row(figs / "p4x_X1_halo_slices.png", [(np.abs(psi) ** 2, "ρ_a Soliton+Halo", "log"),
                           (phi, "Φ", "lin"), (np.angle(psi), "arg ψ", "lin")], g.box, dpi=dpi,
                           suptitle=f"X1: granularer Halo nach {V['t_myr']} Myr")
            viz.isosurface_html(figs3 / "p4x_X1_halo_iso.html", np.abs(psi) ** 2, g.box, title="X1: Soliton + Halo",
                                log=True, label="ρ_a", max_points=cfg["viz"]["iso_max_points_per_axis"])
            viz.ortho_slices_3d_png(figs3 / "p4x_X1_halo_ortho.png", np.abs(psi) ** 2, g.box, title="X1: ρ_a",
                                    dpi=dpi, log=True, label="ρ_a")
            viz.line_plot(figs / "p4x_X1_autocorr.png", np.array(T)[: len(ac)], {"Autokorrelation Φ_c": ac},
                          xlabel="Verzögerung [Myr]", ylabel="ACF", title="X1: Fluktuationszeit", dpi=dpi,
                          hlines={"1/e": 1 / np.e}, vlines={"t_dB": info["t_dB_myr"]})
        Tlast = np.array(T)
        curves = {k: np.interp(np.linspace(0, V["t_myr"], 300), Tlast if len(v) == len(Tlast) else np.linspace(0, V["t_myr"], len(v)), v)
                  for k, v in curves.items()}
    fac = sig["soliton_halo"] / max(sig["isolated"], 1e-300)
    R.add("X1", "Fluktuationsfaktor Halo / isoliert", fac, V["halo_fluct_factor_min"], fac >= V["halo_fluct_factor_min"])
    viz.line_plot(figs / "p4x_X1_phi_c.png", np.linspace(0, V["t_myr"], 300), curves, xlabel="t [Myr]",
                  ylabel="Φ_c/|⟨Φ_c⟩|", title="X1: zentrales Potential – isoliert vs. Halo, mit Filter", dpi=dpi)


def x2_floor(cfg, figs, figs3, dpi):
    V = cfg["validation_p4x"]
    g = Grid3D(cfg["domain"]["box_kpc"], 32)
    rng = np.random.default_rng(3)
    rho = gas_mass_density(10.0, 1.2) * np.exp(3 * rng.standard_normal(g.shape))
    U = hy.cons_from_prim(rho, 0, 0, 0, co.pressure_from_T(rho, 300.0, 1.2), 5 / 3)
    J0 = truelove_number(U, 5 / 3, G_CODE, g.dx)
    e1, m1 = hy.apply_jeans_floor(U, 5 / 3, G_CODE, g.dx, V["floor_NJ"])
    J1 = truelove_number(U, 5 / 3, G_CODE, g.dx)
    e2, _ = hy.apply_jeans_floor(U, 5 / 3, G_CODE, g.dx, V["floor_NJ"])
    lim = 1 / V["floor_NJ"]
    R.add("X2", "J vor Boden", J0)
    R.add("X2", "J nach Boden (≤ 1/N_J)", J1, lim, J1 <= lim * (1 + V["floor_rel_tol"]))  # Toleranz 1e-12 im Kriterium
    R.add("X2", "gebodete Zellen [%]", 100 * m1.mean())
    R.add("X2", "zweite Anwendung: injizierte Energie (idempotent = 0)", e2, 0.0, e2 == 0.0)
    viz.slices_row(figs / "p4x_X2_floor.png", [(rho, "ρ (Lognormal-Test)", "log"), (m1.astype(float), "gebodet", "lin")],
                   g.box, dpi=dpi, suptitle=f"X2: Jeans-Boden N_J = {V['floor_NJ']}")


def x3_cooling(cfg, figs, figs3, dpi):
    V = cfg["validation_p4x"]
    gp = cfg["gas"]
    cp, mu, gam = gp["cooling_params"], gp["mu"], gp["gamma"]
    r_ = co.primordial_rate_cgs(np.array(1.0), np.array(cp["T_floor_K"] * 0.99), cp["x_H2"], cp["T_floor_K"])
    R.add("X3", "Λ unter T_floor", float(r_), 0.0, float(r_) == 0.0)
    rho = gas_mass_density(0.1, mu)
    T0 = 2000.0
    tt = 0.5 * float(co.cooling_time_model_myr("primordial_h2", np.array(rho), np.array(T0), mu, gam, cp))
    dedT = co.pressure_from_T(rho, 1.0, mu) / (gam - 1)
    rate = lambda T: float(co._rate_code("primordial_h2", np.array(rho), np.array(T), mu, cp))  # noqa: E731
    ref = solve_ivp(lambda t, y: [-rate(y[0]) / dedT], (0, tt), [T0], method="LSODA", rtol=1e-12, atol=1e-8).y[0, -1]
    errs = []
    for ns in [4, 16, 64, 256]:
        U = hy.cons_from_prim(np.full((2, 2, 2), rho), 0, 0, 0, co.pressure_from_T(np.full((2, 2, 2), rho), T0, mu), gam)
        for _ in range(ns):
            co.apply_cooling(U, tt / ns, "primordial_h2", gam, mu, None, cparams=cp)
        errs.append(abs(co.temperature(U, gam, mu)[0, 0, 0] / ref - 1))
    p = float(np.log(errs[-2] / errs[-1]) / np.log(4))
    R.add("X3", "Ordnung implizite Kühlung (n=0.1, T=2000 K)", p, 1.0, abs(p - 1) < V["cool_order_tol"],
          note=f"T_ref = {ref:.0f} K nach {tt:.3g} Myr")
    for n in (0.27, 10.0):
        for T in (300.0, 1e3, 1e4):
            th = float(co.cooling_time_myr(np.array(gas_mass_density(n, mu)), np.array(T), mu, gam))
            tp = float(co.cooling_time_model_myr("primordial_h2", np.array(gas_mass_density(n, mu)), np.array(T), mu, gam, cp))
            R.add("X3", f"t_cool n={n:g}, T={T:g} K: primordial [Myr]", tp, note=f"historisch: {th:.3g} Myr")
    TT = np.geomspace(30, 1e7, 400)
    nH = 1.0
    x = co.h_cie_ionized_fraction(TT)
    comp = {"gesamt (n_H=1)": co.primordial_rate_cgs(nH, TT, cp["x_H2"], cp["T_floor_K"]) / nH**2 + 1e-40,
            "H2 (GP98)·x_H2": np.where(TT < 1e4, cp["x_H2"] * co.lambda_h2_gp98_low_density(TT), 0) + 1e-40,
            "historisch": co.lambda_historical(TT) + 1e-40}
    viz.line_plot(figs / "p4x_X3_lambda.png", TT, comp, xlabel="T [K]", ylabel="Λ/n_H² [erg cm³ s⁻¹]",
                  title="X3: primordiale vs. historische Kühlfunktion", dpi=dpi, logx=True, logy=True)
    viz.line_plot(figs / "p4x_X3_ionization.png", TT, {"x_e (CIE)": x + 1e-30}, xlabel="T [K]", ylabel="x",
                  title="X3: Ionisationsgrad CIE", dpi=dpi, logx=True, logy=True)
    ns, Ts = np.geomspace(1e-3, 1e3, 50), np.geomspace(60, 1e6, 60)
    TC = np.array([[float(co.cooling_time_model_myr("primordial_h2", np.array(gas_mass_density(n, mu)), np.array(T), mu, gam, cp))
                    for n in ns] for T in Ts])
    TC = np.where(np.isfinite(TC), TC, np.nan)
    viz.heatmap(figs / "p4x_X3_tcool.png", ns, Ts, TC, xlabel="n [cm⁻³]", ylabel="T [K]", zlabel="t_cool [Myr]",
                title="X3: Kühlzeit primordial (B6)", dpi=dpi, logx=True, logy=True, logz=True, contour_levels=[1.0, 30.0])
    viz.surface3d_png(figs3 / "p4x_X3_tcool_surface.png", ns, Ts, np.nan_to_num(TC, nan=np.nanmax(TC)), xlabel="n [cm⁻³]",
                      ylabel="T [K]", zlabel="t_cool [Myr]", title="t_cool(n, T) primordial", dpi=dpi, log_axes=(True, True, True))
    viz.surface3d_html(figs3 / "p4x_X3_tcool_surface.html", ns, Ts, np.nan_to_num(TC, nan=np.nanmax(TC)), xlabel="n [cm⁻³]",
                       ylabel="T [K]", zlabel="t_cool [Myr]", title="t_cool(n, T) primordial", log_axes=(True, True, True))


def x4_seed(cfg):
    g = Grid3D(cfg["domain"]["box_kpc"], 16)
    c2 = copy.deepcopy(cfg)
    c2["meta"]["seed"] = cfg["meta"]["seed"] + 1
    same = np.array_equal(gas_initial_condition(cfg, g, 1.0), gas_initial_condition(cfg, g, 1.0))
    diff = not np.array_equal(gas_initial_condition(cfg, g, 1.0), gas_initial_condition(c2, g, 1.0))
    R.add("X4", "gleicher Seed -> bitgleich", float(same), 1.0, same)
    R.add("X4", "anderer Seed -> verschieden", float(diff), 1.0, diff)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P4X")
    figs, figs3, dpi, w = out / "figs", out / "figs3d", cfg["viz"]["dpi"], cfg["compute"]["fft_workers"]
    for name, fn in [("X1", lambda: x1_halo(cfg, figs, figs3, dpi, w)), ("X2", lambda: x2_floor(cfg, figs, figs3, dpi)),
                     ("X3", lambda: x3_cooling(cfg, figs, figs3, dpi)), ("X4", lambda: x4_seed(cfg))]:
        t = time.perf_counter()
        fn()
        print(f"  {name} fertig ({time.perf_counter()-t:.0f} s)", flush=True)
    R.write(out, "P4X", "Modellerweiterungen B4–B7", ["Schwellen vorab in `validation_p4x`."])
    write_manifest(cfg, out, "P4X", extra=R.results)
    print(f"P4X fertig -> {out} ({'alle bestanden' if R.n_failed() == 0 else f'{R.n_failed()} FEHLGESCHLAGEN'})")
    if R.n_failed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
