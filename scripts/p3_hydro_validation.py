"""P3 – Hydrodynamik (MUSCL-Hancock/HLLC), Gravitationsquellterm, Kühlung.

H1  Sod-Stoßrohr entlang x, y, z gegen exakten Riemann-Löser; L1-Konvergenz
H2  Erhaltung von Masse, Impuls, Energie (periodisch, ohne Quellen) bis Rundung
H3  Lineare Jeans-Dispersion ω² = c_s²k² − 4πGρ₀ mit Selbstgravitation (Research Plan Gl. 13)
H4  Hydrostatisches Gas im Soliton-Potential: Rauschboden künstlicher Strömungen (relevant für O3)
H5  Kühlung: implizite Integration gegen Referenz-ODE; Kühlzeit-Karte des historischen Modells

Aufruf: python -m scripts.p3_hydro_validation --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from hgr import cooling as co
from hgr import hydro as hy
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.grid import Grid3D
from hgr.poisson import gradient_spectral, solve_poisson_periodic
from hgr.report import Recorder
from hgr.riemann_exact import sample
from hgr.soliton import embed_ground_state
from hgr.units import G_CODE, KMS_TO_CODE, AxionUnits, gas_mass_density

R = Recorder()


# ------------------------------------------------------------------ H1 ----
def _sod(N, axis, t_end, cfl, lim, snaps=None):
    g = 1.4
    x = (np.arange(N) + 0.5) / N
    shape = [4, 4, 4]
    shape[axis] = N
    xs = x.reshape([-1 if i == axis else 1 for i in range(3)])
    left = xs < 0.5
    rho = np.where(left, 1.0, 0.125) * np.ones(shape)
    p = np.where(left, 1.0, 0.1) * np.ones(shape)
    U = hy.cons_from_prim(rho, 0 * rho, 0 * rho, 0 * rho, p, g)
    t, k = 0.0, 0
    hist = [(0.0, np.moveaxis(U[0], axis, 0)[:, 0, 0].copy())]
    while t < t_end - 1e-14:
        dt = min(hy.cfl_dt(U, 1 / N, g, cfl), t_end - t)
        hy.step(U, dt, 1 / N, g, lim, k)
        t += dt
        k += 1
        if snaps is not None:
            hist.append((t, np.moveaxis(U[0], axis, 0)[:, 0, 0].copy()))
    W = np.moveaxis(U, axis + 1, 1)[:, :, 0, 0]
    return x, W[0], W[1 + axis] / W[0], hy.pressure(np.moveaxis(U, axis + 1, 1), g)[:, 0, 0], hist


def h1_sod(cfg, figs, figs3, dpi):
    V = cfg["validation_p3"]
    lim = cfg["hydro"]["limiter"]
    t_end = V["sod_t"]
    errs = []
    for N in V["sod_N"]:
        e_ax = []
        for ax in range(3):
            x, r, u, p, _ = _sod(N, ax, t_end, V["sod_cfl"], lim)
            sel = (x > 0.25) & (x < 0.75)
            re, ue, pe = sample((x[sel] - 0.5) / t_end, 1, 0, 1, 0.125, 0, 0.1, 1.4)
            e_ax.append(np.mean(np.abs(r[sel] - re)) * 0.5)  # L1 über Länge 0.5
        errs.append(e_ax)
        R.add("H1", f"L1(ρ) N={N} (x/y/z max)", max(e_ax))
        R.add("H1", f"Achsen-Asymmetrie N={N}", max(e_ax) - min(e_ax), 1e-14, max(e_ax) - min(e_ax) < 1e-14,
              note="Split-Schema muss richtungsunabhängig sein")
    errs = np.array(errs)[:, 0]
    i400 = V["sod_N"].index(400)
    R.add("H1", "L1(ρ) bei N=400", errs[i400], V["sod_l1_max_at_400"], errs[i400] < V["sod_l1_max_at_400"])
    p_obs = np.log2(errs[-2] / errs[-1])
    R.add("H1", "beobachtete Ordnung L1 (letzte Verdopplung)", p_obs, V["sod_order_min"], p_obs > V["sod_order_min"],
          note="Stoß/Kontakt begrenzen auf ~1")
    N = 400
    x, r, u, p, hist = _sod(N, 0, t_end, V["sod_cfl"], lim, snaps=True)
    re, ue, pe = sample((x - 0.5) / t_end, 1, 0, 1, 0.125, 0, 0.1, 1.4)
    for name, num, ex in [("rho", r, re), ("u", u, ue), ("p", p, pe)]:
        viz.line_plot(figs / f"p3_H1_sod_{name}.png", x, {f"{name} numerisch (N={N})": num, f"{name} exakt": ex},
                      xlabel="x", ylabel=name, title=f"H1: Sod t={t_end}", dpi=dpi)
    viz.line_plot(figs / "p3_H1_sod_convergence.png", np.array(V["sod_N"], float),
                  {"L1(ρ)": errs, "∝ 1/N": errs[0] * V["sod_N"][0] / np.array(V["sod_N"], float)},
                  xlabel="N", ylabel="L1-Fehler", title="H1: Konvergenz Sod", dpi=dpi, logx=True, logy=True, markers=True)
    ts = np.array([h[0] for h in hist])
    Z = np.array([h[1] for h in hist])
    viz.heatmap(figs / "p3_H1_sod_spacetime.png", x, ts, Z, xlabel="x", ylabel="t", zlabel="ρ",
                title="H1: Raum-Zeit-Diagramm ρ(x,t)", dpi=dpi)
    viz.surface3d_png(figs3 / "p3_H1_sod_surface.png", x[::2], ts, Z[:, ::2], xlabel="x", ylabel="t", zlabel="ρ",
                      title="H1: ρ(x, t)", dpi=dpi)
    viz.surface3d_html(figs3 / "p3_H1_sod_surface.html", x[::2], ts, Z[:, ::2], xlabel="x", ylabel="t",
                       zlabel="ρ", title="H1: ρ(x, t)")


# ------------------------------------------------------------------ H2 ----
def h2_conservation(cfg, figs, figs3, dpi):
    V = cfg["validation_p3"]
    N = V["conservation_N"]
    g = Grid3D(1.0, N)
    x, y, z = g.mesh_sparse
    k = 2 * np.pi
    rho = 1 + 0.3 * np.sin(k * x) * np.cos(2 * k * y) + 0.2 * np.cos(k * z)
    U = hy.cons_from_prim(rho, 0.3 * np.sin(k * y), 0.2 * np.cos(k * z) * np.sin(k * x), 0.1 * np.sin(2 * k * x),
                          1 + 0.2 * np.cos(k * (x + y + z)), 5 / 3)
    T0 = hy.totals(U, g.cell_volume)
    nfb = 0
    for s in range(V["conservation_steps"]):
        nfb += hy.step(U, hy.cfl_dt(U, g.dx, 5 / 3, 0.4), g.dx, 5 / 3, cfg["hydro"]["limiter"], s)
    T1 = hy.totals(U, g.cell_volume)
    names = ["Masse", "Impuls x", "Impuls y", "Impuls z", "Energie"]
    scale = np.array([T0[0], T0[0], T0[0], T0[0], T0[4]])  # Impuls relativ zur Masse (×1 Geschw.-Einheit)
    rel = np.abs(T1 - T0) / scale
    for n, r_ in zip(names, rel):
        R.add("H2", f"rel. Änderung {n}", r_, V["conservation_rel_max"], r_ < V["conservation_rel_max"])
    R.add("H2", "lokale Rückfälle auf 1. Ordnung", nfb)
    viz.bar_log(figs / "p3_H2_conservation.png", names, [max(v, 1e-18) for v in rel], ylabel="relative Änderung",
                title=f"H2: Erhaltung nach {V['conservation_steps']} Schritten", dpi=dpi)
    viz.ortho_slices_3d_png(figs3 / "p3_H2_rho_end_ortho.png", U[0], 1.0, title="H2: ρ am Ende", dpi=dpi,
                            log=False, label="ρ")


# ------------------------------------------------------------------ H3 ----
def _self_gravity_accel(U, g, G, workers):
    phi = solve_poisson_periodic(U[0], g, G, workers)
    return tuple(-d for d in gradient_spectral(phi, g, workers))


def h3_jeans(cfg, figs, figs3, dpi, workers):
    V = cfg["validation_p3"]
    N = V["jeans_N"]
    g = Grid3D(1.0, N)
    G, rho0, gam = 1.0, 1.0, 5.0 / 3.0
    kJ = V["jeans_kJ_modes"] * 2 * np.pi
    cs = np.sqrt(4 * np.pi * G * rho0) / kJ
    p0 = rho0 * cs**2 / gam
    x = g.mesh_sparse[0] + 0 * g.mesh_sparse[1] + 0 * g.mesh_sparse[2]
    k2s, w2_num, w2_ana = [], [], []
    for m in V["jeans_modes"]:
        k = 2 * np.pi * m
        w2 = cs**2 * k**2 - 4 * np.pi * G * rho0
        A = V["jeans_amp"]
        rho = rho0 * (1 + A * np.cos(k * x))
        U = hy.cons_from_prim(rho, 0 * x, 0 * x, 0 * x, p0 + cs**2 * (rho - rho0), gam)
        T = V["jeans_periods"] * 2 * np.pi / np.sqrt(w2) if w2 > 0 else 3.0 / np.sqrt(-w2)
        t, s, ts, amps, prof = 0.0, 0, [0.0], [A], [rho[:, 0, 0] / rho0 - 1]
        acc = _self_gravity_accel(U, g, G, workers)
        while t < T - 1e-14:
            dt = min(hy.cfl_dt(U, g.dx, gam, cfg["time"]["cfl_hydro"]), T - t, T / 200)
            hy.gravity_half_kick(U, acc, 0.5 * dt)
            hy.step(U, dt, g.dx, gam, cfg["hydro"]["limiter"], s)
            acc = _self_gravity_accel(U, g, G, workers)
            hy.gravity_half_kick(U, acc, 0.5 * dt)
            t += dt
            s += 1
            line = U[0][:, 0, 0] / rho0 - 1
            ts.append(t)
            amps.append(2 * np.mean(line * np.cos(k * g.x1d)))
            prof.append(line)
        ts, amps = np.array(ts), np.array(amps) / A
        if w2 > 0:
            fit = minimize_scalar(lambda w: np.sum((amps - np.cos(w * ts)) ** 2),
                                  bounds=(0.5 * np.sqrt(w2), 1.5 * np.sqrt(w2)), method="bounded")
            wn2 = fit.x**2
            ana = np.cos(np.sqrt(w2) * ts)
            label = "stabil (Oszillation)"
        else:
            fit = minimize_scalar(lambda q: np.sum((np.log(np.abs(amps)) - np.log(np.cosh(q * ts))) ** 2),
                                  bounds=(0.5 * np.sqrt(-w2), 1.5 * np.sqrt(-w2)), method="bounded")
            wn2 = -fit.x**2
            ana = np.cosh(np.sqrt(-w2) * ts)
            label = "instabil (Wachstum)"
        rel = abs(wn2 / w2 - 1)
        R.add("H3", f"Mode {m} (k/k_J={m/V['jeans_kJ_modes']:.2f}) rel. Fehler ω²", rel, V["jeans_rel_err_max"],
              rel < V["jeans_rel_err_max"], note=label)
        k2s.append(k**2), w2_num.append(wn2), w2_ana.append(w2)
        viz.line_plot(figs / f"p3_H3_jeans_mode{m}.png", ts, {"numerisch": amps, "linear exakt": ana},
                      xlabel="t (Code)", ylabel="δρ̂_k(t)/δρ̂_k(0)", title=f"H3: Jeans-Mode {m} – {label}", dpi=dpi)
        P = np.array(prof)
        stride = max(1, len(ts) // 80)
        viz.surface3d_png(figs3 / f"p3_H3_jeans_mode{m}_surface.png", g.x1d, ts[::stride], P[::stride],
                          xlabel="x", ylabel="t", zlabel="δρ/ρ₀", title=f"H3: δρ(x,t), Mode {m}", dpi=dpi)
    kk = np.linspace(0, max(k2s) * 1.1, 200)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(kk, cs**2 * kk - 4 * np.pi * G * rho0, label="exakt")
    ax.plot(k2s, w2_num, "o", label="numerisch")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("k²"), ax.set_ylabel("ω²"), ax.set_title("H3: Jeans-Dispersion"), ax.legend()
    fig.savefig(figs / "p3_H3_dispersion.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ H4 ----
def h4_hydrostatic(cfg, figs, figs3, dpi, workers):
    V = cfg["validation_p3"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["hydrostatic_N"])
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    sp = cfg["sp"]
    pol = {"coarse_n": sp["polish_coarse_n"], "coarse_steps": sp["polish_coarse_steps"],
           "coarse_dtau_T0": sp["polish_coarse_dtau_T0"], "fine_steps": sp["polish_fine_steps"],
           "fine_dtau_T0": sp["polish_fine_dtau_T0"]}
    _, phi, _ = embed_ground_state(g, cfg["axion"]["soliton_mass_msun"], a, G_CODE, pol, workers)
    acc = tuple(-d for d in gradient_spectral(phi, g, workers))
    cs2 = (phi.max() - phi.min()) / V["hydrostatic_log_contrast"]
    gp = cfg["gas"]
    rho_c = gas_mass_density(gp["n_cm3"], gp["mu"])
    rho = rho_c * np.exp(-(phi - phi.min()) / cs2)
    gam = gp["gamma"]
    U = hy.cons_from_prim(rho, 0 * rho, 0 * rho, 0 * rho, rho * cs2, gam)
    T_iso = co.temperature(U, gam, gp["mu"]).max()
    t, s, ts, vmax, dcen = 0.0, 0, [0.0], [0.0], [0.0]
    c = g.center_index
    r00 = rho[c, c, c]
    while t < V["hydrostatic_t_myr"] - 1e-12:
        dt = min(hy.cfl_dt(U, g.dx, gam, cfg["time"]["cfl_hydro"]), V["hydrostatic_t_myr"] - t)
        hy.gravity_half_kick(U, acc, 0.5 * dt)
        hy.step(U, dt, g.dx, gam, cfg["hydro"]["limiter"], s)
        hy.gravity_half_kick(U, acc, 0.5 * dt)
        t += dt
        s += 1
        vmag = np.sqrt(np.sum(hy.velocity(U) ** 2, axis=0))
        ts.append(t), vmax.append(vmag.max() / np.sqrt(cs2)), dcen.append(U[0][c, c, c] / r00 - 1)
    R.add("H4", "Testtemperatur (isotherm) [K]", T_iso, note="aus vorgegebenem Dichtekontrast")
    R.add("H4", "max |v|/c_s nach t_end", vmax[-1], V["hydrostatic_vmax_over_cs_max"],
          vmax[-1] < V["hydrostatic_vmax_over_cs_max"], note="künstliche Strömung = Rauschboden für O3")
    R.add("H4", "max |v| [km/s]", vmax[-1] * np.sqrt(cs2) / KMS_TO_CODE)
    R.add("H4", "rel. Änderung ρ(0)", dcen[-1])
    viz.line_plot(figs / "p3_H4_noise.png", np.array(ts), {"max|v|/c_s": np.array(vmax) + 1e-18,
                  "|Δρ(0)|/ρ(0)": np.abs(np.array(dcen)) + 1e-18}, xlabel="t [Myr]", ylabel="relativ",
                  title=f"H4: Gleichgewichtserhalt im Soliton-Potential (N={g.n})", dpi=dpi, logy=True)
    vfield = np.sqrt(np.sum(hy.velocity(U) ** 2, axis=0)) / np.sqrt(cs2)
    viz.slices_row(figs / "p3_H4_slices.png", [(U[0], "ρ_b", "log"), (vfield, "|v|/c_s", "lin"),
                   (U[0] / rho - 1, "Δρ/ρ", "sym")], L, dpi=dpi, suptitle="H4: hydrostatisches Gas, Endzustand")
    viz.ortho_slices_3d_png(figs3 / "p3_H4_vel_ortho.png", vfield, L, title="H4: |v|/c_s (Rauschboden)", dpi=dpi,
                            log=False, label="|v|/c_s")
    viz.isosurface_html(figs3 / "p3_H4_rho_iso.html", U[0], L, title="H4: ρ_b im Soliton-Potential", log=True,
                        label="ρ_b", max_points=cfg["viz"]["iso_max_points_per_axis"])


# ------------------------------------------------------------------ H5 ----
def h5_cooling(cfg, figs, figs3, dpi):
    V = cfg["validation_p3"]
    gp = cfg["gas"]
    mu, gam, T0 = gp["mu"], gp["gamma"], V["cooling_T0_K"]
    for n in V["cooling_n_cm3"]:
        rho = np.full((2, 2, 2), gas_mass_density(n, mu))
        U = hy.cons_from_prim(rho, 0, 0, 0, co.pressure_from_T(rho, T0, mu), gam)
        co.apply_cooling(U, V["cooling_dt_myr"], "piecewise_historical", gam, mu, None)
        T_be = co.temperature(U, gam, mu)[0, 0, 0]
        # Referenz: dT/dt = −(γ−1) n²Λ μ m_H /(ρ k_B) über dieselbe Zeit, Stop bei T=300 K (Λ springt auf 0)
        r1 = rho[0, 0, 0]
        e_of_T = lambda T: co.pressure_from_T(r1, T, mu) / (gam - 1)  # noqa: E731
        rate = lambda T: co.number_density_cm3(r1, mu) ** 2 * co.lambda_historical(T) * co._RATE_TO_CODE  # noqa: E731
        dedT = e_of_T(1.0)
        ev = lambda t, y: y[0] - 300.0  # noqa: E731
        ev.terminal = True
        sol = solve_ivp(lambda t, y: [-rate(y[0]) / dedT], (0, V["cooling_dt_myr"]), [T0], method="LSODA",
                        rtol=1e-12, atol=1e-10, events=ev)
        T_ref = 300.0 if sol.t_events[0].size else sol.y[0, -1]
        rel = abs(T_be / T_ref - 1)
        tc = float(co.cooling_time_myr(np.array(r1), np.array(T0), mu, gam))
        R.add("H5", f"n={n}: T nach {V['cooling_dt_myr']} Myr (implizit) [K]", T_be)
        R.add("H5", f"n={n}: rel. Abweichung zur Referenz-ODE", rel, V["cooling_rel_err_max"],
              rel < V["cooling_rel_err_max"])
        R.add("H5", f"n={n}: Kühlzeit bei T0={T0:g} K [s]", tc * 3.15576e13, note="historisches Λ")
    # Nicht-steifer Fall (T > 1e4 K, n = 1e-3): Rückwärts-Euler muss mit Ordnung 1 konvergieren
    r1 = gas_mass_density(1e-3, mu)
    T_hot = 1e5
    t_tot = 0.5 * float(co.cooling_time_myr(np.array(r1), np.array(T_hot), mu, gam))
    dedT = co.pressure_from_T(r1, 1.0, mu) / (gam - 1)
    rate = lambda T: co.number_density_cm3(r1, mu) ** 2 * co.lambda_historical(T) * co._RATE_TO_CODE  # noqa: E731
    ref = solve_ivp(lambda t, y: [-rate(y[0]) / dedT], (0, t_tot), [T_hot], method="LSODA", rtol=1e-12,
                    atol=1e-8).y[0, -1]
    nsubs, errs = [1, 4, 16, 64, 256], []
    for ns_ in nsubs:
        rho = np.full((2, 2, 2), r1)
        U = hy.cons_from_prim(rho, 0, 0, 0, co.pressure_from_T(rho, T_hot, mu), gam)
        for _ in range(ns_):
            co.apply_cooling(U, t_tot / ns_, "piecewise_historical", gam, mu, None)
        errs.append(abs(co.temperature(U, gam, mu)[0, 0, 0] / ref - 1))
    p_be = np.log(errs[-2] / errs[-1]) / np.log(4)
    R.add("H5", "Rückwärts-Euler Ordnung (nicht-steif, T=1e5 K)", p_be, 1.0, abs(p_be - 1) < 0.15,
          note=f"T_ref = {ref:.0f} K nach t = {t_tot:.3g} Myr")
    viz.line_plot(figs / "p3_H5_be_convergence.png", np.array(nsubs, float), {"rel. Fehler T": np.array(errs),
                  "∝ 1/n_sub": errs[0] / np.array(nsubs, float)}, xlabel="Teilschritte", ylabel="|T/T_ref − 1|",
                  title="H5: Konvergenz der impliziten Kühlung", dpi=dpi, logx=True, logy=True, markers=True)
    ns = np.geomspace(1e-3, 1e2, 60)
    Ts = np.geomspace(310, 1e5, 60)
    TC = np.array([[float(co.cooling_time_myr(np.array(gas_mass_density(n, mu)), np.array(T), mu, gam))
                    for n in ns] for T in Ts])
    viz.heatmap(figs / "p3_H5_tcool_map.png", ns, Ts, TC, xlabel="n [cm⁻³]", ylabel="T [K]", zlabel="t_cool [Myr]",
                title="H5: Kühlzeit des HISTORISCHEN Modells", dpi=dpi, logx=True, logy=True, logz=True,
                contour_levels=[1e-9, 1e-3, 1.0])
    viz.surface3d_png(figs3 / "p3_H5_tcool_surface.png", ns, Ts, TC, xlabel="n [cm⁻³]", ylabel="T [K]",
                      zlabel="t_cool [Myr]", title="H5: t_cool(n, T) historisch", dpi=dpi, log_axes=(True, True, True))
    viz.surface3d_html(figs3 / "p3_H5_tcool_surface.html", ns, Ts, TC, xlabel="n [cm⁻³]", ylabel="T [K]",
                       zlabel="t_cool [Myr]", title="H5: t_cool(n, T) historisch", log_axes=(True, True, True))
    TT = np.geomspace(10, 1e6, 400)
    viz.line_plot(figs / "p3_H5_lambda.png", TT, {"Λ(T) historisch": co.lambda_historical(TT) + 1e-40},
                  xlabel="T [K]", ylabel="Λ [erg cm³ s⁻¹]", title="H5: historische Kühlfunktion (Sprünge bei 300 K, 10⁴ K)",
                  dpi=dpi, logx=True, logy=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P3")
    figs, figs3 = out / "figs", out / "figs3d"
    dpi, workers = cfg["viz"]["dpi"], cfg["compute"]["fft_workers"]
    for name, fn in [("H1", lambda: h1_sod(cfg, figs, figs3, dpi)),
                     ("H2", lambda: h2_conservation(cfg, figs, figs3, dpi)),
                     ("H3", lambda: h3_jeans(cfg, figs, figs3, dpi, workers)),
                     ("H4", lambda: h4_hydrostatic(cfg, figs, figs3, dpi, workers)),
                     ("H5", lambda: h5_cooling(cfg, figs, figs3, dpi))]:
        t = time.perf_counter()
        fn()
        print(f"  {name} fertig ({time.perf_counter()-t:.0f} s)", flush=True)
    R.write(out, "P3", "Hydrodynamik, Gravitationsquellterm, Kühlung",
            [f"Limiter: `{cfg['hydro']['limiter']}`. Schwellen vorab in `validation_p3` festgelegt."])
    write_manifest(cfg, out, "P3", extra=R.results)
    print(f"P3 fertig -> {out} ({'alle bestanden' if R.n_failed() == 0 else f'{R.n_failed()} FEHLGESCHLAGEN'})")
    if R.n_failed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
