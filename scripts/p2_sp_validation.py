"""P2 – Schrödinger-Poisson-Solver und Soliton-Grundzustand.

S0  Grundzustand: Shooting-Profil, Skalierung, Vergleich mit Schive-Fit, Eigenzustands-Residuum
S1  Stationarität des isolierten Solitons: ρ(0,t), Norm, Energie, Schwerpunkt
S2  Freies Wellenpaket (G=0) gegen analytische Lösung (prüft Drift-Operator exakt)
S3  Galilei-Boost: bewegter Soliton, Schwerpunkt vs. v·t, Formerhalt
S4  Zeitliche Konvergenzordnung des selbstkonsistenten KDK-Schritts (erwartet 2)

Aufruf: python -m scripts.p2_sp_validation --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import plotly.graph_objects as go

from hgr import analytic as an
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.grid import Grid3D
from hgr.report import Recorder
from hgr.schroedinger import center_of_mass, energies, mass, potential, stable_dt, step_kdk
from hgr.soliton import embed_ground_state, ground_state_profile, physical_scales
from hgr.units import G_CODE, AxionUnits, KMS_TO_CODE

R = Recorder()


def radial_profile(field: np.ndarray, grid: Grid3D, edges: np.ndarray) -> np.ndarray:
    r = np.broadcast_to(grid.r, grid.shape).ravel()
    idx = np.digitize(r, edges)
    s = np.bincount(idx, weights=field.ravel(), minlength=len(edges) + 1)
    c = np.bincount(idx, minlength=len(edges) + 1)
    with np.errstate(invalid="ignore"):
        return (s / c)[1:len(edges)]


def polish_cfg(cfg):
    sp = cfg["sp"]
    return {"coarse_n": sp["polish_coarse_n"], "coarse_steps": sp["polish_coarse_steps"],
            "coarse_dtau_T0": sp["polish_coarse_dtau_T0"], "fine_steps": sp["polish_fine_steps"],
            "fine_dtau_T0": sp["polish_fine_dtau_T0"]}


def s0_ground_state(cfg, figs, figs3, dpi, workers, a, M):
    V = cfg["validation_p2"]
    L = cfg["domain"]["box_kpc"]
    prof = ground_state_profile()
    sc = physical_scales(prof, M, a, G_CODE)
    m22 = cfg["axion"]["m22"]
    rc_s = an.schive_core_radius_from_mass(m22, M)
    R.add("S0", "U(0) (Shooting)", prof.u0, note="dimensionslos")
    R.add("S0", "μ = 4π∫f²ξ²dξ", prof.mu, note="Literatur ≈ 25.9 (gleiche Skalierung)")
    R.add("S0", "ξ_c (f²=1/2)", prof.xi_c)
    R.add("S0", "r_c numerisch [kpc]", sc["rc"])
    R.add("S0", "r_c Schive-Fit [kpc]", rc_s, note="empirischer Fit")
    R.add("S0", "r_c numerisch / Schive", sc["rc"] / rc_s, note="Fit-Genauigkeit ~ wenige %")
    R.add("S0", "ρ(0) numerisch [M_sun/kpc³]", sc["rho0"])
    R.add("S0", "Soliton-Zeitskala T0 = L0²/a [Myr]", sc["T0"])
    g = Grid3D(L, V["N_main"])
    t0 = time.perf_counter()
    psi, phi, info = embed_ground_state(g, M, a, G_CODE, polish_cfg(cfg), workers)
    R.add("S0", "Residuum direkt interpoliert", info["residual_direct"], note="Boxeffekt + Interpolation")
    R.add("S0", "Residuum nach Politur", info["residual_final"])
    R.add("S0", "Zeit Grundzustand [s]", time.perf_counter() - t0)
    # Abbildungen
    xi = prof.xi[prof.xi < 8]
    f = prof.f[: len(xi)]
    schive_shape = (1 + an.SCHIVE_A * (xi / prof.xi_c) ** 2) ** -8
    viz.line_plot(figs / "p2_S0_profile_dimless.png", xi, {"f²(ξ) numerisch": f**2, "Schive-Fit (gleiches ξ_c)": schive_shape},
                  xlabel="ξ = r/L₀", ylabel="ρ/ρ(0)", title="S0: SP-Grundzustand (dimensionslos)", dpi=dpi,
                  logy=True, vlines={"ξ_c": prof.xi_c})
    edges = np.linspace(0, L / 2, 65)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    viz.line_plot(figs / "p2_S0_profile_grid.png", rmid,
                  {"|ψ|² Gitter (poliert)": radial_profile(np.abs(psi) ** 2, g, edges),
                   "Shooting (isoliert)": sc["A"] ** 2 * np.interp(rmid / sc["L0"], prof.xi, prof.f) ** 2,
                   "Schive-Fit": an.schive_density(rmid, m22, rc_s)},
                  xlabel="r [kpc]", ylabel="ρ_a [M_sun/kpc³]", title=f"S0: Soliton im Gitter N={g.n}", dpi=dpi,
                  logy=True, vlines={"r_c": sc["rc"]})
    viz.slices_row(figs / "p2_S0_slices.png", [(np.abs(psi) ** 2, "ρ_a = |ψ|²", "log"),
                   (np.angle(psi), "Phase arg ψ", "lin"), (phi, "Φ", "lin")], L, dpi=dpi,
                   suptitle="S0: Grundzustand")
    viz.ortho_slices_3d_png(figs3 / "p2_S0_rho_ortho.png", np.abs(psi) ** 2, L, title="S0: ρ_a", dpi=dpi,
                            log=True, label="ρ_a")
    viz.isosurface_html(figs3 / "p2_S0_rho_iso.html", np.abs(psi) ** 2, L, title="S0: Soliton ρ_a", log=True,
                        label="ρ_a", max_points=cfg["viz"]["iso_max_points_per_axis"])
    return g, psi, phi, sc


def s1_stationarity(cfg, figs, figs3, dpi, workers, a, g, psi, phi):
    V = cfg["validation_p2"]
    tp = cfg["time"]
    c = g.center_index
    m0, e0 = mass(psi, g), energies(psi, phi, g, a, workers)["tot"]
    r00 = abs(psi[c, c, c]) ** 2
    dt = stable_dt(g, phi, a, tp["sp_kinetic_dt_factor"], tp["sp_potential_phase_max_rad"])
    nsteps = int(np.ceil(V["t_stationary_myr"] / dt))
    dt = V["t_stationary_myr"] / nsteps
    edges = np.linspace(0, g.box / 2, 49)
    T, RHO0, NRM, EN, COM, PROF = [0.0], [1.0], [0.0], [0.0], [0.0], [radial_profile(np.abs(psi) ** 2, g, edges)]
    every = max(1, nsteps // 60)
    for k in range(1, nsteps + 1):
        psi, phi = step_kdk(psi, phi, None, g, dt, a, G_CODE, workers)
        if k % every == 0 or k == nsteps:
            T.append(k * dt)
            RHO0.append(abs(psi[c, c, c]) ** 2 / r00)
            NRM.append(abs(mass(psi, g) / m0 - 1))
            EN.append(abs(energies(psi, phi, g, a, workers)["tot"] / e0 - 1))
            COM.append(float(np.linalg.norm(center_of_mass(psi, g))))
            PROF.append(radial_profile(np.abs(psi) ** 2, g, edges))
    T, RHO0 = np.array(T), np.array(RHO0)
    var = float(np.max(np.abs(RHO0 - 1)))
    R.add("S1", f"max|ρ(0,t)/ρ(0,0) − 1| über {V['t_stationary_myr']} Myr", var, V["rho0_variation_max"],
          var < V["rho0_variation_max"])
    R.add("S1", "max Normdrift", max(NRM), V["norm_drift_max"], max(NRM) < V["norm_drift_max"])
    R.add("S1", "max rel. Energiedrift", max(EN), V["energy_drift_max"], max(EN) < V["energy_drift_max"])
    R.add("S1", "max Schwerpunktsverschiebung [kpc]", max(COM), note=f"dx = {g.dx:.3g} kpc")
    R.add("S1", "Zeitschritt dt [Myr]", dt)
    R.add("S1", "Schritte", nsteps)
    viz.line_plot(figs / "p2_S1_rho0.png", T, {"ρ(0,t)/ρ(0,0)": RHO0}, xlabel="t [Myr]", ylabel="ρ(0,t)/ρ(0,0)",
                  title="S1: Zentraldichte des Solitons", dpi=dpi, hlines={"1": 1.0})
    viz.line_plot(figs / "p2_S1_conservation.png", T[1:], {"|ΔN|/N": np.array(NRM[1:]) + 1e-18,
                  "|ΔE|/|E|": np.array(EN[1:]) + 1e-18, "Schwerpunkt [kpc]": np.array(COM[1:]) + 1e-18},
                  xlabel="t [Myr]", ylabel="relativ / kpc", title="S1: Erhaltungsgrößen", dpi=dpi, logy=True)
    P = np.array(PROF)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    viz.heatmap(figs / "p2_S1_profile_heatmap.png", rmid, T, np.log10(P), xlabel="r [kpc]", ylabel="t [Myr]",
                zlabel="log10 ρ_a", title="S1: radiales Profil über die Zeit", dpi=dpi)
    viz.surface3d_png(figs3 / "p2_S1_profile_surface.png", rmid, T, P, xlabel="r [kpc]", ylabel="t [Myr]",
                      zlabel="ρ_a", title="S1: ρ_a(r, t)", dpi=dpi, log_axes=(False, False, True))
    viz.surface3d_html(figs3 / "p2_S1_profile_surface.html", rmid, T, P, xlabel="r [kpc]", ylabel="t [Myr]",
                       zlabel="ρ_a", title="S1: ρ_a(r, t)", log_axes=(False, False, True))


def s2_packet(cfg, figs, figs3, dpi, workers, a):
    V = cfg["validation_p2"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["N_main"])
    sig, M = V["packet_sigma_kpc"], 1e8
    k0 = 2 * np.pi * V["boost_modes"] / L
    v = a * k0
    x, y, z = g.mesh_sparse
    psi = np.sqrt(M) * (2 * np.pi * sig**2) ** -0.75 * np.exp(-g.r**2 / (4 * sig**2) + 1j * k0 * x)
    phi = np.zeros(g.shape)
    nst = 10
    dt = V["packet_t_myr"] / nst
    for _ in range(nst):
        psi, phi = step_kdk(psi, phi, None, g, dt, a, 0.0, workers)
    t = V["packet_t_myr"]
    st2 = sig**2 * (1 + (a * t / (2 * sig**2)) ** 2)
    xs = (x - v * t + L / 2) % L - L / 2
    rho_ex = M * (2 * np.pi * st2) ** -1.5 * np.exp(-(xs**2 + y**2 + z**2) / (2 * st2))
    err = float(np.max(np.abs(np.abs(psi) ** 2 - rho_ex)) / rho_ex.max())
    R.add("S2", "max|ρ−ρ_exakt|/max ρ", err, V["packet_rel_err_max"], err < V["packet_rel_err_max"],
          note=f"v = {v/KMS_TO_CODE:.2f} km/s, σ(t)/σ0 = {np.sqrt(st2)/sig:.2f}")
    viz.slices_row(figs / "p2_S2_packet.png", [(np.abs(psi) ** 2, "numerisch", "lin"), (rho_ex, "exakt", "lin"),
                   (np.abs(psi) ** 2 - rho_ex, "Differenz", "sym")], L, dpi=dpi,
                   suptitle=f"S2: freies Wellenpaket nach {t} Myr")
    viz.field_surface_png(figs3 / "p2_S2_packet_surface.png", np.abs(psi) ** 2, L, zlabel="ρ",
                          title="S2: Wellenpaket (Mittelschnitt)", dpi=dpi)


def s3_boost(cfg, figs, figs3, dpi, workers, a, M):
    V = cfg["validation_p2"]
    tp = cfg["time"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["N_main"])
    psi, phi, _ = embed_ground_state(g, M, a, G_CODE, polish_cfg(cfg), workers)
    k0 = 2 * np.pi * V["boost_modes"] / L
    v = a * k0
    x = g.mesh_sparse[0]
    psi = psi * np.exp(1j * k0 * x)
    dt = stable_dt(g, phi, a, tp["sp_kinetic_dt_factor"], tp["sp_potential_phase_max_rad"])
    n = int(np.ceil(V["boost_t_myr"] / dt))
    dt = V["boost_t_myr"] / n
    rmax0 = np.max(np.abs(psi) ** 2)
    T, C, E = [0.0], [center_of_mass(psi, g)], [0.0]
    for k in range(1, n + 1):
        psi, phi = step_kdk(psi, phi, None, g, dt, a, G_CODE, workers)
        if k % max(1, n // 40) == 0 or k == n:
            T.append(k * dt)
            C.append(center_of_mass(psi, g))
            E.append(abs(np.max(np.abs(psi) ** 2) / rmax0 - 1))
    T, C = np.array(T), np.array(C)
    expect = (v * T + L / 2) % L - L / 2
    err = float(np.max(np.abs(C[:, 0] - expect)))
    R.add("S3", "max |x_com − v t| [kpc]", err, V["com_err_max_kpc"], err < V["com_err_max_kpc"],
          note=f"v = {v/KMS_TO_CODE:.2f} km/s")
    R.add("S3", "max |y_com|,|z_com| [kpc]", float(np.max(np.abs(C[:, 1:]))), V["com_err_max_kpc"],
          float(np.max(np.abs(C[:, 1:]))) < V["com_err_max_kpc"])
    R.add("S3", "max |ρ_max(t)/ρ_max(0) − 1|", max(E), V["rho0_variation_max"], max(E) < V["rho0_variation_max"],
          note="Galilei-Invarianz: Form bleibt erhalten")
    viz.line_plot(figs / "p2_S3_com.png", T, {"x_com numerisch": C[:, 0], "v·t": expect, "y_com": C[:, 1],
                  "z_com": C[:, 2]}, xlabel="t [Myr]", ylabel="kpc", title="S3: bewegter Soliton", dpi=dpi)
    fig = go.Figure(go.Scatter3d(x=C[:, 0], y=C[:, 1], z=C[:, 2], mode="lines+markers",
                                 marker=dict(size=3, color=T, colorscale="Viridis", colorbar=dict(title="t [Myr]"))))
    fig.update_layout(title="S3: Schwerpunktsbahn", scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
    (figs3 / "p2_S3_com_path.html").parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(figs3 / "p2_S3_com_path.html"), include_plotlyjs="directory")
    viz.ortho_slices_3d_png(figs3 / "p2_S3_rho_end_ortho.png", np.abs(psi) ** 2, L, title="S3: ρ_a am Ende",
                            dpi=dpi, log=True, label="ρ_a")


def s4_order(cfg, figs, figs3, dpi, workers, a, M):
    V = cfg["validation_p2"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["order_N"])
    prof = ground_state_profile()
    sc = physical_scales(prof, M, a, G_CODE)
    psi0 = (sc["A"] * np.interp(1.15 * g.r / sc["L0"], prof.xi, prof.f, right=0.0)).astype(np.complex128)
    psi0 *= np.sqrt(M / mass(psi0, g))  # gestauchter, also angeregter Soliton
    phi0 = potential(psi0, None, g, G_CODE, workers)
    res = []
    for f in V["order_kin_factors"]:
        dt = f * g.dx**2 / a
        n = int(np.ceil(V["order_t_myr"] / dt))
        dt = V["order_t_myr"] / n
        psi, phi = psi0.copy(), phi0.copy()
        for _ in range(n):
            psi, phi = step_kdk(psi, phi, None, g, dt, a, G_CODE, workers)
        res.append((dt, psi))
    ref = res[-1][1]
    dts = np.array([d for d, _ in res[:-1]])
    errs = np.array([np.linalg.norm(p - ref) / np.linalg.norm(ref) for _, p in res[:-1]])
    p_obs = np.log2((errs[0] - 0) / errs[1]) if len(errs) > 1 else np.nan
    # Richardson-sauber: Ordnung aus drei aufeinanderfolgenden Lösungen
    e01 = np.linalg.norm(res[0][1] - res[1][1])
    e12 = np.linalg.norm(res[1][1] - res[2][1])
    p_rich = np.log2(e01 / e12)
    R.add("S4", "beobachtete Ordnung (Richardson, 3 Lösungen)", p_rich, V["order_expected"],
          abs(p_rich - V["order_expected"]) < V["order_tol"])
    R.add("S4", "beobachtete Ordnung (gegen feinste Lösung)", p_obs)
    for d, e in zip(dts, errs):
        R.add("S4", f"‖ψ−ψ_ref‖/‖ψ_ref‖ bei dt={d:.4g} Myr", e)
    viz.line_plot(figs / "p2_S4_order.png", dts, {"Fehler": errs, "∝ dt²": errs[0] * (dts / dts[0]) ** 2},
                  xlabel="dt [Myr]", ylabel="relativer Fehler", title=f"S4: Zeitkonvergenz (p ≈ {p_rich:.2f})",
                  dpi=dpi, logx=True, logy=True, markers=True)
    viz.field_surface_png(figs3 / "p2_S4_error_surface.png", np.abs(res[0][1] - ref), L,
                          zlabel="|ψ−ψ_ref|", title="S4: Fehlerfeld gröbster Zeitschritt", dpi=dpi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P2")
    figs, figs3 = out / "figs", out / "figs3d"
    dpi, workers = cfg["viz"]["dpi"], cfg["compute"]["fft_workers"]
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    M = cfg["axion"]["soliton_mass_msun"]
    t = time.perf_counter()
    g, psi, phi, _ = s0_ground_state(cfg, figs, figs3, dpi, workers, a, M)
    print(f"  S0 fertig ({time.perf_counter()-t:.0f} s)", flush=True)
    for name, fn in [("S1", lambda: s1_stationarity(cfg, figs, figs3, dpi, workers, a, g, psi, phi)),
                     ("S2", lambda: s2_packet(cfg, figs, figs3, dpi, workers, a)),
                     ("S3", lambda: s3_boost(cfg, figs, figs3, dpi, workers, a, M)),
                     ("S4", lambda: s4_order(cfg, figs, figs3, dpi, workers, a, M))]:
        t = time.perf_counter()
        fn()
        print(f"  {name} fertig ({time.perf_counter()-t:.0f} s)", flush=True)
    R.write(out, "P2", "Schrödinger-Poisson & Soliton-Grundzustand",
            ["Schwellen vorab in `validation_p2` festgelegt. Soliton: numerischer SP-Grundzustand "
             "(Shooting + imaginäre Zeit im periodischen Gitter)."])
    write_manifest(cfg, out, "P2", extra=R.results)
    print(f"P2 fertig -> {out} ({'alle bestanden' if R.n_failed() == 0 else f'{R.n_failed()} FEHLGESCHLAGEN'})")
    if R.n_failed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
