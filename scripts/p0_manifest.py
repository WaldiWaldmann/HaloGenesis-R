from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from hgr import analytic as an
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.grid import Grid3D
from hgr.units import (
    G_CODE,
    AxionUnits,
    code_velocity_to_kms,
    free_fall_time,
    gas_mass_density,
    jeans_length,
    jeans_mass,
    sound_speed,
)

BYTES_F64 = 8


def soliton_scales(m22: float, mass: float) -> dict:
    au = AxionUnits(m22)
    rc = an.schive_core_radius_from_mass(m22, mass)
    rho_c = an.schive_central_density(m22, rc)
    m_rc = float(an.schive_enclosed_mass(rc, m22, rc))
    v_circ_rc = np.sqrt(G_CODE * m_rc / rc)
    integral, err = quad(lambda r: r * an.schive_density(r, m22, rc), 0.0, np.inf, limit=200)
    assert err < 1e-6 * abs(integral), "quad did not converge"
    phi0 = -4.0 * np.pi * G_CODE * integral
    return {
        "m22": m22,
        "rc_kpc": rc,
        "rho_c": rho_c,
        "M_enc_rc": m_rc,
        "v_circ_rc": v_circ_rc,
        "phi0": phi0,
        "t_star": np.sqrt(rc**3 / (G_CODE * mass)),
        "t_ff_center": float(free_fall_time(rho_c)),
        "t_dB": au.de_broglie_time_myr(v_circ_rc),
        "lambda_dB": au.de_broglie_wavelength_kpc(v_circ_rc),
        "t_compton": au.compton_time_myr,
        "hbar_over_m": au.hbar_over_m,
    }


def gas_scales(n_cm3: float, T_K: float, mu: float, gamma: float) -> dict:
    rho = gas_mass_density(n_cm3, mu)
    cs = sound_speed(T_K, mu, gamma)
    return {
        "rho_b": rho,
        "cs": cs,
        "lambda_J": float(jeans_length(cs, rho)),
        "M_J": float(jeans_mass(cs, rho)),
        "t_ff_gas": float(free_fall_time(rho)),
    }


def numerics(N: int, box: float, sol: dict, gas: dict, cfg: dict, tau: float) -> dict:
    g = Grid3D(box, N)
    hbm = sol["hbar_over_m"]
    dt_kin = cfg["time"]["sp_kinetic_dt_factor"] * g.dx**2 / hbm
    dt_pot = cfg["time"]["sp_potential_phase_max_rad"] * hbm / abs(sol["phi0"])
    dt_sp = min(dt_kin, dt_pot)
    v_esc = np.sqrt(2.0 * abs(sol["phi0"]))
    dt_hydro = cfg["time"]["cfl_hydro"] * g.dx / (v_esc + gas["cs"])
    steps = cfg["time"]["t_end_myr"] / dt_sp
    q = 2 + 10 + 2 + 2 + 6 + 15 + 4
    mem_core = q * BYTES_F64 * N**3
    k_buf = int(np.ceil(tau / dt_hydro)) if tau > 0 else 0
    mem_buf = 2 * k_buf * BYTES_F64 * N**3
    return {
        "N": N,
        "dx": g.dx,
        "cells_per_rc": sol["rc_kpc"] / g.dx,
        "dt_kin": dt_kin,
        "dt_pot": dt_pot,
        "dt_sp": dt_sp,
        "dt_hydro": dt_hydro,
        "steps": steps,
        "mem_core_GB": mem_core / 1e9,
        "filter_slices": k_buf,
        "mem_buffer_GB": mem_buf / 1e9,
        "k_nyq_over_k_dB": g.k_nyquist / (2 * np.pi / sol["lambda_dB"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P0")
    figs, figs3 = out / "figs", out / "figs3d"
    dpi = cfg["viz"]["dpi"]
    H = cfg["historical_reference"]

    M = cfg["axion"]["soliton_mass_msun"]
    L = cfg["domain"]["box_kpc"]
    gp = cfg["gas"]
    sol = soliton_scales(cfg["axion"]["m22"], M)
    gas = gas_scales(gp["n_cm3"], gp["T_K"], gp["mu"], gp["gamma"])
    tau = cfg["filter"]["tau_myr"]
    num = {N: numerics(N, L, sol, gas, cfg, tau) for N in cfg["scans"]["N_convergence"]}
    m22_match = an.schive_m22_for(M, H["core_radius_kpc"])

    R = np.array(cfg["analysis"]["apertures_kpc"])
    m_gas_R = gas["rho_b"] * 4.0 / 3.0 * np.pi * R**3
    Q_R = gas["M_J"] / m_gas_R
    eps_R = m_gas_R / an.schive_enclosed_mass(R, sol["m22"], sol["rc_kpc"])
    rho_mean_sol = M / L**3
    per_corr_rc = an.periodic_background_accel(sol["rc_kpc"], rho_mean_sol, G_CODE) / (
        G_CODE * sol["M_enc_rc"] / sol["rc_kpc"] ** 2
    )

    kms = code_velocity_to_kms
    rows = [
        ("m22", sol["m22"], "-", "chosen", "—"),
        ("hbar/m_a", sol["hbar_over_m"], "kpc²/Myr", "derived", "—"),
        ("soliton mass M_s", M, "M_sun", "historical adopted", "p0_profiles"),
        ("core radius r_c(M_s, m22) [Schive-Fit]", sol["rc_kpc"], "kpc", "derived", "p0_profiles"),
        ("m22 for M_s AND r_c=0.5 kpc", m22_match, "-", "derived", "p0_timescales_m22"),
        ("central soliton density rho_c", sol["rho_c"], "M_sun/kpc³", "derived", "p0_profiles"),
        ("M_s(<r_c)", sol["M_enc_rc"], "M_sun", "derived", "p0_profiles"),
        ("v_circ(r_c)", kms(sol["v_circ_rc"]), "km/s", "derived", "p0_timescales"),
        ("Phi(0) isolated", kms(np.sqrt(abs(sol["phi0"]))) ** 2, "(km/s)²", "derived", "p0_profiles"),
        ("t* = sqrt(r_c³/GM)", sol["t_star"], "Myr", "derived", "p0_timescales"),
        ("t_ff(rho_c)", sol["t_ff_center"], "Myr", "derived", "p0_timescales"),
        ("t_dB = hbar/(m v²)", sol["t_dB"], "Myr", "derived", "p0_timescales"),
        ("lambda_dB", sol["lambda_dB"], "kpc", "derived", "p0_timescales"),
        ("t_Compton = hbar/(m c²)", sol["t_compton"], "Myr", "derived", "p0_timescales"),
        ("rho_b (n, mu)", gas["rho_b"], "M_sun/kpc³", "derived", "p0_profiles"),
        ("rho_b / rho_c(soliton)", gas["rho_b"] / sol["rho_c"], "-", "derived", "p0_profiles"),
        ("c_s", kms(gas["cs"]), "km/s", "derived", "p0_timescales"),
        ("lambda_J (gas)", gas["lambda_J"], "kpc", "derived", "p0_QJ"),
        ("M_J (gas, t=0)", gas["M_J"], "M_sun", "derived", "p0_QJ"),
        ("t_ff (gas)", gas["t_ff_gas"], "Myr", "derived", "p0_timescales"),
        ("t_sound = r_c/c_s", sol["rc_kpc"] / gas["cs"], "Myr", "derived", "p0_timescales"),
        ("gas mass in box", gas["rho_b"] * L**3, "M_sun", "derived", "—"),
        ("periodic force correction at r_c", per_corr_rc, "-", "derived", "P1"),
    ]
    for Ri, Qi, ei in zip(R, Q_R, eps_R):
        rows.append((f"Q_J(R={Ri} kpc, t=0)", Qi, "-", "derived", "p0_QJ"))
        rows.append((f"epsilon(R={Ri} kpc) = M_gas/M_sol", ei, "-", "derived", "p0_epsilon"))
    for N, d in num.items():
        rows += [
            (f"[N={N}] dx", d["dx"], "kpc", "derived", "p0_resolution"),
            (f"[N={N}] cells per r_c", d["cells_per_rc"], "-", "derived", "p0_resolution"),
            (f"[N={N}] dt_SP", d["dt_sp"], "Myr", "derived", "p0_cost"),
            (f"[N={N}] dt_hydro (CFL, v_esc)", d["dt_hydro"], "Myr", "derived", "p0_cost"),
            (f"[N={N}] SP-steps up to t_end", d["steps"], "-", "derived", "p0_cost"),
            (f"[N={N}] memory core (A=0+A=1)", d["mem_core_GB"], "GB", "derived", "p0_cost"),
            (f"[N={N}] filter buffer tau={tau} Myr", d["mem_buffer_GB"], "GB", "derived", "p0_cost"),
        ]

    dt_sp_dev = num[cfg["domain"]["N_dev"]]["dt_sp"]
    comparisons = [
        (
            "sound speed",
            f"{H['sound_speed_reported_kms']} km/s",
            f"{kms(gas['cs']):.3f} km/s",
            "Paper Eq. (11) is wrong; Research Plan corrects to 3.38 km/s.",
        ),
        (
            "Freifallzeit soliton",
            f"{H['t_ff_reported_myr']} Myr",
            f"{sol['t_ff_center']:.2f} Myr",
            "factor ~{:.0f}: gas cannot compress 50x within t_sim=1.2 Myr.".format(
                sol["t_ff_center"] / H["t_ff_reported_myr"]
            ),
        ),
        (
            "simulation time vs. Jeans-Dauer",
            f"t_sim={H['t_sim_myr']} Myr, T_J={H['jeans_duration_myr']} Myr",
            "T_J > t_sim",
            "Impossible within one run; Research Plan assumes 20 Myr.",
        ),
        (
            "Mittelungsfenster",
            f"{H['averaging_window_yr']:.0e} yr",
            f"dt_SP(N={cfg['domain']['N_dev']}) = {dt_sp_dev*1e6:.0f} yr",
            "Window < time step ⇒ filter had no numerical effect (≡ instantaneous).",
        ),
        (
            "Relevante fluctuation time",
            "tau_Phi = 0.10 yr (paper Eq. 9)",
            f"t_dB = {sol['t_dB']:.1f} Myr",
            "Compton oscillation is removed in the SP limit.",
        ),
        (
            "mass ratio epsilon",
            f"{H['epsilon_reported'][0]:.0e}–{H['epsilon_reported'][1]:.0e}",
            f"{eps_R[list(R).index(cfg['analysis']['primary_aperture_kpc'])]:.3g} (R=0.5 kpc)",
            "gas self-gravity is not perturbative.",
        ),
        (
            "min M_J/M_gas",
            f"{H['min_jeans_ratio']:.1e}",
            f"{Q_R[list(R).index(cfg['analysis']['primary_aperture_kpc'])]:.2e} already at t=0",
            "Q<1 already holds without soliton ⇒ diagnostic must be differential (ΔO2).",
        ),
        (
            "core radius at M_s, m22=1",
            f"{H['core_radius_kpc']} kpc",
            f"{sol['rc_kpc']:.3f} kpc",
            f"Consistent only for m22 ≈ {m22_match:.2f}.",
        ),
    ]

    ts_labels = [
        "t_Compton",
        "dt_SP (N=128)",
        "dt_hydro (N=128)",
        "t*",
        "t_dB",
        "t_ff soliton",
        "t_ff gas",
        "t_sound(r_c)",
        "t_end",
    ]
    ts_vals = [
        sol["t_compton"],
        num[128]["dt_sp"],
        num[128]["dt_hydro"],
        sol["t_star"],
        sol["t_dB"],
        sol["t_ff_center"],
        gas["t_ff_gas"],
        sol["rc_kpc"] / gas["cs"],
        cfg["time"]["t_end_myr"],
    ]
    viz.bar_log(
        figs / "p0_timescales.png",
        ts_labels,
        ts_vals,
        ylabel="time [Myr]",
        title="timescale ladder (Default-Config)",
        dpi=dpi,
        colors_list=["#999", "#4c72b0", "#4c72b0", "#dd8452", "#c44e52", "#dd8452", "#55a868", "#55a868", "#000"],
    )

    m22s = np.geomspace(0.3, 5.0, 60)
    scs = [soliton_scales(m, M) for m in m22s]
    viz.line_plot(
        figs / "p0_timescales_m22.png",
        m22s,
        {
            "t_dB": np.array([s["t_dB"] for s in scs]),
            "t_ff(rho_c)": np.array([s["t_ff_center"] for s in scs]),
            "t*": np.array([s["t_star"] for s in scs]),
            "r_c [kpc] (same axis)": np.array([s["rc_kpc"] for s in scs]),
        },
        xlabel="m22",
        ylabel="Myr or. kpc",
        title="soliton scales vs. axion mass (M_s fest)",
        dpi=dpi,
        logx=True,
        logy=True,
        vlines={"m22=1": 1.0, f"m22={m22_match:.2f}": m22_match},
        hlines={"t_end": cfg["time"]["t_end_myr"]},
    )
    vs = np.geomspace(5, 200, 50) * (1.0 / code_velocity_to_kms(1.0))
    TDB = np.array([[AxionUnits(m).de_broglie_time_myr(v) for m in m22s] for v in vs])
    vs_kms = np.array([code_velocity_to_kms(v) for v in vs])
    for fn in (viz.surface3d_png,):
        fn(
            figs3 / "p0_tdB_surface.png",
            m22s,
            vs_kms,
            TDB,
            xlabel="m22",
            ylabel="v [km/s]",
            zlabel="t_dB [Myr]",
            title="coherence time t_dB(m22, v)",
            dpi=dpi,
            log_axes=(True, True, True),
        )
    viz.surface3d_html(
        figs3 / "p0_tdB_surface.html",
        m22s,
        vs_kms,
        TDB,
        xlabel="m22",
        ylabel="v [km/s]",
        zlabel="t_dB [Myr]",
        title="coherence time t_dB(m22, v)",
        log_axes=(True, True, True),
    )
    viz.heatmap(
        figs / "p0_tdB_heatmap.png",
        m22s,
        vs_kms,
        TDB,
        xlabel="m22",
        ylabel="v [km/s]",
        zlabel="t_dB [Myr]",
        title="t_dB(m22, v) – contours at 1, 3, 10, 30 Myr",
        dpi=dpi,
        logx=True,
        logy=True,
        logz=True,
        contour_levels=[1, 3, 10, 30],
    )

    r = np.geomspace(1e-2, L / 2, 300)
    prof = {
        f"soliton m22={m}": an.schive_density(r, m, an.schive_core_radius_from_mass(m, M)) for m in cfg["scans"]["m22"]
    }
    hl = {f"gas n={n} cm⁻³": gas_mass_density(n, gp["mu"]) for n in cfg["scans"]["n_cm3"]}
    viz.line_plot(
        figs / "p0_profiles.png",
        r,
        prof,
        xlabel="r [kpc]",
        ylabel="ρ [M_sun/kpc³]",
        title="soliton density profiles (Schive-Fit) vs. homogeneous gas densities",
        dpi=dpi,
        logx=True,
        logy=True,
        hlines=hl,
    )
    mprof = {
        f"M_sol(<r) m22={m}": an.schive_enclosed_mass(r, m, an.schive_core_radius_from_mass(m, M))
        for m in cfg["scans"]["m22"]
    }
    mprof.update(
        {f"M_gas(<r) n={n}": gas_mass_density(n, gp["mu"]) * 4 / 3 * np.pi * r**3 for n in cfg["scans"]["n_cm3"]}
    )
    viz.line_plot(
        figs / "p0_enclosed_mass.png",
        r,
        mprof,
        xlabel="r [kpc]",
        ylabel="M(<r) [M_sun]",
        title="enclosed mass: soliton vs. gas",
        dpi=dpi,
        logx=True,
        logy=True,
    )

    g3 = Grid3D(L, 96)
    rho_sol3 = an.schive_density(g3.r, sol["m22"], sol["rc_kpc"])
    viz.ortho_slices_3d_png(
        figs3 / "p0_soliton_ortho.png",
        rho_sol3,
        L,
        title="soliton ρ_a (Schive-Fit)",
        dpi=dpi,
        log=True,
        label="ρ [M_sun/kpc³]",
    )
    viz.field_surface_png(
        figs3 / "p0_soliton_surface.png",
        np.log10(rho_sol3),
        L,
        zlabel="log10 ρ",
        title="soliton density, mid-plane slice as surface",
        dpi=dpi,
        stride=2,
    )
    viz.isosurface_html(
        figs3 / "p0_soliton_iso.html",
        rho_sol3,
        L,
        title="soliton ρ_a isosurfaces",
        log=True,
        label="ρ",
        max_points=cfg["viz"]["iso_max_points_per_axis"],
    )

    Rg = np.geomspace(0.05, 2.0, 60)
    ng = np.geomspace(1e-3, 1e2, 60)
    rho_g = np.array([gas_mass_density(n, gp["mu"]) for n in ng])
    cs0 = gas["cs"]
    QJ = jeans_mass(cs0, rho_g)[:, None] / (rho_g[:, None] * 4 / 3 * np.pi * Rg[None, :] ** 3)
    EPS = (rho_g[:, None] * 4 / 3 * np.pi * Rg[None, :] ** 3) / an.schive_enclosed_mass(Rg, sol["m22"], sol["rc_kpc"])[
        None, :
    ]
    viz.heatmap(
        figs / "p0_QJ.png",
        Rg,
        ng,
        QJ,
        xlabel="aperture R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="Q_J = M_J/M_gas(<R)",
        title="Jeans diagnostic at t=0 WITHOUT soliton (T=10³ K)",
        dpi=dpi,
        logx=True,
        logy=True,
        logz=True,
        contour_levels=[1.0, H["min_jeans_ratio"]],
    )
    viz.surface3d_png(
        figs3 / "p0_QJ_surface.png",
        Rg,
        ng,
        QJ,
        xlabel="R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="Q_J",
        title="Q_J(R, n) at t=0",
        dpi=dpi,
        log_axes=(True, True, True),
    )
    viz.surface3d_html(
        figs3 / "p0_QJ_surface.html",
        Rg,
        ng,
        QJ,
        xlabel="R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="Q_J",
        title="Q_J(R, n) at t=0",
        log_axes=(True, True, True),
    )
    viz.heatmap(
        figs / "p0_epsilon.png",
        Rg,
        ng,
        EPS,
        xlabel="R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="ε = M_gas(<R)/M_sol(<R)",
        title="mass ratio gas/soliton",
        dpi=dpi,
        logx=True,
        logy=True,
        logz=True,
        contour_levels=[0.01, 0.1, 1.0],
    )
    viz.surface3d_png(
        figs3 / "p0_epsilon_surface.png",
        Rg,
        ng,
        EPS,
        xlabel="R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="ε",
        title="ε(R, n)",
        dpi=dpi,
        log_axes=(True, True, True),
    )
    viz.surface3d_html(
        figs3 / "p0_epsilon_surface.html",
        Rg,
        ng,
        EPS,
        xlabel="R [kpc]",
        ylabel="n [cm⁻³]",
        zlabel="ε",
        title="ε(R, n)",
        log_axes=(True, True, True),
    )

    Ns = np.array([32, 48, 64, 96, 128, 192, 256, 384, 512])
    CPR = np.array([[an.schive_core_radius_from_mass(m, M) / (L / N) for N in Ns] for m in m22s])
    viz.heatmap(
        figs / "p0_resolution.png",
        Ns,
        m22s,
        CPR,
        xlabel="N",
        ylabel="m22",
        zlabel="cells per r_c",
        title=f"resolution of the soliton core (L={L} kpc)",
        dpi=dpi,
        logx=True,
        logy=True,
        logz=True,
        contour_levels=[5, 10, 20],
    )
    viz.surface3d_png(
        figs3 / "p0_resolution_surface.png",
        Ns,
        m22s,
        CPR,
        xlabel="N",
        ylabel="m22",
        zlabel="cells/r_c",
        title="cells per r_c",
        dpi=dpi,
        log_axes=(True, True, True),
    )
    taus = np.array([0.0, 0.3, 1.0, 3.0, 10.0])
    MEM = np.array(
        [
            [
                numerics(int(N), L, sol, gas, cfg, t)["mem_core_GB"]
                + numerics(int(N), L, sol, gas, cfg, t)["mem_buffer_GB"]
                for N in Ns
            ]
            for t in taus
        ]
    )
    STEPS = np.array([numerics(int(N), L, sol, gas, cfg, 0.0)["steps"] for N in Ns])
    viz.line_plot(
        figs / "p0_cost.png",
        Ns,
        {f"memory [GB], τ={t} Myr": MEM[i] for i, t in enumerate(taus)} | {"SP-steps up to t_end / 1000": STEPS / 1000},
        xlabel="N",
        ylabel="GB or. 10³ steps",
        title="cost: memory (A=0+A=1, float64-filter buffer) and step count",
        dpi=dpi,
        logx=True,
        logy=True,
        hlines={"16 GB Laptop": 16.0, "32 GB": 32.0},
        markers=True,
    )
    viz.surface3d_png(
        figs3 / "p0_cost_surface.png",
        Ns,
        taus + 0.1,
        MEM,
        xlabel="N",
        ylabel="τ + 0.1 [Myr]",
        zlabel="memory [GB]",
        title="memory(N, τ)",
        dpi=dpi,
        log_axes=(True, True, True),
    )
    viz.surface3d_html(
        figs3 / "p0_cost_surface.html",
        Ns,
        taus + 0.1,
        MEM,
        xlabel="N",
        ylabel="τ + 0.1 [Myr]",
        zlabel="memory [GB]",
        title="memory(N, τ)",
        log_axes=(True, True, True),
    )

    lines = [
        f"# P0 – parameter manifest & consistency check ({cfg['meta']['run_label']})",
        "",
        f"**Provenienz:** {cfg['meta']['provenance']}",
        "",
        "Evidence class of all numbers: *derived* from the config (no simulation). "
        "soliton quantities use the empirical Schive fit; P2 replaces it with the numerical "
        "SP ground state.",
        "",
        "## Key numbers",
        "",
        "| Quantity | Value | Unit | Class | Figure |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {a} | {b:.4g} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    lines += [
        "",
        "## Comparison with the original paper",
        "",
        "| Quantity | reported | reconstructed | consequence |",
        "|---|---|---|---|",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in comparisons]
    lines += [
        "",
        "## Figures",
        "",
        "2D: `figs/`, 3D static: `figs3d/*.png`, "
        "3D interactive: `figs3d/*.html` (open in a browser, works offline).",
    ]
    (out / "P0_report.md").write_text("\n".join(lines), encoding="utf-8")

    results = {
        "rows": [{"name": a, "value": float(b), "unit": c, "class": d} for a, b, c, d, _ in rows],
        "comparisons": [dict(zip(["quantity", "reported", "reconstructed", "consequence"], c)) for c in comparisons],
    }
    (out / "P0_values.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    mpath = write_manifest(cfg, out, "P0", extra=results)
    print(f"P0 done -> {out}\n  Report: {out/'P0_report.md'}\n  Manifest: {mpath}")


if __name__ == "__main__":
    main()
