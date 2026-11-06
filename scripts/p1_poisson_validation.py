from __future__ import annotations

import argparse
import json
import os
import time

import astropy.constants as const
import astropy.units as u
import numpy as np

from hgr import analytic as an
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.grid import Grid3D
from hgr.poisson import gradient_spectral, poisson_residual, solve_poisson_periodic
from hgr.units import G_CODE, KMS_TO_CODE, AxionUnits, sound_speed

RESULTS: dict[str, dict] = {}


def record(test: str, name: str, value: float, threshold: float | None, passed: bool | None, note: str = "") -> None:
    RESULTS.setdefault(test, {})[name] = {"value": float(value), "threshold": threshold, "passed": passed, "note": note}


def t0_units(cfg: dict) -> None:
    G_ref = const.G.value * const.M_sun.value * (u.Myr.to(u.s)) ** 2 / (u.kpc.to(u.m)) ** 3
    record(
        "T0",
        "G_code [kpc³/(Msun Myr²)]",
        G_CODE,
        1e-12,
        bool(np.isclose(G_CODE, G_ref, rtol=1e-12)),
        f"SI cross-check {G_ref:.6e}",
    )
    au = AxionUnits(1.0)
    m_kg = 1e-22 * const.e.value / const.c.value**2
    hbm_si = const.hbar.value / m_kg
    hbm_ref = hbm_si * (u.Myr.to(u.s)) / (u.kpc.to(u.m)) ** 2
    record(
        "T0",
        "hbar/m (m22=1) [kpc²/Myr]",
        au.hbar_over_m,
        1e-10,
        bool(np.isclose(au.hbar_over_m, hbm_ref, rtol=1e-10)),
        f"SI cross-check {hbm_ref:.6e}",
    )
    M, rc = 2.92e8, 0.5
    gstar_si = (G_CODE * M / rc**2) * (u.kpc / u.Myr**2).to(u.m / u.s**2)
    vstar = np.sqrt(G_CODE * M / rc) / KMS_TO_CODE
    tstar = np.sqrt(rc**3 / (G_CODE * M))
    cs = sound_speed(1.0e3, 1.2, 5.0 / 3.0) / KMS_TO_CODE
    for name, val, ref in [
        ("g*(r_c) [m/s²]", gstar_si, 1.63e-10),
        ("v*(r_c) [km/s]", vstar, 50.1),
        ("t*(r_c) [Myr]", tstar, 9.75),
        ("c_s(10³K) [km/s]", cs, 3.38),
    ]:
        ok = bool(abs(val / ref - 1) < 5e-3)
        record("T0", name, val, 5e-3, ok, f"Research Plan: {ref}")


def t1_fourier_mode(cfg: dict, figs, figs3, dpi: int, workers: int) -> None:
    V = cfg["validation_p1"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["N_main"])
    x, y, z = g.mesh_sparse
    kvec = 2 * np.pi / L * np.array([1.0, 2.0, 3.0])
    amp, rho0 = 1.0e7, 3.0e8
    phase = kvec[0] * x + kvec[1] * y + kvec[2] * z
    rho = rho0 + amp * np.cos(phase)
    k2 = float(kvec @ kvec)
    phi_ex = -4 * np.pi * G_CODE * amp * np.cos(phase) / k2
    gx_ex = -4 * np.pi * G_CODE * amp * (-kvec[0]) * np.sin(phase) / k2
    phi = solve_poisson_periodic(rho, g, G_CODE, workers)
    gx = gradient_spectral(phi, g, workers)[0]
    e_phi = np.max(np.abs(phi - phi_ex)) / np.max(np.abs(phi_ex))
    e_gx = np.max(np.abs(gx - gx_ex)) / np.max(np.abs(gx_ex))
    thr = V["fourier_mode_rel_err_max"]
    record("T1", "max|ΔΦ|/max|Φ|", e_phi, thr, bool(e_phi < thr))
    record("T1", "max|Δ∂xΦ|/max|∂xΦ|", e_gx, thr, bool(e_gx < thr))
    record(
        "T1", "ε_P spektral", poisson_residual(phi, rho, g, G_CODE, "spectral", workers), thr, None, "only informative"
    )
    viz.slices_row(
        figs / "p1_T1_mode_slices.png",
        [(phi, "Φ numerical", "sym"), (phi_ex, "Φ exact", "sym"), (phi - phi_ex, "error", "sym")],
        L,
        dpi=dpi,
        suptitle="T1: single Fourier mode k=(1,2,3)·2π/L",
    )
    viz.field_surface_png(
        figs3 / "p1_T1_mode_error_surface.png",
        phi - phi_ex,
        L,
        zlabel="ΔΦ",
        title="T1: error landscape (round-off level)",
        dpi=dpi,
        stride=2,
    )
    viz.isosurface_html(
        figs3 / "p1_T1_mode_iso.html",
        phi,
        L,
        title="T1: Φ of a Fourier mode",
        log=False,
        label="Φ",
        max_points=cfg["viz"]["iso_max_points_per_axis"],
    )


def t2_gaussian(cfg: dict, figs, figs3, dpi: int, workers: int) -> None:
    V = cfg["validation_p1"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, V["N_main"])
    M, sig = V["gauss_mass_msun"], V["gauss_sigma_kpc"]
    rho = an.gaussian_density(g.r, M, sig)
    m_grid = g.integrate(rho)
    assert abs(m_grid / M - 1) < 1e-6, f"Gaussian not resolved/truncated: {m_grid/M}"
    phi = solve_poisson_periodic(rho, g, G_CODE, workers)
    rho_mean = m_grid / L**3
    phi_iso = an.gaussian_potential_isolated(g.r, M, sig, G_CODE)
    phi_bg = an.periodic_background_potential(g.r, rho_mean, G_CODE)
    inner = g.r < L / 4
    scale = abs(phi_iso.min())
    d_with = phi - phi_iso - phi_bg
    d_with -= d_with[inner].mean()
    d_wo = phi - phi_iso
    d_wo -= d_wo[inner].mean()
    e_with = np.max(np.abs(d_with[inner])) / scale
    e_wo = np.max(np.abs(d_wo[inner])) / scale
    thr = V["gauss_bgcorr_rel_err_max"]
    record("T2", "rel. error WITH background correction (r<L/4)", e_with, thr, bool(e_with < thr))
    record("T2", "rel. error WITHOUT correction (r<L/4)", e_wo, None, None, "shows: periodic box ≠ isolated system")
    record("T2", "ε_P spektral", poisson_residual(phi, rho, g, G_CODE, "spectral", workers), None, None)
    record(
        "T2",
        "ε_P FD2",
        poisson_residual(phi, rho, g, G_CODE, "fd2", workers),
        None,
        None,
        "same solution, different Laplacian -> different value",
    )
    c = g.center_index
    xr = g.x1d[c:]
    line = lambda f: f[c:, c, c]  # noqa: E731
    C_with = (phi - phi_iso - phi_bg)[inner].mean()
    viz.line_plot(
        figs / "p1_T2_gauss_profile.png",
        xr[1:],
        {
            "Φ numerical (periodic) − C": line(phi)[1:] - C_with,
            "Φ_iso + Φ_bg": line(phi_iso + phi_bg)[1:],
            "Φ_iso (isolated)": line(phi_iso)[1:],
        },
        xlabel="x [kpc]",
        ylabel="Φ [kpc²/Myr²]",
        title="T2: Gaussian sphere, potential along x",
        dpi=dpi,
    )
    viz.line_plot(
        figs / "p1_T2_gauss_residual.png",
        xr[1:],
        {
            "|error| without correction": np.abs(line(d_wo)[1:]) / scale,
            "|error| with (2π/3)Gρ̄r²-correction": np.abs(line(d_with)[1:]) / scale + 1e-17,
        },
        xlabel="x [kpc]",
        ylabel="|ΔΦ|/|Φ_iso(0)|",
        title="T2: residual along x",
        dpi=dpi,
        logy=True,
        hlines={"threshold": thr},
        vlines={"L/4": L / 4},
    )
    viz.slices_row(
        figs / "p1_T2_gauss_slices.png",
        [
            (rho, "ρ", "log"),
            (phi, "Φ numerical", "lin"),
            (np.where(inner, d_with, 0.0) / scale, "residual (r<L/4), corrected", "sym"),
        ],
        L,
        dpi=dpi,
        suptitle="T2: Gaussian sphere",
    )
    viz.field_surface_png(
        figs3 / "p1_T2_gauss_phi_surface.png",
        phi,
        L,
        zlabel="Φ",
        title="T2: potential well (mid-plane slice)",
        dpi=dpi,
        stride=2,
    )
    viz.field_surface_png(
        figs3 / "p1_T2_gauss_residual_surface.png",
        np.where(inner, d_with, 0.0) / scale,
        L,
        zlabel="ΔΦ/|Φ(0)|",
        title="T2: residual error landscape",
        dpi=dpi,
        stride=2,
    )
    viz.isosurface_html(
        figs3 / "p1_T2_gauss_phi_iso.html",
        phi,
        L,
        title="T2: Φ isosurfaces",
        log=False,
        label="Φ",
        max_points=cfg["viz"]["iso_max_points_per_axis"],
    )


def _radial_force_errors(L: float, N: int, m22: float, M: float, workers: int):
    g = Grid3D(L, N)
    rc = an.schive_core_radius_from_mass(m22, M)
    rho = an.schive_density(g.r, m22, rc)
    phi = solve_poisson_periodic(rho, g, G_CODE, workers)
    gx, gy, gz = gradient_spectral(phi, g, workers)
    x, y, z = g.mesh_sparse
    rr = np.where(g.r > 0, g.r, 1.0)
    acc_r = -(gx * x + gy * y + gz * z) / rr
    m_box = g.integrate(rho)
    acc_iso = -G_CODE * an.schive_enclosed_mass(g.r, m22, rc) / rr**2
    acc_bg = an.periodic_background_accel(g.r, m_box / L**3, G_CODE)
    sel = (g.r >= 2 * g.dx) & (g.r <= L / 4)
    e_corr = np.abs(acc_r - acc_iso - acc_bg)[sel] / np.abs(acc_iso)[sel]
    e_raw = np.abs(acc_r - acc_iso)[sel] / np.abs(acc_iso)[sel]
    return g, rc, phi, acc_r, acc_iso, acc_bg, sel, e_corr, e_raw, m_box


def t3_soliton_force(cfg: dict, figs, figs3, dpi: int, workers: int) -> None:
    V = cfg["validation_p1"]
    m22, M = cfg["axion"]["m22"], cfg["axion"]["soliton_mass_msun"]
    thr = V["soliton_force_rel_err_max"]
    curves_corr, curves_raw = {}, {}
    rbins = np.linspace(0.1, 4.0, 40)
    surf = []
    for L in V["box_sizes_kpc"]:
        N = int(round(L / V["box_test_dx_kpc"]))
        N += N % 2
        g, rc, phi, acc_r, acc_iso, acc_bg, sel, e_corr, e_raw, m_box = _radial_force_errors(L, N, m22, M, workers)
        r_sel = g.r[sel] if np.ndim(g.r) == 3 else np.broadcast_to(g.r, g.shape)[sel]
        idx = np.digitize(r_sel, rbins)
        med_c = np.array([np.median(e_corr[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(rbins))])
        med_r = np.array([np.median(e_raw[idx == i]) if np.any(idx == i) else np.nan for i in range(1, len(rbins))])
        curves_corr[f"L={L} kpc (corrected)"] = med_c
        curves_raw[f"L={L} kpc (roh)"] = med_r
        surf.append(med_r)
        e_rc_raw = np.median(e_raw[np.abs(r_sel - rc) < g.dx])
        record(
            "T3",
            f"L={L}: median rel. force error corrected (2dx≤r≤L/4)",
            np.median(e_corr),
            thr,
            bool(np.median(e_corr) < thr),
        )
        record("T3", f"L={L}: max rel. force error corrected", np.max(e_corr), None, None)
        record(
            "T3",
            f"L={L}: rel. deviation from the ISOLATED soliton at r_c",
            e_rc_raw,
            None,
            None,
            "physikalischer Boxeffekt",
        )
        record("T3", f"L={L}: mass on the grid / M_s", m_box / M, None, None, "truncation/overlap")
        if L == cfg["domain"]["box_kpc"]:
            c = g.center_index
            xr = g.x1d[c + 1 :]
            viz.line_plot(
                figs / "p1_T3_force_profile.png",
                xr,
                {
                    "|g_r| numerical": np.abs(acc_r[c + 1 :, c, c]),
                    "GM(<r)/r² (isolated)": np.abs(acc_iso[c + 1 :, c, c]),
                    "isolated + background term": np.abs((acc_iso + acc_bg)[c + 1 :, c, c]),
                },
                xlabel="x [kpc]",
                ylabel="|g| [kpc/Myr²]",
                dpi=dpi,
                logx=True,
                logy=True,
                title=f"T3: soliton force, L={L} kpc, N={N}",
                vlines={"r_c": rc},
            )
            viz.slices_row(
                figs / "p1_T3_soliton_slices.png",
                [
                    (an.schive_density(g.r, m22, rc), "ρ_a (Schive)", "log"),
                    (phi, "Φ", "lin"),
                    (np.sqrt(sum(a**2 for a in gradient_spectral(phi, g, workers))), "|∇Φ|", "lin"),
                ],
                L,
                dpi=dpi,
                suptitle="T3: soliton – density, potential, force magnitude",
            )
            viz.field_surface_png(
                figs3 / "p1_T3_soliton_phi_surface.png",
                phi,
                L,
                zlabel="Φ",
                title="T3: soliton potential well",
                dpi=dpi,
                stride=2,
            )
            viz.ortho_slices_3d_png(
                figs3 / "p1_T3_soliton_phi_ortho.png", phi, L, title="T3: Φ (3 slices)", dpi=dpi, log=False, label="Φ"
            )
            viz.isosurface_html(
                figs3 / "p1_T3_soliton_phi_iso.html",
                phi,
                L,
                title="T3: soliton-potential",
                log=False,
                label="Φ",
                max_points=cfg["viz"]["iso_max_points_per_axis"],
            )
    rmid = 0.5 * (rbins[1:] + rbins[:-1])
    viz.line_plot(
        figs / "p1_T3_box_sensitivity.png",
        rmid,
        curves_raw | curves_corr,
        xlabel="r [kpc]",
        ylabel="median |Δg_r|/|g_iso|",
        title="T3: box-size sensitivity the soliton force",
        dpi=dpi,
        logy=True,
        hlines={"threshold": thr},
    )
    S = np.array(surf)
    viz.surface3d_png(
        figs3 / "p1_T3_box_sensitivity_surface.png",
        rmid,
        np.array(V["box_sizes_kpc"]),
        np.where(np.isfinite(S), S, np.nan),
        xlabel="r [kpc]",
        ylabel="L [kpc]",
        zlabel="|Δg|/|g_iso|",
        title="deviation from the isolated soliton (r, L)",
        dpi=dpi,
        log_axes=(False, True, True),
    )
    viz.surface3d_html(
        figs3 / "p1_T3_box_sensitivity_surface.html",
        rmid,
        np.array(V["box_sizes_kpc"]),
        np.where(np.isfinite(S), S, np.nan),
        xlabel="r [kpc]",
        ylabel="L [kpc]",
        zlabel="|Δg|/|g_iso|",
        title="deviation from the isolated soliton (r, L)",
        log_axes=(False, True, True),
    )


def t4_convergence(cfg: dict, figs, figs3, dpi: int, workers: int) -> None:
    V = cfg["validation_p1"]
    L = cfg["domain"]["box_kpc"]
    Nref = V["convergence_N_ref"]
    Ns = V["convergence_N"]
    sigmas = V["convergence_sigmas_kpc"]
    M = V["gauss_mass_msun"]
    E = np.zeros((len(sigmas), len(Ns)))
    EPfd = np.zeros_like(E)
    EPsp = np.zeros_like(E)
    for i, sig in enumerate(sigmas):
        gref = Grid3D(L, Nref)
        phi_ref = solve_poisson_periodic(an.gaussian_density(gref.r, M, sig), gref, G_CODE, workers)
        for j, N in enumerate(Ns):
            assert Nref % N == 0
            s = Nref // N
            g = Grid3D(L, N)
            rho = an.gaussian_density(g.r, M, sig)
            phi = solve_poisson_periodic(rho, g, G_CODE, workers)
            ref = phi_ref[::s, ::s, ::s]
            E[i, j] = np.max(np.abs(phi - ref)) / np.max(np.abs(ref))
            EPfd[i, j] = poisson_residual(phi, rho, g, G_CODE, "fd2", workers)
            EPsp[i, j] = poisson_residual(phi, rho, g, G_CODE, "spectral", workers)
        del phi_ref
    isig = sigmas.index(cfg["validation_p1"]["gauss_sigma_kpc"])
    for j, N in enumerate(Ns):
        record("T4", f"σ={sigmas[isig]}: rel. error N={N} vs N_ref={Nref}", E[isig, j], None, None)
        record("T4", f"σ={sigmas[isig]}: ε_P FD2 N={N}", EPfd[isig, j], None, None)
    p_fd = np.log2(EPfd[isig, -2] / EPfd[isig, -1])
    record(
        "T4",
        "observed order ε_P FD2 (last refinement)",
        p_fd,
        2.0,
        bool(abs(p_fd - 2) < 0.3),
        "expected 2 for a smooth source",
    )
    dx = np.array([L / N for N in Ns])
    viz.line_plot(
        figs / "p1_T4_convergence.png",
        np.array(Ns),
        {f"error σ={s} kpc": E[i] for i, s in enumerate(sigmas)}
        | {f"ε_P FD2 σ={sigmas[isig]}": EPfd[isig], f"ε_P spektral σ={sigmas[isig]}": EPsp[isig]},
        xlabel="N",
        ylabel="relative error",
        title="T4: spectral convergence & residual convention",
        dpi=dpi,
        logx=True,
        logy=True,
        markers=True,
    )
    viz.line_plot(
        figs / "p1_T4_convergence_vs_resolution.png",
        sigmas_arr := np.array(sigmas),
        {f"N={N} (σ/dx={sigmas_arr[0]/dx[j]:.1f}…)": E[:, j] for j, N in enumerate(Ns)},
        xlabel="σ [kpc]",
        ylabel="rel. error",
        title="T4: error vs. source width",
        dpi=dpi,
        logy=True,
        markers=True,
    )
    viz.surface3d_png(
        figs3 / "p1_T4_error_surface.png",
        np.array(Ns, float),
        np.array(sigmas),
        E,
        xlabel="N",
        ylabel="σ [kpc]",
        zlabel="rel. error",
        title="T4: error(N, σ)",
        dpi=dpi,
        log_axes=(True, False, True),
    )
    viz.surface3d_html(
        figs3 / "p1_T4_error_surface.html",
        np.array(Ns, float),
        np.array(sigmas),
        E,
        xlabel="N",
        ylabel="σ [kpc]",
        zlabel="rel. error",
        title="T4: error(N, σ)",
        log_axes=(True, False, True),
    )
    viz.heatmap(
        figs / "p1_T4_error_heatmap.png",
        np.array(Ns, float),
        np.array(sigmas),
        E,
        xlabel="N",
        ylabel="σ [kpc]",
        zlabel="rel. error",
        title="T4: error(N, σ)",
        dpi=dpi,
        logx=True,
        logz=True,
        contour_levels=[1e-6, 1e-3],
    )


def t5_invariance(cfg: dict, figs, figs3, dpi: int, workers: int) -> None:
    V = cfg["validation_p1"]
    L = cfg["domain"]["box_kpc"]
    g = Grid3D(L, 64)
    rng = np.random.default_rng(cfg["meta"]["seed"])
    r1 = an.gaussian_density(g.r, 1e8, 0.4)
    r2 = np.abs(rng.normal(size=g.shape)) * 1e6
    p1 = solve_poisson_periodic(r1, g, G_CODE, workers)
    p2 = solve_poisson_periodic(r2, g, G_CODE, workers)
    s = np.max(np.abs(p1))
    thr = V["invariance_rel_err_max"]
    e_const = np.max(np.abs(solve_poisson_periodic(r1 + 5e8, g, G_CODE, workers) - p1)) / s
    lin = solve_poisson_periodic(2.5 * r1 - 0.7 * r2, g, G_CODE, workers) - (2.5 * p1 - 0.7 * p2)
    e_lin = np.max(np.abs(lin)) / np.max(np.abs(2.5 * p1 - 0.7 * p2))
    shift = (5, -3, 7)
    e_tr = (
        np.max(
            np.abs(
                solve_poisson_periodic(np.roll(r1, shift, (0, 1, 2)), g, G_CODE, workers)
                - np.roll(p1, shift, (0, 1, 2))
            )
        )
        / s
    )
    e_mean = abs(p1.mean()) / s
    sym = np.max(np.abs(p1 - p1.transpose(1, 0, 2))) / s
    for name, val in [
        ("constant density added", e_const),
        ("linearity", e_lin),
        ("translation (integer)", e_tr),
        ("|⟨Φ⟩|", e_mean),
        ("symmetry x↔y", sym),
    ]:
        record("T5", name, val, thr, bool(val < thr))
    vals = [e_const, e_lin, e_tr, e_mean, sym]
    viz.bar_log(
        figs / "p1_T5_invariances.png",
        ["constant", "linearity", "translation", "⟨Φ⟩", "x↔y"],
        [max(v, 1e-18) for v in vals],
        ylabel="relative error",
        title=f"T5: invariances (threshold {thr:g})",
        dpi=dpi,
    )
    viz.ortho_slices_3d_png(
        figs3 / "p1_T5_linearity_error_ortho.png",
        lin / np.max(np.abs(2.5 * p1 - 0.7 * p2)),
        L,
        title="T5: linearity error field",
        dpi=dpi,
        log=False,
        label="rel. error",
    )


def t6_timing(cfg: dict, figs, figs3, dpi: int) -> None:
    V = cfg["validation_p1"]
    L = cfg["domain"]["box_kpc"]
    ncpu = os.cpu_count() or 1
    wlist = sorted({1, 2, 4, ncpu} & set(range(1, ncpu + 1)))
    Ns = V["timing_N"]
    T = np.full((len(wlist), len(Ns)), np.nan)
    for j, N in enumerate(Ns):
        g = Grid3D(L, N)
        rho = an.gaussian_density(g.r, 1e8, 0.3)
        _ = g.k2_r
        for i, w in enumerate(wlist):
            solve_poisson_periodic(rho, g, G_CODE, w)
            ts = []
            for _ in range(V["timing_repeats"]):
                t0 = time.perf_counter()
                solve_poisson_periodic(rho, g, G_CODE, w)
                ts.append(time.perf_counter() - t0)
            T[i, j] = min(ts)
        del rho, g
    for j, N in enumerate(Ns):
        record("T6", f"seconds/Poisson N={N} ({wlist[-1]} Threads)", T[-1, j], None, None)
    Nn = np.array(Ns, float)
    model = T[-1, -1] * (Nn**3 * np.log2(Nn)) / (Nn[-1] ** 3 * np.log2(Nn[-1]))
    viz.line_plot(
        figs / "p1_T6_timing.png",
        Nn,
        {f"{w} Thread(s)": T[i] for i, w in enumerate(wlist)} | {"∝ N³ log N": model},
        xlabel="N",
        ylabel="seconds per solution",
        title=f"T6: runtime Poisson (CPU cores: {ncpu})",
        dpi=dpi,
        logx=True,
        logy=True,
        markers=True,
    )
    if len(wlist) > 1:
        viz.surface3d_png(
            figs3 / "p1_T6_timing_surface.png",
            Nn,
            np.array(wlist, float),
            T,
            xlabel="N",
            ylabel="Threads",
            zlabel="s",
            title="T6: runtime(N, Threads)",
            dpi=dpi,
            log_axes=(True, False, True),
        )
    else:
        viz.surface3d_png(
            figs3 / "p1_T6_timing_surface.png",
            Nn,
            np.array([1.0, 1.0001]),
            np.vstack([T[0], T[0]]),
            xlabel="N",
            ylabel="Threads",
            zlabel="s",
            title="T6: runtime (only 1 core available)",
            dpi=dpi,
            log_axes=(True, False, True),
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-timing", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P1")
    figs, figs3 = out / "figs", out / "figs3d"
    dpi, workers = cfg["viz"]["dpi"], cfg["compute"]["fft_workers"]

    steps = [
        ("T0", lambda: t0_units(cfg)),
        ("T1", lambda: t1_fourier_mode(cfg, figs, figs3, dpi, workers)),
        ("T2", lambda: t2_gaussian(cfg, figs, figs3, dpi, workers)),
        ("T3", lambda: t3_soliton_force(cfg, figs, figs3, dpi, workers)),
        ("T4", lambda: t4_convergence(cfg, figs, figs3, dpi, workers)),
        ("T5", lambda: t5_invariance(cfg, figs, figs3, dpi, workers)),
    ]
    if not args.skip_timing:
        steps.append(("T6", lambda: t6_timing(cfg, figs, figs3, dpi)))
    for name, fn in steps:
        t0 = time.perf_counter()
        fn()
        print(f"  {name} done ({time.perf_counter() - t0:.1f} s)")

    n_fail = sum(1 for t in RESULTS.values() for r in t.values() if r["passed"] is False)
    lines = [
        f"# P1 – Verification units/grid/Poisson ({cfg['meta']['run_label']})",
        "",
        f"**Overall:** {'PASSED' if n_fail == 0 else f'{n_fail} Test(s) NOT passed'}",
        "",
        "Thresholds were set in the config before the run (`validation_p1`).",
        "",
    ]
    for test, rows in RESULTS.items():
        lines += [f"## {test}", "", "| Quantity | Value | Threshold | Status | Note |", "|---|---|---|---|---|"]
        for name, r in rows.items():
            st = {True: "✅", False: "❌", None: "info"}[r["passed"]]
            thr = "—" if r["threshold"] is None else f"{r['threshold']:.3g}"
            lines.append(f"| {name} | {r['value']:.4g} | {thr} | {st} | {r['note']} |")
        lines.append("")
    (out / "P1_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "P1_values.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    write_manifest(cfg, out, "P1", extra=RESULTS)
    print(f"P1 done -> {out}  ({'all passed' if n_fail == 0 else f'{n_fail} FAILED'})")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
