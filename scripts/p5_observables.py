from __future__ import annotations

import argparse
import json
import time

import numpy as np

from hgr import cooling as co
from hgr import hydro as hy
from hgr import observables as ob
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.coupled import CoupledPair
from hgr.grid import Grid3D
from hgr.report import Recorder
from hgr.setup import build_axion_field, coupling_params, gas_initial_condition
from hgr.soliton import embed_ground_state
from hgr.units import G_CODE, KMS_TO_CODE, AxionUnits, gas_mass_density

R = Recorder()


def run(cfg, N, t_end, tau, n_cm3, out):
    figs, figs3, dpi = out / "figs", out / "figs3d", cfg["viz"]["dpi"]
    P5, gp = cfg["p5"], cfg["gas"]
    grid = Grid3D(cfg["domain"]["box_kpc"], N)
    dV, gam = grid.cell_volume, gp["gamma"]
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    psi, ginfo = build_axion_field(cfg, grid, cfg["compute"]["fft_workers"])
    pair = CoupledPair(grid, coupling_params(cfg, tau), psi, gas_initial_condition(cfg, grid, n_cm3))
    b1, b0 = pair.br

    Rs_an = list(cfg["analysis"]["apertures_kpc"])
    Rp = cfg["analysis"]["primary_aperture_kpc"]
    if min(Rs_an) < grid.dx:
        raise ValueError(
            f"analysis aperture {min(Rs_an)} kpc < dx = {grid.dx:.3g} kpc: at N={N} unresolved. "
            "Choose a larger N (pre-declared apertures are not adjusted)."
        )
    skipped = sorted(r for r in P5["viz_radii_kpc"] if r < grid.dx)
    Rs_viz = sorted({r for r in P5["viz_radii_kpc"] if r >= grid.dx} | set(Rs_an))
    AP = {r: ob.Aperture(grid, r, with_gradient=False) for r in Rs_viz}
    APF = {r: ob.Aperture(grid, r, with_gradient=True) for r in Rs_an}
    TRK = ob.Aperture(grid, cfg["analysis"]["track_radius_kpc"], with_gradient=False)
    tracking = cfg["analysis"]["aperture_center"] == "soliton"
    if cfg["analysis"]["aperture_center"] not in ("soliton", "box"):
        raise ValueError("analysis.aperture_center must be 'soliton' or 'box'")
    ctr = [(grid.center_index,) * 3]
    SLU = AP[max(Rs_an)].sl
    WU = {}
    for r in Rs_an:
        wf = np.zeros(grid.shape)
        wf[AP[r].sl] = AP[r].w
        WU[r] = np.ascontiguousarray(wf[SLU])
    WpU = WU[Rp]

    snaps = sorted(t for t in cfg["analysis"]["full_snapshot_times_myr"] if t <= t_end)
    edges = np.linspace(0, grid.box / 2, 41)
    rmid = 0.5 * (edges[1:] + edges[:-1])
    rr = np.broadcast_to(grid.r, grid.shape).ravel()
    ridx = np.digitize(rr, edges)
    cnt = np.bincount(ridx, minlength=len(edges) + 1)[1 : len(edges)]
    prof_every = 5

    def radial(f):
        sh = tuple(grid.center_index - int(a) for a in ctr[-1])
        f = np.roll(f, sh, axis=(0, 1, 2)) if any(sh) else f
        tot = np.bincount(ridx, weights=f.ravel(), minlength=len(edges) + 1)[1 : len(edges)]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)

    ts, rec = [], {k: [] for k in ["dt", "J1", "J0", "c1", "c0", "o3p", "mb", "fl1", "fl0", "phic", "phibc", "nmax"]}
    M = {(A, r): [] for A in (1, 0) for r in Rs_viz}
    Q = {(A, r): [] for A in (1, 0) for r in Rs_viz}
    F = {(A, r): [] for A in (1, 0) for r in Rs_an}
    MF = {(A, r): [] for A in (1, 0) for r in Rs_an}
    O3 = {r: [] for r in Rs_an}
    prof1, prof0, prof_t = [], [], []
    Ig = [np.zeros(grid.shape, np.float32) for _ in range(3)]
    dg_prev = None
    c = grid.center_index
    status = "completed"

    def measure(dt_last, force_profile=False):
        nonlocal dg_prev
        if tracking:
            ci = ob.track_center(b1.psi.real**2 + b1.psi.imag**2, ctr[-1], TRK)
            ctr.append(ci) if len(ts) else ctr.__setitem__(0, ci)
            for ap in AP.values():
                ap.set_center(ci)
        ts.append(pair.t)
        rec["dt"].append(dt_last)
        for A, b in ((1, b1), (0, b0)):
            rec[f"J{A}"].append(ob.truelove_from_state(b.U, gam, G_CODE, grid.dx))
            for r in Rs_viz:
                q, _, m = AP[r].jeans_Q(b.U, None, gam, G_CODE)
                M[(A, r)].append(m), Q[(A, r)].append(q)
            for r in Rs_an:
                F[(A, r)].append(APF[r].flux(b.U))
                MF[(A, r)].append(APF[r].mass(b.U[0]))
        rec["c1"].append(b1.U[0][ctr[-1]]), rec["c0"].append(b0.U[0][ctr[-1]])
        UA = AP[max(Rs_an)]
        r0 = UA.take(b0.U[0])
        dg_full = tuple((b1.acc[d] - b0.acc[d]).astype(np.float32) for d in range(3))
        dg = tuple(UA.take(dg_full[d]) for d in range(3))
        dv = tuple(UA.take(b1.U[1 + d]) / UA.take(b1.U[0]) - UA.take(b0.U[1 + d]) / r0 for d in range(3))
        for r in Rs_an:
            O3[r].append(ob.o3_coherence(r0, dg, dv, WU[r]))
        if dg_prev is not None:
            for d in range(3):
                Ig[d] += 0.5 * (dg_prev[d] + dg_full[d]) * np.float32(dt_last)
        dg_prev = dg_full
        nv = ob.weighted_norm(r0, dv, WpU)
        rec["mb"].append(
            ob.weighted_norm(r0, tuple(dv[d] - UA.take(Ig[d]) for d in range(3)), WpU) / nv if nv > 0 else np.nan
        )
        if force_profile or len(ts) % prof_every == 1:
            prof_t.append(pair.t), prof1.append(radial(b1.U[0])), prof0.append(radial(b0.U[0]))
        for A, b in ((1, b1), (0, b0)):
            m = UA.take(pair.floor_mask[0 if A == 1 else 1])
            rb = UA.take(b.U[0])
            rec[f"fl{A}"].append(float(np.sum(WpU * rb * m) / np.sum(WpU * rb)))
        ci = ctr[-1]
        rec["phic"].append(float(b1.phi[ci])), rec["phibc"].append(b1.phibar_at(ci))
        rec["nmax"].append(float(co.number_density_cm3(b1.U[0].max(), gp["mu"])))
        probes["rho_a"].append((b1.psi[pI].real ** 2 + b1.psi[pI].imag ** 2).tolist())
        probes["rho_b1"].append(b1.U[0][pI].tolist())
        Up = b1.U[(slice(None),) + pI]
        probes["p_b1"].append(((gam - 1) * (Up[4] - 0.5 * (Up[1] ** 2 + Up[2] ** 2 + Up[3] ** 2) / Up[0])).tolist())
        probes["rho_b0"].append(b0.U[0][pI].tolist())

    snapdata = []
    RS = cfg["response"]
    from hgr.setup import fibonacci_probes

    pidx, prad = fibonacci_probes(grid, RS["probe_shells_kpc"], RS["probes_per_shell"])
    pI = tuple(pidx.T)
    probes = {"r_kpc": prad.tolist(), "rho_a": [], "rho_b1": [], "rho_b0": [], "p_b1": []}

    def snapshot(tag):
        k = int(ctr[-1][2])
        v1 = b1.U[1:4, :, :, k] / b1.U[0][None, :, :, k]
        v0 = b0.U[1:4, :, :, k] / b0.U[0][None, :, :, k]
        snapdata.append(
            {
                "t": pair.t,
                "r1": b1.U[0][:, :, k].copy(),
                "r0": b0.U[0][:, :, k].copy(),
                "dv_kms": np.sqrt(np.sum((v1 - v0) ** 2, axis=0)) / KMS_TO_CODE,
                "ra": (b1.psi[:, :, k].real ** 2 + b1.psi[:, :, k].imag ** 2).copy(),
                "cx": (ctr[-1][0] - grid.center_index) * grid.dx,
                "cy": (ctr[-1][1] - grid.center_index) * grid.dx,
            }
        )

    measure(0.0)
    snapshot(f"{0.0:05.1f}")
    snaps = [s for s in snaps if s > 0]
    t0 = time.perf_counter()
    while pair.t < t_end - 1e-12:
        dt = pair.max_dt()
        if dt < P5["dt_min_abort_myr"]:
            status = f"ABORTED at t = {pair.t:.3f} Myr: Δt = {dt:.2e} Myr < dt_min (collapse to grid scale)"
            break
        nxt = min([s for s in snaps if s > pair.t + 1e-12] + [t_end])
        dt = min(dt, nxt - pair.t)
        pair.step(dt)
        measure(dt)
        if any(abs(pair.t - s) < 1e-9 for s in snaps):
            prof_t.append(pair.t), prof1.append(radial(b1.U[0])), prof0.append(radial(b0.U[0]))
            snapshot(f"{pair.t:05.1f}")
        if pair.steps % 25 == 0:
            print(
                f"    t = {pair.t:6.2f} Myr, steps {pair.steps}, dt = {dt:.3g}, " f"{time.perf_counter()-t0:.0f} s",
                flush=True,
            )
    ts = np.array(ts)
    R.add("Run", "Status: " + status, pair.t, note="t_final [Myr]")
    R.add("Run", "N / τ [Myr] / n [cm⁻³]", N, note=f"τ = {tau}, n = {n_cm3}")
    R.add("Run", "macro steps", pair.steps)
    R.add("Run", "skipped visualisation radii (< dx)", len(skipped), note=", ".join(map(str, skipped)))
    R.add("Run", "wall time [s]", time.perf_counter() - t0)
    R.add("Run", "ground-state residual", ginfo["residual"])
    R.add(
        "Run",
        f"setup: {ginfo['kind']}",
        ginfo.get("M_total", cfg["axion"]["soliton_mass_msun"]),
        note="M_axion total [M_sun]"
        + (
            f"; pre-relaxation {ginfo['relax_myr']:.1f} Myr, σ(Φ_c)/|Φ_c| = "
            f"{ginfo['phi_c_rel_std_relax']:.3g}, t_dB = {ginfo['t_dB_myr']:.1f} Myr"
            if "relax_myr" in ginfo
            else ""
        ),
    )
    phic, phibc = np.array(rec["phic"]), np.array(rec["phibc"])
    R.add(
        "Run",
        "potential fluctuation σ(Φ_c)/|⟨Φ_c⟩| (A=1)",
        float(np.std(phic) / abs(np.mean(phic))),
        note="isolated ground state ≈ 0 -> filter has no effect; halo > 0",
    )
    R.add("Run", "max|Φ̄_c − Φ_c|/|⟨Φ_c⟩| (A=1)", float(np.max(np.abs(phibc - phic)) / abs(np.mean(phic))))
    R.add(
        "Run",
        "max Jeans-floor mass fraction in R_p (A=1 / A=0)",
        max(rec["fl1"]),
        note=f"A=0: {max(rec['fl0']):.3g}; N_J = {cfg['hydro']['jeans_floor_NJ']}",
    )
    R.add("Run", "max n_H (A=1) [cm⁻³]", max(rec["nmax"]), note="GP98-Niedrigdichte-Limes: n ≲ 10³ cm⁻³")
    R.add("Run", "local first-order fallbacks (hydro)", pair.fallbacks)
    U_ic = gas_initial_condition(cfg, grid, n_cm3)
    E_ref = float(np.sum(co.pressure_from_T(U_ic[0], gp["T_K"], gp["mu"]) / (gam - 1)) * dV)
    R.add("Run", "ε_filter A=1 (Eq. 45), E_ref = E_int(0)", abs(pair.filter_work[0]) / E_ref)
    R.add("Run", "energy injected by Jeans floor / E_int(0) (A=1)", pair.floor_energy[0] / E_ref)
    Jmax = max(max(rec["J1"]), max(rec["J0"]))
    cpos = (np.array(ctr, float) - grid.center_index) * grid.dx
    shift = float(np.max(np.linalg.norm(cpos, axis=1)))
    R.add(
        "Run",
        f"Aperturzentrum: {cfg['analysis']['aperture_center']}, max displacement [kpc]",
        shift,
        note=f"Endposition {np.round(cpos[-1], 3).tolist()}; rounded to cell (dx = {grid.dx:.3g} kpc)",
    )
    R.add(
        "Run",
        "max Truelove number J = dx/λ_J (A=1, A=0)",
        Jmax,
        0.25,
        Jmax <= 0.25,
        note="J > 1/4: artificial fragmentation possible -> not resolution-robust",
    )

    O1 = {r: np.log((np.array(M[(1, r)]) / M[(1, r)][0]) / (np.array(M[(0, r)]) / M[(0, r)][0])) for r in Rs_viz}
    for r in Rs_an:
        R.add("O1", f"O1(R={r}) at t_final", O1[r][-1])
        R.add("O1", f"max O1(R={r})", np.max(O1[r]))
        for A in (1, 0):
            m = np.array(MF[(A, r)])
            f = np.array(F[(A, r)])
            integ = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ts))])
            dm = m - m[0]
            scale = np.max(np.abs(dm))
            res = np.max(np.abs(dm - integ)) / scale if scale > 0 else np.nan
            R.add(
                "O1",
                f"flux cross-check R={r}, A={A}: max|ΔM − ∫F dt|/max|ΔM| (alt)",
                res,
                note="ill-conditioned if ΔM ≈ 0 – info only",
            )
            res0 = float(np.max(np.abs(dm - integ)) / m[0])
            thr0 = cfg["p6"]["flux_residual_M0_max"]
            R.add(
                "O1",
                f"Massenbilanz R={r}, A={A}: max|ΔM − ∫F dt|/M(0)",
                res0,
                thr0,
                res0 <= thr0,
                note="fixed aperture; normalisation pre-declared in p6",
            )
            if r == Rp:
                viz.line_plot(
                    figs / f"p5_O1_flux_check_A{A}.png",
                    ts,
                    {
                        "ΔM/M(0) direct": dm / m[0],
                        "∫ surface flux dt / M(0)": integ / m[0],
                        "|Differenz|/M(0)": np.abs(dm - integ) / m[0],
                    },
                    xlabel="t [Myr]",
                    ylabel="relative to M(<R, t=0)",
                    title=f"mass balance (fixed aperture) A={A}, R={r} kpc",
                    subtitle=f"max |Differenz|/M(0) = {np.max(np.abs(dm - integ)) / m[0]:.1e}",
                    dpi=dpi,
                )
    viz.line_plot(
        figs / "p5_O1.png",
        ts,
        {f"R = {r} kpc": O1[r] for r in Rs_an},
        xlabel="t [Myr]",
        ylabel="O1",
        title="O1: differential concentration (0 = no effect)",
        dpi=dpi,
        hlines={"0": 0.0},
    )
    Z1 = np.array([O1[r] for r in Rs_viz]).T
    viz.heatmap(
        figs / "p5_O1_map.png",
        np.array(Rs_viz),
        ts,
        Z1,
        xlabel="R [kpc]",
        ylabel="t [Myr]",
        zlabel="O1",
        title="O1(R, t)",
        dpi=dpi,
        logx=True,
    )
    viz.surface3d_png(
        figs3 / "p5_O1_surface.png",
        np.array(Rs_viz),
        ts,
        Z1,
        xlabel="R [kpc]",
        ylabel="t [Myr]",
        zlabel="O1",
        title="O1(R, t)",
        dpi=dpi,
        log_axes=(True, False, False),
    )
    viz.surface3d_html(
        figs3 / "p5_O1_surface.html",
        np.array(Rs_viz),
        ts,
        Z1,
        xlabel="R [kpc]",
        ylabel="t [Myr]",
        zlabel="O1",
        title="O1(R, t)",
        log_axes=(True, False, False),
    )

    occ, intervals = {}, {}
    for A in (1, 0):
        qmin = np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0)
        occ[A], intervals[A] = ob.occupancy(ts, qmin, cfg["analysis"]["jeans_threshold"])
        R.add(
            "O2",
            f"O2(A={A}) time fraction min_R Q < 1",
            occ[A],
            note="Intervalle: " + ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in intervals[A][:6]),
        )
        R.add("O2", f"min Q (A={A})", float(qmin.min()))
    R.add("O2", "ΔO2 = O2(1) − O2(0)", occ[1] - occ[0], note="hypothesis: > 0")
    for A in (1, 0):
        Zq = np.array([Q[(A, r)] for r in Rs_viz]).T
        viz.heatmap(
            figs / f"p5_O2_Q_A{A}.png",
            np.array(Rs_viz),
            ts,
            Zq,
            xlabel="R [kpc]",
            ylabel="t [Myr]",
            zlabel="Q = M_J/M_b",
            title=f"Jeans ratio A={A}",
            dpi=dpi,
            logx=True,
            logz=True,
            contour_levels=[1.0],
        )
        viz.surface3d_html(
            figs3 / f"p5_O2_Q_A{A}_surface.html",
            np.array(Rs_viz),
            ts,
            Zq,
            xlabel="R [kpc]",
            ylabel="t [Myr]",
            zlabel="Q",
            title=f"Q(R, t), A={A}",
            log_axes=(True, False, True),
        )
        viz.surface3d_png(
            figs3 / f"p5_O2_Q_A{A}_surface.png",
            np.array(Rs_viz),
            ts,
            Zq,
            xlabel="R [kpc]",
            ylabel="t [Myr]",
            zlabel="Q",
            title=f"Q(R, t), A={A}",
            dpi=dpi,
            log_axes=(True, False, True),
        )
    viz.line_plot(
        figs / "p5_O2_Qmin.png",
        ts,
        {f"min_R Q, A={A}": np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0) for A in (1, 0)},
        xlabel="t [Myr]",
        ylabel="Q",
        title="O2: Jeans diagnostic (pre-declared R)",
        dpi=dpi,
        logy=True,
        hlines={"Q = 1": 1.0},
    )

    early = (ts > 0) & (ts <= P5["early_window_myr"])
    o3e = np.array(O3[Rp])[early]
    o3e_mean = float(np.nanmean(o3e)) if o3e.size else np.nan
    R.add(
        "O3",
        f"O3(R={Rp}) mean for t ≤ {P5['early_window_myr']} Myr",
        o3e_mean,
        P5["o3_early_min"],
        (o3e_mean >= P5["o3_early_min"]) if np.isfinite(o3e_mean) else None,
        note="Eq. 30: O3 → 1 early",
    )
    mb = np.array(rec["mb"])
    R.add(
        "O3",
        "early momentum budget: max ‖δv − ∫δg dt‖/‖δv‖",
        float(np.nanmax(mb[early])) if early.any() else np.nan,
        note="residual = Druck/Advektion R(t)",
    )
    for r in Rs_an:
        R.add("O3", f"O3(R={r}) time-averaged", float(np.nanmean(O3[r][1:])))
    viz.line_plot(
        figs / "p5_O3.png",
        ts[1:],
        {f"R = {r} kpc": np.array(O3[r][1:]) for r in Rs_an},
        xlabel="t [Myr]",
        ylabel="O3",
        title="O3: coherence of extra force ↔ velocity response",
        dpi=dpi,
        hlines={"0": 0.0},
    )
    viz.line_plot(
        figs / "p5_O3_momentum_budget.png",
        ts[1:],
        {"‖δv − ∫δg dt‖/‖δv‖": mb[1:]},
        xlabel="t [Myr]",
        ylabel="relative",
        title="O3 cross-check: momentum budget",
        dpi=dpi,
        logy=True,
    )
    Z3 = np.array([O3[r][1:] for r in Rs_an]).T
    viz.surface3d_png(
        figs3 / "p5_O3_surface.png",
        np.array(Rs_an),
        ts[1:],
        Z3,
        xlabel="R [kpc]",
        ylabel="t [Myr]",
        zlabel="O3",
        title="O3(R, t)",
        dpi=dpi,
    )

    H = cfg["historical_reference"]
    rs = min(Rs_an)
    ap = np.array(M[(1, rs)]) / M[(1, rs)][0]
    R.add(
        "Comparison", f"C_ρ aperture mean R={rs} (A=1), max", ap.max(), note=f"paper: ~{H['central_gas_enhancement']}"
    )
    i12 = int(np.argmin(np.abs(ts - H["t_sim_myr"])))
    R.add("Comparison", f"C_ρ aperture mean R={rs} at t≈{ts[i12]:.2f} Myr", ap[i12])
    R.add(
        "Comparison",
        "C_ρ single cell at centre (A=1), max",
        max(rec["c1"]) / rec["c1"][0],
        note="NOT equivalent to a resolved density (Research Plan §2)",
    )
    TJ1 = occ[1] * (ts[-1] - ts[0])
    R.add("Comparison", "duration min_R Q<1 (A=1) [Myr]", TJ1, note=f"paper: {H['jeans_duration_myr']} Myr")

    viz.line_plot(
        figs / "p5_truelove.png",
        ts,
        {"J (A=1)": np.array(rec["J1"]), "J (A=0)": np.array(rec["J0"])},
        xlabel="t [Myr]",
        ylabel="dx/λ_J",
        title="Truelove-Kriterium",
        dpi=dpi,
        logy=True,
        hlines={"1/4": 0.25},
    )
    viz.line_plot(
        figs / "p5_central_density.png",
        ts,
        {"ρ_b(0) A=1": np.array(rec["c1"]), "ρ_b(0) A=0": np.array(rec["c0"])},
        xlabel="t [Myr]",
        ylabel="M_sun/kpc³",
        title="Zentrale Gasdichte (Einzelzelle)",
        dpi=dpi,
        logy=True,
    )
    viz.line_plot(
        figs / "p5_dt.png",
        ts[1:],
        {"Δt": np.array(rec["dt"][1:])},
        xlabel="t [Myr]",
        ylabel="Myr",
        title="macro time step",
        subtitle="dotted: snapshot times (Δt shortened there, harmless)",
        dpi=dpi,
        logy=True,
        hlines={"dt_max": P5["dt_max_myr"]},
        vlines={f"{t:g}": t for t in cfg["analysis"]["full_snapshot_times_myr"] if 0 < t < pair.t},
    )
    order = np.argsort(prof_t)
    prof_ts = np.array(prof_t)[order]
    P1, P0 = np.array(prof1)[order], np.array(prof0)[order]
    viz.heatmap(
        figs / "p5_profile_A1.png",
        rmid,
        prof_ts,
        np.log10(P1),
        xlabel="r [kpc]",
        ylabel="t [Myr]",
        zlabel="log10 ρ_b",
        title="radial gas profile A=1",
        dpi=dpi,
    )
    viz.surface3d_html(
        figs3 / "p5_profile_ratio_surface.html",
        rmid,
        prof_ts,
        P1 / P0,
        xlabel="r [kpc]",
        ylabel="t [Myr]",
        zlabel="ρ₁/ρ₀",
        title="radial profile ratio ρ_b(A=1)/ρ_b(A=0)",
        log_axes=(False, False, True),
    )
    viz.surface3d_png(
        figs3 / "p5_profile_ratio_surface.png",
        rmid,
        prof_ts,
        P1 / P0,
        xlabel="r [kpc]",
        ylabel="t [Myr]",
        zlabel="ρ₁/ρ₀",
        title="ρ_b(A=1)/ρ_b(A=0)",
        dpi=dpi,
        log_axes=(False, False, True),
    )
    viz.isosurface_html(
        figs3 / "p5_gas_A1_iso.html",
        b1.U[0],
        grid.box,
        title=f"ρ_b A=1, t={pair.t:.1f} Myr",
        log=True,
        label="ρ_b",
        max_points=cfg["viz"]["iso_max_points_per_axis"],
    )
    viz.ortho_slices_3d_png(
        figs3 / "p5_delta_rho_ortho.png",
        b1.U[0] / b0.U[0] - 1,
        grid.box,
        title="ρ₁/ρ₀ − 1 (final state)",
        dpi=dpi,
        log=False,
        label="Δρ/ρ₀",
    )
    viz.snapshot_series(
        figs,
        snapdata,
        rho_mean=gas_mass_density(n_cm3, gp["mu"]),
        noise_amp=gp["perturbation_rel_amplitude"],
        extent_kpc=grid.box,
        apertures=Rs_an,
        dpi=dpi,
        title=f"{cfg['meta']['run_label']} · N={N} · τ={tau:g} Myr · n={n_cm3:g} cm⁻³ · " f"{cfg['setup']['kind']}",
    )
    qmin = {A: np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0) for A in (1, 0)}
    info = {
        "_title": f"P5 – {cfg['meta']['run_label']} | N={N}, τ={tau:g} Myr, n={n_cm3:g} cm⁻³, "
        f"Setup={cfg['setup']['kind']}, floor N_J={cfg['hydro']['jeans_floor_NJ']:g}, "
        f"cooling={cfg['gas']['cooling_model']}",
        "Status": status,
        "t_final [Myr]": f"{pair.t:.2f}",
        f"O1(R={Rp}) end": f"{O1[Rp][-1]:+.3f}  (×{np.exp(O1[Rp][-1]):.2f} mass)",
        f"ΔlnQ(R={Rp}) end": f"{np.log(Q[(1, Rp)][-1] / Q[(0, Rp)][-1]):+.3f}",
        "O2(A=1) / O2(A=0)": f"{occ[1]:.2f} / {occ[0]:.2f}  (ΔO2 = {occ[1]-occ[0]:+.2f})",
        f"O3 early (≤{P5['early_window_myr']:g} Myr)": f"{o3e_mean:.3f}",
        "max Truelove J": f"{Jmax:.3g}" + ("  ⚠ unresolved" if Jmax > 0.25 else "  ✓"),
        "max Jeans-floor fraction": f"{max(rec['fl1']):.2f}",
        "aperture centre / max displacement": f"{cfg['analysis']['aperture_center']} / {shift:.2f} kpc",
        "Wall time": f"{time.perf_counter() - t0:.0f} s, {pair.steps} steps",
    }
    from hgr.soliton import ground_state_profile, physical_scales

    rc_kpc = physical_scales(
        ground_state_profile(), cfg["axion"]["soliton_mass_msun"], AxionUnits(cfg["axion"]["m22"]).hbar_over_m, G_CODE
    )["rc"]
    kz = int(ctr[-1][2])
    viz.p5_dashboard(
        figs / "p5_DASHBOARD.png",
        ts=ts,
        O1={r: O1[r] for r in Rs_an},
        dlnQ={r: np.log(np.array(Q[(1, r)]) / np.array(Q[(0, r)])) for r in Rs_an},
        qmin1=qmin[1],
        qmin0=qmin[0],
        o3={r: O3[r] for r in Rs_an},
        mb=mb,
        J1=rec["J1"],
        J0=rec["J0"],
        floor1=rec["fl1"],
        rho1=b1.U[0][:, :, kz],
        rho0=b0.U[0][:, :, kz],
        extent_kpc=grid.box,
        apertures=Rs_an,
        prof_r=rmid,
        prof_t=prof_ts,
        prof_ratio=P1 / P0,
        info=info,
        dpi=dpi,
        truelove_max=cfg["p6"]["truelove_max"],
        floor_max=cfg["p6"]["floor_mass_fraction_max"],
        q1p=Q[(1, Rp)],
        q0p=Q[(0, Rp)],
        phic=rec["phic"],
        phibc=rec["phibc"],
        center_kpc=cpos,
        floor_active=cfg["hydro"]["jeans_floor_NJ"] > 0,
        rc_kpc=rc_kpc,
        center_xy=(float(cpos[-1][0]), float(cpos[-1][1])),
    )
    series = {
        "t": ts.tolist(),
        "O1": {str(r): O1[r].tolist() for r in Rs_an},
        "O3": {str(r): list(map(float, O3[r])) for r in Rs_an},
        "Q": {f"A{A}_R{r}": list(map(float, Q[(A, r)])) for A in (1, 0) for r in Rs_an},
        "M": {f"A{A}_R{r}": list(map(float, M[(A, r)])) for A in (1, 0) for r in Rs_an},
        "truelove": {"A1": rec["J1"], "A0": rec["J0"]},
        "momentum_budget": list(map(float, mb)),
        "O2_intervals": {str(A): intervals[A] for A in (1, 0)},
        "F": {f"A{A}_R{r}": list(map(float, F[(A, r)])) for A in (1, 0) for r in Rs_an},
        "MF": {f"A{A}_R{r}": list(map(float, MF[(A, r)])) for A in (1, 0) for r in Rs_an},
        "probes": probes,
        "dt": list(map(float, rec["dt"])),
        "floor_frac": {"A1": rec["fl1"], "A0": rec["fl0"]},
        "center_kpc": cpos.tolist(),
        "phi_c": rec["phic"],
        "phibar_c": rec["phibc"],
        "meta": {
            "N": N,
            "tau": tau,
            "n_cm3": n_cm3,
            "t_end": t_end,
            "t_final": float(pair.t),
            "status": status,
            "run_label": cfg["meta"]["run_label"],
            "dx": grid.dx,
            "apertures": Rs_an,
            "primary": Rp,
            "config_path": cfg["_config_path"],
            "setup": cfg["setup"]["kind"],
            "jeans_NJ": cfg["hydro"]["jeans_floor_NJ"],
            "cooling": cfg["gas"]["cooling_model"],
            "seed": cfg["meta"]["seed"],
            "dt_max": cfg["p5"]["dt_max_myr"],
            "aperture_center": cfg["analysis"]["aperture_center"],
            "halo_seed": cfg["setup"]["halo_seed"],
            "filter_kind": cfg["filter"]["kernel"],
            "T_K": cfg["gas"]["T_K"],
            "halo_sigma_kms": cfg["setup"]["halo_sigma_kms"],
            "m22": cfg["axion"]["m22"],
            "gamma": cfg["gas"]["gamma"],
            "mu": cfg["gas"]["mu"],
        },
    }
    (out / "P5_timeseries.json").write_text(json.dumps(series), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--N", type=int)
    ap.add_argument("--t-end", type=float)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--n-cm3", type=float, help="override gas density (n scan)")
    ap.add_argument("--seed", type=int, help="seed of the gas perturbation")
    ap.add_argument("--setup", choices=["isolated", "soliton_halo"], help="B4: axion-Anfangskonfiguration")
    ap.add_argument("--jeans-floor", type=float, help="Jeans floor N_J (0 = off)")
    ap.add_argument("--cooling", choices=["piecewise_historical", "primordial_h2", "isothermal", "none"], help="B6")
    ap.add_argument("--dt-max", type=float, help="upper bound of macro time step [Myr]")
    ap.add_argument("--halo-seed", type=int, help="halo realisation (setup soliton_halo only)")
    ap.add_argument(
        "--filter-kind",
        choices=["trailing_boxcar", "zero_lag_boxcar", "delay_half_tau", "ema"],
        help="temporal filter of the gas potential",
    )
    ap.add_argument("--T-K", type=float, help="initial gas temperature [K] (Mach scan)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    N = args.N if args.N is not None else cfg["p5"]["N"]
    t_end = args.t_end if args.t_end is not None else cfg["p5"]["t_end_myr"]
    tau = args.tau if args.tau is not None else cfg["filter"]["tau_myr"]
    n_cm3 = args.n_cm3 if args.n_cm3 is not None else cfg["gas"]["n_cm3"]
    tags = ""
    for val, (sec, key), tag in (
        (args.seed, ("meta", "seed"), "s"),
        (args.setup, ("setup", "kind"), ""),
        (args.jeans_floor, ("hydro", "jeans_floor_NJ"), "jf"),
        (args.cooling, ("gas", "cooling_model"), "c-"),
        (args.dt_max, ("p5", "dt_max_myr"), "dt"),
        (args.halo_seed, ("setup", "halo_seed"), "hs"),
        (args.filter_kind, ("filter", "kernel"), "fk-"),
        (args.T_K, ("gas", "T_K"), "T"),
    ):
        if val is not None and val != cfg[sec][key]:
            cfg[sec][key] = val
            tags += f"_{tag}{val:g}" if isinstance(val, (int, float)) else f"_{tag}{val}"
    stage = f"P5_N{N}_tau{tau:g}_n{n_cm3:g}_t{t_end:g}{tags}"
    out = output_dir(cfg, stage)
    run(cfg, N, t_end, tau, n_cm3, out)
    R.write(
        out,
        stage,
        "Matched run A=1/A=0 and observables O1–O3",
        [
            f"N = {N}, t_end = {t_end} Myr, τ = {tau} Myr, n = {n_cm3} cm⁻³. "
            "Test apertures pre-declared (`analysis.apertures_kpc`). Resolution and "
            "ablation robustness (falsification matrix) follow in P6."
        ],
    )
    write_manifest(
        cfg,
        out,
        stage,
        extra={
            "overrides": {
                "N": N,
                "t_end": t_end,
                "tau": tau,
                "n_cm3": n_cm3,
                "seed": cfg["meta"]["seed"],
                "setup": cfg["setup"]["kind"],
                "jeans_NJ": cfg["hydro"]["jeans_floor_NJ"],
                "cooling": cfg["gas"]["cooling_model"],
            },
            **R.results,
        },
    )
    print(f"{stage} done -> {out} ({'all tests passed' if R.n_failed() == 0 else f'{R.n_failed()} test(s) failed'})")


if __name__ == "__main__":
    main()
