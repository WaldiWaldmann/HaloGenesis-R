"""P5 – Gematchter Lauf A=1 (Soliton) / A=0 (Kontrolle) und Observablen O1–O3.

Aufruf:  python -m scripts.p5_observables --config configs/default.yaml [--N 128] [--t-end 30] [--tau 3]
Überschreibungen werden im Manifest protokolliert. Ergebnis: runs/<label>/P5[_suffix]/
"""
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

    Rs_an = list(cfg["analysis"]["apertures_kpc"])          # VORAB deklariert (O1/O2/O3-Tests)
    Rp = cfg["analysis"]["primary_aperture_kpc"]
    if min(Rs_an) < grid.dx:
        raise ValueError(f"Analyse-Apertur {min(Rs_an)} kpc < dx = {grid.dx:.3g} kpc: bei N={N} nicht aufgelöst. "
                         "Größeres N wählen (vorab deklarierte Aperturen werden nicht angepasst).")
    skipped = sorted(r for r in P5["viz_radii_kpc"] if r < grid.dx)
    Rs_viz = sorted({r for r in P5["viz_radii_kpc"] if r >= grid.dx} | set(Rs_an))  # nur Visualisierung
    # Aperturen mit Zuschnitt (nur Bounding-Box wird summiert)
    AP = {r: ob.Aperture(grid, r, with_gradient=False) for r in Rs_viz}          # mitlaufend (Observablen)
    APF = {r: ob.Aperture(grid, r, with_gradient=True) for r in Rs_an}          # fest (Fluss-Gegenprobe)
    TRK = ob.Aperture(grid, cfg["analysis"]["track_radius_kpc"], with_gradient=False)
    tracking = cfg["analysis"]["aperture_center"] == "soliton"
    if cfg["analysis"]["aperture_center"] not in ("soliton", "box"):
        raise ValueError("analysis.aperture_center muss 'soliton' oder 'box' sein")
    ctr = [(grid.center_index,) * 3]
    # gemeinsamer Zuschnitt für O3/Impulsbudget = Box der größten Analyse-Apertur
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
    cnt = np.bincount(ridx, minlength=len(edges) + 1)[1:len(edges)]
    prof_every = 5  # Radialprofile (volles Gitter) nur jeden 5. Schritt + Snapshots

    def radial(f):
        """Kugelschalen-Mittel um das (verfolgte) Soliton-Zentrum: ganzzahlige periodische Verschiebung."""
        sh = tuple(grid.center_index - int(a) for a in ctr[-1])
        f = np.roll(f, sh, axis=(0, 1, 2)) if any(sh) else f
        return np.bincount(ridx, weights=f.ravel(), minlength=len(edges) + 1)[1:len(edges)] / cnt

    ts, rec = [], {k: [] for k in ["dt", "J1", "J0", "c1", "c0", "o3p", "mb", "fl1", "fl0", "phic", "phibc", "nmax"]}
    M = {(A, r): [] for A in (1, 0) for r in Rs_viz}
    Q = {(A, r): [] for A in (1, 0) for r in Rs_viz}
    F = {(A, r): [] for A in (1, 0) for r in Rs_an}
    O3 = {r: [] for r in Rs_an}
    prof1, prof0, prof_t = [], [], []
    Ig = [np.zeros(grid.shape, np.float32) for _ in range(3)]  # volles Gitter (Zentrum kann wandern)
    dg_prev = None
    c = grid.center_index
    status = "regulär beendet"

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
            P = hy.pressure(b.U, gam)
            rec[f"J{A}"].append(ob.truelove_from_pressure(b.U[0], P, gam, G_CODE, grid.dx))
            for r in Rs_viz:
                q, _, m = AP[r].jeans_Q(b.U, P, gam, G_CODE)
                M[(A, r)].append(m), Q[(A, r)].append(q)
            for r in Rs_an:
                F[(A, r)].append(APF[r].flux(b.U))
        rec["c1"].append(b1.U[0][c, c, c]), rec["c0"].append(b0.U[0][c, c, c])
        UA = AP[max(Rs_an)]  # gemeinsamer Zuschnitt (größte Analyse-Apertur), mitlaufend
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
        rec["mb"].append(ob.weighted_norm(r0, tuple(dv[d] - UA.take(Ig[d]) for d in range(3)), WpU) / nv
                         if nv > 0 else np.nan)
        if force_profile or len(ts) % prof_every == 1:
            prof_t.append(pair.t), prof1.append(radial(b1.U[0])), prof0.append(radial(b0.U[0]))
        for A, b in ((1, b1), (0, b0)):  # B5: gebodeter Massenanteil in Primärapertur
            m = UA.take(pair.floor_mask[0 if A == 1 else 1])
            rb = UA.take(b.U[0])
            rec[f"fl{A}"].append(float(np.sum(WpU * rb * m) / np.sum(WpU * rb)))
        ci = ctr[-1]
        rec["phic"].append(float(b1.phi[ci])), rec["phibc"].append(b1.phibar_at(ci))
        rec["nmax"].append(float(co.number_density_cm3(b1.U[0].max(), gp["mu"])))

    snapdata = []  # Mittelschnitte; gezeichnet wird am Ende mit gemeinsamer Skala

    def snapshot(tag):
        k = int(ctr[-1][2])
        v1 = b1.U[1:4, :, :, k] / b1.U[0][None, :, :, k]
        v0 = b0.U[1:4, :, :, k] / b0.U[0][None, :, :, k]
        snapdata.append({"t": pair.t, "r1": b1.U[0][:, :, k].copy(), "r0": b0.U[0][:, :, k].copy(),
                         "dv_kms": np.sqrt(np.sum((v1 - v0) ** 2, axis=0)) / KMS_TO_CODE,
                         "ra": (b1.psi[:, :, k].real ** 2 + b1.psi[:, :, k].imag ** 2).copy(),
                         "cx": (ctr[-1][0] - grid.center_index) * grid.dx,
                         "cy": (ctr[-1][1] - grid.center_index) * grid.dx})

    measure(0.0)
    snapshot(f"{0.0:05.1f}")
    snaps = [s for s in snaps if s > 0]
    t0 = time.perf_counter()
    while pair.t < t_end - 1e-12:
        dt = pair.max_dt()
        if dt < P5["dt_min_abort_myr"]:
            status = f"ABBRUCH bei t = {pair.t:.3f} Myr: Δt = {dt:.2e} Myr < dt_min (Kollaps auf Gitterskala)"
            break
        nxt = min([s for s in snaps if s > pair.t + 1e-12] + [t_end])
        dt = min(dt, nxt - pair.t)
        pair.step(dt)
        measure(dt)
        if any(abs(pair.t - s) < 1e-9 for s in snaps):
            prof_t.append(pair.t), prof1.append(radial(b1.U[0])), prof0.append(radial(b0.U[0]))
            snapshot(f"{pair.t:05.1f}")
        if pair.steps % 25 == 0:
            print(f"    t = {pair.t:6.2f} Myr, Schritte {pair.steps}, dt = {dt:.3g}, "
                  f"{time.perf_counter()-t0:.0f} s", flush=True)
    ts = np.array(ts)
    R.add("Lauf", "Status: " + status, pair.t, note="t_final [Myr]")
    R.add("Lauf", "N / τ [Myr] / n [cm⁻³]", N, note=f"τ = {tau}, n = {n_cm3}")
    R.add("Lauf", "Makroschritte", pair.steps)
    R.add("Lauf", "ausgelassene Visualisierungsradien (< dx)", len(skipped), note=", ".join(map(str, skipped)))
    R.add("Lauf", "Rechenzeit [s]", time.perf_counter() - t0)
    R.add("Lauf", "Grundzustands-Residuum", ginfo["residual"])
    R.add("Lauf", f"Setup: {ginfo['kind']}", ginfo.get("M_total", cfg["axion"]["soliton_mass_msun"]),
          note="M_Axion gesamt [M_sun]" + (f"; Vorrelaxation {ginfo['relax_myr']:.1f} Myr, σ(Φ_c)/|Φ_c| = "
                                          f"{ginfo['phi_c_rel_std_relax']:.3g}, t_dB = {ginfo['t_dB_myr']:.1f} Myr"
                                          if "relax_myr" in ginfo else ""))
    phic, phibc = np.array(rec["phic"]), np.array(rec["phibc"])
    R.add("Lauf", "Potentialfluktuation σ(Φ_c)/|⟨Φ_c⟩| (A=1)", float(np.std(phic) / abs(np.mean(phic))),
          note="isolierter Grundzustand ≈ 0 -> Filter wirkungslos; Halo > 0")
    R.add("Lauf", "max|Φ̄_c − Φ_c|/|⟨Φ_c⟩| (A=1)", float(np.max(np.abs(phibc - phic)) / abs(np.mean(phic))))
    R.add("Lauf", "max Jeans-Boden-Massenanteil in R_p (A=1 / A=0)", max(rec["fl1"]),
          note=f"A=0: {max(rec['fl0']):.3g}; N_J = {cfg['hydro']['jeans_floor_NJ']}")
    R.add("Lauf", "max n_H (A=1) [cm⁻³]", max(rec["nmax"]), note="GP98-Niedrigdichte-Limes: n ≲ 10³ cm⁻³")
    R.add("Lauf", "lokale Rückfälle 1. Ordnung (Hydro)", pair.fallbacks)
    U_ic = gas_initial_condition(cfg, grid, n_cm3)
    E_ref = float(np.sum(co.pressure_from_T(U_ic[0], gp["T_K"], gp["mu"]) / (gam - 1)) * dV)
    R.add("Lauf", "ε_filter A=1 (Gl. 45), E_ref = E_int(0)", abs(pair.filter_work[0]) / E_ref)
    R.add("Lauf", "vom Jeans-Boden injizierte Energie / E_int(0) (A=1)", pair.floor_energy[0] / E_ref)
    Jmax = max(max(rec["J1"]), max(rec["J0"]))
    cpos = (np.array(ctr, float) - grid.center_index) * grid.dx
    shift = float(np.max(np.linalg.norm(cpos, axis=1)))
    R.add("Lauf", f"Aperturzentrum: {cfg['analysis']['aperture_center']}, max Verschiebung [kpc]", shift,
          note=f"Endposition {np.round(cpos[-1], 3).tolist()}; Rundung auf Zelle (dx = {grid.dx:.3g} kpc)")
    R.add("Lauf", "max Truelove-Zahl J = dx/λ_J (A=1, A=0)", Jmax, 0.25, Jmax <= 0.25,
          note="J > 1/4: künstliche Fragmentation möglich -> Ergebnis nicht auflösungsrobust")

    # ------------------------------------------------------------------ O1 --
    O1 = {r: np.log((np.array(M[(1, r)]) / M[(1, r)][0]) / (np.array(M[(0, r)]) / M[(0, r)][0])) for r in Rs_viz}
    for r in Rs_an:
        R.add("O1", f"O1(R={r}) bei t_final", O1[r][-1])
        R.add("O1", f"max O1(R={r})", np.max(O1[r]))
        for A in (1, 0):
            m = np.array(M[(A, r)])
            f = np.array(F[(A, r)])
            integ = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ts))])
            dm = m - m[0]
            scale = np.max(np.abs(dm))
            res = np.max(np.abs(dm - integ)) / scale if scale > 0 else np.nan
            R.add("O1", f"Fluss-Gegenprobe R={r}, A={A}: max|ΔM − ∫F dt|/max|ΔM| (alt)", res,
                  note="schlecht konditioniert, wenn ΔM ≈ 0 – nur Info")
            res0 = float(np.max(np.abs(dm - integ)) / m[0])
            thr0 = cfg["p6"]["flux_residual_M0_max"]
            R.add("O1", f"Massenbilanz R={r}, A={A}: max|ΔM − ∫F dt|/M(0)", res0, thr0, res0 <= thr0,
                  note="feste Apertur; Normierung vorab in p6 festgelegt")
            if r == Rp:
                viz.line_plot(figs / f"p5_O1_flux_check_A{A}.png", ts,
                              {"ΔM/M(0) direkt": dm / m[0], "∫ Oberflächenfluss dt / M(0)": integ / m[0],
                               "|Differenz|/M(0)": np.abs(dm - integ) / m[0]},
                              xlabel="t [Myr]", ylabel="relativ zu M(<R, t=0)",
                              title=f"Massenbilanz (feste Apertur) A={A}, R={r} kpc",
                              subtitle=f"max |Differenz|/M(0) = {np.max(np.abs(dm - integ)) / m[0]:.1e}", dpi=dpi)
    viz.line_plot(figs / "p5_O1.png", ts, {f"R = {r} kpc": O1[r] for r in Rs_an}, xlabel="t [Myr]", ylabel="O1",
                  title="O1: differentielle Konzentration (0 = kein Effekt)", dpi=dpi, hlines={"0": 0.0})
    Z1 = np.array([O1[r] for r in Rs_viz]).T
    viz.heatmap(figs / "p5_O1_map.png", np.array(Rs_viz), ts, Z1, xlabel="R [kpc]", ylabel="t [Myr]", zlabel="O1",
                title="O1(R, t)", dpi=dpi, logx=True)
    viz.surface3d_png(figs3 / "p5_O1_surface.png", np.array(Rs_viz), ts, Z1, xlabel="R [kpc]", ylabel="t [Myr]",
                      zlabel="O1", title="O1(R, t)", dpi=dpi, log_axes=(True, False, False))
    viz.surface3d_html(figs3 / "p5_O1_surface.html", np.array(Rs_viz), ts, Z1, xlabel="R [kpc]", ylabel="t [Myr]",
                       zlabel="O1", title="O1(R, t)", log_axes=(True, False, False))

    # ------------------------------------------------------------------ O2 --
    occ, intervals = {}, {}
    for A in (1, 0):
        qmin = np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0)
        occ[A], intervals[A] = ob.occupancy(ts, qmin, cfg["analysis"]["jeans_threshold"])
        R.add("O2", f"O2(A={A}) Zeitanteil min_R Q < 1", occ[A],
              note="Intervalle: " + ", ".join(f"[{x:.2f}, {y:.2f}]" for x, y in intervals[A][:6]))
        R.add("O2", f"min Q (A={A})", float(qmin.min()))
    R.add("O2", "ΔO2 = O2(1) − O2(0)", occ[1] - occ[0], note="Hypothese: > 0")
    for A in (1, 0):
        Zq = np.array([Q[(A, r)] for r in Rs_viz]).T
        viz.heatmap(figs / f"p5_O2_Q_A{A}.png", np.array(Rs_viz), ts, Zq, xlabel="R [kpc]", ylabel="t [Myr]",
                    zlabel="Q = M_J/M_b", title=f"Jeans-Verhältnis A={A}", dpi=dpi, logx=True, logz=True,
                    contour_levels=[1.0])
        viz.surface3d_html(figs3 / f"p5_O2_Q_A{A}_surface.html", np.array(Rs_viz), ts, Zq, xlabel="R [kpc]",
                           ylabel="t [Myr]", zlabel="Q", title=f"Q(R, t), A={A}", log_axes=(True, False, True))
        viz.surface3d_png(figs3 / f"p5_O2_Q_A{A}_surface.png", np.array(Rs_viz), ts, Zq, xlabel="R [kpc]",
                          ylabel="t [Myr]", zlabel="Q", title=f"Q(R, t), A={A}", dpi=dpi, log_axes=(True, False, True))
    viz.line_plot(figs / "p5_O2_Qmin.png", ts, {f"min_R Q, A={A}": np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0)
                  for A in (1, 0)}, xlabel="t [Myr]", ylabel="Q", title="O2: Jeans-Diagnostik (vorab deklarierte R)",
                  dpi=dpi, logy=True, hlines={"Q = 1": 1.0})

    # ------------------------------------------------------------------ O3 --
    early = (ts > 0) & (ts <= P5["early_window_myr"])
    o3e = np.array(O3[Rp])[early]
    o3e_mean = float(np.nanmean(o3e)) if o3e.size else np.nan
    R.add("O3", f"O3(R={Rp}) Mittel für t ≤ {P5['early_window_myr']} Myr", o3e_mean, P5["o3_early_min"],
          (o3e_mean >= P5["o3_early_min"]) if np.isfinite(o3e_mean) else None, note="Gl. 30: O3 → 1 früh")
    mb = np.array(rec["mb"])
    R.add("O3", "Impulsbudget früh: max ‖δv − ∫δg dt‖/‖δv‖", float(np.nanmax(mb[early])) if early.any() else np.nan,
          note="Rest = Druck/Advektion R(t)")
    for r in Rs_an:
        R.add("O3", f"O3(R={r}) zeitgemittelt", float(np.nanmean(O3[r][1:])))
    viz.line_plot(figs / "p5_O3.png", ts[1:], {f"R = {r} kpc": np.array(O3[r][1:]) for r in Rs_an}, xlabel="t [Myr]",
                  ylabel="O3", title="O3: Kohärenz Zusatzkraft ↔ Geschwindigkeitsantwort", dpi=dpi, hlines={"0": 0.0})
    viz.line_plot(figs / "p5_O3_momentum_budget.png", ts[1:], {"‖δv − ∫δg dt‖/‖δv‖": mb[1:]}, xlabel="t [Myr]",
                  ylabel="relativ", title="O3-Gegenprobe: Impulsbudget", dpi=dpi, logy=True)
    Z3 = np.array([O3[r][1:] for r in Rs_an]).T
    viz.surface3d_png(figs3 / "p5_O3_surface.png", np.array(Rs_an), ts[1:], Z3, xlabel="R [kpc]", ylabel="t [Myr]",
                      zlabel="O3", title="O3(R, t)", dpi=dpi)

    # -------------------------------------------- Vergleich Originalarbeit --
    H = cfg["historical_reference"]
    rs = min(Rs_an)
    ap = np.array(M[(1, rs)]) / M[(1, rs)][0]
    R.add("Vergleich", f"C_ρ Aperturmittel R={rs} (A=1), max", ap.max(), note=f"Arbeit: ~{H['central_gas_enhancement']}")
    i12 = int(np.argmin(np.abs(ts - H["t_sim_myr"])))
    R.add("Vergleich", f"C_ρ Aperturmittel R={rs} bei t≈{ts[i12]:.2f} Myr", ap[i12])
    R.add("Vergleich", "C_ρ Einzelzelle Zentrum (A=1), max", max(rec["c1"]) / rec["c1"][0],
          note="NICHT gleichwertig zu aufgelöster Dichte (Research Plan §2)")
    TJ1 = occ[1] * (ts[-1] - ts[0])
    R.add("Vergleich", "Dauer min_R Q<1 (A=1) [Myr]", TJ1, note=f"Arbeit: {H['jeans_duration_myr']} Myr")

    # ----------------------------------------------------------- Übersicht --
    viz.line_plot(figs / "p5_truelove.png", ts, {"J (A=1)": np.array(rec["J1"]), "J (A=0)": np.array(rec["J0"])},
                  xlabel="t [Myr]", ylabel="dx/λ_J", title="Truelove-Kriterium", dpi=dpi, logy=True,
                  hlines={"1/4": 0.25})
    viz.line_plot(figs / "p5_central_density.png", ts, {"ρ_b(0) A=1": np.array(rec["c1"]), "ρ_b(0) A=0": np.array(rec["c0"])},
                  xlabel="t [Myr]", ylabel="M_sun/kpc³", title="Zentrale Gasdichte (Einzelzelle)", dpi=dpi, logy=True)
    viz.line_plot(figs / "p5_dt.png", ts[1:], {"Δt": np.array(rec["dt"][1:])}, xlabel="t [Myr]", ylabel="Myr",
                  title="Makro-Zeitschritt", subtitle="gepunktet: Snapshot-Zeiten (Δt wird dort gekürzt, harmlos)",
                  dpi=dpi, logy=True, hlines={"dt_max": P5["dt_max_myr"]},
                  vlines={f"{t:g}": t for t in cfg["analysis"]["full_snapshot_times_myr"] if 0 < t < pair.t})
    order = np.argsort(prof_t)
    prof_ts = np.array(prof_t)[order]
    P1, P0 = np.array(prof1)[order], np.array(prof0)[order]
    viz.heatmap(figs / "p5_profile_A1.png", rmid, prof_ts, np.log10(P1), xlabel="r [kpc]", ylabel="t [Myr]",
                zlabel="log10 ρ_b", title="Radialprofil Gas A=1", dpi=dpi)
    viz.surface3d_html(figs3 / "p5_profile_ratio_surface.html", rmid, prof_ts, P1 / P0, xlabel="r [kpc]", ylabel="t [Myr]",
                       zlabel="ρ₁/ρ₀", title="Radialprofil-Verhältnis ρ_b(A=1)/ρ_b(A=0)", log_axes=(False, False, True))
    viz.surface3d_png(figs3 / "p5_profile_ratio_surface.png", rmid, prof_ts, P1 / P0, xlabel="r [kpc]", ylabel="t [Myr]",
                      zlabel="ρ₁/ρ₀", title="ρ_b(A=1)/ρ_b(A=0)", dpi=dpi, log_axes=(False, False, True))
    viz.isosurface_html(figs3 / "p5_gas_A1_iso.html", b1.U[0], grid.box, title=f"ρ_b A=1, t={pair.t:.1f} Myr", log=True,
                        label="ρ_b", max_points=cfg["viz"]["iso_max_points_per_axis"])
    viz.ortho_slices_3d_png(figs3 / "p5_delta_rho_ortho.png", b1.U[0] / b0.U[0] - 1, grid.box,
                            title="ρ₁/ρ₀ − 1 (Endzustand)", dpi=dpi, log=False, label="Δρ/ρ₀")
    viz.snapshot_series(figs, snapdata, rho_mean=gas_mass_density(n_cm3, gp["mu"]),
                        noise_amp=gp["perturbation_rel_amplitude"], extent_kpc=grid.box, apertures=Rs_an, dpi=dpi,
                        title=f"{cfg['meta']['run_label']} · N={N} · τ={tau:g} Myr · n={n_cm3:g} cm⁻³ · "
                              f"{cfg['setup']['kind']}")
    # ------------------------------------------------------- Dashboard --
    qmin = {A: np.min(np.array([Q[(A, r)] for r in Rs_an]), axis=0) for A in (1, 0)}
    info = {"_title": f"P5 – {cfg['meta']['run_label']} | N={N}, τ={tau:g} Myr, n={n_cm3:g} cm⁻³, "
                      f"Setup={cfg['setup']['kind']}, Boden N_J={cfg['hydro']['jeans_floor_NJ']:g}, "
                      f"Kühlung={cfg['gas']['cooling_model']}",
            "Status": status, "t_final [Myr]": f"{pair.t:.2f}",
            f"O1(R={Rp}) Ende": f"{O1[Rp][-1]:+.3f}  (×{np.exp(O1[Rp][-1]):.2f} Masse)",
            f"ΔlnQ(R={Rp}) Ende": f"{np.log(Q[(1, Rp)][-1] / Q[(0, Rp)][-1]):+.3f}",
            "O2(A=1) / O2(A=0)": f"{occ[1]:.2f} / {occ[0]:.2f}  (ΔO2 = {occ[1]-occ[0]:+.2f})",
            f"O3 früh (≤{P5['early_window_myr']:g} Myr)": f"{o3e_mean:.3f}",
            "max Truelove J": f"{Jmax:.3g}" + ("  ⚠ nicht aufgelöst" if Jmax > 0.25 else "  ✓"),
            "max Jeans-Boden-Anteil": f"{max(rec['fl1']):.2f}",
            "Aperturzentrum / max Verschiebung": f"{cfg['analysis']['aperture_center']} / {shift:.2f} kpc",
            "Rechenzeit": f"{time.perf_counter() - t0:.0f} s, {pair.steps} Schritte"}
    from hgr.soliton import ground_state_profile, physical_scales
    rc_kpc = physical_scales(ground_state_profile(), cfg["axion"]["soliton_mass_msun"],
                             AxionUnits(cfg["axion"]["m22"]).hbar_over_m, G_CODE)["rc"]
    kz = int(ctr[-1][2])
    viz.p5_dashboard(figs / "p5_DASHBOARD.png", ts=ts, O1={r: O1[r] for r in Rs_an},
                     dlnQ={r: np.log(np.array(Q[(1, r)]) / np.array(Q[(0, r)])) for r in Rs_an},
                     qmin1=qmin[1], qmin0=qmin[0], o3={r: O3[r] for r in Rs_an}, mb=mb, J1=rec["J1"], J0=rec["J0"],
                     floor1=rec["fl1"], rho1=b1.U[0][:, :, kz], rho0=b0.U[0][:, :, kz], extent_kpc=grid.box,
                     apertures=Rs_an, prof_r=rmid, prof_t=prof_ts, prof_ratio=P1 / P0, info=info, dpi=dpi,
                     truelove_max=cfg["p6"]["truelove_max"], floor_max=cfg["p6"]["floor_mass_fraction_max"],
                     q1p=Q[(1, Rp)], q0p=Q[(0, Rp)], phic=rec["phic"], phibc=rec["phibc"], center_kpc=cpos,
                     floor_active=cfg["hydro"]["jeans_floor_NJ"] > 0, rc_kpc=rc_kpc,
                     center_xy=(float(cpos[-1][0]), float(cpos[-1][1])))
    viz.setup_schematic(figs / "p5_SETUP.png", dpi=dpi, params={
        "N": N, "dx": f"{grid.dx:.3g} kpc", "L": f"{grid.box:g} kpc", "τ": f"{tau:g} Myr", "n": f"{n_cm3:g} cm⁻³",
        "Setup": cfg["setup"]["kind"], "Kühlung": cfg["gas"]["cooling_model"],
        "N_J": f"{cfg['hydro']['jeans_floor_NJ']:g}", "dt_max": f"{P5['dt_max_myr']:g} Myr",
        "Apertur": cfg["analysis"]["aperture_center"], "m22": f"{cfg['axion']['m22']:g}"})
    series = {"t": ts.tolist(), "O1": {str(r): O1[r].tolist() for r in Rs_an},
              "O3": {str(r): list(map(float, O3[r])) for r in Rs_an},
              "Q": {f"A{A}_R{r}": list(map(float, Q[(A, r)])) for A in (1, 0) for r in Rs_an},
              "M": {f"A{A}_R{r}": list(map(float, M[(A, r)])) for A in (1, 0) for r in Rs_an},
              "truelove": {"A1": rec["J1"], "A0": rec["J0"]}, "momentum_budget": list(map(float, mb)),
              "O2_intervals": {str(A): intervals[A] for A in (1, 0)},
              "F": {f"A{A}_R{r}": list(map(float, F[(A, r)])) for A in (1, 0) for r in Rs_an},
              "dt": list(map(float, rec["dt"])),
              "floor_frac": {"A1": rec["fl1"], "A0": rec["fl0"]}, "center_kpc": cpos.tolist(), "phi_c": rec["phic"], "phibar_c": rec["phibc"],
              "meta": {"N": N, "tau": tau, "n_cm3": n_cm3, "t_end": t_end, "t_final": float(pair.t),
                       "status": status, "run_label": cfg["meta"]["run_label"], "dx": grid.dx,
                       "apertures": Rs_an, "primary": Rp, "config_path": cfg["_config_path"],
                       "setup": cfg["setup"]["kind"], "jeans_NJ": cfg["hydro"]["jeans_floor_NJ"],
                       "cooling": cfg["gas"]["cooling_model"], "seed": cfg["meta"]["seed"],
                       "dt_max": cfg["p5"]["dt_max_myr"], "aperture_center": cfg["analysis"]["aperture_center"]}}
    (out / "P5_timeseries.json").write_text(json.dumps(series), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--N", type=int)
    ap.add_argument("--t-end", type=float)
    ap.add_argument("--tau", type=float)
    ap.add_argument("--n-cm3", type=float, help="Gasdichte überschreiben (n-Scan)")
    ap.add_argument("--seed", type=int, help="B7: Seed der Gas-Störung")
    ap.add_argument("--setup", choices=["isolated", "soliton_halo"], help="B4: Axion-Anfangskonfiguration")
    ap.add_argument("--jeans-floor", type=float, help="B5: N_J (0 = aus)")
    ap.add_argument("--cooling", choices=["piecewise_historical", "primordial_h2", "isothermal", "none"], help="B6")
    ap.add_argument("--dt-max", type=float, help="Obergrenze Makro-Zeitschritt [Myr] (Zeitverfeinerung)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    N = args.N if args.N is not None else cfg["p5"]["N"]
    t_end = args.t_end if args.t_end is not None else cfg["p5"]["t_end_myr"]
    tau = args.tau if args.tau is not None else cfg["filter"]["tau_myr"]
    n_cm3 = args.n_cm3 if args.n_cm3 is not None else cfg["gas"]["n_cm3"]
    tags = ""
    for val, (sec, key), tag in ((args.seed, ("meta", "seed"), "s"), (args.setup, ("setup", "kind"), ""),
                                 (args.jeans_floor, ("hydro", "jeans_floor_NJ"), "jf"),
                                 (args.cooling, ("gas", "cooling_model"), "c-"),
                                 (args.dt_max, ("p5", "dt_max_myr"), "dt")):
        if val is not None and val != cfg[sec][key]:
            cfg[sec][key] = val
            tags += f"_{tag}{val:g}" if isinstance(val, (int, float)) else f"_{tag}{val}"
    stage = f"P5_N{N}_tau{tau:g}_n{n_cm3:g}_t{t_end:g}{tags}"
    out = output_dir(cfg, stage)
    run(cfg, N, t_end, tau, n_cm3, out)
    R.write(out, stage, "Gematchter Lauf A=1/A=0 und Observablen O1–O3",
            [f"N = {N}, t_end = {t_end} Myr, τ = {tau} Myr, n = {n_cm3} cm⁻³. "
             "Aperturen für Tests vorab deklariert (`analysis.apertures_kpc`). Auflösungs- und "
             "Ablationsrobustheit (Falsifikationsmatrix) folgt in P6."])
    write_manifest(cfg, out, stage, extra={"overrides": {"N": N, "t_end": t_end, "tau": tau, "n_cm3": n_cm3, "seed": cfg["meta"]["seed"],
                                                        "setup": cfg["setup"]["kind"], "jeans_NJ": cfg["hydro"]["jeans_floor_NJ"],
                                                        "cooling": cfg["gas"]["cooling_model"]}, **R.results})
    print(f"{stage} fertig -> {out} ({'alle Tests bestanden' if R.n_failed() == 0 else f'{R.n_failed()} Test(s) nicht bestanden'})")


if __name__ == "__main__":
    main()
