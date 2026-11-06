from __future__ import annotations

import argparse
import time

import numpy as np

from hgr import hydro as hy
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.coupled import CoupledPair, TrailingBoxcar
from hgr.grid import Grid3D
from hgr.report import Recorder
from hgr.setup import coupling_params, gas_initial_condition, polish_cfg
from hgr.soliton import embed_ground_state
from hgr.units import G_CODE, AxionUnits

R = Recorder()


def c1_filter(cfg, figs, figs3, dpi):
    V = cfg["validation_p4"]
    rng = np.random.default_rng(1)
    dts = rng.uniform(0.05, 0.4, 200)
    ts = np.concatenate([[0.0], np.cumsum(dts)])
    tau = 3.0
    f = TrailingBoxcar(tau)
    err = 0.0
    for i in range(1, len(ts)):
        f.push(ts[i - 1], ts[i], np.array([2.5 * (ts[i] - ts[i - 1])]))
        err = max(err, abs(f.mean(ts[i], np.array([2.5]))[0] / 2.5 - 1))
    R.add(
        "C1",
        "filter exact for constant Φ (unequal steps)",
        err,
        V["filter_rel_err_max"],
        err < V["filter_rel_err_max"],
    )
    alpha, beta = 1.0, 0.7
    f = TrailingBoxcar(tau)
    rel, ratio = 0.0, 0.0
    for i in range(1, len(ts)):
        f.push(ts[i - 1], ts[i], np.array([(alpha + beta * 0.5 * (ts[i - 1] + ts[i])) * (ts[i] - ts[i - 1])]))
        w = min(tau, ts[i])
        ex = alpha + beta * (ts[i] - w / 2)
        e = abs(f.mean(ts[i], np.array([alpha + beta * ts[i]]))[0] - ex)
        rel = max(rel, e / ex)
        ratio = max(ratio, e / (beta * dts.max() ** 2 / (8 * w)))
    R.add(
        "C1",
        "linear Φ: rel. error (pre-declared 'exact' threshold was WRONGLY justified)",
        rel,
        None,
        None,
        note="Eq. 42 is exact only for constant Φ; value kept for transparency",
    )
    R.add("C1", "linear Φ: error / bound βΔt²/(8w) (hergeleitet)", ratio, 1.0, ratio <= 1.0 + 1e-12)
    taus, oms = np.array(V["bias_taus_myr"]), np.array(V["bias_omegas_per_myr"])
    B = np.zeros((len(oms), len(taus)))
    tt = np.linspace(0, 60, 2401)
    for i, om in enumerate(oms):
        for j, ta in enumerate(taus):
            f = TrailingBoxcar(ta)
            worst = 0.0
            for k in range(1, len(tt)):
                integ = (np.cos(om * tt[k - 1]) - np.cos(om * tt[k])) / om
                f.push(tt[k - 1], tt[k], np.array([integ]))
                if tt[k] >= ta:
                    worst = max(worst, abs(f.mean(tt[k], np.array([np.sin(om * tt[k])]))[0] - np.sin(om * tt[k])))
            B[i, j] = worst / (om * ta / 2)
    R.add(
        "C1",
        "max Bias / bound Lτ/2 (Eq. 43)",
        B.max(),
        1.0,
        B.max() <= 1.0 + 1e-12,
        note="≤ 1 confirms the bound",
    )
    viz.heatmap(
        figs / "p4_C1_bias.png",
        taus,
        oms,
        B,
        xlabel="τ [Myr]",
        ylabel="ω [1/Myr]",
        zlabel="Bias / (Lτ/2)",
        title="C1: filter bias relative to bound Eq. 43",
        dpi=dpi,
        logx=True,
        logy=True,
        contour_levels=[0.5, 1.0],
    )
    viz.surface3d_png(
        figs3 / "p4_C1_bias_surface.png",
        taus,
        oms,
        B,
        xlabel="τ [Myr]",
        ylabel="ω [1/Myr]",
        zlabel="Bias/(Lτ/2)",
        title="C1: filter bias",
        dpi=dpi,
        log_axes=(True, True, False),
    )
    viz.surface3d_html(
        figs3 / "p4_C1_bias_surface.html",
        taus,
        oms,
        B,
        xlabel="τ [Myr]",
        ylabel="ω [1/Myr]",
        zlabel="Bias/(Lτ/2)",
        title="C1: filter bias",
        log_axes=(True, True, False),
    )
    f = TrailingBoxcar(3.0)
    m = []
    for k in range(1, len(tt)):
        f.push(tt[k - 1], tt[k], np.array([(np.cos(0.5 * tt[k - 1]) - np.cos(0.5 * tt[k])) / 0.5]))
        m.append(f.mean(tt[k], np.array([np.sin(0.5 * tt[k])]))[0])
    viz.line_plot(
        figs / "p4_C1_filter_demo.png",
        tt[1:],
        {"Φ(t) = sin(0.5 t)": np.sin(0.5 * tt[1:]), "Φ̄_τ, τ = 3 Myr": np.array(m)},
        xlabel="t [Myr]",
        ylabel="Φ",
        title="C1: filter effect",
        dpi=dpi,
    )


def _run(pair, t_end, record):
    while pair.t < t_end - 1e-12:
        dt = min(pair.max_dt(), t_end - pair.t)
        pair.step(dt)
        record(pair)


def c2_limit(cfg, figs, figs3, dpi, grid, psi):
    V = cfg["validation_p4"]
    pair = CoupledPair(
        grid, coupling_params(cfg, cfg["filter"]["tau_myr"]), psi, gas_initial_condition(cfg, grid, V["limit_n_cm3"])
    )
    c = grid.center_index
    r0 = abs(psi[c, c, c]) ** 2
    T, Y = [0.0], [1.0]
    _run(pair, V["limit_t_myr"], lambda p: (T.append(p.t), Y.append(abs(p.br[0].psi[c, c, c]) ** 2 / r0)))
    var = float(np.max(np.abs(np.array(Y) - 1)))
    R.add("C2", "max|ρ_a(0,t)/ρ_a(0,0) − 1| (gas negligible)", var, V["soliton_var_max"], var < V["soliton_var_max"])
    viz.line_plot(
        figs / "p4_C2_soliton.png",
        np.array(T),
        {"ρ_a(0,t)/ρ_a(0,0)": np.array(Y)},
        xlabel="t [Myr]",
        ylabel="relative",
        title="C2: soliton on the coupled code",
        dpi=dpi,
    )


def c3_null(cfg, grid):
    V = cfg["validation_p4"]
    pair = CoupledPair(
        grid,
        coupling_params(cfg, cfg["filter"]["tau_myr"]),
        np.zeros(grid.shape, complex),
        gas_initial_condition(cfg, grid, cfg["gas"]["n_cm3"]),
    )
    for _ in range(V["null_steps"]):
        pair.step(pair.max_dt())
    d = float(np.max(np.abs(pair.br[0].U - pair.br[1].U)) / np.max(np.abs(pair.br[1].U)))
    R.add(
        "C3",
        f"max|U(A=1)−U(A=0)|/max|U| after {V['null_steps']} steps (ψ≡0)",
        d,
        V["null_identity_rel_max"],
        d < V["null_identity_rel_max"],
        note="ablation changes ONLY the axion source",
    )


def c4_momentum(cfg, figs, figs3, dpi, grid, psi):
    V = cfg["validation_p4"]
    curves = {}
    for tau in V["momentum_tau_values_myr"]:
        pair = CoupledPair(
            grid, coupling_params(cfg, tau), psi.copy(), gas_initial_condition(cfg, grid, cfg["gas"]["n_cm3"])
        )
        b = pair.br[0]
        P0 = pair.momentum(b)
        T, D, S = [0.0], [0.0], [0.0]

        def rec(p):
            P = p.momentum(b)
            gas_abs = float(np.sum(np.sqrt(np.sum(b.U[1:4] ** 2, axis=0))) * grid.cell_volume)
            T.append(p.t), D.append(float(np.linalg.norm(P - P0))), S.append(gas_abs)

        _run(pair, V["momentum_t_myr"], rec)
        rel = max(D) / max(S)
        thr = V["momentum_rel_max_tau0"] if tau == 0 else None
        R.add(
            "C4",
            f"τ={tau}: max|ΔP_ges| / max Σ|ρv|",
            rel,
            thr,
            (rel < thr) if thr else None,
            note="τ>0: axion↔gas asymmetry (Research Plan §8.3)" if tau > 0 else "Splitting-error",
        )
        if tau > 0:
            gp = cfg["gas"]
            E_ref = float(
                np.sum(hy.pressure(gas_initial_condition(cfg, grid, gp["n_cm3"]), gp["gamma"]) / (gp["gamma"] - 1))
                * grid.cell_volume
            )
            R.add("C4", f"τ={tau}: ε_filter (Eq. 45), E_ref = E_int(0)", abs(pair.filter_work[0]) / E_ref)
        curves[f"τ = {tau} Myr"] = np.array(D) / max(S) + 1e-18
        R.add("C4", f"τ={tau}: macro steps / mittlere SP-substeps", pair.steps)
        if tau == V["momentum_tau_values_myr"][-1]:
            viz.ortho_slices_3d_png(
                figs3 / "p4_C4_gas_ortho.png",
                b.U[0],
                grid.box,
                title=f"C4: ρ_b (A=1) after {pair.t:.0f} Myr",
                dpi=dpi,
                log=True,
                label="ρ_b",
            )
            viz.isosurface_html(
                figs3 / "p4_C4_gas_iso.html",
                b.U[0],
                grid.box,
                title="C4: ρ_b (A=1)",
                log=True,
                label="ρ_b",
                max_points=cfg["viz"]["iso_max_points_per_axis"],
            )
            viz.slices_row(
                figs / "p4_C4_slices.png",
                [
                    (b.U[0], "ρ_b A=1", "log"),
                    (pair.br[1].U[0], "ρ_b A=0", "log"),
                    (b.phibar - b.phi, "Φ̄ − Φ (A=1)", "sym"),
                ],
                grid.box,
                dpi=dpi,
                suptitle=f"C4: final state t = {pair.t:.1f} Myr, N={grid.n}",
            )
    viz.line_plot(
        figs / "p4_C4_momentum.png",
        np.array(T),
        curves,
        xlabel="t [Myr]",
        ylabel="|ΔP| / Σ|ρv|",
        title="C4: total momentum gas+axion",
        dpi=dpi,
        logy=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P4")
    figs, figs3 = out / "figs", out / "figs3d"
    dpi = cfg["viz"]["dpi"]
    t = time.perf_counter()
    c1_filter(cfg, figs, figs3, dpi)
    print(f"  C1 done ({time.perf_counter()-t:.0f} s)", flush=True)
    grid = Grid3D(cfg["domain"]["box_kpc"], cfg["validation_p4"]["N"])
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    psi, _, _ = embed_ground_state(
        grid, cfg["axion"]["soliton_mass_msun"], a, G_CODE, polish_cfg(cfg), cfg["compute"]["fft_workers"]
    )
    for name, fn in [
        ("C2", lambda: c2_limit(cfg, figs, figs3, dpi, grid, psi)),
        ("C3", lambda: c3_null(cfg, grid)),
        ("C4", lambda: c4_momentum(cfg, figs, figs3, dpi, grid, psi)),
    ]:
        t = time.perf_counter()
        fn()
        print(f"  {name} done ({time.perf_counter()-t:.0f} s)", flush=True)
    R.write(
        out,
        "P4",
        "coupling axion–gas, potential filter, lockstep",
        ["Thresholds pre-declared in `validation_p4`. gas responds to Φ̄_τ, axions to Φ."],
    )
    write_manifest(cfg, out, "P4", extra=R.results)
    print(f"P4 done -> {out} ({'all passed' if R.n_failed() == 0 else f'{R.n_failed()} FAILED'})")
    if R.n_failed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
