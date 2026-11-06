from __future__ import annotations

import argparse
import time

import numpy as np
import scipy.fft as sfft

from hgr import hydro as hy
from hgr import response as rp
from hgr import viz
from hgr.config import load_config, output_dir, write_manifest
from hgr.filters import KERNELS, make_filter, transfer_function
from hgr.grid import Grid3D
from hgr.poisson import gradient_spectral, solve_poisson_periodic
from hgr.report import Recorder
from hgr.units import KMS_TO_CODE, AxionUnits

R = Recorder()


def v1_filters(cfg, figs, figs3, dpi):
    V = cfg["validation_p7"]
    tau = 1.0
    dt = V["filter_dt_over_tau"] * tau
    wt = np.array(V["filter_omega_tau"], float)
    wfine = np.geomspace(0.05, 30, 400)
    num = {k: [] for k in KERNELS}
    for kind in KERNELS:
        for x in wt:
            w = x / tau
            f = make_filter(kind, tau)
            f.reset(0.0, np.array([1.0]))
            ts, ys = [], []
            t = 0.0
            t_end = 8 * tau + 6 * 2 * np.pi / w
            while t < t_end - 1e-12:
                t1 = t + dt
                integ = np.array([(np.sin(w * t1) - np.sin(w * t)) / w])
                f.push(t, t1, integ, np.array([np.cos(w * t1)]))
                t = t1
                if t > 8 * tau:
                    ts.append(t), ys.append(f.value(t, np.array([np.cos(w * t)]))[0])
            ts, ys = np.array(ts), np.array(ys)
            A = np.vstack([np.cos(w * ts), np.sin(w * ts)]).T
            (c, s_), *_ = np.linalg.lstsq(A, ys, rcond=None)
            H_num = c - 1j * s_
            H_th = transfer_function(kind, np.array([w]), tau)[0]
            err = abs(H_num - H_th)
            num[kind].append(H_num)
            R.add("V1", f"{kind}: |H_num − H| at ωτ={x:g}", err, V["filter_rel_err_max"], err < V["filter_rel_err_max"])
    fig_series = {}
    for kind in KERNELS:
        fig_series[f"|H| {kind}"] = np.abs(transfer_function(kind, wfine, tau))
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    for i, kind in enumerate(KERNELS):
        H = transfer_function(kind, wfine, tau)
        Hn = np.array(num[kind])
        axs[0].plot(wfine, np.abs(H), color=viz.OKABE[i], label=kind)
        axs[0].plot(wt, np.abs(Hn), "o", color=viz.OKABE[i], ms=5)
        axs[1].plot(wfine, -np.unwrap(np.angle(H)) / wfine, color=viz.OKABE[i])
        axs[1].plot(wt, -np.angle(Hn) / wt, "o", color=viz.OKABE[i], ms=5)
        axs[2].plot(H.real, H.imag, color=viz.OKABE[i])
        axs[2].plot(Hn.real, Hn.imag, "o", color=viz.OKABE[i], ms=5)
    axs[0].set_xscale("log"), axs[0].set_xlabel("ωτ"), axs[0].set_ylabel("|H|  (smoothing)"), axs[0].legend(fontsize=8)
    axs[0].set_title("amplitude response: line analytic, points implementation", loc="left")
    axs[1].set_xscale("log"), axs[1].set_xlabel("ωτ"), axs[1].set_ylabel("lag −φ/ω  [τ]")
    axs[1].set_title("lag (zero_lag ≈ 0 for slow signals)", loc="left")
    axs[2].set_aspect("equal"), axs[2].set_xlabel("Re H"), axs[2].set_ylabel("Im H")
    axs[2].set_title("locus H(ω) (Nyquist)", loc="left"), axs[2].axhline(0, color="k", lw=0.5), axs[2].axvline(
        0, color="k", lw=0.5
    )
    fig.suptitle("V1 – temporal filters: delay and smoothing separated", x=0.01, ha="left", fontweight="bold")
    fig.savefig(figs / "p7_V1_filters.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def v2_drive(cfg, figs, figs3, dpi, workers):
    V = cfg["validation_p7"]
    N = V["drive_N"]
    g = Grid3D(1.0, N)
    G, rho0, gam = 1.0, 0.1, 5.0 / 3.0
    k = 2 * np.pi * 2
    cs = 0.2
    ws2 = cs**2 * k**2 - 4 * np.pi * G * rho0
    ws = np.sqrt(ws2)
    p0 = rho0 * cs**2 / gam
    x = g.mesh_sparse[0] + 0 * g.mesh_sparse[1] + 0 * g.mesh_sparse[2]
    cosx = np.cos(k * g.x1d)
    curves, pts = {}, []
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, len(V["drive_omega_over_ws"]), figsize=(4.2 * len(V["drive_omega_over_ws"]), 3.6))
    for j, ratio in enumerate(V["drive_omega_over_ws"]):
        w = ratio * ws
        chi = -(k**2) / (ws2 - w**2)
        eps = 1e-3 / abs(chi)
        U = hy.cons_from_prim(np.full(g.shape, rho0), 0 * x, 0 * x, 0 * x, np.full(g.shape, p0), gam)
        T_end = 3 * 2 * np.pi / min(w, ws)
        t, s, ts, dk = 0.0, 0, [0.0], [0.0]

        def accel(U, tt):
            phi = solve_poisson_periodic(U[0], g, G, workers)
            gs = [-d for d in gradient_spectral(phi, g, workers)]
            gs[0] = gs[0] + eps * k * np.sin(k * x) * np.cos(w * tt)
            return tuple(gs)

        acc = accel(U, 0.0)
        while t < T_end - 1e-12:
            dt = min(hy.cfl_dt(U, g.dx, gam, 0.3), T_end - t, 2 * np.pi / max(w, ws) / 80)
            hy.gravity_half_kick(U, acc, 0.5 * dt)
            hy.step(U, dt, g.dx, gam, cfg["hydro"]["limiter"], s)
            t += dt
            s += 1
            acc = accel(U, t)
            hy.gravity_half_kick(U, acc, 0.5 * dt)
            ts.append(t), dk.append(2 * np.mean((U[0][:, 0, 0] / rho0 - 1) * cosx))
        ts, dk = np.array(ts), np.array(dk)
        th = chi * eps * (np.cos(w * ts) - np.cos(ws * ts))
        err = np.max(np.abs(dk - th)) / np.max(np.abs(th))
        R.add(
            "V2",
            f"ω/ω_s = {ratio:g}: max|δ_num − δ_exakt|/max|δ_exakt| (N={N}, info)",
            err,
            note="contains numerical damping of the FREE oscillation cos ω_s t over ~10 periods",
        )
        A = np.vstack(
            [
                np.cos(w * ts),
                np.sin(w * ts),
                np.cos(ws * ts),
                np.sin(ws * ts),
                ts * np.cos(ws * ts),
                ts * np.sin(ws * ts),
            ]
        ).T
        coef = np.linalg.lstsq(A, dk / eps, rcond=None)[0]
        chi_num = float(coef[0])
        errc = abs(chi_num / chi - 1)
        R.add(
            "V2",
            f"ω/ω_s = {ratio:g}: |χ_num/χ − 1| (driven component)",
            errc,
            V["drive_rel_err_max"],
            errc < V["drive_rel_err_max"],
            note=f"χ = {chi:.3g}, χ_num = {chi_num:.3g}",
        )
        pts.append((ratio, chi_num, chi))
        ax = axs[j]
        ax.plot(ts, th * 1e3, color="k", lw=1.2, label="exact (linear)")
        ax.plot(ts, dk * 1e3, color=viz.OKABE[1], lw=1.4, ls="--", label="simulation")
        ax.set_title(f"ω/ω_s = {ratio:g}  (error {err:.1%})", loc="left", fontsize=9.5)
        ax.set_xlabel("t"), ax.set_ylabel("δ_k  [10⁻³]") if j == 0 else None
    axs[0].legend(fontsize=8)
    R.add("V2", "Note", N, note="convergence of the curve deviation with N: see P3-H3 (Jeans, N=64: ≤1 %)")
    fig.suptitle(
        "V2 – driven gas mode with self-gravity: simulation vs exact linear solution",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.savefig(figs / "p7_V2_driven_modes.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    rr = np.linspace(0.05, 3.0, 600)
    chi_line = -(k**2) / (ws2 - (rr * ws) ** 2)
    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.plot(rr, np.abs(chi_line), color="k", label="|χ| theory")
    ax.plot([p[0] for p in pts], [abs(p[1]) for p in pts], "o", color=viz.OKABE[1], ms=7, label="|χ| measured")
    ax.axvline(1, color="0.5", ls=":"), ax.set_yscale("log"), ax.set_xlabel("ω/ω_s"), ax.set_ylabel("|χ| = |δ/Φ_ext|")
    ax.set_title("V2 – susceptibility spectroscopy: resonance at ω = ω_s", loc="left"), ax.legend()
    fig.savefig(figs / "p7_V2_resonance.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    ks = np.linspace(2, 30, 120)
    ws_ = np.linspace(0.05, 8, 160)
    CH = np.abs(rp.susceptibility(ks[None, :], ws_[:, None], cs, rho0, 0.15, G=G))
    viz.surface3d_png(
        figs3 / "p7_V2_chi_surface.png",
        ks,
        ws_,
        CH,
        xlabel="k",
        ylabel="ω",
        zlabel="|χ|",
        title="|χ(k,ω)| – resonance ridge ω = ω_s(k)",
        dpi=dpi,
        log_axes=(False, False, True),
    )
    viz.surface3d_html(
        figs3 / "p7_V2_chi_surface.html",
        ks,
        ws_,
        CH,
        xlabel="k",
        ylabel="ω",
        zlabel="|χ|",
        title="|χ(k,ω)| – resonance ridge",
        log_axes=(False, False, True),
    )


def v3_granules(cfg, figs, figs3, dpi, workers):
    V = cfg["validation_p7"]
    N, L = V["granule_N"], V["granule_box_kpc"]
    g = Grid3D(L, N)
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    sig = cfg["setup"]["halo_sigma_kms"] * KMS_TO_CODE
    rng = np.random.default_rng(3)
    amp = np.exp(-(a**2) * g.k2_full / (4 * sig**2))
    psik = amp * (rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape))
    wk = 0.5 * a * g.k2_full
    nt, dt = 512, 2.0
    kmag = np.sqrt(g.k2_full)
    edges = np.linspace(0.6, 6.5, 14)
    sh = np.digitize(kmag.ravel(), edges)
    series = np.zeros((nt, len(edges) - 1), complex)
    Pt = []
    var = []
    for it in range(nt):
        psi = sfft.ifftn(psik * np.exp(-1j * wk * it * dt), workers=workers)
        rho = np.abs(psi) ** 2
        d = rho / rho.mean() - 1
        var.append(np.mean(d**2))
        Pt.append(sfft.fftn(d, workers=workers).ravel())
    Pt = np.array(Pt)
    win = np.hanning(nt)[:, None]
    Pw = np.abs(sfft.fft(Pt * win, axis=0)) ** 2
    om = 2 * np.pi * sfft.fftfreq(nt, dt)
    MIN_MODES = 150
    kc, sw_meas, sw_th, S_meas, nmodes = [], [], [], [], []
    for i in range(1, len(edges)):
        m = sh == i
        if m.sum() < 10:
            continue
        spec = Pw[:, m].sum(axis=1)
        kc.append(kmag.ravel()[m].mean())
        sw_meas.append(np.sqrt(np.sum(om**2 * spec) / np.sum(spec)))
        nmodes.append(int(m.sum()))
        sw_th.append(rp.temporal_width(kc[-1], sig))
        S_meas.append(np.mean(np.abs(Pt[:, m]) ** 2))
    kc, sw_meas, sw_th, S_meas, nmodes = map(np.array, (kc, sw_meas, sw_th, S_meas, nmodes))
    ok_shell = nmodes >= MIN_MODES
    rel = np.abs(sw_meas / sw_th - 1)[ok_shell]
    for kk_, nm_, rr_ in zip(kc[~ok_shell], nmodes[~ok_shell], (sw_meas / sw_th)[~ok_shell]):
        R.add(
            "V3",
            f"σ_ω-ratio shell k≈{kk_:.2f} (only {nm_} modes, info)",
            rr_,
            note=f"sampling scatter ~1/√N ≈ {1/np.sqrt(nm_):.2f}; sign changes between realisations",
        )
    R.add(
        "V3",
        f"max |σ_ω,mess/σ_ω,theo − 1| (shells ≥ {MIN_MODES} modes, Hann window)",
        rel.max(),
        V["granule_rel_err_max"],
        rel.max() < V["granule_rel_err_max"],
        note="σ_ω = σk/√2, independent of ħ",
    )
    slope = np.polyfit(kc**2, np.log(S_meas), 1)[0]
    slope_th = -(a**2) / (4 * sig**2)
    rs = abs(slope / slope_th - 1)
    R.add("V3", "slope ln S(k) vs k² relative to −a²/4σ²", rs, V["granule_rel_err_max"], rs < V["granule_rel_err_max"])
    vm = float(np.mean(var))
    R.add(
        "V3",
        "Var(δ) (exponential intensity statistics: 1)",
        vm,
        V["granule_var_rel_err_max"],
        abs(vm - 1) < V["granule_var_rel_err_max"],
    )
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    axs[0].plot(kc, sw_th, color="k", label="σk/√2 (theory)")
    axs[0].plot(kc, sw_meas, "o", color=viz.OKABE[1], label="measured")
    axs[0].set_xlabel("k [1/kpc]"), axs[0].set_ylabel("σ_ω [1/Myr]"), axs[0].legend()
    axs[0].set_title("temporal width: sweeping, independent of ħ", loc="left")
    kk = np.linspace(kc.min(), kc.max(), 200)
    axs[1].plot(kk, np.exp(np.log(S_meas[0]) + slope_th * (kk**2 - kc[0] ** 2)), color="k", label="exp(−a²k²/4σ²)")
    axs[1].plot(kc, S_meas, "o", color=viz.OKABE[1], label="measured")
    axs[1].set_yscale("log"), axs[1].set_xlabel("k [1/kpc]"), axs[1].set_ylabel("S(k) (arb. units.)"), axs[1].legend()
    axs[1].set_title("spatial spectrum", loc="left")
    i = len(kc) // 2
    m = sh == (i + 1)
    spec = Pw[:, m].sum(axis=1)
    o = np.argsort(om)
    sth = rp.temporal_width(kc[i], sig)
    gauss = np.exp(-0.5 * om[o] ** 2 / sth**2)
    axs[2].plot(om[o], spec[o] / spec.max(), color=viz.OKABE[1], label=f"measured, k≈{kc[i]:.2f}")
    axs[2].plot(om[o], gauss, color="k", ls="--", label="Gaussian, σ_ω = σk/√2")
    axs[2].set_xlim(-5 * sth, 5 * sth), axs[2].set_xlabel("ω [1/Myr]"), axs[2].set_ylabel("P(k,ω)/max"), axs[2].legend()
    axs[2].set_title("frequency profile of a k-shell", loc="left")
    fig.suptitle(
        "V3 – granule spectrum of a free random-phase field vs the analytic formula",
        x=0.01,
        ha="left",
        fontweight="bold",
    )
    fig.savefig(figs / "p7_V3_granule_spectrum.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    ks = np.linspace(0.3, 7, 120)
    ws_ = np.linspace(-0.6, 0.6, 160)
    PP = rp.granule_spectrum(ks[None, :], ws_[:, None], sig, a)
    viz.surface3d_html(
        figs3 / "p7_V3_granule_spectrum.html",
        ks,
        ws_,
        PP,
        xlabel="k [1/kpc]",
        ylabel="ω [1/Myr]",
        zlabel="P_δ",
        title="analytic granule spectrum P_δ(k,ω)",
    )
    viz.heatmap(
        figs / "p7_V3_granule_spectrum_map.png",
        ks,
        ws_,
        PP,
        xlabel="k [1/kpc]",
        ylabel="ω [1/Myr]",
        zlabel="P_δ(k,ω)",
        title="Analytic granule spectrum (sweeping wedge |ω| ~ σk/√2)",
        dpi=dpi,
        logz=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg, "P7_validation")
    figs, figs3, dpi, w = out / "figs", out / "figs3d", cfg["viz"]["dpi"], cfg["compute"]["fft_workers"]
    for name, fn in [
        ("V1", lambda: v1_filters(cfg, figs, figs3, dpi)),
        ("V2", lambda: v2_drive(cfg, figs, figs3, dpi, w)),
        ("V3", lambda: v3_granules(cfg, figs, figs3, dpi, w)),
    ]:
        t = time.perf_counter()
        fn()
        print(f"  {name} done ({time.perf_counter()-t:.0f} s)", flush=True)
    R.write(
        out,
        "P7_validation",
        "Validation of filters (A) and response theory (B)",
        ["Thresholds pre-declared in `validation_p7`."],
    )
    write_manifest(cfg, out, "P7_validation", extra=R.results)
    print(f"P7-Validation done -> {out} ({'all passed' if R.n_failed() == 0 else f'{R.n_failed()} FAILED'})")
    if R.n_failed():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
