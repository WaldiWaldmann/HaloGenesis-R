from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hgr import viz  # noqa: E402
from hgr.config import PACKAGE_ROOT, load_config, output_dir, write_manifest  # noqa: E402
from hgr.report import Recorder  # noqa: E402

R = Recorder()


def load_runs(label):
    runs = []
    for f in sorted((PACKAGE_ROOT / "runs" / label).glob("P5_*/P5_timeseries.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "meta" not in d:
            continue
        vals = json.loads((f.parent / f"{f.parent.name}_values.json").read_text(encoding="utf-8"))
        d["values"], d["dir"] = vals, f.parent
        runs.append(d)
    return runs


def series(run, key_fn):
    return np.array(run["t"]), np.asarray(key_fn(run), dtype=float)


def at(run, key_fn, t):
    ts, y = series(run, key_fn)
    return float(np.interp(t, ts, y)) if t <= ts[-1] + 1e-9 else np.nan


def jmax_until(run, t):
    ts = np.array(run["t"])
    m = ts <= t + 1e-9
    return float(max(np.max(np.array(run["truelove"]["A1"])[m]), np.max(np.array(run["truelove"]["A0"])[m])))


def flux_res_M0(run, Rp, A):
    ts = np.array(run["t"])
    m = np.array(run.get("MF", run["M"])[f"A{A}_R{Rp}"])
    f = np.array(run["F"][f"A{A}_R{Rp}"])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(ts))])
    return float(np.max(np.abs(m - m[0] - integ)) / m[0])


def robust(values):
    v = [x for x in values if np.isfinite(x)]
    if len(v) < 2:
        return None, np.nan
    err = abs(v[-1] - v[-2])
    return (len({np.sign(x) for x in v}) == 1 and abs(v[-1]) > err), err


def p_obs(values, Ns):
    if len(values) < 3:
        return np.nan
    f1, f2, f3 = values[-3:]
    r = Ns[-1] / Ns[-2]
    if not np.isclose(Ns[-2] / Ns[-3], r) or f2 == f3 or f1 == f2:
        return np.nan
    return float(np.log(abs((f1 - f2) / (f2 - f3))) / np.log(r))


def variant(r, cfg):
    m = r["meta"]
    return (m.get("setup", "isolated"), float(m.get("jeans_NJ", 0.0)), m.get("cooling", "piecewise_historical"))


def analyze(cfg, runs, label, P, out, figs, figs3, dpi, tag):
    global R
    R = Recorder()
    all_runs = runs
    fk0 = cfg["filter"]["kernel"]
    runs = [r for r in runs if r["meta"]["filter_kind"] == fk0 and np.isclose(r["meta"]["T_K"], cfg["gas"]["T_K"])]
    if not runs:
        print(f"  [{tag}] no runs with default filter/temperature – decomposition only")
    tau0, n0 = cfg["filter"]["tau_myr"], cfg["gas"]["n_cm3"]
    Rp = str(cfg["analysis"]["primary_aperture_kpc"])
    k = lambda r: (r["meta"]["N"], r["meta"]["tau"], r["meta"]["n_cm3"])  # noqa: E731

    O1 = lambda r: r["O1"][Rp]  # noqa: E731
    DQ = lambda r: np.log(np.array(r["Q"][f"A1_R{Rp}"]) / np.array(r["Q"][f"A0_R{Rp}"]))  # noqa: E731
    O3 = lambda r: [np.nan if x is None else x for x in r["O3"][Rp]]  # noqa: E731

    seed0 = cfg["meta"]["seed"]
    dt0 = cfg["p5"]["dt_max_myr"]
    base = sorted(
        [
            r
            for r in runs
            if np.isclose(r["meta"]["tau"], tau0)
            and np.isclose(r["meta"]["n_cm3"], n0)
            and r["meta"]["seed"] == seed0
            and np.isclose(r["meta"]["dt_max"], dt0)
            and r["meta"]["halo_seed"] == cfg["setup"]["halo_seed"]
        ],
        key=k,
    )
    e_time = {}
    fine_dt = [
        r
        for r in runs
        if np.isclose(r["meta"]["tau"], tau0)
        and np.isclose(r["meta"]["n_cm3"], n0)
        and r["meta"]["seed"] == seed0
        and r["meta"]["dt_max"] < dt0 - 1e-12
    ]
    if fine_dt:
        rf = max(fine_dt, key=lambda r: r["meta"]["N"])
        rc = [r for r in base if r["meta"]["N"] == rf["meta"]["N"]]
        if rc:
            for nm, fn in (
                ("O1", lambda r: r["O1"][Rp]),
                ("ΔlnQ", lambda r: np.log(np.array(r["Q"][f"A1_R{Rp}"]) / np.array(r["Q"][f"A0_R{Rp}"]))),
            ):
                for t in P["eval_times_myr"]:
                    a_, b_ = at(rc[0], fn, t), at(rf, fn, t)
                    if np.isfinite(a_) and np.isfinite(b_):
                        e_time[(nm, t)] = abs(a_ - b_)
                        R.add(
                            f"Time refinement (N={rf['meta']['N']}, dt_max {dt0:g} → {rf['meta']['dt_max']:g} Myr)",
                            f"{nm} at t={t:g}: e_Zeit",
                            e_time[(nm, t)],
                            note=f"values {a_:.4g} / {b_:.4g}",
                        )
    sigma = {}
    cand = [r for r in runs if np.isclose(r["meta"]["tau"], tau0) and np.isclose(r["meta"]["n_cm3"], n0)]
    if cand:
        Nseed = max(
            {r["meta"]["N"] for r in cand},
            key=lambda N: (len({r["meta"]["seed"] for r in cand if r["meta"]["N"] == N}), N),
        )
        sruns = [r for r in cand if r["meta"]["N"] == Nseed]
        if len({r["meta"]["seed"] for r in sruns}) >= 2:
            for nm, fn in (
                ("O1", lambda r: r["O1"][Rp]),
                ("ΔlnQ", lambda r: np.log(np.array(r["Q"][f"A1_R{Rp}"]) / np.array(r["Q"][f"A0_R{Rp}"]))),
            ):
                for t in P["eval_times_myr"]:
                    v = np.array([at(r, fn, t) for r in sruns])
                    if np.isfinite(v).sum() >= 2:
                        sigma[(nm, t)] = float(np.nanstd(v, ddof=1))
                        R.add(
                            f"Seeds (N={Nseed}, {len(sruns)} realisations)",
                            f"σ_Seeds {nm} at t={t:g}",
                            sigma[(nm, t)],
                            note="E_sample (Eq. 34); values: " + ", ".join(f"{x:.3g}" for x in v),
                        )
    Ns = [r["meta"]["N"] for r in base]
    R.add("Runs", "P5 runs found", len(runs), note="; ".join(f"N{a}/τ{b:g}/n{c:g}" for a, b, c in map(k, runs)))
    R.add("Runs", "resolution series (τ, n = base)", len(base), note="N = " + ", ".join(map(str, Ns)))

    T = [t for t in P["eval_times_myr"]]
    err_space = {}
    matrix, rows = {}, [
        "O1 (concentration)",
        "ΔlnQ (Jeans, continuous)",
        "O2 original",
        "O3 (coherence)",
        "Numerics: Truelove",
        "Numerics: mass balance",
    ]
    o1_tab = np.full((len(base), len(T)), np.nan)
    dq_tab = np.full_like(o1_tab, np.nan)
    for j, t in enumerate(T):
        v1 = [at(r, O1, t) for r in base]
        v2 = [at(r, DQ, t) for r in base]
        o1_tab[:, j], dq_tab[:, j] = v1, v2
        fin = base[-1] if base else None
        floored = fin is not None and fin["meta"]["jeans_NJ"] > 0
        if floored:
            tsf = np.array(fin["t"]) <= t + 1e-9
            ffrac = float(np.max(np.array(fin["floor_frac"]["A1"])[tsf])) if tsf.any() else np.nan
            resolved = t <= fin["meta"]["t_final"] + 1e-9 and ffrac <= P["floor_mass_fraction_max"]
            R.add(
                f"Resolution t={t:g} Myr",
                "max Jeans-floor mass fraction (A=1)",
                ffrac,
                P["floor_mass_fraction_max"],
                ffrac <= P["floor_mass_fraction_max"],
            )
        else:
            resolved = (
                fin is not None and t <= fin["meta"]["t_final"] + 1e-9 and jmax_until(fin, t) <= P["truelove_max"]
            )
        for name, vals, sk in (("O1 (concentration)", v1, "O1"), ("ΔlnQ (Jeans, continuous)", v2, "ΔlnQ")):
            ok, err = robust(vals)
            err_space[(sk, t)] = err
            if ok and (sk, t) in e_time:
                ok = abs(vals[-1]) > err + e_time[(sk, t)]
            if ok and (sk, t) in sigma:
                ok = abs(vals[-1]) > P["seed_sigma_factor"] * sigma[(sk, t)]
            po = p_obs(vals, Ns)
            R.add(
                f"Resolution t={t:g} Myr",
                f"{name}: values N={Ns}",
                vals[-1] if vals else np.nan,
                note=", ".join(f"{x:.3g}" for x in vals) + f"; error estimate {err:.2g}; p_obs {po:.2f}",
            )
            matrix[(name, t)] = "na" if ok is None else ("ok" if ok and resolved else ("unres" if ok else "fail"))
        matrix[("Numerics: Truelove", t)] = (
            "na" if fin is None or t > fin["meta"]["t_final"] + 1e-9 else ("ok" if resolved else "fail")
        )
        if fin is not None and not floored:
            R.add(
                f"Resolution t={t:g} Myr",
                f"max J up to t (N={fin['meta']['N']})",
                jmax_until(fin, t),
                P["truelove_max"],
                jmax_until(fin, t) <= P["truelove_max"],
            )
    for r in base:
        N = r["meta"]["N"]
        v = r["values"]
        o2_0 = v["O2"]["O2(A=0) time fraction min_R Q < 1"]["value"]
        R.add(
            "O2 original",
            f"N={N}: O2(A=0) / ΔO2",
            o2_0,
            P["o2_saturation"],
            None,
            note=f"ΔO2 = {v['O2']['ΔO2 = O2(1) − O2(0)']['value']:.3g}",
        )
        o3e = v["O3"][
            f"O3(R={cfg['analysis']['primary_aperture_kpc']}) mean for t ≤ {cfg['p5']['early_window_myr']} Myr"
        ]["value"]
        mb = v["O3"]["early momentum budget: max ‖δv − ∫δg dt‖/‖δv‖"]["value"]
        R.add("O3", f"N={N}: early coherence", o3e, P["o3_early_min"], o3e >= P["o3_early_min"])
        R.add(
            "O3",
            f"N={N}: early momentum budget",
            mb,
            P["momentum_budget_early_max"],
            mb <= P["momentum_budget_early_max"],
        )
        for A in (1, 0):
            fr = flux_res_M0(r, Rp, A)
            R.add(
                "Mass balance",
                f"N={N}, A={A}: max|ΔM−∫F|/M(0)",
                fr,
                P["flux_residual_M0_max"],
                fr <= P["flux_residual_M0_max"],
            )
    if base:
        last = base[-1]["values"]
        sat = last["O2"]["O2(A=0) time fraction min_R Q < 1"]["value"] >= P["o2_saturation"]
        o3ok = all(
            R.results["O3"][f"N={r['meta']['N']}: early coherence"]["passed"]
            and R.results["O3"][f"N={r['meta']['N']}: early momentum budget"]["passed"]
            for r in base
        )
        flok = all(
            R.results["Mass balance"][f"N={base[-1]['meta']['N']}, A={A}: max|ΔM−∫F|/M(0)"]["passed"] for A in (1, 0)
        )
        for t in T:
            if t > base[-1]["meta"]["t_final"] + 1e-9:
                continue
            matrix[("O2 original", t)] = "sat" if sat else "ok"
            matrix[("O3 (coherence)", t)] = "ok" if o3ok else "fail"
            matrix[("Numerics: mass balance", t)] = "ok" if flok else "fail"

    code = {"ok": 3, "unres": 2, "sat": 1, "fail": 0, "na": np.nan}
    lab = {"ok": "robust", "unres": "robust,\nunresolved", "sat": "saturated", "fail": "rejected", "na": "—"}
    Mx = np.array([[code[matrix.get((r_, t), "na")] for t in T] for r_ in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cmap = matplotlib.colors.ListedColormap(["#d62728", "#bbbbbb", "#ff9f1c", "#2ca02c"])
    ax.imshow(Mx, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="auto", origin="upper")
    for i, r_ in enumerate(rows):
        for j, t in enumerate(T):
            ax.text(j, i, lab[matrix.get((r_, t), "na")], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(T)), [f"t = {t:g} Myr" for t in T])
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(f"Falsification matrix ({label} / {tag}, N up to {Ns[-1] if Ns else '—'})")
    ax.grid(False)
    fig.savefig(figs / "p6_falsification_matrix.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if len(base) >= 2:
        for nm, tab, fn in (("O1", o1_tab, O1), ("dlnQ", dq_tab, DQ)):
            viz.line_plot(
                figs / f"p6_conv_{nm}.png",
                np.array(Ns, float),
                {f"t = {t:g} Myr": tab[:, j] for j, t in enumerate(T)},
                xlabel="N",
                ylabel=nm,
                title=f"Resolution study {nm}(R={Rp} kpc)",
                dpi=dpi,
                logx=True,
                markers=True,
                hlines={"0": 0.0},
            )
            viz.surface3d_png(
                figs3 / f"p6_conv_{nm}_surface.png",
                np.array(Ns, float),
                np.array(T),
                np.nan_to_num(tab.T),
                xlabel="N",
                ylabel="t [Myr]",
                zlabel=nm,
                title=f"{nm}(N, t)",
                dpi=dpi,
                log_axes=(True, False, False),
            )
            viz.surface3d_html(
                figs3 / f"p6_conv_{nm}_surface.html",
                np.array(Ns, float),
                np.array(T),
                np.nan_to_num(tab.T),
                xlabel="N",
                ylabel="t [Myr]",
                zlabel=nm,
                title=f"{nm}(N, t)",
                log_axes=(True, False, False),
            )
            curves = {
                f"N = {r['meta']['N']}": np.interp(np.linspace(0, r["meta"]["t_final"], 200), r["t"], fn(r))
                for r in base
            }
            tt = np.linspace(0, min(r["meta"]["t_final"] for r in base), 200)
            viz.line_plot(
                figs / f"p6_conv_{nm}_time.png",
                tt,
                {
                    kk: np.interp(tt, np.linspace(0, r["meta"]["t_final"], 200), v)
                    for (kk, v), r in zip(curves.items(), base)
                },
                xlabel="t [Myr]",
                ylabel=nm,
                title=f"{nm}(t) for all resolutions",
                dpi=dpi,
            )
        viz.line_plot(
            figs / "p6_truelove.png",
            np.array(base[-1]["t"]),
            {
                f"N = {r['meta']['N']}": np.interp(
                    base[-1]["t"], r["t"], np.maximum(r["truelove"]["A1"], r["truelove"]["A0"])
                )
                for r in base
            },
            xlabel="t [Myr]",
            ylabel="J = dx/λ_J",
            title="Truelove number per resolution",
            dpi=dpi,
            logy=True,
            hlines={"1/4": P["truelove_max"]},
        )

    for var, other, ovalue, title in (("tau", "n_cm3", n0, "τ [Myr]"), ("n_cm3", "tau", tau0, "n [cm⁻³]")):
        sel = [
            r
            for r in runs
            if np.isclose(r["meta"][other], ovalue)
            and r["meta"]["seed"] == cfg["meta"]["seed"]
            and np.isclose(r["meta"]["dt_max"], cfg["p5"]["dt_max_myr"])
        ]
        if not sel:
            continue
        Nsc = max(
            {r["meta"]["N"] for r in sel}, key=lambda N: (len({r["meta"][var] for r in sel if r["meta"]["N"] == N}), N)
        )
        sel = sorted([r for r in sel if r["meta"]["N"] == Nsc], key=lambda r: r["meta"][var])
        if len(sel) < 2:
            continue
        xs = np.array([r["meta"][var] for r in sel], float)
        for nm, fn in (("O1", O1), ("dlnQ", DQ), ("O3", O3)):
            Z = np.array([[at(r, fn, t) for t in T] for r in sel])
            for i, r in enumerate(sel):
                R.add(
                    f"Scan {var} (N={Nsc})",
                    f"{nm} at {var}={r['meta'][var]:g}",
                    (
                        Z[i, -1]
                        if np.isfinite(Z[i, -1])
                        else Z[i][np.isfinite(Z[i])][-1] if np.isfinite(Z[i]).any() else np.nan
                    ),
                    note=", ".join(f"t{t:g}:{z:.3g}" for t, z in zip(T, Z[i])),
                )
            xp = xs + (0.01 if var == "tau" else 0.0)
            viz.line_plot(
                figs / f"p6_scan_{var}_{nm}.png",
                xp,
                {f"t = {t:g} Myr": Z[:, j] for j, t in enumerate(T)},
                xlabel=title + (" (+0.01)" if var == "tau" else ""),
                ylabel=nm,
                title=f"{nm} vs. {title}, N={Nsc}",
                dpi=dpi,
                logx=True,
                markers=True,
            )
            viz.surface3d_png(
                figs3 / f"p6_scan_{var}_{nm}_surface.png",
                xp,
                np.array(T),
                np.nan_to_num(Z.T),
                xlabel=title,
                ylabel="t [Myr]",
                zlabel=nm,
                title=f"{nm}({var}, t)",
                dpi=dpi,
                log_axes=(True, False, False),
            )
            viz.surface3d_html(
                figs3 / f"p6_scan_{var}_{nm}_surface.html",
                xp,
                np.array(T),
                np.nan_to_num(Z.T),
                xlabel=title,
                ylabel="t [Myr]",
                zlabel=nm,
                title=f"{nm}({var}, t)",
                log_axes=(True, False, False),
            )

    hs0 = cfg["setup"]["halo_seed"]
    allc = [r for r in runs if np.isclose(r["meta"]["n_cm3"], n0) and r["meta"]["seed"] == seed0]
    cand = [r for r in allc if np.isclose(r["meta"]["dt_max"], dt0) and r["meta"]["halo_seed"] == hs0]

    def delta_at(rlist, tau, tau_ref, fn, t):
        a_ = [r for r in rlist if np.isclose(r["meta"]["tau"], tau)]
        b_ = [r for r in rlist if np.isclose(r["meta"]["tau"], tau_ref)]
        return at(a_[0], fn, t) - at(b_[0], fn, t) if a_ and b_ else np.nan

    def diff_error(nm, fn, tau, tau_ref, t, Nh):
        comp = {}
        Ns_tau = sorted(
            {r["meta"]["N"] for r in cand if np.isclose(r["meta"]["tau"], tau)}
            & {r["meta"]["N"] for r in cand if np.isclose(r["meta"]["tau"], tau_ref)}
        )
        dN = [delta_at([r for r in cand if r["meta"]["N"] == N], tau, tau_ref, fn, t) for N in Ns_tau]
        dN = [d for d in dN if np.isfinite(d)]
        sign_ok = len({np.sign(d) for d in dN}) <= 1
        if len(dN) >= 2:
            comp["resolution"] = abs(dN[-1] - dN[-2])
        seeds = [
            delta_at(
                [
                    r
                    for r in allc
                    if r["meta"]["N"] == Nh and r["meta"]["halo_seed"] == hs and np.isclose(r["meta"]["dt_max"], dt0)
                ],
                tau,
                tau_ref,
                fn,
                t,
            )
            for hs in sorted({r["meta"]["halo_seed"] for r in allc})
        ]
        seeds = [d for d in seeds if np.isfinite(d)]
        if len(seeds) >= 2:
            comp["2σ realisation"] = 2 * float(np.std(seeds, ddof=1))
        fine = [
            r
            for r in allc
            if r["meta"]["N"] == Nh and r["meta"]["halo_seed"] == hs0 and r["meta"]["dt_max"] < dt0 - 1e-12
        ]
        dfine = delta_at(fine, tau, tau_ref, fn, t)
        dcoarse = delta_at([r for r in cand if r["meta"]["N"] == Nh], tau, tau_ref, fn, t)
        if np.isfinite(dfine) and np.isfinite(dcoarse):
            comp["time step"] = abs(dcoarse - dfine)
        e = float(sum(comp.values())) if "resolution" in comp else np.nan
        return e, comp, sign_ok

    if cand:
        Nh = max(
            {r["meta"]["N"] for r in cand},
            key=lambda N: (len({r["meta"]["tau"] for r in cand if r["meta"]["N"] == N}), N),
        )
        hr = sorted([r for r in cand if r["meta"]["N"] == Nh], key=lambda r: r["meta"]["tau"])
        if len(hr) >= 2:
            taus = [r["meta"]["tau"] for r in hr]
            iref = 0
            fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), gridspec_kw={"width_ratios": [1.35, 1]})
            tmax = min(r["meta"]["t_final"] for r in hr)
            Tv = [t for t in T if t <= tmax + 1e-9]
            summary = []
            for row, (nm, sk, fn) in enumerate((("O1", "O1", O1), ("ΔlnQ", "ΔlnQ", DQ))):
                etot_t = {}
                ediff = {}
                for t in Tv:
                    for r in hr:
                        if np.isclose(r["meta"]["tau"], taus[iref]):
                            continue
                        ediff[(r["meta"]["tau"], t)] = diff_error(nm, fn, r["meta"]["tau"], taus[iref], t, Nh)
                    vals_ = [ediff[(r["meta"]["tau"], t)][0] for r in hr if (r["meta"]["tau"], t) in ediff]
                    vals_ = [v for v in vals_ if np.isfinite(v)]
                    etot_t[t] = max(vals_) if vals_ else np.nan
                ax = axes[row, 0]
                tr = np.array(hr[iref]["t"])
                yr = np.asarray(fn(hr[iref]), float)
                for i, r in enumerate(hr):
                    if i == iref:
                        continue
                    yi = np.interp(tr, r["t"], fn(r))
                    ax.plot(tr, yi - yr, color=viz.OKABE[i % 8], lw=1.8, label=f"τ = {r['meta']['tau']:g} Myr")
                tb = [t for t in Tv if np.isfinite(etot_t[t])]
                if tb:
                    eb = np.interp(tr, [0.0] + tb, [0.0] + [etot_t[t] for t in tb])
                    ax.fill_between(
                        tr, -eb, eb, color="0.75", alpha=0.45, lw=0, label="error of the difference (rule v0.9.2)"
                    )
                else:
                    ax.text(
                        0.98,
                        0.04,
                        "error band unknown – resolution/dt/seed runs missing",
                        transform=ax.transAxes,
                        ha="right",
                        fontsize=8,
                        color="0.4",
                    )
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xlabel("t [Myr]"), ax.set_ylabel(f"Δ{nm} = {nm}(τ) − {nm}(τ={taus[iref]:g})")
                ax.set_title(f"Effect of the temporal filter on {nm}(R={Rp} kpc), N={Nh}", loc="left")
                ax.legend(fontsize=8, loc="upper left")
                ins = ax.inset_axes([0.62, 0.08, 0.35, 0.33])
                for i, r in enumerate(hr):
                    ins.plot(r["t"], fn(r), color=viz.OKABE[i % 8], lw=1)
                ins.set_title(f"{nm} absolut", fontsize=7.5), ins.tick_params(labelsize=6.5), ins.grid(alpha=0.2)
                ax = axes[row, 1]
                for j, t in enumerate(Tv):
                    ref = at(hr[iref], fn, t)
                    xs, ys = [], []
                    for i, r in enumerate(hr):
                        if i == iref:
                            continue
                        d = at(r, fn, t) - ref
                        xs.append(i + (j - (len(Tv) - 1) / 2) * 0.08)
                        ys.append(d)
                        e_d, comp, sign_ok = ediff[(r["meta"]["tau"], t)]
                        state = (
                            "error unknown (no resolution series)"
                            if not np.isfinite(e_d)
                            else ("nachweisbar" if (abs(d) > e_d and sign_ok) else "not detectable")
                        )
                        summary.append((nm, r["meta"]["tau"], t, state))
                        R.add(
                            f"Hypothesis: effect of τ (N={Nh}, Ref τ={taus[iref]:g})",
                            f"Δ{nm} τ={r['meta']['tau']:g}, t={t:g}",
                            d,
                            e_d if np.isfinite(e_d) else None,
                            None,
                            note=state
                            + "; "
                            + ", ".join(f"{k}={v:.2g}" for k, v in comp.items())
                            + ("" if sign_ok else "; sign change across N"),
                        )
                    yerr = [ediff[(r["meta"]["tau"], t)][0] for i_, r in enumerate(hr) if i_ != iref]
                    yerr = None if not np.all(np.isfinite(yerr)) else yerr
                    ax.errorbar(
                        xs, ys, yerr=yerr, fmt="o", ms=5, capsize=3, color=viz.OKABE[j % 8], label=f"t = {t:g} Myr"
                    )
                ax.axhline(0, color="k", lw=0.8)
                ax.set_xticks(
                    range(len(taus)), [f"τ={x:g}" + (" (Ref.)" if i == iref else "") for i, x in enumerate(taus)]
                )
                ax.set_ylabel(f"Δ{nm}")
                ax.set_title(f"Δ{nm} at evaluation times (error of the difference)", loc="left"), ax.legend(
                    fontsize=7.5
                )
            pos = sorted({f"τ={tau_:g} @ {t:g} Myr ({nm})" for nm, tau_, t, st in summary if st == "nachweisbar"})
            unk = any(st.startswith("error unknown") for *_, st in summary)
            verdict = (
                ("τ effect DETECTABLE at: " + "; ".join(pos))
                if pos
                else ("No τ effect detectable above the errors" + (" (errors partly unknown)" if unk else ""))
            )
            fig.text(
                0.01,
                -0.035,
                verdict,
                fontsize=10.5,
                fontweight="bold",
                bbox=dict(fc="#fff7d6" if pos else "#eeeeee", ec="0.6", pad=5),
            )
            fig.suptitle(
                f"Hypothesis test: time averaging – {label} / {tag}", x=0.01, ha="left", fontsize=12, fontweight="bold"
            )
            fig.savefig(figs / "p6_HYPOTHESIS_tau.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    dec_runs = [
        r
        for r in all_runs
        if np.isclose(r["meta"]["n_cm3"], n0)
        and r["meta"]["seed"] == seed0
        and np.isclose(r["meta"]["dt_max"], dt0)
        and r["meta"]["halo_seed"] == cfg["setup"]["halo_seed"]
        and np.isclose(r["meta"]["T_K"], cfg["gas"]["T_K"])
    ]
    by_N = {}
    for r in dec_runs:
        by_N.setdefault(r["meta"]["N"], []).append(r)
    dec_done = False
    for Nd in sorted(by_N, reverse=True):
        rl = by_N[Nd]
        ref = [r for r in rl if r["meta"]["tau"] == 0]
        taus_d = sorted(
            {
                r["meta"]["tau"]
                for r in rl
                if r["meta"]["tau"] > 0
                and {"trailing_boxcar", "zero_lag_boxcar", "delay_half_tau"}
                <= {q["meta"]["filter_kind"] for q in rl if np.isclose(q["meta"]["tau"], r["meta"]["tau"])}
            }
        )
        if not ref or not taus_d or dec_done:
            continue
        ref = ref[0]
        fig, axs = plt.subplots(len(taus_d), 2, figsize=(13.5, 3.9 * len(taus_d)), squeeze=False)
        for i, tau_ in enumerate(taus_d):
            pick = {q["meta"]["filter_kind"]: q for q in rl if np.isclose(q["meta"]["tau"], tau_)}
            tr = np.array(ref["t"])
            for col, (nm, fn) in enumerate((("O1", O1), ("ΔlnQ", DQ))):
                yr = np.asarray(fn(ref), float)
                d = {
                    k_: np.interp(tr, pick[k_]["t"], fn(pick[k_])) - yr
                    for k_ in ("trailing_boxcar", "zero_lag_boxcar", "delay_half_tau")
                }
                closure = d["trailing_boxcar"] - d["zero_lag_boxcar"] - d["delay_half_tau"]
                ax = axs[i, col]
                ax.plot(tr, d["trailing_boxcar"], color="k", lw=2, label="total (causal boxcar)")
                ax.plot(tr, d["delay_half_tau"], color=viz.OKABE[1], lw=1.6, label="delay only (τ/2)")
                ax.plot(tr, d["zero_lag_boxcar"], color=viz.OKABE[0], lw=1.6, label="smoothing only (zero-lag)")
                ax.plot(tr, closure, color="0.5", ls=":", lw=1.4, label="residual = nonlinearity")
                ax.axhline(0, color="k", lw=0.6)
                ax.set_xlabel("t [Myr]"), ax.set_ylabel(f"Δ{nm} vs τ = 0")
                ax.set_title(f"τ = {tau_:g} Myr, N = {Nd}: decomposition of Δ{nm}", loc="left"), ax.legend(fontsize=7.5)
                for t in T:
                    if t <= tr[-1] + 1e-9:
                        tot = float(np.interp(t, tr, d["trailing_boxcar"]))
                        R.add(
                            f"Decomposition τ={tau_:g} (N={Nd})",
                            f"Δ{nm} t={t:g}: total / delay / smoothing / residual",
                            tot,
                            note=f"{np.interp(t, tr, d['delay_half_tau']):+.4g} / {np.interp(t, tr, d['zero_lag_boxcar']):+.4g}"
                            f" / {np.interp(t, tr, closure):+.4g}; delay share "
                            f"{(np.interp(t, tr, d['delay_half_tau']) / tot if tot else np.nan):.0%}",
                        )
        fig.suptitle(
            f"Idea A – does averaging act through delay or through smoothing of the granules? ({label} / {tag})",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
        )
        fig.savefig(figs / "p6_DECOMPOSITION.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        dec_done = True

    if base:
        for kk, vv in base[-1]["values"]["Comparison"].items():
            R.add("Vergleich (finest base run)", kk, vv["value"], note=vv["note"])

    intro = [
        "Rules (pre-declared, config `p6`): O1/ΔlnQ robust ⇔ sign stable across all N AND |value(N_max)| > "
        "|value(N_max) − value(N_prev)|; a time only counts as resolved if J ≤ 1/4 up to then in the "
        "finest run. Original O2 is saturated if O2(A=0) ≥ saturation threshold. Matrix: "
        "`figs/p6_falsification_matrix.png`.",
        "",
        "| Zeile | " + " | ".join(f"t={t:g}" for t in T) + " |",
        "|---|" + "---|" * len(T),
    ]
    intro += [
        f"| {r_} | " + " | ".join(lab[matrix.get((r_, t), "na")].replace("\n", " ") for t in T) + " |" for r_ in rows
    ]
    R.write(out, "P6", "Resolution, ablation, sensitivity, falsification matrix", intro)
    write_manifest(cfg, out, "P6", extra={"matrix": {f"{a}@{b}": c for (a, b), c in matrix.items()}, **R.results})
    print(f"P6 [{tag}] done -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    label, P = cfg["meta"]["run_label"], cfg["p6"]
    runs = load_runs(label)
    if not runs:
        raise SystemExit(f"No P5 runs in P6 format under runs/{label}/ gefunden.")
    for r in runs:
        r["meta"].setdefault("seed", cfg["meta"]["seed"])
        r["meta"].setdefault("jeans_NJ", 0.0)
        r["meta"].setdefault("setup", "isolated")
        r["meta"].setdefault("cooling", cfg["gas"]["cooling_model"])
        r["meta"].setdefault("dt_max", cfg["p5"]["dt_max_myr"])
        r["meta"].setdefault("halo_seed", cfg["setup"]["halo_seed"])
        r["meta"].setdefault("filter_kind", "trailing_boxcar")
        r["meta"].setdefault("T_K", cfg["gas"]["T_K"])
    groups = {}
    for r in runs:
        groups.setdefault(variant(r, cfg), []).append(r)
    for (setup, nj, cool), grp in sorted(groups.items()):
        tag = f"{setup}_jf{nj:g}_{cool}"
        out = output_dir(cfg, f"P6/{tag}")
        analyze(cfg, grp, label, P, out, out / "figs", out / "figs3d", cfg["viz"]["dpi"], tag)


if __name__ == "__main__":
    main()
