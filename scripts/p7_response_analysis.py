from __future__ import annotations

import argparse
import json
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from hgr import response as rp  # noqa: E402
from hgr import viz  # noqa: E402
from hgr.config import PACKAGE_ROOT, load_config, output_dir, write_manifest  # noqa: E402
from hgr.report import Recorder  # noqa: E402
from hgr.units import KMS_TO_CODE, sound_speed  # noqa: E402

R = Recorder()


def centered_mean(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    c = np.cumsum(np.vstack([np.zeros((1,) + x.shape[1:]), x]), axis=0)
    i = np.arange(x.shape[0])
    lo = np.maximum(i - n // 2, 0)
    hi = np.minimum(i + n // 2 + 1, x.shape[0])
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def probe_stats(run: dict, RS: dict) -> dict | None:
    pr = run.get("probes")
    if not pr or not pr.get("rho_a"):
        return None
    t = np.array(run["t"])
    dt = RS["resample_dt_myr"]
    tu = np.arange(RS["t_skip_myr"], t[-1] + 1e-9, dt)
    if len(tu) < 16:
        return None

    def prep(key):
        X = np.array(pr[key], float)
        Xu = np.array([np.interp(tu, t, X[:, j]) for j in range(X.shape[1])]).T
        d = Xu / Xu.mean(axis=0) - 1
        return d - centered_mean(d, int(round(RS["highpass_myr"] / dt)))

    da, dg, d0 = prep("rho_a"), prep("rho_b1"), prep("rho_b0")
    lags = np.arange(0, min(20.0, 0.5 * (tu[-1] - tu[0])) + 1e-9, dt)
    nl = len(lags)
    r_l = []
    for li in range(nl):
        L_ = int(round(lags[li] / dt))
        a_ = da[: len(tu) - L_] if L_ else da
        g_ = dg[L_:]
        num = np.sum((a_ - a_.mean(0)) * (g_ - g_.mean(0)), axis=0)
        den = np.sqrt(np.sum((a_ - a_.mean(0)) ** 2, axis=0) * np.sum((g_ - g_.mean(0)) ** 2, axis=0))
        r_l.append(num / np.where(den > 0, den, np.nan))
    r_l = np.array(r_l)
    rm = np.nanmean(r_l, axis=1)
    j = int(np.nanargmax(np.abs(rm)))
    eta = np.var(dg, axis=0) / np.var(da, axis=0)
    eta0 = np.var(d0, axis=0) / np.var(da, axis=0)
    npb = r_l.shape[1]
    return {
        "r0": float(rm[0]),
        "r0_err": float(np.nanstd(r_l[0]) / np.sqrt(npb)),
        "r_lag": rm,
        "lags": lags,
        "r_max": float(rm[j]),
        "lag_at_max": float(lags[j]),
        "eta": float(np.median(eta)),
        "eta_noise": float(np.median(eta0)),
        "rho_dm": float(np.mean(pr["rho_a"])),
        "rho_g": float(np.mean(pr["rho_b1"])),
        "n_probes": npb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    RS = cfg["response"]
    label = cfg["meta"]["run_label"]
    out = output_dir(cfg, "P7_response")
    figs, figs3, dpi = out / "figs", out / "figs3d", cfg["viz"]["dpi"]
    runs = []
    for f in sorted((PACKAGE_ROOT / "runs" / label).glob("P5_*/P5_timeseries.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        if "meta" in d and "probes" in d and "T_K" in d["meta"] and d["meta"].get("setup") == "soliton_halo":
            d["dir"] = f.parent.name
            runs.append(d)
    if not runs:
        raise SystemExit("No halo runs with probes found.")
    groups: dict = {}
    for r in runs:
        m = r["meta"]
        key = (
            m["N"],
            m["tau"],
            m.get("filter_kind", "trailing_boxcar"),
            m["cooling"],
            m["n_cm3"],
            m["halo_seed"],
            m["dt_max"],
            m["seed"],
        )
        groups.setdefault(key, []).append(r)
    theory_cache = {}
    summary = []
    base_setup = None
    for key, grp in sorted(groups.items()):
        grp = sorted(grp, key=lambda r: r["meta"]["T_K"])
        tag = f"N{key[0]}_tau{key[1]:g}_{key[2]}_{key[3]}"
        rows = []
        for r in grp:
            ps = probe_stats(r, RS)
            if ps is None:
                continue
            m = r["meta"]
            cs = sound_speed(m["T_K"], m["mu"], m["gamma"])
            sig = m["halo_sigma_kms"] * KMS_TO_CODE
            st = rp.setup_from_run_meta(
                m, ps["rho_dm"], ps["rho_g"], cfg["domain"]["box_kpc"], RS["damping_per_myr"], RS["highpass_myr"]
            )
            th = rp.point_statistics(st, lags=ps["lags"])
            rows.append({"T": m["T_K"], "M": cs / sig, "sim": ps, "th": th, "st": st, "dir": r["dir"]})
            measurable = ps["eta"] > 3 * ps["eta_noise"]
            R.add(
                f"Mach-Scan {tag}",
                f"T={m['T_K']:g} K (M={cs/sig:.3f}): r0 sim / theo",
                ps["r0"],
                note=f"theo {th['r0']:+.3f}; r_max sim {ps['r_max']:+.3f} @ {ps['lag_at_max']:g} Myr, theo "
                f"{th['r_max']:+.3f} @ {th['lag_at_max']:g} Myr; η sim {ps['eta']:.2e} (noise "
                f"{ps['eta_noise']:.1e}), theo {th['eta']:.2e}" + ("" if measurable else "; NOT MEASURABLE"),
            )
        if len(rows) < 2:
            continue
        Ms = np.array([x["M"] for x in rows])
        r0s = np.array([x["sim"]["r0"] for x in rows])
        r0t = np.array([x["th"]["r0"] for x in rows])
        meas = np.array([x["sim"]["eta"] > 3 * x["sim"]["eta_noise"] for x in rows])
        st0 = rows[len(rows) // 2]["st"]
        Mg = np.geomspace(min(0.02, Ms.min() / 1.5), max(5.0, Ms.max() * 1.5), 28)
        ck = (tag, round(st0.rho_dm), round(st0.rho_g))
        if ck not in theory_cache:
            theory_cache[ck] = rp.mach_scan(st0, Mg)
        TC = theory_cache[ck]

        def sign_change(M, r):
            idx = np.where(np.diff(np.sign(r)) != 0)[0]
            if not len(idx):
                return np.nan
            i = idx[0]
            x0, x1, y0, y1 = np.log(M[i]), np.log(M[i + 1]), r[i], r[i + 1]
            return float(np.exp(x0 + (0 - y0) * (x1 - x0) / (y1 - y0)))

        Mx_t = sign_change(TC["mach"], TC["r0"])
        Mx_s = sign_change(Ms[meas], r0s[meas]) if meas.sum() >= 2 else np.nan
        f = RS["verdict_sign_change_factor"]
        ok_i = bool(np.isfinite(Mx_s) and np.isfinite(Mx_t) and (Mx_t / f <= Mx_s <= Mx_t * f))
        R.add(
            f"Verdict {tag}",
            "(i) sign change of r0: M_x simulation",
            Mx_s,
            None,
            ok_i if np.isfinite(Mx_s) else None,
            note=f"theory M_x = {Mx_t:.3g}; allowed factor {f}",
        )
        rho_s = spearmanr(r0s[meas], r0t[meas]).statistic if meas.sum() >= 3 else np.nan
        ok_ii = bool(np.isfinite(rho_s) and rho_s >= RS["verdict_rank_corr_min"])
        R.add(
            f"Verdict {tag}",
            "(ii) Spearman r0_sim vs r0_theo",
            rho_s if np.isfinite(rho_s) else np.nan,
            RS["verdict_rank_corr_min"],
            ok_ii if np.isfinite(rho_s) else None,
            note=f"{int(meas.sum())} measurable runs",
        )
        R.add(f"Verdict {tag}", "(iii) measurable runs / total", float(meas.sum()) / len(rows))
        summary.append((tag, Mx_s, Mx_t, rho_s, ok_i, ok_ii))
        base_setup = base_setup or st0
        fig, axs = plt.subplots(2, 2, figsize=(13.5, 9))
        ax = axs[0, 0]
        ax.plot(TC["mach"], TC["r0"], color="k", lw=1.8, label="theory r₀ (linear response)")
        ax.errorbar(
            Ms[meas],
            r0s[meas],
            yerr=[x["sim"]["r0_err"] for x in np.array(rows)[meas]],
            fmt="o",
            color=viz.OKABE[1],
            ms=7,
            capsize=3,
            label="simulation (probes)",
        )
        if (~meas).any():
            ax.plot(Ms[~meas], r0s[~meas], "x", color="0.5", ms=8, label="below noise floor")
        ax.axhline(0, color="k", lw=0.7)
        for Mx, c, lab in ((Mx_t, "k", "M_x theory"), (Mx_s, viz.OKABE[1], "M_x simulation")):
            if np.isfinite(Mx):
                ax.axvline(Mx, color=c, ls=":", lw=1.2)
                ax.annotate(lab, (Mx, 0.02), xycoords=("data", "axes fraction"), rotation=90, fontsize=8, color=c)
        ax.axvspan(Mg.min(), rp.jeans_mach(st0), color="0.85", alpha=0.4, lw=0, label="M < M_J (box Jeans-unstable)")
        ax.set_xscale("log"), ax.set_xlabel("M = c_s/σ_DM"), ax.set_ylabel("r₀ = corr(δ_DM, δ_gas) at zero lag")
        ax.set_title("(a) zero-lag correlation: ballistic → resonant → quasi-static", loc="left"), ax.legend(
            fontsize=7.5
        )
        ax = axs[0, 1]
        ax.plot(TC["mach"], TC["lag"], color="k", lw=1.8, label="theory")
        ax.plot(
            Ms[meas],
            [x["sim"]["lag_at_max"] for x in np.array(rows)[meas]],
            "o",
            color=viz.OKABE[1],
            ms=7,
            label="simulation",
        )
        ax.set_xscale("log"), ax.set_xlabel("M"), ax.set_ylabel("lag of maximum |r| [Myr]")
        ax.set_title("(b) gas response lag", loc="left"), ax.legend(fontsize=8)
        ax = axs[1, 0]
        ax.plot(TC["mach"], TC["eta"], color="k", lw=1.8, label="theory η")
        ax.plot(Ms, [x["sim"]["eta"] for x in rows], "o", color=viz.OKABE[1], ms=7, label="simulation η")
        ax.plot(Ms, [x["sim"]["eta_noise"] for x in rows], "s", color="0.6", ms=5, label="noise floor (A=0)")
        ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlabel("M"), ax.set_ylabel("η = Var δ_Gas / Var δ_DM")
        ax.set_title("(c) response efficiency (resonant enhancement)", loc="left"), ax.legend(fontsize=8)
        ax = axs[1, 1]
        for i, x in enumerate([rows[0], rows[len(rows) // 2], rows[-1]]):
            ax.plot(x["sim"]["lags"], x["sim"]["r_lag"], color=viz.OKABE[i], lw=1.8, label=f"Sim M={x['M']:.2f}")
            ax.plot(
                x["th"]["lags"], x["th"]["r_lag"], color=viz.OKABE[i], ls="--", lw=1.2, label=f"Theo M={x['M']:.2f}"
            )
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("lag ℓ [Myr] (gas behind DM)"), ax.set_ylabel("r(ℓ)"), ax.legend(fontsize=7, ncol=2)
        ax.set_title("(d) cross-correlation functions", loc="left")
        verdict = f"(i) sign change {'✓' if ok_i else '✗'}   (ii) rank correlation {'✓' if ok_ii else '✗'}"
        fig.suptitle(
            f"Idea B – gas–granule coupling vs Mach number ({label}, {tag})   ·   {verdict}",
            x=0.01,
            ha="left",
            fontsize=12,
            fontweight="bold",
        )
        fig.savefig(figs / f"p7_MACH_{tag}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    base = base_setup
    if base is not None:
        Mg = np.geomspace(0.03, 5, 18)
        taus = np.array([0.0, 1.0, 3.0, 10.0])
        Z = {}
        for kind in ("trailing_boxcar", "zero_lag_boxcar", "delay_half_tau"):
            Z[kind] = np.array(
                [
                    (
                        rp.filter_sensitivity(replace(base, highpass=0.0), Mg, t, kind)["d_eta_rel"]
                        if t > 0
                        else np.zeros_like(Mg)
                    )
                    for t in taus
                ]
            )
        fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3), sharey=True)
        for ax, (kind, z) in zip(axs, Z.items()):
            for i, t in enumerate(taus[1:]):
                ax.plot(Mg, 100 * z[i + 1], color=viz.OKABE[i], label=f"τ = {t:g} Myr")
            ax.axhline(0, color="k", lw=0.7), ax.set_xscale("log"), ax.set_xlabel("M = c_s/σ")
            ax.set_title(kind, loc="left")
        axs[0].set_ylabel("Δη/η [%] (gas response with filter vs instantaneous)"), axs[0].legend(fontsize=8)
        fig.suptitle(
            "Theory A × B: where is the averaging approximation valid? gas-response error vs M and τ",
            x=0.01,
            ha="left",
            fontweight="bold",
        )
        fig.savefig(figs / "p7_THEORY_filter_error_vs_mach.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        ZZ = Z["trailing_boxcar"][1:] * 100
        viz.surface3d_html(
            figs3 / "p7_THEORY_filter_error_surface.html",
            Mg,
            taus[1:],
            ZZ,
            xlabel="M = c_s/σ",
            ylabel="τ [Myr]",
            zlabel="Δη/η [%]",
            title="error of the averaging approximation (boxcar)",
            log_axes=(True, False, False),
        )
        viz.heatmap(
            figs / "p7_THEORY_filter_error_map.png",
            Mg,
            taus[1:],
            ZZ,
            xlabel="M = c_s/σ",
            ylabel="τ [Myr]",
            zlabel="Δη/η [%] (Boxcar)",
            title="error of the averaging approximation",
            dpi=dpi,
            logx=True,
            contour_levels=[-10, -1, 1, 10],
        )
    intro = [
        "Rules pre-declared (config `response.verdict_*`). Theory: linear response (hgr/response.py), validated in "
        "P7_validation (V1–V3). Probes: spherical shells in the granule region; high-pass = centred mean."
    ]
    R.write(out, "P7_response", "Idea B – gas–granule coupling vs Mach number", intro)
    write_manifest(cfg, out, "P7_response", extra=R.results)
    print(f"P7 analysis done -> {out}")


if __name__ == "__main__":
    main()
