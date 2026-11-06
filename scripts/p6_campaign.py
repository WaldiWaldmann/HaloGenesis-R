from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

from hgr.config import PACKAGE_ROOT, load_config

REF_SECONDS_N48_30MYR = 70.0


def stage_name(N, tau, n, t_end):
    return f"P5_N{N}_tau{tau:g}_n{n:g}_t{t_end:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--jobs", type=int, default=1, help="runs in parallel")
    ap.add_argument("--threads", type=int, default=0, help="threads per run (0 = cores/jobs)")
    args = ap.parse_args()
    runs = yaml.safe_load(Path(args.campaign).read_text(encoding="utf-8"))["runs"]
    todo = []
    for r in runs:
        if r.get("optional", False) and not args.include_optional:
            continue
        cfg = load_config(PACKAGE_ROOT / r["config"])
        N, tau = int(r["N"]), float(r["tau"])
        n = float(r.get("n_cm3", cfg["gas"]["n_cm3"]))
        t_end = float(r.get("t_end", cfg["p5"]["t_end_myr"]))
        st = stage_name(N, tau, n, t_end)
        extra, tags = [], ""
        for key, flag, sec, ck, tag in (
            ("seed", "--seed", "meta", "seed", "s"),
            ("setup", "--setup", "setup", "kind", ""),
            ("jeans_floor", "--jeans-floor", "hydro", "jeans_floor_NJ", "jf"),
            ("cooling", "--cooling", "gas", "cooling_model", "c-"),
            ("dt_max", "--dt-max", "p5", "dt_max_myr", "dt"),
            ("halo_seed", "--halo-seed", "setup", "halo_seed", "hs"),
            ("filter", "--filter-kind", "filter", "kernel", "fk-"),
            ("T_K", "--T-K", "gas", "T_K", "T"),
        ):
            if key in r and r[key] != cfg[sec][ck]:
                extra += [flag, str(r[key])]
                tags += f"_{tag}{r[key]:g}" if isinstance(r[key], (int, float)) else f"_{tag}{r[key]}"
        st += tags
        out = PACKAGE_ROOT / "runs" / cfg["meta"]["run_label"] / st
        done = (out / f"{st}_report.md").exists()
        est = (
            REF_SECONDS_N48_30MYR
            * (N / 48) ** 4
            * t_end
            / 30
            * (cfg["p5"]["dt_max_myr"] / float(r.get("dt_max", cfg["p5"]["dt_max_myr"])))
        )
        todo.append((r, N, tau, n, t_end, st, out, done, est, extra))
    print(f"{'status':10s} {'run':48s} {'estimate (1 core)':>20s}")
    for r, N, tau, n, t_end, st, out, done, est, extra in todo:
        print(f"{'done' if done else 'open':10s} {r['config'].split('/')[-1] + ' ' + st:72s} {est/3600:17.2f} h")
    print("Note: estimate scales ∝ N⁴ on 1 core; real runs use all cores.")
    if args.dry_run:
        return
    import os
    from concurrent.futures import ThreadPoolExecutor

    ncpu = os.cpu_count() or 1
    threads = args.threads or max(1, ncpu // max(1, args.jobs))
    print(f"CPU cores: {ncpu}, parallel runs: {args.jobs}, threads per run: {threads}", flush=True)

    def run_one(item):
        r, N, tau, n, t_end, st, out, done, est, extra = item
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "scripts.p5_observables",
            "--config",
            r["config"],
            "--N",
            str(N),
            "--tau",
            str(tau),
            "--n-cm3",
            str(n),
            "--t-end",
            str(t_end),
        ] + extra
        env = dict(
            os.environ,
            NUMBA_NUM_THREADS=str(threads),
            HGR_FFT_WORKERS=str(threads),
            OMP_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            OPENBLAS_NUM_THREADS=str(threads),
        )
        print(f"\n>>> {st}  ({' '.join(cmd[2:])})", flush=True)
        t0 = time.time()
        with open(out / "run.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd, cwd=PACKAGE_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
            )
            for line in proc.stdout:
                if args.jobs == 1:
                    print(line, end="", flush=True)
                log.write(line)
                log.flush()
            rc = proc.wait()
        print(
            f"<<< {st}: Exit {rc}, {time.time()-t0:.0f} s" + ("" if rc == 0 else f"  FAILED – log: {out/'run.log'}"),
            flush=True,
        )

    pending = [it for it in todo if not it[7]]
    pending.sort(key=lambda it: -it[8]) if args.jobs > 1 else None
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        list(ex.map(run_one, pending))


if __name__ == "__main__":
    main()
