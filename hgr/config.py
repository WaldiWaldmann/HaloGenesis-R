from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata as _md
import json
import platform
import re
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

REQUIRED: dict[str, list[str]] = {
    "meta": ["run_label", "provenance", "seed"],
    "axion": ["m22", "soliton_mass_msun", "self_interaction_lambda"],
    "gas": [
        "n_cm3",
        "T_K",
        "mu",
        "gamma",
        "initial_velocity_kms",
        "perturbation_rel_amplitude",
        "cooling_model",
        "cooling_params",
    ],
    "setup": [
        "kind",
        "halo_mass_ratio",
        "halo_sigma_kms",
        "halo_core_kpc",
        "halo_seed",
        "halo_relax_myr",
        "halo_ref_n",
    ],
    "domain": ["box_kpc", "N_dev", "N_prod", "boundary"],
    "time": [
        "t_end_myr",
        "cfl_hydro",
        "sp_kinetic_dt_factor",
        "sp_potential_phase_max_rad",
        "grav_cfl",
        "freefall_dt_fraction",
    ],
    "filter": ["kernel", "tau_myr", "tau_scan_myr"],
    "analysis": [
        "apertures_kpc",
        "primary_aperture_kpc",
        "aperture_center",
        "track_radius_kpc",
        "jeans_threshold",
        "t_window_myr",
        "full_snapshot_times_myr",
    ],
    "scans": ["n_cm3", "N_convergence", "m22"],
    "compute": ["fft_workers", "dtype"],
    "viz": ["dpi", "iso_max_points_per_axis"],
    "historical_reference": [],
    "validation_p1": [
        "N_main",
        "fourier_mode_rel_err_max",
        "gauss_mass_msun",
        "gauss_sigma_kpc",
        "gauss_bgcorr_rel_err_max",
        "soliton_force_rel_err_max",
        "box_sizes_kpc",
        "box_test_dx_kpc",
        "convergence_N",
        "convergence_N_ref",
        "convergence_sigmas_kpc",
        "invariance_rel_err_max",
        "timing_N",
        "timing_repeats",
    ],
    "sp": [
        "polish_coarse_n",
        "polish_coarse_steps",
        "polish_coarse_dtau_T0",
        "polish_fine_steps",
        "polish_fine_dtau_T0",
    ],
    "hydro": ["limiter", "jeans_floor_NJ"],
    "validation_p2": [
        "N_main",
        "t_stationary_myr",
        "rho0_variation_max",
        "norm_drift_max",
        "energy_drift_max",
        "packet_sigma_kpc",
        "packet_t_myr",
        "packet_rel_err_max",
        "boost_modes",
        "boost_t_myr",
        "com_err_max_kpc",
        "order_N",
        "order_t_myr",
        "order_kin_factors",
        "order_expected",
        "order_tol",
    ],
    "validation_p3": [
        "sod_N",
        "sod_t",
        "sod_cfl",
        "sod_l1_max_at_400",
        "sod_order_min",
        "conservation_N",
        "conservation_steps",
        "conservation_rel_max",
        "jeans_N",
        "jeans_kJ_modes",
        "jeans_modes",
        "jeans_amp",
        "jeans_periods",
        "jeans_rel_err_max",
        "hydrostatic_N",
        "hydrostatic_log_contrast",
        "hydrostatic_t_myr",
        "hydrostatic_vmax_over_cs_max",
        "cooling_n_cm3",
        "cooling_T0_K",
        "cooling_dt_myr",
        "cooling_rel_err_max",
    ],
    "validation_p4": [
        "N",
        "filter_rel_err_max",
        "bias_taus_myr",
        "bias_omegas_per_myr",
        "limit_n_cm3",
        "limit_t_myr",
        "soliton_var_max",
        "null_identity_rel_max",
        "null_steps",
        "momentum_t_myr",
        "momentum_tau_values_myr",
        "momentum_rel_max_tau0",
    ],
    "p5": [
        "N",
        "t_end_myr",
        "dt_max_myr",
        "dt_min_abort_myr",
        "viz_radii_kpc",
        "early_window_myr",
        "o3_early_min",
        "flux_consistency_max",
    ],
    "validation_p4x": ["N", "t_myr", "halo_fluct_factor_min", "floor_NJ", "floor_rel_tol", "cool_order_tol"],
    "response": [
        "probe_shells_kpc",
        "probes_per_shell",
        "highpass_myr",
        "damping_per_myr",
        "t_skip_myr",
        "resample_dt_myr",
        "verdict_sign_change_factor",
        "verdict_rank_corr_min",
    ],
    "validation_p7": [
        "filter_omega_tau",
        "filter_dt_over_tau",
        "filter_rel_err_max",
        "drive_N",
        "drive_omega_over_ws",
        "drive_rel_err_max",
        "granule_N",
        "granule_box_kpc",
        "granule_rel_err_max",
        "granule_var_rel_err_max",
    ],
    "p6": [
        "eval_times_myr",
        "truelove_max",
        "o2_saturation",
        "o3_early_min",
        "momentum_budget_early_max",
        "flux_residual_M0_max",
        "floor_mass_fraction_max",
        "seed_sigma_factor",
    ],
}

TRACKED_PACKAGES = ["numpy", "scipy", "matplotlib", "astropy", "plotly", "PyYAML", "numba"]


class _Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 floats (PyYAML would read '2.92e8' as a string)."""


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                  |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                  |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                  |[-+]?\.(?:inf|Inf|INF)
                  |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _assert_no_numeric_strings(node, where: str) -> None:
    if isinstance(node, dict):
        for k, v in node.items():
            _assert_no_numeric_strings(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            _assert_no_numeric_strings(v, f"{where}[{i}]")
    elif isinstance(node, str):
        try:
            float(node)
        except ValueError:
            return
        raise TypeError(f"config value {where}={node!r} was parsed as a string")


def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.load(fh, Loader=_Loader)  # noqa: S506 (SafeLoader-Subklasse)
    _assert_no_numeric_strings(cfg, "cfg")
    for section, keys in REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"Config {path}: section '{section}' missing")
        for key in keys:
            if key not in cfg[section]:
                raise KeyError(f"Config {path}: key '{section}.{key}' missing")
    if cfg["domain"]["boundary"] != "periodic":
        raise ValueError("Only periodic boundaries are implemented.")
    if cfg["compute"]["dtype"] != "float64":
        raise ValueError("Only float64 is supported.")
    if cfg["axion"]["self_interaction_lambda"] != 0.0:
        raise NotImplementedError("lambda != 0 is not implemented.")
    cfg["_config_path"] = str(path.resolve())
    apply_compute_settings(cfg)
    return cfg


def apply_compute_settings(cfg: dict) -> None:
    import os

    if "HGR_FFT_WORKERS" in os.environ:
        cfg["compute"]["fft_workers"] = int(os.environ["HGR_FFT_WORKERS"])
    backend = cfg["compute"].get("fft_backend", "scipy")
    if backend == "pyfftw":
        try:
            import pyfftw
            import pyfftw.interfaces.scipy_fft as pfft
        except ImportError as exc:
            raise ImportError(
                "compute.fft_backend = pyfftw, but pyFFTW is not installed " "(pip install pyfftw)"
            ) from exc
        import scipy.fft

        pyfftw.interfaces.cache.enable()
        pyfftw.interfaces.cache.set_keepalive_time(300)
        scipy.fft.set_global_backend(pfft)
    elif backend != "scipy":
        raise ValueError(f"unknown fft_backend {backend}")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = _md.version(name)
        except _md.PackageNotFoundError:
            out[name] = None
    return out


def write_manifest(cfg: dict, out_dir: Path, stage: str, extra: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    src_hashes = {
        str(p.relative_to(PACKAGE_ROOT)).replace("\\", "/"): _sha256_file(p)
        for p in sorted(PACKAGE_ROOT.glob("hgr/*.py")) + sorted(PACKAGE_ROOT.glob("scripts/*.py"))
    }
    manifest = {
        "stage": stage,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "run_label": cfg["meta"]["run_label"],
        "provenance": cfg["meta"]["provenance"],
        "config_path": cfg["_config_path"],
        "config_sha256": _sha256_file(Path(cfg["_config_path"])),
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "source_sha256": src_hashes,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
    if extra is not None:
        manifest["results"] = extra
    path = out_dir / f"manifest_{stage}.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def output_dir(cfg: dict, stage: str) -> Path:
    d = PACKAGE_ROOT / "runs" / cfg["meta"]["run_label"] / stage
    (d / "figs").mkdir(parents=True, exist_ok=True)
    (d / "figs3d").mkdir(parents=True, exist_ok=True)
    return d
