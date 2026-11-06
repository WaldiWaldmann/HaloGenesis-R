"""Visualisierung HaloGenesis-R (v2).

Gestaltungsregeln
  * Farbenblind-sichere Palette (Okabe & Ito 2008); Dichten: magma, Potentiale: viridis,
    vorzeichenbehaftete Größen: RdBu_r symmetrisch um 0 (bzw. log-Verhältnisse um 0).
  * Farbgrenzen robust über Perzentile (Ausreißerzellen dominieren nicht die Skala).
  * Schwellen (Q = 1, J = 1/4, Toleranzen) als beschriftete Linien/Bänder direkt im Plot.
  * 3D statisch: schattierte Fläche (LightSource) + projizierte Konturen; 3D interaktiv (plotly):
    Isoflächen bei eingeschlossenen Massenanteilen 50/80/95 % – physikalisch interpretierbar.
  * Dashboards fassen pro Lauf die Aussage auf einer Seite zusammen (p5_dashboard).
Alle bisherigen Funktionssignaturen bleiben erhalten.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib import cm, colors, ticker  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
plt.rcParams.update({
    "figure.figsize": (6.4, 4.0), "figure.dpi": 100, "figure.constrained_layout.use": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.labelsize": 9.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
    "grid.linestyle": "-", "grid.linewidth": 0.6, "axes.prop_cycle": matplotlib.cycler(color=OKABE),
    "lines.linewidth": 1.7, "legend.frameon": False, "legend.fontsize": 8.5, "image.origin": "lower",
    "savefig.facecolor": "white",
})
PLOTLY_FONT = dict(family="DejaVu Sans, Arial", size=12)


def _save(fig, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_html(fig: go.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template="simple_white", font=PLOTLY_FONT, margin=dict(l=10, r=10, t=50, b=10))
    fig.write_html(str(path), include_plotlyjs="directory", full_html=True)
    return path


def _robust_norm(data, mode: str):
    d = np.asarray(data)[np.isfinite(data)]
    if mode == "log":
        pos = d[d > 0]
        if pos.size == 0:
            return colors.Normalize(0, 1), "magma"
        lo, hi = np.percentile(pos, [0.5, 99.9])
        return colors.LogNorm(vmin=max(lo, hi * 1e-8), vmax=hi), "magma"
    if mode == "sym":
        v = float(np.percentile(np.abs(d), 99.5)) if d.size else 1.0
        v = v or 1.0
        return colors.TwoSlopeNorm(vcenter=0.0, vmin=-v, vmax=v), "RdBu_r"
    lo, hi = np.percentile(d, [0.5, 99.5]) if d.size else (0.0, 1.0)
    return colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1), "viridis"


def _finish_axis(ax, xlabel, ylabel, logx, logy):
    ax.set_xscale("log" if logx else "linear")
    ax.set_yscale("log" if logy else "linear")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


# ======================================================================= 2D ==

def line_plot(path: Path, x, series: dict, *, xlabel: str, ylabel: str, title: str, dpi: int,
              logx: bool = False, logy: bool = False, hlines: dict | None = None, vlines: dict | None = None,
              markers: bool = False, subtitle: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    x = np.asarray(x, float)
    for i, (label, y) in enumerate(series.items()):
        y = np.asarray(y, float)
        if logy:
            y = np.where(y > 0, y, np.nan)
        ax.plot(x, y, marker="o" if markers else None, ms=3.5, label=label, color=OKABE[i % len(OKABE)])
    for label, val in (hlines or {}).items():
        ax.axhline(val, ls="--", lw=1.0, color="0.35", zorder=0)
        ax.annotate(label, xy=(1.0, val), xycoords=("axes fraction", "data"), xytext=(3, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color="0.3", annotation_clip=False)
    for label, val in (vlines or {}).items():
        ax.axvline(val, ls=":", lw=1.0, color="0.45", zorder=0)
        ax.annotate(label, xy=(val, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                    textcoords="offset points", va="top", fontsize=7.5, color="0.35", rotation=90)
    _finish_axis(ax, xlabel, ylabel, logx, logy)
    ax.set_title(title + (f"\n{subtitle}" if subtitle else ""), loc="left")
    if len(series) > 1:
        if len(series) > 4:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax.legend()
    return _save(fig, path, dpi)


def bar_log(path: Path, labels, values, *, ylabel: str, title: str, dpi: int, colors_list=None) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 0.45 * len(values) + 1.4))
    vals = np.asarray(values, float)
    ax.barh(range(len(vals)), vals, color=colors_list or [OKABE[0]] * len(vals), height=0.62)
    ax.set_xscale("log")
    ax.set_yticks(range(len(vals)), labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3g}", (v, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    ax.set_xlabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", visible=False)
    return _save(fig, path, dpi)


def heatmap(path: Path, x, y, Z, *, xlabel: str, ylabel: str, zlabel: str, title: str, dpi: int,
            logx: bool = False, logy: bool = False, logz: bool = False, contour_levels=None) -> Path:
    Z = np.asarray(Z, float)
    assert Z.shape == (len(y), len(x)), (Z.shape, len(y), len(x))
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    finite = Z[np.isfinite(Z)]
    norm = None
    if finite.size:
        if logz:
            pos = finite[finite > 0]
            norm = colors.LogNorm(*np.percentile(pos, [0.5, 99.5])) if pos.size else None
        else:
            lo, hi = np.percentile(finite, [0.5, 99.5])
            norm = colors.Normalize(lo, hi if hi > lo else lo + 1)
    shading = "gouraud" if min(len(x), len(y)) >= 8 else "nearest"
    pcm = ax.pcolormesh(x, y, np.ma.masked_invalid(Z), shading=shading, norm=norm, cmap="viridis", rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.01, fraction=0.05).set_label(zlabel)
    if contour_levels and finite.size:
        lv = sorted(l for l in contour_levels if np.nanmin(Z) < l < np.nanmax(Z))
        if lv:
            cs = ax.contour(x, y, Z, levels=lv, colors="white", linewidths=1.3)
            ax.clabel(cs, fmt=lambda v: f"{v:.3g}", fontsize=8)
    _finish_axis(ax, xlabel, ylabel, logx, logy)
    ax.grid(False)
    ax.set_title(title, loc="left")
    return _save(fig, path, dpi)


def slices_row(path: Path, fields, extent_kpc: float, *, dpi: int, suptitle: str, axis: int = 2,
               circles_kpc=None) -> Path:
    """Mittelschnitte; fields: (feld3d, titel, modus) mit modus in {"lin", "log", "sym"}."""
    nf = len(fields)
    fig, axes = plt.subplots(1, nf, figsize=(3.5 * nf, 3.45))
    axes = np.atleast_1d(axes)
    ext = [-extent_kpc / 2, extent_kpc / 2, -extent_kpc / 2, extent_kpc / 2]
    for ax, (f, title, mode) in zip(axes, fields):
        sl = np.take(f, f.shape[axis] // 2, axis=axis).T
        norm, cmap = _robust_norm(sl, mode)
        im = ax.imshow(sl, extent=ext, norm=norm, cmap=cmap, interpolation="bicubic")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
        for R in circles_kpc or []:
            ax.add_patch(Circle((0, 0), R, fill=False, ec="k" if mode == "sym" else "white", lw=0.8, ls="--"))
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.set_xlabel("x [kpc]")
        ax.set_ylabel("y [kpc]")
        ax.grid(False)
    fig.suptitle(suptitle, fontsize=10.5, x=0.01, ha="left")
    return _save(fig, path, dpi)


# ======================================================================= 3D ==

def _log_formatter(enabled):
    if not enabled:
        return ticker.ScalarFormatter()
    return ticker.FuncFormatter(lambda v, _: f"$10^{{{v:.0f}}}$" if abs(v - round(v)) < 1e-6 else f"{10**v:.2g}")


def surface3d_png(path: Path, x, y, Z, *, xlabel: str, ylabel: str, zlabel: str, title: str, dpi: int,
                  log_axes=(False, False, False)) -> Path:
    """Schattierte Fläche + projizierte Konturen am Boden; log-Achsen als 10^x beschriftet."""
    X, Y = np.meshgrid(np.asarray(x, float), np.asarray(y, float))
    Zf = np.asarray(Z, float)
    Xp = np.log10(X) if log_axes[0] else X
    Yp = np.log10(Y) if log_axes[1] else Y
    Zp = np.log10(np.where(Zf > 0, Zf, np.nan)) if log_axes[2] else Zf
    if not np.isfinite(Zp).any():
        Zp = np.zeros_like(Zf)
    Zp = np.where(np.isfinite(Zp), Zp, np.nanmin(Zp))
    zlo, zhi = float(np.min(Zp)), float(np.max(Zp))
    zhi = zhi if zhi > zlo else zlo + 1.0
    fig = plt.figure(figsize=(6.8, 5.0))
    ax = fig.add_subplot(111, projection="3d")
    norm = colors.Normalize(zlo, zhi)
    rgb = LightSource(azdeg=315, altdeg=40).shade(Zp, cmap=cm.viridis, norm=norm, blend_mode="soft")
    ax.plot_surface(Xp, Yp, Zp, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=True, shade=False)
    zmin = zlo - 0.15 * (zhi - zlo)
    if min(Zp.shape) >= 2:
        try:
            ax.contour(Xp, Yp, Zp, zdir="z", offset=zmin, cmap="viridis", levels=10, linewidths=0.8)
        except (ValueError, TypeError):
            pass
    ax.set_zlim(zmin, zhi)
    for axis_, enabled in zip((ax.xaxis, ax.yaxis, ax.zaxis), log_axes):
        axis_.set_major_formatter(_log_formatter(enabled))
        axis_.pane.set_alpha(0.0)
    ax.set_xlabel(xlabel, labelpad=6)
    ax.set_ylabel(ylabel, labelpad=6)
    ax.set_zlabel(zlabel, labelpad=6)
    ax.view_init(elev=26, azim=-58)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=0.55, pad=0.08,
                 label=("log₁₀ " if log_axes[2] else "") + zlabel)
    ax.set_title(title, loc="left")
    return _save(fig, path, dpi)


def surface3d_html(path: Path, x, y, Z, *, xlabel: str, ylabel: str, zlabel: str, title: str,
                   log_axes=(False, False, False)) -> Path:
    Xp = np.log10(x) if log_axes[0] else np.asarray(x)
    Yp = np.log10(y) if log_axes[1] else np.asarray(y)
    Zf = np.asarray(Z, float)
    Zp = np.log10(np.where(Zf > 0, Zf, np.nan)) if log_axes[2] else Zf
    pre = lambda b: "log₁₀ " if b else ""  # noqa: E731
    fig = go.Figure(go.Surface(x=Xp, y=Yp, z=Zp, colorscale="Viridis",
                               contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=2)),
                               lighting=dict(ambient=0.6, diffuse=0.8, specular=0.15),
                               colorbar=dict(title=pre(log_axes[2]) + zlabel)))
    fig.update_layout(title=title, scene=dict(xaxis_title=pre(log_axes[0]) + xlabel,
                                              yaxis_title=pre(log_axes[1]) + ylabel,
                                              zaxis_title=pre(log_axes[2]) + zlabel,
                                              camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9))))
    return _write_html(fig, path)


def field_surface_png(path: Path, field, extent_kpc: float, *, zlabel: str, title: str, dpi: int,
                      stride: int = 1) -> Path:
    n = field.shape[0]
    stride = max(stride, int(np.ceil(n / 96)))
    sl = field[:, :, n // 2][::stride, ::stride]
    x = (np.arange(n)[::stride] - n // 2) * extent_kpc / n
    return surface3d_png(path, x, x, sl.T, xlabel="x [kpc]", ylabel="y [kpc]", zlabel=zlabel, title=title, dpi=dpi)


def ortho_slices_3d_png(path: Path, field, extent_kpc: float, *, title: str, dpi: int, log: bool, label: str,
                        stride: int = 2) -> Path:
    """Drei orthogonale Mittelschnitte als Kreuz im Raum (gemeinsame, robuste Farbskala)."""
    n = field.shape[0]
    stride = max(stride, int(np.ceil(n / 64)))
    f = np.asarray(field[::stride, ::stride, ::stride], float)
    x = (np.arange(n)[::stride] - n // 2) * extent_kpc / n
    data = np.log10(np.where(f > 0, f, np.nan)) if log else f
    if log:
        lo, hi = np.nanpercentile(data, [1, 99.9])
        norm, cmap = colors.Normalize(lo, hi if hi > lo else lo + 1), cm.magma
    elif np.nanmin(data) < 0 < np.nanmax(data):
        v = float(np.nanpercentile(np.abs(data), 99.5)) or 1.0
        norm, cmap = colors.TwoSlopeNorm(0, -v, v), cm.RdBu_r
    else:
        lo, hi = np.nanpercentile(data, [0.5, 99.5])
        norm, cmap = colors.Normalize(lo, hi if hi > lo else lo + 1), cm.viridis
    c = f.shape[0] // 2
    C1, C2 = np.meshgrid(x, x, indexing="ij")
    full = np.full_like(C1, x[c])
    fig = plt.figure(figsize=(6.4, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    kw = dict(rstride=1, cstride=1, shade=False, linewidth=0, antialiased=False)
    ax.plot_surface(C1, C2, full, facecolors=cmap(norm(data[:, :, c])), **kw)
    ax.plot_surface(C1, full, C2, facecolors=cmap(norm(data[:, c, :])), **kw)
    ax.plot_surface(full, C1, C2, facecolors=cmap(norm(data[c, :, :])), **kw)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.0)
    ax.view_init(elev=24, azim=-50)
    ax.set_box_aspect((1, 1, 1))
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.55, pad=0.06,
                 label=("log₁₀ " if log else "") + label)
    ax.set_xlabel("x [kpc]")
    ax.set_ylabel("y [kpc]")
    ax.set_zlabel("z [kpc]")
    ax.set_title(title, loc="left")
    return _save(fig, path, dpi)


def _mass_levels(vals, fractions=(0.5, 0.8, 0.95)) -> list:
    """Dichteschwellen, oberhalb derer der Anteil f der Gesamtmasse liegt."""
    v = np.sort(np.asarray(vals).ravel())[::-1]
    cum = np.cumsum(v) / v.sum()
    return [float(v[min(np.searchsorted(cum, f), v.size - 1)]) for f in fractions]


def isosurface_html(path: Path, field, extent_kpc: float, *, title: str, log: bool, label: str, max_points: int,
                    n_surfaces: int = 5) -> Path:
    """Dichten (log=True): Isoflächen, die 50/80/95 % der Masse einschließen (in der Legende benannt).
    Sonst (z.B. Potential): Niveaus im unteren Wertebereich (Potentialtopf)."""
    n = field.shape[0]
    stride = max(1, int(np.ceil(n / max_points)))
    f = np.asarray(field[::stride, ::stride, ::stride], float)
    x = (np.arange(n)[::stride] - n // 2) * extent_kpc / n
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    fig = go.Figure()
    if log:
        fr = (0.5, 0.8, 0.95)
        for level, frac, col, op in zip(_mass_levels(np.clip(f, 0, None), fr), fr,
                                        ("#7a0177", "#dd3497", "#fa9fb5"), (0.9, 0.45, 0.16)):
            fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=f.ravel(), isomin=level,
                                        isomax=level, surface_count=1, opacity=op, showscale=False,
                                        colorscale=[[0, col], [1, col]],
                                        caps=dict(x_show=False, y_show=False, z_show=False),
                                        name=f"{int(frac * 100)} % der Masse", showlegend=True))
    else:
        lo, hi = float(np.min(f)), float(np.percentile(f, 50))
        fig.add_trace(go.Isosurface(x=X.ravel(), y=Y.ravel(), z=Z.ravel(), value=f.ravel(),
                                    isomin=lo + 0.1 * (hi - lo), isomax=hi, surface_count=n_surfaces, opacity=0.3,
                                    colorscale="Viridis", caps=dict(x_show=False, y_show=False, z_show=False),
                                    colorbar=dict(title=label)))
    fig.update_layout(title=title, legend=dict(x=0.02, y=0.98),
                      scene=dict(xaxis_title="x [kpc]", yaxis_title="y [kpc]", zaxis_title="z [kpc]",
                                 aspectmode="cube", camera=dict(eye=dict(x=1.5, y=-1.5, z=1.1))))
    return _write_html(fig, path)


# =================================================================== Dashboard ==

def p5_dashboard(path: Path, *, ts, O1: dict, dlnQ: dict, qmin1, qmin0, o3: dict, mb, J1, J0, floor1,
                 rho1, rho0, extent_kpc: float, apertures, prof_r, prof_t, prof_ratio, info: dict, dpi: int,
                 truelove_max: float, floor_max: float, q1p=None, q0p=None, phic=None, phibc=None,
                 center_kpc=None, floor_active: bool = False, rc_kpc: float | None = None,
                 center_xy=(0.0, 0.0)) -> Path:
    """Eine Seite pro Lauf. Grau: J > 1/4 (nicht aufgelöst); orange: Jeans-Boden dominiert.
    (g) rot nur dort, wo Q < 1 AUSSCHLIESSLICH mit Soliton gilt (diskriminierend)."""
    fig = plt.figure(figsize=(15.5, 10.4))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1])
    ts = np.asarray(ts, float)
    bad = np.maximum(np.asarray(J1), np.asarray(J0)) > truelove_max
    floored = np.asarray(floor1) > floor_max

    def shade(ax):
        for mask, col in ((bad, "0.82"), (floored, "#fdd0a2")):
            if mask.any():
                ax.fill_between(ts, 0, 1, where=mask, transform=ax.get_xaxis_transform(), color=col, alpha=0.7,
                                lw=0, step="mid", zorder=0)

    ext = [-extent_kpc / 2, extent_kpc / 2, -extent_kpc / 2, extent_kpc / 2]
    cx, cy = center_xy
    s1, s0 = rho1.T, rho0.T
    rho_mean = float(np.mean(s0))
    d1, d0 = s1 / rho_mean - 1, s0 / rho_mean - 1
    vd = float(np.percentile(np.abs(np.concatenate([d1.ravel(), d0.ravel()])), 99.9)) or 1e-3
    lin = max(1e-3, 0.03 * vd)
    for k, (sl, ttl) in enumerate(((d1, "(a) Überdichte δ mit Soliton  A=1"), (d0, "(b) Überdichte δ Kontrolle  A=0"))):
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(sl, extent=ext, norm=colors.SymLogNorm(lin, vmin=-vd, vmax=vd), cmap="RdBu_r",
                       interpolation="bicubic")
        for R in apertures:
            ax.add_patch(Circle((cx, cy), R, fill=False, ec="0.2", lw=0.7, ls="--"))
        ax.plot(cx, cy, "+", color="k", ms=8)
        ax.set_title(ttl, loc="left"), ax.set_xlabel("x [kpc]"), ax.set_ylabel("y [kpc]"), ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="δ = ρ/ρ̄ − 1 (gleiche Skala)")
    ax = fig.add_subplot(gs[0, 2])
    lr = np.log10(s1 / s0)
    v = float(np.percentile(np.abs(lr), 99.5)) or 1e-4
    im = ax.imshow(lr, extent=ext, cmap="PuOr_r", norm=colors.TwoSlopeNorm(0, -v, v), interpolation="bicubic")
    for R in apertures:
        ax.add_patch(Circle((cx, cy), R, fill=False, ec="k", lw=0.7, ls="--"))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="log₁₀(ρ₁/ρ₀)")
    ax.set_title("(c) Soliton-Effekt ρ₁/ρ₀", loc="left"), ax.set_xlabel("x [kpc]"), ax.grid(False)
    ax = fig.add_subplot(gs[0, 3])
    prof_t = np.asarray(prof_t, float)
    tn = colors.Normalize(prof_t.min(), prof_t.max() if prof_t.max() > prof_t.min() else prof_t.min() + 1)
    for k in np.unique(np.linspace(0, len(prof_t) - 1, min(9, len(prof_t))).astype(int)):
        ax.plot(prof_r, prof_ratio[k], color=cm.viridis(tn(prof_t[k])), lw=1.5)
    ax.axhline(1, color="0.4", lw=0.8, ls="--")
    for R in apertures:
        ax.axvspan(0, R, color="0.5", alpha=0.05, lw=0)
    if rc_kpc:
        ax.axvline(rc_kpc, color="#D55E00", lw=1, ls=":")
        ax.annotate("r_c Soliton", (rc_kpc, 1.0), xytext=(3, 4), textcoords="offset points", fontsize=7.5,
                    color="#D55E00")
    ax.set_xscale("log"), ax.set_yscale("log")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
    ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter("%.3g"))
    fig.colorbar(cm.ScalarMappable(tn, "viridis"), ax=ax, fraction=0.05, pad=0.02, label="t [Myr]")
    ax.set_xlabel("r [kpc] (um Soliton-Zentrum)"), ax.set_ylabel("ρ₁/ρ₀ in Kugelschale")
    ax.set_title("(d) Radiales Konzentrationsprofil", loc="left")

    ax = fig.add_subplot(gs[1, 0])
    shade(ax)
    for i, (R, y) in enumerate(O1.items()):
        y = np.asarray(y)
        ax.plot(ts, y, color=OKABE[i], label=f"R = {R} kpc")
        ax.annotate(f"{y[-1]:+.2f}", (ts[-1], y[-1]), xytext=(3, 0), textcoords="offset points", fontsize=8,
                    color=OKABE[i], va="center", annotation_clip=False)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("(e) O1 – zusätzliche Konzentration", loc="left"), ax.set_xlabel("t [Myr]")
    ax.set_ylabel("O1 = ln[(M₁/M₁⁰)/(M₀/M₀⁰)]"), ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[1, 1])
    shade(ax)
    for i, (R, y) in enumerate(dlnQ.items()):
        ax.plot(ts, y, color=OKABE[i], label=f"R = {R} kpc")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("(f) ΔlnQ = ln(Q₁/Q₀)  (< 0: instabiler)", loc="left"), ax.set_xlabel("t [Myr]")
    ax.set_ylabel("ΔlnQ")
    ax = fig.add_subplot(gs[1, 2])
    shade(ax)
    if q1p is not None:
        q1p, q0p = np.asarray(q1p), np.asarray(q0p)
        only = (q1p < 1) & (q0p >= 1)
        if only.any():
            ax.fill_between(ts, 0, 1, where=only, transform=ax.get_xaxis_transform(), color="#D55E00", alpha=0.18,
                            lw=0, step="mid", label="nur mit Soliton Q < 1")
        ax.plot(ts, q1p, color=OKABE[1], label="Q(R_p) A=1")
        ax.plot(ts, q0p, color=OKABE[0], label="Q(R_p) A=0")
    ax.plot(ts, qmin1, color=OKABE[1], ls=":", lw=1.1, label="min_R Q A=1")
    ax.plot(ts, qmin0, color=OKABE[0], ls=":", lw=1.1, label="min_R Q A=0")
    ax.axhline(1, color="#D55E00", lw=1, ls="--")
    ax.set_yscale("log"), ax.set_xlabel("t [Myr]"), ax.set_ylabel("Q = M_J/M_b")
    ax.set_title("(g) Jeans-Verhältnis (rot: nur A=1 instabil)", loc="left")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax = fig.add_subplot(gs[1, 3])
    for i, (R, y) in enumerate(o3.items()):
        ax.plot(ts[1:], np.asarray(y, float)[1:], color=OKABE[i], label=f"R = {R} kpc")
    ax.set_ylim(-1.05, 1.05), ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("t [Myr]"), ax.set_ylabel("O3 = cos∠(δg, δv)")
    ax2 = ax.twinx()
    ax2.plot(ts[1:], np.asarray(mb, float)[1:], color="0.35", ls=":", lw=1.4)
    ax2.set_yscale("log"), ax2.grid(False), ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("Impulsbudget-Rest (gepunktet)", color="0.35")
    ax.set_title("(h) O3 – Kraft ↔ Antwort", loc="left"), ax.legend(fontsize=7, loc="lower left")

    ax = fig.add_subplot(gs[2, 0])
    J = np.maximum(np.asarray(J1), np.asarray(J0))
    ax.plot(ts, J1, color=OKABE[1], label="A=1"), ax.plot(ts, J0, color=OKABE[0], label="A=0")
    ax.axhline(truelove_max, color="k", ls="--", lw=1)
    ax.axhspan(truelove_max, max(np.nanmax(J), truelove_max) * 3, color="0.82", alpha=0.7, zorder=0)
    ax.set_yscale("log"), ax.set_xlabel("t [Myr]"), ax.set_ylabel("J = dx/λ_J")
    ax.set_title("(i) Auflösung – Truelove (grau: J > 1/4)", loc="left"), ax.legend(fontsize=7.5)
    ax = fig.add_subplot(gs[2, 1])
    if floor_active:
        ax.plot(ts, floor1, color=OKABE[3])
        ax.axhline(floor_max, color="k", ls="--", lw=1)
        ax.set_ylim(-0.02, 1.02), ax.set_ylabel("gebodeter Massenanteil in R_p")
        ax.set_title("(j) Jeans-Boden-Anteil", loc="left")
    elif phic is not None:
        ref = abs(float(np.mean(phic)))
        ax.plot(ts, np.asarray(phic) / ref, color=OKABE[4], lw=1.2, label="Φ (Axionen sehen)")
        ax.plot(ts, np.asarray(phibc) / ref, color=OKABE[0], lw=1.8, label="Φ̄_τ (Gas sieht)")
        ax.set_ylabel("Potential im Zentrum / |⟨Φ⟩|")
        if center_kpc is not None:
            ax2 = ax.twinx()
            ax2.plot(ts, np.linalg.norm(np.asarray(center_kpc), axis=1), color="0.45", ls="--", lw=1)
            ax2.set_ylabel("Soliton-Verschiebung [kpc] (gestrichelt)", color="0.45")
            ax2.grid(False), ax2.spines["right"].set_visible(True)
        ax.set_title("(j) Filter Φ → Φ̄ & Soliton-Position", loc="left"), ax.legend(fontsize=7, loc="lower left")
    ax.set_xlabel("t [Myr]")
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis("off")
    rows = [[k, v] for k, v in info.items() if not k.startswith("_")]
    tab = ax.table(cellText=rows, colLabels=["Kennzahl", "Wert"], loc="upper left", colWidths=[0.58, 0.42],
                   cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.28)
    for (r_, _c), cell in tab.get_celld().items():
        cell.set_edgecolor("0.85")
        if r_ == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")
    fig.text(0.5, -0.012, "Schnitte durch die Ebene des Soliton-Zentrums (+); Kreise = Analyse-Aperturen. "
             "Zeitreihen: grau = nicht aufgelöst (J > 1/4), orange = Jeans-Boden dominiert.",
             ha="center", fontsize=8, color="0.35")
    fig.suptitle(info.get("_title", "P5 – Ergebnis"), x=0.01, ha="left", fontsize=13, fontweight="bold")
    return _save(fig, path, dpi)


def setup_schematic(path: Path, *, params: dict, dpi: int) -> Path:
    """Schema des gekoppelten Lockstep-Algorithmus (eine Seite, für Bericht/Jury)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 7.6))
    ax.set_xlim(0, 13), ax.set_ylim(0, 7.6), ax.axis("off")

    def box(x, y, w, h, txt, fc, ec="0.3", fs=9.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.1))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x0, y0, x1, y1, txt="", col="0.25", style="-|>", ls="-"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, color=col, lw=1.3,
                                     linestyle=ls))
        if txt:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, txt, ha="center", fontsize=8, color=col)

    ax.text(0.2, 7.3, "HaloGenesis-R – gekoppelter, gematchter Lauf (ein Makroschritt Δt)", fontsize=13, fontweight="bold")
    # Zweig A=1
    ax.text(0.3, 6.65, "Zweig A=1  (mit Axionen)", fontsize=10.5, fontweight="bold", color=OKABE[1])
    box(0.3, 5.3, 2.7, 1.1, "Axionfeld ψ\nSchrödinger, Split-Step\n(n Teilschritte)", "#fde9d9")
    box(3.6, 5.3, 2.3, 1.1, "Poisson\n∇²Φ = 4πG(ρ_a+ρ_b)", "#eeeeee")
    box(6.5, 5.3, 2.5, 1.1, "Zeitfilter\nΦ̄_τ = (1/τ)∫Φ dt\n(Gl. 42, Fourierraum)", "#dbe9f6")
    box(9.6, 5.3, 3.1, 1.1, "Gas: Euler (MUSCL-HLLC)\n+ Kühlung + Jeans-Boden\nKick mit −∇Φ̄_τ", "#e2f0d9")
    arrow(3.0, 5.85, 3.6, 5.85, "ρ_a = |ψ|²")
    arrow(5.9, 5.85, 6.5, 5.85, "Φ")
    arrow(9.0, 5.85, 9.6, 5.85, "Φ̄_τ")
    arrow(4.75, 5.3, 1.65, 5.3, "Φ (instantan) → ψ", col=OKABE[1], style="-|>", ls="--")
    arrow(11.15, 6.4, 4.75, 6.4, "ρ_b (Rückkopplung)", col="0.45", ls=":")
    # Zweig A=0
    ax.text(0.3, 4.55, "Zweig A=0  (Kontrolle, ψ ≡ 0 – sonst identischer Code und identische Anfangswerte)",
            fontsize=10.5, fontweight="bold", color=OKABE[0])
    box(3.6, 3.2, 2.3, 1.1, "Poisson\n∇²Φ = 4πG ρ_b", "#eeeeee")
    box(6.5, 3.2, 2.5, 1.1, "Zeitfilter Φ̄_τ\n(gleiches τ)", "#dbe9f6")
    box(9.6, 3.2, 3.1, 1.1, "Gas: identische Numerik\nund Anfangsbedingung", "#e2f0d9")
    arrow(5.9, 3.75, 6.5, 3.75, "Φ"), arrow(9.0, 3.75, 9.6, 3.75, "Φ̄_τ")
    arrow(11.15, 4.3, 4.75, 4.3, "ρ_b", col="0.45", ls=":")
    # Observablen
    box(0.3, 0.9, 6.0, 1.75, "Observablen an mitlaufenden Aperturen (Soliton-Schwerpunkt)\n"
        "O1 = ln[(M₁/M₁⁰)/(M₀/M₀⁰)]   Konzentration\nΔlnQ = ln(Q₁/Q₀),  O2   Jeans-Verhältnis\n"
        "O3 = cos∠(δg, δv)   Kraft ↔ Antwort", "#fff7d6", fs=9.2)
    box(6.7, 0.9, 6.0, 1.75, "Gegenproben & Gültigkeit\nMassenbilanz ΔM = −∮ρv·dS (feste Apertur)\n"
        "Impulsbudget δv vs ∫δg dt   ·   Truelove J ≤ 1/4\nAuflösung 64/128/256, dt-Halbierung, Seeds, τ-Scan",
        "#f2f2f2", fs=9.2)
    arrow(11.15, 3.2, 3.3, 2.65, col="0.5", style="-|>")
    arrow(11.15, 5.3, 9.7, 2.65, col="0.5", style="-|>")
    ptxt = "   ·   ".join(f"{k} = {v}" for k, v in params.items())
    ax.text(0.3, 0.35, ptxt, fontsize=8.5, color="0.3")
    return _save(fig, path, dpi)


# ============================================================ Snapshot-Serien ==

def _symlog(vmax: float, linthresh: float):
    vmax = max(vmax, 1.5 * linthresh)
    return colors.SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax, base=10)


def snapshot_series(outdir: Path, snaps: list, *, rho_mean: float, noise_amp: float, extent_kpc: float,
                    apertures, dpi: int, title: str) -> Path:
    """Alle Snapshots eines Laufs mit EINER gemeinsamen Skala je Größe.
    snaps: Liste von dicts {t, r1, r0, dv_kms} (Mittelschnitte, 2D).
    Dichte als Überdichte δ = ρ/ρ̄₀ − 1, symmetrisch-logarithmisch (linear für |δ| < 3·Rauschamplitude)."""
    lin = 3.0 * noise_amp
    d1 = [s["r1"] / rho_mean - 1 for s in snaps]
    d0 = [s["r0"] / rho_mean - 1 for s in snaps]
    lr = [np.log10(s["r1"] / s["r0"]) for s in snaps]
    vd = float(np.percentile(np.abs(np.concatenate([np.ravel(x) for x in d1 + d0])), 99.95))
    vr = float(np.percentile(np.abs(np.concatenate([np.ravel(x) for x in lr])), 99.95))
    vv = float(max(np.max(s["dv_kms"]) for s in snaps)) or 1.0
    has_ax = all("ra" in s and np.max(s["ra"]) > 0 for s in snaps)
    norms = [_symlog(vd, lin), _symlog(vd, lin), _symlog(vr, lin / np.log(10)), colors.Normalize(0, vv)]
    cmaps = ["RdBu_r", "RdBu_r", "PuOr_r", "viridis"]
    rows = [("δ = ρ/ρ̄₀ − 1  (A=1, Soliton)", d1), ("δ = ρ/ρ̄₀ − 1  (A=0, Kontrolle)", d0),
            ("log₁₀ ρ₁/ρ₀  (Wirkung)", lr), ("|δv| [km/s]", [s["dv_kms"] for s in snaps])]
    if has_ax:
        allra = np.concatenate([s["ra"].ravel() for s in snaps])
        hi = float(np.percentile(allra, 99.95))
        rows.insert(0, ("ρ_a Axionen [M☉/kpc³]", [s["ra"] for s in snaps]))
        norms.insert(0, colors.LogNorm(vmin=hi * 1e-4, vmax=hi))
        cmaps.insert(0, "cividis")
    ext = [-extent_kpc / 2, extent_kpc / 2, -extent_kpc / 2, extent_kpc / 2]
    nt = len(snaps)
    fig, axes = plt.subplots(len(rows), nt, figsize=(2.55 * nt + 1.6, 2.55 * len(rows)), squeeze=False)
    for i, (lab, data) in enumerate(rows):
        for j, s in enumerate(snaps):
            ax = axes[i, j]
            im = ax.imshow(data[j].T, extent=ext, norm=norms[i], cmap=cmaps[i], interpolation="bicubic")
            for R in apertures:
                ax.add_patch(Circle((s.get("cx", 0.0), s.get("cy", 0.0)), R, fill=False, ec="0.25", lw=0.6, ls="--"))
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            gx = np.linspace(*ext[:2], data[j].shape[0])
            XX, YY = np.meshgrid(gx, gx, indexing="ij")
            inner = data[j][np.hypot(XX - s.get("cx", 0.0), YY - s.get("cy", 0.0)) < 0.5]
            lab_i = rows[i][0]
            ax.text(0.03, 0.04, f"max {np.max(inner):+.2g}" if ("δ" in lab_i or "log" in lab_i)
                    else f"max {np.max(data[j]):.2g}",
                    transform=ax.transAxes, fontsize=7.5, bbox=dict(fc="white", ec="none", alpha=0.75, pad=1.2))
            if i == 0:
                ax.set_title(f"t = {s['t']:.1f} Myr", fontsize=10)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8.5)
        cb = fig.colorbar(im, ax=axes[i, :].tolist(), fraction=0.02, pad=0.01)
        cb.ax.tick_params(labelsize=7.5)
    fig.suptitle(title + f"\nGemeinsame Skala über alle Zeiten · Kreise: Aperturen · Box {extent_kpc:g} kpc · "
                         f"linear für |δ| < {lin:.0e} (Rauschniveau)", x=0.01, ha="left", fontsize=11)
    out = _save(fig, outdir / "p5_SNAPSHOTS.png", dpi)
    # Einzelbilder mit derselben globalen Skala
    for j, s in enumerate(snaps):
        fig, axs = plt.subplots(1, len(rows), figsize=(3.5 * len(rows), 3.5))
        for i, (lab, data) in enumerate(rows):
            im = axs[i].imshow(data[j].T, extent=ext, norm=norms[i], cmap=cmaps[i], interpolation="bicubic")
            for R in apertures:
                axs[i].add_patch(Circle((s.get("cx", 0.0), s.get("cy", 0.0)), R, fill=False, ec="0.25", lw=0.6, ls="--"))
            fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
            axs[i].set_title(lab, fontsize=9, loc="left"), axs[i].set_xlabel("x [kpc]"), axs[i].grid(False)
        axs[0].set_ylabel("y [kpc]")
        fig.suptitle(f"{title} · t = {s['t']:.2f} Myr (globale Skala)", x=0.01, ha="left", fontsize=10.5)
        _save(fig, outdir / f"p5_snap_t{s['t']:05.1f}.png", dpi)
    return out
