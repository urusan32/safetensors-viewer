"""Plotting helpers: downsampling, colour scaling, and the figures themselves."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .lora import LoraFile, LoraModule, sort_key

SIGNED_CMAP = "RdBu_r"      # white at zero: right for deltas around zero
MAGNITUDE_CMAP = "magma"

POOL_MODES = ("extreme", "mean", "rms")


# ---------------------------------------------------------------- downsample
def pool(mat: np.ndarray, max_side: int = 1024, mode: str = "extreme") -> np.ndarray:
    """Block-reduce `mat` so neither side exceeds `max_side`.

    'extreme' keeps, per block, the element with the largest magnitude
    (sign preserved).  Plain averaging cancels the +/- structure that makes a
    LoRA delta interesting, so it is not the default.
    """
    if mode not in POOL_MODES:
        raise ValueError(f"pool mode must be one of {POOL_MODES}, got {mode!r}")
    h, w = mat.shape
    fh = max(1, math.ceil(h / max_side))
    fw = max(1, math.ceil(w / max_side))
    if fh == 1 and fw == 1:
        return mat
    out_h, out_w = math.ceil(h / fh), math.ceil(w / fw)
    fill = 0.0 if mode == "extreme" else np.nan
    padded = np.full((out_h * fh, out_w * fw), fill, dtype=np.float64)
    padded[:h, :w] = mat
    blocks = padded.reshape(out_h, fh, out_w, fw).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(out_h, out_w, fh * fw)
    if mode == "extreme":
        idx = np.abs(blocks).argmax(axis=-1)[..., None]
        return np.take_along_axis(blocks, idx, axis=-1)[..., 0]
    if mode == "mean":
        return np.nanmean(blocks, axis=-1)
    return np.sqrt(np.nanmean(blocks ** 2, axis=-1))


def symmetric_limit(data: np.ndarray, percentile: float = 99.5) -> float:
    """A robust +/- limit for a diverging colour scale."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    if limit <= 0:
        limit = float(np.abs(finite).max())
    return limit or 1.0


def _fmt(x: float) -> str:
    if x == 0:
        return "0"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e4 else f"{x:.2e}"


# ------------------------------------------------------------------- pieces
def draw_matrix(
    ax: Axes,
    mat: np.ndarray,
    *,
    title: str = "",
    vlim: float | None = None,
    percentile: float = 99.5,
    cmap: str = SIGNED_CMAP,
    max_side: int = 1024,
    pool_mode: str = "extreme",
    colorbar: bool = True,
    cbar_label: str = "weight",
) -> tuple[mpl.image.AxesImage, float]:
    """imshow a (possibly huge) matrix on a symmetric colour scale."""
    small = pool(mat, max_side=max_side, mode=pool_mode)
    if vlim is None:
        vlim = symmetric_limit(small, percentile)
    im = ax.imshow(
        small, cmap=cmap, vmin=-vlim, vmax=vlim,
        aspect="auto", interpolation="nearest", origin="upper",
    )
    ax.set_title(title, fontsize=9)
    xlabel = f"input dim ({mat.shape[1]})"
    if small.shape != mat.shape:
        xlabel += (f"   [pooled to {small.shape[0]}x{small.shape[1]}, "
                   f"{pool_mode}]")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(f"output dim ({mat.shape[0]})", fontsize=8)
    ax.tick_params(labelsize=7)
    if colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(f"{cbar_label} (+/- {_fmt(vlim)})", fontsize=7)
    return im, vlim


def draw_histogram(ax: Axes, values: np.ndarray, *, bins: int = 200,
                   title: str = "value distribution", log: bool = True,
                   label: str | None = None) -> None:
    finite = values[np.isfinite(values)]
    lim = symmetric_limit(finite, 99.9)
    # Values outside the robust range are dropped rather than clipped, so the
    # tails do not pile up into fake spikes at the edges.
    outside = int(np.count_nonzero(np.abs(finite) > lim))
    ax.hist(finite, bins=bins, range=(-lim, lim), histtype="stepfilled",
            alpha=0.55 if label else 0.8, label=label)
    if outside and not label:
        ax.text(0.99, 0.97, f"{outside / finite.size:.2%} beyond +/-{_fmt(lim)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
                color="0.35")
    if log:
        ax.set_yscale("log")
    ax.axvline(0.0, color="0.3", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("value", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-2, 3))


def draw_spectrum(ax: Axes, spectra: Sequence[tuple[str, np.ndarray]], *,
                  title: str = "singular values of dW") -> None:
    for name, s in spectra:
        ax.plot(np.arange(1, len(s) + 1), s, lw=1.0,
                label=name if len(spectra) <= 12 else None)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("index", fontsize=8)
    ax.set_ylabel("singular value", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    if 1 < len(spectra) <= 12:
        ax.legend(fontsize=6)


def draw_grid(
    ax: Axes,
    values: np.ndarray,
    blocks: Sequence[int],
    roles: Sequence[str],
    *,
    title: str,
    cbar_label: str,
    cmap: str = MAGNITUDE_CMAP,
    vmin: float | None = None,
    vmax: float | None = None,
    annotate: bool = True,
) -> None:
    """The block x role map: one cell per LoRA module."""
    im = ax.imshow(values, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(blocks)))
    ax.set_yticklabels([f"block {b}" for b in blocks], fontsize=7)
    ax.set_title(title, fontsize=10)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.tick_params(labelsize=7)
    cb.set_label(cbar_label, fontsize=8)
    if annotate and values.size <= 400:
        finite = values[np.isfinite(values)]
        mid = (finite.min() + finite.max()) / 2 if finite.size else 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                v = values[i, j]
                if not np.isfinite(v):
                    continue
                ax.text(j, i, _fmt(v), ha="center", va="center", fontsize=5.5,
                        color="white" if v < mid else "black")


# ------------------------------------------------------------------ figures
def figure_module(module: LoraModule, *, max_side: int = 1024,
                  pool_mode: str = "extreme", percentile: float = 99.5,
                  what: str = "delta") -> Figure:
    """Heatmap + histogram + spectrum for a single module."""
    import matplotlib.pyplot as plt

    mat = {"delta": module.delta, "down": module.down, "up": module.up}[what]()
    fig = plt.figure(figsize=(13, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 1, 1])
    ax_map = fig.add_subplot(gs[:, 0])
    label = {"delta": "dW = (alpha/r) up @ down", "down": "lora_down",
             "up": "lora_up"}[what]
    draw_matrix(ax_map, mat, title=f"{module.name}\n{label}", max_side=max_side,
                pool_mode=pool_mode, percentile=percentile)

    ax_hist = fig.add_subplot(gs[0, 1])
    draw_histogram(ax_hist, mat.ravel(), title="value distribution")

    ax_spec = fig.add_subplot(gs[0, 2])
    draw_spectrum(ax_spec, [(module.name, module.spectrum())], title="dW spectrum")

    ax_rows = fig.add_subplot(gs[1, 1])
    rn = np.linalg.norm(mat, axis=1)
    ax_rows.plot(rn, np.arange(len(rn)), lw=0.6)
    ax_rows.invert_yaxis()
    ax_rows.set_title("row norm (per output unit)", fontsize=9)
    ax_rows.tick_params(labelsize=7)

    ax_cols = fig.add_subplot(gs[1, 2])
    cn = np.linalg.norm(mat, axis=0)
    ax_cols.plot(cn, lw=0.6)
    ax_cols.set_title("column norm (per input unit)", fontsize=9)
    ax_cols.tick_params(labelsize=7)

    stats = module.stats() if what == "delta" else {
        "mean": float(mat.mean()), "std": float(mat.std()),
        "absmax": float(np.abs(mat).max()), "fro": float(np.linalg.norm(mat)),
    }
    fig.suptitle(
        f"{module.parent.path}   shape {module.shape[0]}x{module.shape[1]}   "
        f"rank {module.rank}   alpha {_fmt(module.alpha)}   scale {_fmt(module.scale)}   "
        f"std {_fmt(stats['std'])}   absmax {_fmt(stats['absmax'])}   "
        f"||.||F {_fmt(stats['fro'])}",
        fontsize=9,
    )
    return fig


def figure_compare_module(a: LoraModule, b: LoraModule, *, max_side: int = 1024,
                          pool_mode: str = "extreme",
                          percentile: float = 99.5) -> Figure:
    """Side-by-side dW for two files plus their difference, on one colour scale."""
    import matplotlib.pyplot as plt

    da, db = a.delta(), b.delta()
    diff = da - db
    shared = max(symmetric_limit(pool(da, max_side, pool_mode), percentile),
                 symmetric_limit(pool(db, max_side, pool_mode), percentile))

    fig = plt.figure(figsize=(15, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.15])
    for col, (mat, title) in enumerate((
        (da, f"A: {a.parent.path}\nrank {a.rank}, alpha {_fmt(a.alpha)}"),
        (db, f"B: {b.parent.path}\nrank {b.rank}, alpha {_fmt(b.alpha)}"),
    )):
        ax = fig.add_subplot(gs[0, col])
        draw_matrix(ax, mat, title=title, vlim=shared, max_side=max_side,
                    pool_mode=pool_mode, cbar_label="dW")
    ax_d = fig.add_subplot(gs[0, 2])
    draw_matrix(ax_d, diff, title="A - B", vlim=shared, max_side=max_side,
                pool_mode=pool_mode, cbar_label="dW difference")

    ax_h = fig.add_subplot(gs[1, 0])
    draw_histogram(ax_h, da.ravel(), title="dW distribution", label="A")
    draw_histogram(ax_h, db.ravel(), label="B")
    draw_histogram(ax_h, diff.ravel(), label="A - B")
    ax_h.legend(fontsize=7)

    ax_s = fig.add_subplot(gs[1, 1])
    draw_spectrum(ax_s, [("A", a.spectrum()), ("B", b.spectrum())],
                  title="singular values")

    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.axis("off")
    na, nb = a.fro(), b.fro()
    dn = float(np.linalg.norm(diff))
    cos = a.inner(b) / (na * nb) if na and nb else float("nan")
    rows = [
        ("||dW_A||_F", _fmt(na)),
        ("||dW_B||_F", _fmt(nb)),
        ("||A - B||_F", _fmt(dn)),
        ("relative change", f"{dn / na:.2%}" if na else "n/a"),
        ("cosine similarity", f"{cos:+.4f}"),
        ("norm ratio B/A", f"{nb / na:.4f}" if na else "n/a"),
        ("rank A -> B", f"{a.rank} -> {b.rank}"),
        ("max |A - B|", _fmt(float(np.abs(diff).max()))),
    ]
    ax_t.table(cellText=rows, colWidths=[0.55, 0.45], loc="center",
               cellLoc="left").auto_set_font_size(False)
    for cell in ax_t.tables[0].get_celld().values():
        cell.set_fontsize(8)
        cell.set_linewidth(0.3)
    ax_t.set_title(f"{a.name}", fontsize=9)
    return fig


def collect_values(modules: Iterable[LoraModule], *, max_elements: int = 4_000_000,
                   seed: int = 0) -> np.ndarray:
    """Pool dW values from many modules, subsampling to stay in memory."""
    modules = list(modules)
    if not modules:
        return np.zeros(0, dtype=np.float32)
    per = max(1, max_elements // len(modules))
    rng = np.random.default_rng(seed)
    chunks = []
    for m in modules:
        d = m.delta().ravel()
        if d.size > per:
            d = d[rng.choice(d.size, per, replace=False)]
        chunks.append(d.astype(np.float32, copy=False))
        m.release()
    return np.concatenate(chunks)


def grid_values(lora: LoraFile, fn, pattern: str | None = None
                ) -> tuple[np.ndarray, list[int], list[str]]:
    """Evaluate `fn(module)` over the block x role grid, NaN where absent."""
    blocks, roles, cells = lora.grid()
    if pattern:
        keep = set(lora.names(pattern))
        cells = {k: m for k, m in cells.items() if m.name in keep}
        blocks = sorted({b for b, _ in cells})
        roles = sorted({r for _, r in cells})
    values = np.full((len(blocks), len(roles)), np.nan)
    for i, b in enumerate(blocks):
        for j, r in enumerate(roles):
            m = cells.get((b, r))
            if m is not None:
                values[i, j] = fn(m)
                m.release()
    return values, blocks, roles


def role_order(roles: Sequence[str]) -> list[str]:
    return sorted(roles, key=sort_key)
