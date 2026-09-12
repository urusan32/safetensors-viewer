"""Command line interface: `python -m loraview <command> ...`."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Callable

import numpy as np

from . import render
from .lora import LoraFile, LoraModule, common_modules, sort_key

# Metrics usable on a single file's block x role map.
METRICS: dict[str, tuple[Callable[[LoraModule], float], str]] = {
    "fro": (lambda m: m.fro(), "||dW||_F"),
    "rms": (lambda m: m.rms(), "RMS of dW"),
    "absmax": (lambda m: m.stats()["absmax"], "max |dW|"),
    "std": (lambda m: m.stats()["std"], "std of dW"),
    "rank": (lambda m: float(m.rank), "stored rank"),
    "alpha": (lambda m: m.alpha, "alpha"),
    "scale": (lambda m: m.scale, "alpha / rank"),
    "spectral": (lambda m: float(m.spectrum()[0]), "top singular value of dW"),
    "effrank": (lambda m: float(m.effective_rank()), "rank holding 99% of energy"),
}

COMPARE_METRICS = ("relative", "absolute", "cosine", "ratio", "norm_a", "norm_b")


# ------------------------------------------------------------------ output
def _default_backend(show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use("Agg")
        return
    for backend in ("QtAgg", "TkAgg", "GTK3Agg", "MacOSX"):
        try:
            matplotlib.use(backend)
            return
        except Exception:
            continue
    matplotlib.use("Agg")


def _finish(fig, args, default_name: str) -> None:
    import matplotlib.pyplot as plt

    if args.out:
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.out}")
    elif args.show:
        plt.show()
    else:
        path = default_name
        fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {path}")
    plt.close(fig)


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)[:120]


# ----------------------------------------------------------------- commands
def cmd_info(args) -> int:
    with LoraFile(args.file) as lora:
        meta = lora.metadata
        print(f"file      : {lora.path}")
        print(f"size      : {os.path.getsize(lora.path) / 2**20:.1f} MiB")
        print(f"tensors   : {len(lora.file)}")
        print(f"modules   : {len(lora)}")
        blocks, roles, _ = lora.grid()
        if blocks:
            print(f"blocks    : {len(blocks)} ({blocks[0]}..{blocks[-1]})")
        print(f"roles     : {len(roles)}")
        for r in render.role_order(roles):
            print(f"            {r}")
        ranks = sorted({m.rank for m in lora})
        alphas = sorted({round(m.alpha, 4) for m in lora})
        print(f"ranks     : {ranks if len(ranks) <= 12 else f'{min(ranks)}..{max(ranks)} ({len(ranks)} distinct)'}")
        print(f"alphas    : {alphas if len(alphas) <= 12 else f'{min(alphas)}..{max(alphas)} ({len(alphas)} distinct)'}")
        dtypes = sorted({lora.file.dtype_of(k) for k in lora.file.keys()})
        print(f"dtypes    : {', '.join(dtypes)}")
        unmatched = lora.unmatched_keys()
        if unmatched:
            print(f"unmatched : {len(unmatched)} keys not part of a LoRA pair")
        if meta:
            print(f"\nmetadata  : {len(meta)} entries")
            keys = sorted(meta) if args.all_metadata else [
                k for k in sorted(meta) if k in _KEY_METADATA
            ]
            width = max((len(k) for k in keys), default=0)
            for k in keys:
                v = str(meta[k])
                if not args.all_metadata and len(v) > 160:
                    v = v[:157] + "..."
                print(f"  {k:<{width}} : {v}")
            if not args.all_metadata:
                print("  (--all-metadata for the rest)")
    return 0


_KEY_METADATA = {
    "modelspec.architecture", "modelspec.title", "modelspec.date",
    "modelspec.resolution", "modelspec.implementation",
    "ss_base_model_version", "ss_sd_model_name", "ss_network_module",
    "ss_network_dim", "ss_network_alpha", "ss_network_dropout",
    "ss_learning_rate", "ss_optimizer", "ss_lr_scheduler", "ss_steps",
    "ss_num_train_images", "ss_num_epochs", "ss_resolution", "ss_seed",
    "ss_mixed_precision", "ss_training_comment", "ss_output_name",
    "ss_dataset_dirs", "ss_scale_weight_norms", "ss_min_snr_gamma",
}


def cmd_list(args) -> int:
    with LoraFile(args.file) as lora:
        modules = lora.select(args.pattern)
        if not modules:
            print("no modules matched", file=sys.stderr)
            return 1
        rows = []
        for m in modules:
            fro = m.fro()
            row = {
                "name": m.name,
                "out": m.shape[0], "in": m.shape[1],
                "rank": m.rank, "alpha": m.alpha, "scale": m.scale,
                "fro": fro, "rms": m.rms(),
                "spectral": float(m.spectrum()[0]),
                "effrank": m.effective_rank(),
            }
            if args.full:
                row.update(m.stats())
            rows.append(row)
            m.release()

        if args.sort in rows[0]:
            rows.sort(key=lambda r: r[args.sort],
                      reverse=args.sort not in ("name",))
        if args.top:
            rows = rows[: args.top]

        if args.json:
            print(json.dumps(rows, indent=2))
            return 0

        cols = ["name", "out", "in", "rank", "alpha", "scale", "fro", "rms",
                "spectral", "effrank"]
        if args.full:
            cols += ["std", "absmax", "min", "max"]
        widths = {c: max(len(c), max(len(_cell(r[c])) for r in rows)) for c in cols}
        print("  ".join(c.rjust(widths[c]) if c != "name" else c.ljust(widths[c])
                        for c in cols))
        print("  ".join("-" * widths[c] for c in cols))
        for r in rows:
            print("  ".join(
                _cell(r[c]).rjust(widths[c]) if c != "name"
                else _cell(r[c]).ljust(widths[c]) for c in cols))
        print(f"\n{len(rows)} modules"
              f"   total ||dW||_F = {np.sqrt(sum(r['fro'] ** 2 for r in rows)):.6g}")
    return 0


def _cell(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return render._fmt(float(v))


def cmd_overview(args) -> int:
    import matplotlib.pyplot as plt

    with LoraFile(args.file) as lora:
        fn, label = METRICS[args.metric]
        values, blocks, roles = render.grid_values(lora, fn, args.pattern)
        if values.size == 0:
            print("no modules matched", file=sys.stderr)
            return 1
        order = [roles.index(r) for r in render.role_order(roles)]
        values, roles = values[:, order], render.role_order(roles)
        fig, ax = plt.subplots(figsize=(1.15 * len(roles) + 5,
                                        0.32 * len(blocks) + 3),
                               constrained_layout=True)
        render.draw_grid(
            ax, values, blocks, roles,
            title=f"{os.path.basename(lora.path)}\n{label} per LoRA module",
            cbar_label=label, annotate=not args.no_annotate,
        )
        finite = values[np.isfinite(values)]
        ax.set_xlabel(
            f"min {render._fmt(finite.min())}   median {render._fmt(np.median(finite))}"
            f"   max {render._fmt(finite.max())}", fontsize=8)
    _finish(fig, args, f"overview_{_slug(os.path.basename(args.file))}_{args.metric}.png")
    return 0


def cmd_heatmap(args) -> int:
    with LoraFile(args.file) as lora:
        module = lora.resolve(args.module)
        fig = render.figure_module(
            module, max_side=args.max_side, pool_mode=args.pool,
            percentile=args.percentile, what=args.what,
        )
    _finish(fig, args, f"heatmap_{_slug(module.name)}_{args.what}.png")
    return 0


def cmd_hist(args) -> int:
    import matplotlib.pyplot as plt

    with LoraFile(args.file) as lora:
        modules = lora.select(args.pattern)
        if not modules:
            print("no modules matched", file=sys.stderr)
            return 1
        values = render.collect_values(modules, max_elements=args.max_elements)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
        render.draw_histogram(
            axes[0], values, bins=args.bins,
            title=f"dW values, {len(modules)} modules ({values.size:,} sampled)")
        render.draw_spectrum(
            axes[1], [(m.name, m.spectrum()) for m in modules],
            title="singular values of dW, all matched modules")
        fig.suptitle(
            f"{os.path.basename(lora.path)}"
            + (f"   filter: {args.pattern}" if args.pattern else "")
            + f"\nmean {render._fmt(float(values.mean()))}   "
              f"std {render._fmt(float(values.std()))}   "
              f"absmax {render._fmt(float(np.abs(values).max()))}", fontsize=9)
    _finish(fig, args, f"hist_{_slug(os.path.basename(args.file))}.png")
    return 0


def cmd_spectrum(args) -> int:
    import matplotlib.pyplot as plt

    with LoraFile(args.file) as lora:
        modules = lora.select(args.pattern)
        if not modules:
            print("no modules matched", file=sys.stderr)
            return 1
        spectra = [(m.name, m.spectrum()) for m in modules]
        width = max(s.size for _, s in spectra)
        grid = np.full((len(spectra), width), np.nan)
        for i, (_, s) in enumerate(spectra):
            grid[i, : s.size] = s
        fig, axes = plt.subplots(
            1, 2, figsize=(14, max(4.5, 0.13 * len(spectra) + 2)),
            constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
        render.draw_spectrum(axes[0], spectra)
        im = axes[1].imshow(np.log10(np.where(grid > 0, grid, np.nan)),
                            aspect="auto", cmap=render.MAGNITUDE_CMAP,
                            interpolation="nearest")
        axes[1].set_title("log10 singular value per module", fontsize=9)
        axes[1].set_xlabel("singular value index", fontsize=8)
        step = max(1, len(spectra) // 40)
        axes[1].set_yticks(range(0, len(spectra), step))
        axes[1].set_yticklabels([spectra[i][0] for i in range(0, len(spectra), step)],
                                fontsize=5.5)
        cb = fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.015)
        cb.ax.tick_params(labelsize=7)
        fig.suptitle(f"{os.path.basename(lora.path)}   {len(spectra)} modules",
                     fontsize=10)
    _finish(fig, args, f"spectrum_{_slug(os.path.basename(args.file))}.png")
    return 0


def cmd_compare(args) -> int:
    import matplotlib.pyplot as plt

    with LoraFile(args.file_a) as a, LoraFile(args.file_b) as b:
        if args.module:
            ma, mb = a.resolve(args.module), b.resolve(args.module)
            if ma.shape != mb.shape:
                print(f"shape mismatch: {ma.shape} vs {mb.shape}", file=sys.stderr)
                return 1
            fig = render.figure_compare_module(
                ma, mb, max_side=args.max_side, pool_mode=args.pool,
                percentile=args.percentile)
            _finish(fig, args, f"compare_{_slug(ma.name)}.png")
            return 0

        shared = common_modules(a, b, args.pattern)
        if not shared:
            print("no comparable modules (no shared names with matching shapes)",
                  file=sys.stderr)
            return 1
        only_a = sorted(set(a.names(args.pattern)) - set(shared), key=sort_key)
        only_b = sorted(set(b.names(args.pattern)) - set(shared), key=sort_key)

        stats = {}
        for name in shared:
            ma, mb = a[name], b[name]
            na, nb = ma.fro(), mb.fro()
            ip = ma.inner(mb)
            diff = float(np.sqrt(max(na * na + nb * nb - 2 * ip, 0.0)))
            stats[name] = {
                "absolute": diff,
                "relative": diff / na if na else np.nan,
                "cosine": ip / (na * nb) if na and nb else np.nan,
                "ratio": nb / na if na else np.nan,
                "norm_a": na, "norm_b": nb,
            }
            ma.release()
            mb.release()

        if args.json:
            print(json.dumps(stats, indent=2))
            return 0

        _print_compare_table(stats, args)

        _, _, cells_a = a.grid()
        blocks = sorted({k[0] for k in cells_a if cells_a[k].name in stats})
        roles = render.role_order({k[1] for k in cells_a if cells_a[k].name in stats})
        values = np.full((len(blocks), len(roles)), np.nan)
        for (blk, role), m in cells_a.items():
            if m.name in stats and blk in blocks and role in roles:
                values[blocks.index(blk), roles.index(role)] = \
                    stats[m.name][args.metric]

        diverging = args.metric in ("cosine", "ratio")
        fig, ax = plt.subplots(
            figsize=(1.15 * len(roles) + 5, 0.32 * len(blocks) + 3.4),
            constrained_layout=True)
        kwargs = {}
        if diverging:
            centre = 1.0
            spread = np.nanmax(np.abs(values - centre)) or 1.0
            kwargs = {"cmap": render.SIGNED_CMAP, "vmin": centre - spread,
                      "vmax": centre + spread}
        render.draw_grid(
            ax, values, blocks, roles,
            title=f"A: {os.path.basename(a.path)}    B: {os.path.basename(b.path)}\n"
                  f"{_COMPARE_LABEL[args.metric]} per module",
            cbar_label=_COMPARE_LABEL[args.metric],
            annotate=not args.no_annotate, **kwargs)
        note = f"{len(shared)} shared modules"
        if only_a or only_b:
            note += f"   |   only in A: {len(only_a)}   only in B: {len(only_b)}"
        ax.set_xlabel(note, fontsize=8)
    _finish(fig, args,
            f"compare_{_slug(os.path.basename(args.file_a))}_vs_"
            f"{_slug(os.path.basename(args.file_b))}_{args.metric}.png")
    return 0


_COMPARE_LABEL = {
    "relative": "||dW_A - dW_B|| / ||dW_A||",
    "absolute": "||dW_A - dW_B||_F",
    "cosine": "cosine(dW_A, dW_B)",
    "ratio": "||dW_B|| / ||dW_A||",
    "norm_a": "||dW_A||_F",
    "norm_b": "||dW_B||_F",
}


def _print_compare_table(stats: dict, args) -> None:
    tot_a = np.sqrt(sum(s["norm_a"] ** 2 for s in stats.values()))
    tot_b = np.sqrt(sum(s["norm_b"] ** 2 for s in stats.values()))
    tot_d = np.sqrt(sum(s["absolute"] ** 2 for s in stats.values()))
    print(f"{len(stats)} comparable modules")
    print(f"  total ||dW_A||_F   {tot_a:.6g}")
    print(f"  total ||dW_B||_F   {tot_b:.6g}   (ratio {tot_b / tot_a:.4f})")
    print(f"  total ||A - B||_F  {tot_d:.6g}   (relative {tot_d / tot_a:.2%})")
    cos = np.array([s["cosine"] for s in stats.values()])
    print(f"  cosine similarity  min {np.nanmin(cos):.4f}"
          f"   median {np.nanmedian(cos):.4f}   max {np.nanmax(cos):.4f}")
    n = args.top or 10
    ranked = sorted(stats.items(), key=lambda kv: kv[1]["relative"], reverse=True)
    print(f"\n  most changed {n} modules (by relative Frobenius difference)")
    width = max(len(k) for k, _ in ranked[:n])
    print(f"    {'module'.ljust(width)}  {'rel':>8}  {'cos':>8}  "
          f"{'||A||':>10}  {'||B||':>10}")
    for name, s in ranked[:n]:
        print(f"    {name.ljust(width)}  {s['relative']:8.2%}  {s['cosine']:8.4f}  "
              f"{s['norm_a']:10.4g}  {s['norm_b']:10.4g}")
    print()


def cmd_browse(args) -> int:
    from .browse import browse

    browse(args.file, args.file_b, pattern=args.pattern, what=args.what,
           max_side=args.max_side, pool_mode=args.pool,
           percentile=args.percentile)
    return 0


# ------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="loraview",
        description="Visualize the weights inside a LoRA safetensors file. "
                    "Everything is expressed in terms of the effective update "
                    "dW = (alpha/rank) * lora_up @ lora_down, which is what the "
                    "base model actually sees and is comparable across files of "
                    "different rank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def plot_opts(sp, *, module_level=False):
        sp.add_argument("-o", "--out", help="save the figure to this path")
        sp.add_argument("--show", action="store_true",
                        help="open an interactive window instead of saving")
        sp.add_argument("--dpi", type=int, default=140)
        if module_level:
            sp.add_argument("--max-side", type=int, default=1024,
                            help="downsample so neither side exceeds this")
            sp.add_argument("--pool", choices=render.POOL_MODES, default="extreme",
                            help="how to reduce each block when downsampling")
            sp.add_argument("--percentile", type=float, default=99.5,
                            help="percentile of |value| used as the colour limit")

    sp = sub.add_parser("info", help="file, module and metadata summary")
    sp.add_argument("file")
    sp.add_argument("--all-metadata", action="store_true")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("list", help="per-module table of shapes and dW statistics")
    sp.add_argument("file")
    sp.add_argument("-p", "--pattern", help="regex filter on module names")
    sp.add_argument("--sort", default="name",
                    choices=["name", "fro", "rms", "rank", "alpha", "scale",
                             "spectral", "effrank"])
    sp.add_argument("--top", type=int, help="only show the first N rows")
    sp.add_argument("--full", action="store_true",
                    help="also compute element-wise stats (slower)")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_list)

    sp = sub.add_parser("overview",
                        help="heat map of one statistic across all modules")
    sp.add_argument("file")
    sp.add_argument("-p", "--pattern")
    sp.add_argument("-m", "--metric", choices=sorted(METRICS), default="fro")
    sp.add_argument("--no-annotate", action="store_true")
    plot_opts(sp)
    sp.set_defaults(func=cmd_overview)

    sp = sub.add_parser("heatmap", help="heat map of one module's weights")
    sp.add_argument("file")
    sp.add_argument("module", help="module name, or a unique substring/regex")
    sp.add_argument("--what", choices=["delta", "down", "up"], default="delta")
    plot_opts(sp, module_level=True)
    sp.set_defaults(func=cmd_heatmap)

    sp = sub.add_parser("hist", help="value distribution across modules")
    sp.add_argument("file")
    sp.add_argument("-p", "--pattern")
    sp.add_argument("--bins", type=int, default=200)
    sp.add_argument("--max-elements", type=int, default=4_000_000,
                    help="cap on sampled dW elements")
    plot_opts(sp)
    sp.set_defaults(func=cmd_hist)

    sp = sub.add_parser("spectrum", help="singular values of dW per module")
    sp.add_argument("file")
    sp.add_argument("-p", "--pattern")
    plot_opts(sp)
    sp.set_defaults(func=cmd_spectrum)

    sp = sub.add_parser("compare", help="compare two files module by module")
    sp.add_argument("file_a")
    sp.add_argument("file_b")
    sp.add_argument("-p", "--pattern")
    sp.add_argument("--module", help="drill into one module instead of the map")
    sp.add_argument("-m", "--metric", choices=COMPARE_METRICS, default="relative")
    sp.add_argument("--top", type=int, default=10,
                    help="how many most-changed modules to print")
    sp.add_argument("--no-annotate", action="store_true")
    sp.add_argument("--json", action="store_true")
    plot_opts(sp, module_level=True)
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("browse", help="interactive module browser")
    sp.add_argument("file")
    sp.add_argument("file_b", nargs="?",
                    help="second file: browse both and their difference")
    sp.add_argument("-p", "--pattern")
    sp.add_argument("--what", choices=["delta", "down", "up"], default="delta")
    sp.add_argument("--max-side", type=int, default=768)
    sp.add_argument("--pool", choices=render.POOL_MODES, default="extreme")
    sp.add_argument("--percentile", type=float, default=99.5)
    sp.set_defaults(func=cmd_browse)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "browse":
        _default_backend(True)
    elif hasattr(args, "show"):
        _default_backend(args.show and not args.out)
    try:
        return args.func(args)
    except (KeyError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
