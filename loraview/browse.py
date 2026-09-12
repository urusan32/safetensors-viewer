"""Interactive module browser built on matplotlib widgets.

    left / right arrows   previous / next module
    up / down             jump a whole role (i.e. same layer type, next block)
    d                     cycle dW / lora_down / lora_up
    p                     cycle pooling mode
    s                     save the current view as a PNG
    q                     quit
"""

from __future__ import annotations

import numpy as np

from . import render
from .lora import LoraFile, common_modules

WHATS = ("delta", "down", "up")


class Browser:
    def __init__(self, path_a: str, path_b: str | None = None, *,
                 pattern: str | None = None, what: str = "delta",
                 max_side: int = 768, pool_mode: str = "extreme",
                 percentile: float = 99.5):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        self.a = LoraFile(path_a)
        self.b = LoraFile(path_b) if path_b else None
        if self.b is not None:
            self.names = common_modules(self.a, self.b, pattern)
            if not self.names:
                raise ValueError("the two files share no comparable modules")
        else:
            self.names = self.a.names(pattern)
            if not self.names:
                raise ValueError("no modules matched")
        self.index = 0
        self.what = what
        self.max_side = max_side
        self.pool_mode = pool_mode
        self.percentile = percentile

        ncols = 3 if self.b is not None else 2
        self.fig, self.axes = plt.subplots(
            1, ncols, figsize=(6.0 * ncols, 5.6))
        self.fig.subplots_adjust(bottom=0.22, top=0.84, wspace=0.28)
        self.slider_ax = self.fig.add_axes([0.12, 0.07, 0.76, 0.03])
        self.slider = Slider(self.slider_ax, "module", 0, len(self.names) - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
        self.slider.valtext.set_fontsize(8)
        self.slider_ax.set_xlabel(
            "arrows: prev/next module   up/down: same role, next block   "
            "d: dW/down/up   p: pool mode   s: save   q: quit",
            fontsize=8)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._colorbars = []
        self.draw()

    # -- navigation ---------------------------------------------------------
    def _on_slider(self, value) -> None:
        idx = int(value)
        if idx != self.index:
            self.index = idx
            self.draw()

    def _step(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.names)
        self.slider.set_val(self.index)   # triggers draw

    def _step_role(self, delta: int) -> None:
        role = self.a[self.names[self.index]].parsed.role
        same = [i for i, n in enumerate(self.names)
                if self.a[n].parsed.role == role]
        pos = same.index(self.index) if self.index in same else 0
        self.index = same[(pos + delta) % len(same)]
        self.slider.set_val(self.index)

    def _on_key(self, event) -> None:
        import matplotlib.pyplot as plt

        if event.key in ("right", "n"):
            self._step(1)
        elif event.key in ("left", "N"):
            self._step(-1)
        elif event.key == "down":
            self._step_role(1)
        elif event.key == "up":
            self._step_role(-1)
        elif event.key == "d":
            self.what = WHATS[(WHATS.index(self.what) + 1) % len(WHATS)]
            self.draw()
        elif event.key == "p":
            modes = render.POOL_MODES
            self.pool_mode = modes[(modes.index(self.pool_mode) + 1) % len(modes)]
            self.draw()
        elif event.key == "s":
            path = f"browse_{self.names[self.index]}_{self.what}.png"
            self.fig.savefig(path, dpi=140, bbox_inches="tight")
            print(f"wrote {path}")
        elif event.key == "q":
            plt.close(self.fig)

    # -- drawing ------------------------------------------------------------
    def _matrix(self, lora: LoraFile, name: str) -> np.ndarray:
        m = lora[name]
        return {"delta": m.delta, "down": m.down, "up": m.up}[self.what]()

    def draw(self) -> None:
        for cb in self._colorbars:
            cb.remove()
        self._colorbars.clear()
        for ax in self.axes:
            ax.clear()

        name = self.names[self.index]
        ma = self.a[name]
        mat_a = self._matrix(self.a, name)
        label = {"delta": "dW", "down": "lora_down", "up": "lora_up"}[self.what]

        if self.b is None:
            self._draw_matrix(self.axes[0], mat_a, f"{label}")
            render.draw_histogram(self.axes[1], mat_a.ravel(),
                                  title=f"{label} distribution")
            head = (f"{name}   [{self.index + 1}/{len(self.names)}]\n"
                    f"{self.a.path}   shape {mat_a.shape[0]}x{mat_a.shape[1]}   "
                    f"rank {ma.rank}   alpha {render._fmt(ma.alpha)}   "
                    f"std {render._fmt(float(mat_a.std()))}   "
                    f"absmax {render._fmt(float(np.abs(mat_a).max()))}")
        else:
            mb = self.b[name]
            mat_b = self._matrix(self.b, name)
            if mat_a.shape == mat_b.shape:
                diff = mat_a - mat_b
                vlim = max(
                    render.symmetric_limit(
                        render.pool(mat_a, self.max_side, self.pool_mode),
                        self.percentile),
                    render.symmetric_limit(
                        render.pool(mat_b, self.max_side, self.pool_mode),
                        self.percentile))
                self._draw_matrix(self.axes[0], mat_a, f"A  {label}", vlim=vlim)
                self._draw_matrix(self.axes[1], mat_b, f"B  {label}", vlim=vlim)
                self._draw_matrix(self.axes[2], diff, "A - B", vlim=vlim)
                na, nb = ma.fro(), mb.fro()
                dn = float(np.linalg.norm(diff))
                cos = ma.inner(mb) / (na * nb) if na and nb else float("nan")
                head = (f"{name}   [{self.index + 1}/{len(self.names)}]\n"
                        f"A {self.a.path} (rank {ma.rank})    "
                        f"B {self.b.path} (rank {mb.rank})\n"
                        f"||A||F {render._fmt(na)}   ||B||F {render._fmt(nb)}   "
                        f"||A-B||F {render._fmt(dn)}   "
                        f"relative {dn / na:.2%}   cosine {cos:+.4f}")
            else:
                self._draw_matrix(self.axes[0], mat_a, f"A  {label}")
                self._draw_matrix(self.axes[1], mat_b, f"B  {label}")
                self.axes[2].axis("off")
                self.axes[2].text(0.5, 0.5, "shapes differ\nno difference shown",
                                  ha="center", va="center", fontsize=10)
                head = f"{name}   [{self.index + 1}/{len(self.names)}]"
            mb.release()
        ma.release()

        self.fig.suptitle(head + f"      (pool: {self.pool_mode})", fontsize=9)
        self.fig.canvas.draw_idle()

    def _draw_matrix(self, ax, mat, title, vlim=None) -> None:
        im, _ = render.draw_matrix(
            ax, mat, title=title, vlim=vlim, percentile=self.percentile,
            max_side=self.max_side, pool_mode=self.pool_mode, colorbar=False)
        cb = self.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        self._colorbars.append(cb)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()
        self.a.close()
        if self.b is not None:
            self.b.close()


def browse(path_a: str, path_b: str | None = None, **kwargs) -> None:
    Browser(path_a, path_b, **kwargs).show()
