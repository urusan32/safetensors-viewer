"""Grouping of safetensors keys into LoRA modules, plus the linear algebra.

A kohya-style LoRA stores, per patched Linear layer:

    <module>.lora_down.weight   [r, in]
    <module>.lora_up.weight     [out, r]
    <module>.alpha              scalar

and the update actually applied to the base model is

    dW = (alpha / r) * up @ down            shape [out, in]

`dW` is the interesting object: it lives in the base model's weight space, so
it is comparable across files even when they have different ranks (which is
exactly the case for a resized LoRA).  Everything below is built around it.

Materializing dW is expensive (an mlp layer here is 8192x2048), so norms,
inner products and singular values are computed from r-by-r Gram matrices
instead -- exactly, not approximately.  See `inner`, `fro` and `spectrum`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .reader import SafeTensorsFile

DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight", ".lora_down")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".lora_up")

_BLOCK_RE = re.compile(r"^(?P<prefix>.*?)_?blocks?_(?P<index>\d+)_(?P<rest>.*)$")


@dataclass(frozen=True)
class ModuleName:
    """A module name split into a block index and a layer role, for grouping."""

    full: str
    block: int | None
    role: str

    @classmethod
    def parse(cls, name: str) -> "ModuleName":
        m = _BLOCK_RE.match(name)
        if m:
            return cls(name, int(m.group("index")), m.group("rest"))
        return cls(name, None, name)

    def __str__(self) -> str:
        return self.full


class LoraModule:
    """One LoRA-patched layer inside a file."""

    def __init__(self, parent: "LoraFile", name: str, down_key: str, up_key: str,
                 alpha_key: str | None):
        self.parent = parent
        self.name = name
        self.down_key = down_key
        self.up_key = up_key
        self.alpha_key = alpha_key
        self.parsed = ModuleName.parse(name)
        self._cache: dict[str, np.ndarray] = {}

    # -- shape / scaling ----------------------------------------------------
    @property
    def rank(self) -> int:
        return self.parent.file.shape_of(self.down_key)[0]

    @property
    def in_features(self) -> int:
        return int(np.prod(self.parent.file.shape_of(self.down_key)[1:]))

    @property
    def out_features(self) -> int:
        return self.parent.file.shape_of(self.up_key)[0]

    @property
    def shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def alpha(self) -> float:
        if self.alpha_key is None:
            return float(self.rank)
        return self.parent.file.scalar(self.alpha_key)

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    # -- factors ------------------------------------------------------------
    def _factor(self, key: str) -> np.ndarray:
        if key not in self._cache:
            # Conv LoRAs carry trailing spatial dims; flatten to 2-D so the
            # same algebra applies.
            arr = self.parent.file.get(key)
            self._cache[key] = arr.reshape(arr.shape[0], -1)
        return self._cache[key]

    def down(self) -> np.ndarray:
        """[r, in]"""
        return self._factor(self.down_key)

    def up(self) -> np.ndarray:
        """[out, r]"""
        return self._factor(self.up_key)

    def delta(self) -> np.ndarray:
        """The effective weight update, [out, in]. Materializes the matrix."""
        return self.scale * (self.up() @ self.down())

    def release(self) -> None:
        self._cache.clear()

    # -- cheap exact statistics --------------------------------------------
    def inner(self, other: "LoraModule") -> float:
        """Frobenius inner product <dW_self, dW_other>, without forming dW.

        <A,B> = tr(A^T B) = sa*sb * tr((Ua^T Ub) (Db Da^T)), and both Grams
        are only r-by-r.
        """
        if self.shape != other.shape:
            raise ValueError(
                f"{self.name}: shape {self.shape} != {other.shape}, not comparable"
            )
        gu = self.up().T @ other.up()            # [ra, rb]
        gd = other.down() @ self.down().T        # [rb, ra]
        return float(self.scale * other.scale * np.einsum("ij,ji->", gu, gd))

    def fro(self) -> float:
        """||dW||_F, without forming dW."""
        return float(np.sqrt(max(self.inner(self), 0.0)))

    def rms(self) -> float:
        return self.fro() / np.sqrt(self.out_features * self.in_features)

    def spectrum(self) -> np.ndarray:
        """Exact singular values of dW (there are at most `rank` non-zero)."""
        qu, ru = np.linalg.qr(self.up())          # [out,r] [r,r]
        qd, rd = np.linalg.qr(self.down().T)      # [in,r]  [r,r]
        core = self.scale * (ru @ rd.T)
        return np.linalg.svd(core, compute_uv=False)

    def effective_rank(self, energy: float = 0.99) -> int:
        """Number of singular values needed to capture `energy` of ||dW||_F^2."""
        s = self.spectrum() ** 2
        total = s.sum()
        if total <= 0:
            return 0
        return int(np.searchsorted(np.cumsum(s) / total, energy) + 1)

    def stats(self) -> dict[str, float]:
        """Element-wise statistics of dW. Materializes the matrix."""
        d = self.delta()
        return {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "min": float(d.min()),
            "max": float(d.max()),
            "absmax": float(np.abs(d).max()),
            "fro": float(np.linalg.norm(d)),
        }


class LoraFile:
    """A safetensors file viewed as a collection of LoRA modules."""

    def __init__(self, path: str):
        self.file = SafeTensorsFile(path)
        self.path = path
        self.modules: dict[str, LoraModule] = {}
        self._build()

    def _build(self) -> None:
        keys = set(self.file.keys())
        for key in self.file.keys():
            base = _strip(key, DOWN_SUFFIXES)
            if base is None:
                continue
            up_key = next(
                (base + s for s in UP_SUFFIXES if base + s in keys), None
            )
            if up_key is None:
                continue
            alpha_key = base + ".alpha" if base + ".alpha" in keys else None
            self.modules[base] = LoraModule(self, base, key, up_key, alpha_key)

    # -- container protocol -------------------------------------------------
    def __len__(self) -> int:
        return len(self.modules)

    def __iter__(self) -> Iterator[LoraModule]:
        return iter(self.modules.values())

    def __contains__(self, name: str) -> bool:
        return name in self.modules

    def __getitem__(self, name: str) -> LoraModule:
        try:
            return self.modules[name]
        except KeyError:
            raise KeyError(f"{self.path}: no LoRA module named {name!r}") from None

    def __enter__(self) -> "LoraFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.file.close()

    # -- selection ----------------------------------------------------------
    def names(self, pattern: str | None = None) -> list[str]:
        names = sorted(self.modules, key=sort_key)
        if pattern:
            rx = re.compile(pattern)
            names = [n for n in names if rx.search(n)]
        return names

    def select(self, pattern: str | None = None) -> list[LoraModule]:
        return [self.modules[n] for n in self.names(pattern)]

    def resolve(self, name: str) -> LoraModule:
        """Look a module up by exact name, then by unique substring/regex."""
        if name in self.modules:
            return self.modules[name]
        matches = self.names(re.escape(name))
        if not matches:
            matches = self.names(name)
        if len(matches) == 1:
            return self.modules[matches[0]]
        if not matches:
            raise KeyError(f"no module matches {name!r}")
        raise KeyError(
            f"{name!r} is ambiguous, {len(matches)} matches: "
            + ", ".join(matches[:5])
            + (" ..." if len(matches) > 5 else "")
        )

    # -- layout -------------------------------------------------------------
    def blocks(self) -> list[int]:
        return sorted({m.parsed.block for m in self if m.parsed.block is not None})

    def roles(self) -> list[str]:
        return sorted({m.parsed.role for m in self})

    def grid(self) -> tuple[list[int], list[str], dict[tuple[int, str], LoraModule]]:
        """Modules arranged as (blocks x roles), the natural layout for a map."""
        cells = {
            (m.parsed.block, m.parsed.role): m
            for m in self
            if m.parsed.block is not None
        }
        return self.blocks(), self.roles(), cells

    # -- misc ---------------------------------------------------------------
    @property
    def metadata(self) -> dict[str, str]:
        return self.file.metadata

    def unmatched_keys(self) -> list[str]:
        used = set()
        for m in self:
            used.update({m.down_key, m.up_key})
            if m.alpha_key:
                used.add(m.alpha_key)
        return [k for k in self.file.keys() if k not in used]


def _strip(key: str, suffixes: tuple[str, ...]) -> str | None:
    for s in suffixes:
        if key.endswith(s):
            return key[: -len(s)]
    return None


def sort_key(name: str):
    """Sort names so that block_2 comes before block_10."""
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", name)
    )


def common_modules(a: LoraFile, b: LoraFile, pattern: str | None = None) -> list[str]:
    """Names present in both files with matching dW shape."""
    shared = set(a.names(pattern)) & set(b.names(pattern))
    return sorted(
        (n for n in shared if a[n].shape == b[n].shape), key=sort_key
    )
