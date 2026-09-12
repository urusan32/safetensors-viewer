"""Minimal, memory-mapped safetensors reader.

Only numpy is required.  BF16 tensors -- which is what kohya-style LoRAs are
usually stored as -- are widened to float32 on read, since numpy has no native
bfloat16.
"""

from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Any

import numpy as np

# safetensors dtype string -> numpy dtype.  BF16 is read as uint16 and widened
# by hand (see _read_raw).
_DTYPES: dict[str, np.dtype] = {
    "F64": np.dtype("<f8"),
    "F32": np.dtype("<f4"),
    "F16": np.dtype("<f2"),
    "BF16": np.dtype("<u2"),
    "I64": np.dtype("<i8"),
    "I32": np.dtype("<i4"),
    "I16": np.dtype("<i2"),
    "I8": np.dtype("i1"),
    "U8": np.dtype("u1"),
    "BOOL": np.dtype("?"),
}

MAX_HEADER_BYTES = 100 * 1024 * 1024


class SafeTensorsFile:
    """Lazily reads tensors out of a safetensors file via mmap."""

    def __init__(self, path: str):
        self.path = path
        self._file = open(path, "rb")
        try:
            raw = self._file.read(8)
            if len(raw) != 8:
                raise ValueError(f"{path}: too short to be a safetensors file")
            header_len = struct.unpack("<Q", raw)[0]
            if header_len == 0 or header_len > MAX_HEADER_BYTES:
                raise ValueError(f"{path}: implausible header length {header_len}")
            header = json.loads(self._file.read(header_len))
            self._data_start = 8 + header_len
            self.metadata: dict[str, str] = header.pop("__metadata__", {}) or {}
            self.header: dict[str, Any] = header
            self._check_complete()
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise

    def _check_complete(self) -> None:
        """Fail early and clearly on a truncated file."""
        end = max(
            (entry["data_offsets"][1] for entry in self.header.values()), default=0
        )
        actual = os.fstat(self._file.fileno()).st_size
        needed = self._data_start + end
        if actual < needed:
            raise ValueError(
                f"{self.path}: file is truncated -- header describes "
                f"{needed} bytes but the file is {actual}"
            )

    # -- container protocol -------------------------------------------------
    def keys(self) -> list[str]:
        return list(self.header)

    def __contains__(self, name: str) -> bool:
        return name in self.header

    def __len__(self) -> int:
        return len(self.header)

    def __iter__(self):
        return iter(self.header)

    def __enter__(self) -> "SafeTensorsFile":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_mmap", None) is not None:
            self._mmap.close()
            self._mmap = None
        if not self._file.closed:
            self._file.close()

    # -- tensor access ------------------------------------------------------
    def dtype_of(self, name: str) -> str:
        return self.header[name]["dtype"]

    def shape_of(self, name: str) -> tuple[int, ...]:
        return tuple(self.header[name]["shape"])

    def nbytes_of(self, name: str) -> int:
        begin, end = self.header[name]["data_offsets"]
        return end - begin

    def _read_raw(self, name: str) -> np.ndarray:
        try:
            entry = self.header[name]
        except KeyError:
            raise KeyError(f"{self.path}: no tensor named {name!r}") from None
        dtype_str = entry["dtype"]
        if dtype_str not in _DTYPES:
            raise ValueError(f"{name}: unsupported dtype {dtype_str}")
        dtype = _DTYPES[dtype_str]
        shape = tuple(entry["shape"])
        begin, end = entry["data_offsets"]
        count = int(np.prod(shape)) if shape else 1
        if (end - begin) != count * dtype.itemsize:
            raise ValueError(f"{name}: data_offsets do not match shape/dtype")
        flat = np.frombuffer(
            self._mmap, dtype=dtype, count=count, offset=self._data_start + begin
        )
        return flat.reshape(shape)

    def get(self, name: str, dtype: str | np.dtype = "float32") -> np.ndarray:
        """Return tensor `name` as a numpy array (a copy, safe to mutate)."""
        raw = self._read_raw(name)
        if self.header[name]["dtype"] == "BF16":
            # bfloat16 is the top 16 bits of a float32, so widening is a shift.
            widened = np.left_shift(raw.astype(np.uint32), 16)
            out = widened.view(np.float32)
        else:
            out = raw
        return np.ascontiguousarray(out, dtype=np.dtype(dtype))

    def scalar(self, name: str) -> float:
        return float(self.get(name).reshape(-1)[0])

    def total_tensor_bytes(self) -> int:
        return sum(self.nbytes_of(k) for k in self.header)
