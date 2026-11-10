from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


@dataclass(frozen=True)
class Grid3D:
    box: float
    n: int

    def __post_init__(self) -> None:
        if not (np.isfinite(self.box) and self.box > 0):
            raise ValueError(f"box must be > 0, got {self.box}")
        if self.n < 8 or self.n % 2 != 0:
            raise ValueError(f"n must be even and >= 8, got {self.n}")

    @property
    def dx(self) -> float:
        return self.box / self.n

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n, self.n, self.n)

    @property
    def cell_volume(self) -> float:
        return self.dx**3

    @property
    def center_index(self) -> int:
        return self.n // 2

    @cached_property
    def x1d(self) -> np.ndarray:
        return (np.arange(self.n) - self.n // 2) * self.dx

    @cached_property
    def mesh_sparse(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.meshgrid(self.x1d, self.x1d, self.x1d, indexing="ij", sparse=True))

    @cached_property
    def r(self) -> np.ndarray:
        x, y, z = self.mesh_sparse
        return np.sqrt(x**2 + y**2 + z**2)

    @cached_property
    def k1d(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.fftfreq(self.n, d=self.dx)

    @cached_property
    def k1d_r(self) -> np.ndarray:
        return 2.0 * np.pi * np.fft.rfftfreq(self.n, d=self.dx)

    @cached_property
    def k_r(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.meshgrid(self.k1d, self.k1d, self.k1d_r, indexing="ij", sparse=True))

    @cached_property
    def k2_r(self) -> np.ndarray:
        kx, ky, kz = self.k_r
        return kx**2 + ky**2 + kz**2

    @cached_property
    def k_full(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return tuple(np.meshgrid(self.k1d, self.k1d, self.k1d, indexing="ij", sparse=True))

    @cached_property
    def k2_full(self) -> np.ndarray:
        kx, ky, kz = self.k_full
        return kx**2 + ky**2 + kz**2

    @cached_property
    def inv_k2_r(self) -> np.ndarray:
        k2 = self.k2_r
        out = np.zeros_like(k2)
        np.divide(1.0, k2, out=out, where=k2 > 0)
        return out

    @cached_property
    def ik_r(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        out = []
        for axis, k in enumerate(self.k_r):
            kk = np.array(k, copy=True)
            if axis < 2:
                idx = [slice(None)] * 3
                idx[axis] = self.n // 2
                kk[tuple(idx)] = 0.0
            else:
                kk[..., -1] = 0.0
            out.append(1j * kk)
        return tuple(out)

    @property
    def k_nyquist(self) -> float:
        return np.pi / self.dx

    def integrate(self, field: np.ndarray) -> float:
        assert field.shape == self.shape, (field.shape, self.shape)
        return float(np.sum(field) * self.cell_volume)
