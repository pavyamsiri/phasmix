"""Implementation of Tavangar et al. 2026's BFE phase spiral method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp

__all__: Final[list[str]] = [
    "OrthogonalBFECoeffs",
    "TavBFECoeffs",
    "compute_orthogonal_bfe_coefficients",
    "compute_tavbfe_coefficients",
]


@dataclass(frozen=True)
class TavBFECoeffs:
    coefficients: onp.Array2D[np.complex128]

    def strength(self, m: int) -> float:
        return np.sqrt(np.sum(np.square(np.abs(self.coefficients)), axis=0))[m]  # pyright: ignore[reportAny]


def compute_tavbfe_coefficients(
    jz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    max_radial_mode: int = 19,
    max_azimuthal_mode: int = 2,
) -> TavBFECoeffs:

    num_radial_modes: int = max_radial_mode + 1
    num_azimuthal_modes: int = max_azimuthal_mode + 1
    anm = np.zeros((num_radial_modes, num_azimuthal_modes), dtype=np.complex128)

    for n in range(num_radial_modes):
        for m in range(num_azimuthal_modes):
            factor: float = 1 / np.sqrt(np.pi / 2) if m != 0 else 1 / np.sqrt(np.pi)  # pyright: ignore[reportAny]
            anm[n, m] = factor * np.sum(radial_basis(n, jz) * np.exp(-1j * m * thetaz))
    return TavBFECoeffs(coefficients=anm)


def radial_basis(n: int, jz: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
    a = np.mean(jz)
    x = jz / a
    prefactor: float = 1.0 / (a * np.sqrt(n + 1))  # pyright: ignore[reportAny]
    exp_factor = np.exp(-x)
    laguerre_factor = special.eval_genlaguerre(n, 1, 2 * x)
    return prefactor * exp_factor * laguerre_factor


@dataclass(frozen=True)
class OrthogonalBFECoeffs:
    """Order-0 Laguerre expansion of a probability density in dJ dtheta.

    Only nonnegative angular modes are stored, with angular normalization
    exp(i*m*theta)/sqrt(2*pi). These are not Tavangar's coefficients.
    """

    coefficients: onp.Array2D[np.complex128]
    action_scale: float

    def strength(self, m: int) -> float:
        return np.sqrt(np.sum(np.square(np.abs(self.coefficients)), axis=0))[m]  # pyright: ignore[reportAny]

    def reconstruct(
        self,
        jz: onp.ArrayND[np.float64],
        thetaz: onp.ArrayND[np.float64],
    ) -> onp.ArrayND[np.float64]:
        """Evaluate density per dJ dtheta on broadcast-compatible arrays.

        Truncation can produce negative densities; values are not clipped.
        The fitted action scale is reused on every evaluation grid.
        """
        jz, thetaz = np.broadcast_arrays(jz, thetaz)
        if np.any(~np.isfinite(jz)) or np.any(jz < 0) or np.any(~np.isfinite(thetaz)):
            msg = "Evaluation actions must be finite and nonnegative, and angles finite"
            raise ValueError(msg)
        density = np.zeros_like(jz, dtype=np.float64)
        for n, modes in enumerate(self.coefficients):
            angular = np.full(jz.shape, modes[0].real, dtype=np.float64)
            for m in range(1, len(modes)):
                angular += 2 * np.real(modes[m] * np.exp(1j * m * thetaz))
            density += _orthogonal_radial_basis(n, jz, self.action_scale) * angular
        return density / np.sqrt(2 * np.pi)

    def reconstruct_cartesian(
        self,
        x: onp.ArrayND[np.float64],
        y: onp.ArrayND[np.float64],
    ) -> onp.ArrayND[np.float64]:
        """Evaluate density per dx dy for x=sqrt(J) cos(theta), y=sqrt(J) sin(theta)."""
        x, y = np.broadcast_arrays(x, y)
        return 2 * self.reconstruct(x**2 + y**2, np.arctan2(y, x))


def _orthogonal_radial_basis(n: int, jz: onp.ArrayND[np.float64], action_scale: float) -> onp.ArrayND[np.float64]:
    x = jz / action_scale
    return np.sqrt(2 / action_scale) * np.exp(-x) * special.eval_laguerre(n, 2 * x)


def compute_orthogonal_bfe_coefficients(
    jz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    max_radial_mode: int = 19,
    max_azimuthal_mode: int = 2,
    action_scale: float | None = None,
) -> OrthogonalBFECoeffs:
    """Fit an orthogonal Fourier--Laguerre probability-density expansion.

    Project equally weighted particles onto order-0 Laguerre functions,
    orthonormal under dJ on [0, infinity). No Gram solve is required.
    The action scale defaults to the sample mean. Mode limits are inclusive.
    """
    if jz.ndim != 1 or thetaz.shape != jz.shape or jz.size == 0:
        msg = "Actions and angles must be nonempty one-dimensional arrays of equal shape"
        raise ValueError(msg)
    if np.any(~np.isfinite(jz)) or np.any(jz < 0) or np.any(~np.isfinite(thetaz)):
        msg = "Actions must be finite and nonnegative, and angles finite"
        raise ValueError(msg)
    if max_radial_mode < 0 or max_azimuthal_mode < 0:
        msg = "Mode limits must be nonnegative"
        raise ValueError(msg)
    scale = float(np.mean(jz)) if action_scale is None else action_scale
    if not np.isfinite(scale) or scale <= 0:
        msg = "Action scale must be finite and positive"
        raise ValueError(msg)
    coefficients = np.empty((max_radial_mode + 1, max_azimuthal_mode + 1), dtype=np.complex128)
    for n in range(max_radial_mode + 1):
        radial = _orthogonal_radial_basis(n, jz, scale)
        for m in range(max_azimuthal_mode + 1):
            coefficients[n, m] = np.mean(radial * np.exp(-1j * m * thetaz)) / np.sqrt(2 * np.pi)
    return OrthogonalBFECoeffs(coefficients=coefficients, action_scale=scale)
