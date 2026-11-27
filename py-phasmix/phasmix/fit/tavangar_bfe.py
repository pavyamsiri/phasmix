"""Implementation of Tavangar et al. 2026's BFE phase spiral method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import special

if TYPE_CHECKING:
    from typing import Final, Literal

    from optype import numpy as onp

__all__: Final[list[str]] = [
    "BfeCoeffs",
    "compute_bfe_coeffs",
    "compute_orthogonal_bfe_coefficients",
    "compute_tavbfe_coefficients",
]


@dataclass(frozen=True)
class BfeCoeffs:
    """BFE coefficients of the vertical action-angle distribution.

    Attributes
    ----------
    coefficients : Array2D[complex128]
        The BFE coefficients.
    alpha : 0, 1
        The index of the Laguerre radial basis.
    action_scale : float
        The fitted radial scale.
    particle_count : int or None
        The number of fitted particles, required for count estimates.

    """

    coefficients: onp.Array2D[np.complex128]
    alpha: Literal[0, 1]
    action_scale: float
    particle_count: int | None = None

    def strength(self, m: int) -> float:
        """Return the strength of the m-th azimuthal mode.

        Parameters
        ----------
        m : int
            The azimuthal mode.

        Returns
        -------
        strength : float
            The strength of the m-th azimuthal mode.

        """
        if m not in range(self.coefficients.shape[1]):
            return float("nan")
        prefactor = np.sqrt(2) if m > 0 else 1.0
        squares = np.square(np.abs(self.coefficients[:, m]))
        return float(prefactor * np.sqrt(np.sum(squares)))  # pyright: ignore[reportAny]

    def reconstruct(
        self,
        jz: onp.Array2D[np.float64],
        thetaz: onp.Array2D[np.float64],
        *,
        number_counts: bool = False,
    ) -> onp.Array2D[np.float64]:
        """Reconstruct the basis function expansion of the coefficients.

        Parameters
        ----------
        jz : Array2D[f64]
            The Jz mesh.
        thetaz : Array2D[f64]
            The thetaz mesh.
        number_counts : bool
            Set this to return the expansion in terms of number count.

        Returns
        -------
        result : Array2D[f64]
            The probability density if `number_counts` was not set otherwise the expected number counts.

        """
        _check_mesh_shapes(("jz", jz), ("thetaz", thetaz))
        density = self._evaluate(jz, thetaz)
        return self._number_counts(density, jz, thetaz) if number_counts else density

    def reconstruct_cartesian(
        self,
        xz: onp.Array2D[np.float64],
        yz: onp.Array2D[np.float64],
        *,
        number_counts: bool = False,
    ) -> onp.Array2D[np.float64]:
        """Reconstruct the basis function expansion of the coefficients.

        Parameters
        ----------
        xz : Array2D[f64]
            The Xz = sqrt(Jz)cos(thetaz) mesh.
        yz : Array2D[f64]
            The Yz = sqrt(Jz)sin(thetaz) mesh.
        number_counts : bool
            Set this to return the expansion in terms of number count.

        Returns
        -------
        result : Array2D[f64]
            The probability density if `number_counts` was not set otherwise the expected number counts.

        """
        _check_mesh_shapes(("xz", xz), ("yz", yz))
        # Factor of 2.0 is the Jacobian between the (Xz, Yz) plane and the (Jz, thetaz) plane.
        density = 2.0 * self._evaluate(np.square(xz) + np.square(yz), np.arctan2(yz, xz))
        return self._number_counts(density, xz, yz) if number_counts else density

    def _evaluate(self, jz: onp.Array2D[np.float64], thetaz: onp.Array2D[np.float64]) -> onp.Array2D[np.float64]:
        """Evaluate the probability density on the given Jz and thetaz mesh.

        Parameters
        ----------
        jz : Array2D[f64]
            The Jz mesh.
        thetaz : Array2D[f64]
            The thetaz mesh.

        Returns
        -------
        density : Array2D[f64]
            The probability density.

        """
        _check_action_angle_mesh(jz, thetaz)

        x = jz / self.action_scale
        density = np.zeros_like(jz, dtype=np.float64)
        angular = np.zeros_like(jz, dtype=np.float64)

        azimuthal_modes: onp.Array1D[np.complex128]
        for n, azimuthal_modes in enumerate(self.coefficients):
            angular[:] = np.real(azimuthal_modes[0])  # pyright: ignore[reportAny]
            for m in range(1, len(azimuthal_modes)):
                angular += 2.0 * np.real(azimuthal_modes[m] * np.exp(1j * m * thetaz))  # pyright: ignore[reportAny]
            density += _radial_basis(n, self.alpha, x, self.action_scale) * angular / np.sqrt(2.0 * np.pi)  # pyright: ignore[reportAny]

        return density

    def _number_counts(
        self,
        density: onp.Array2D[np.float64],
        first: onp.Array2D[np.float64],
        second: onp.Array2D[np.float64],
    ) -> onp.Array2D[np.float64]:
        """Transform the given density to expected number counts.

        Parameters
        ----------
        density : Array2D[f64]
            The density evaluated at the mesh centres.
        first : Array2D[f64]
            The first coordinate mesh.
        second : Array2D[f64]
            The second coordinate mesh.

        Returns
        -------
        counts : Array2D[f64]
            The expected number counts.

        Raises
        ------
        ValueError
            This method fails if `particle_counts` is not set.

        """
        if self.particle_count is None:
            msg = "Particle count is required for number counts"
            raise ValueError(msg)

        area = _mesh_bin_areas(first, second)

        return density * area * self.particle_count


def _check_mesh_shapes(first: tuple[str, onp.Array2D[np.float64]], second: tuple[str, onp.Array2D[np.float64]]) -> None:
    """Check that the two given arrays are 2D, have the same shape and are non-empty.

    Parameters
    ----------
    first : tuple[str, Array2D[f64]]
        The name of the first array and the array itself.
    second : tuple[str, Array2D[f64]]
        The name of the second array and the array itself.

    Raises
    ------
    ValueError
        If the checks do not pass.

    """
    for name, arr in (first, second):
        if arr.ndim != 2:
            msg = f"{name} must be a 2D array."
            raise ValueError(msg)
        if arr.size == 0:
            msg = f"{name} must be non-empty."
            raise ValueError(msg)

    if first[1].shape != second[1].shape:
        msg = f"{first[0]} and {second[0]} must have the same shape: {first[1].shape} != {second[1].shape}"
        raise ValueError(msg)


def _check_action_angle_mesh(jz: onp.Array2D[np.float64], thetaz: onp.Array2D[np.float64]) -> None:
    """Check that Jz and thetaz meshes are valid.

    Parameters
    ----------
    jz : Array2D[f64]
        The Jz mesh.
    thetaz : Array2D[f64]
        The thetaz mesh.

    Raises
    ------
    ValueError
        If the checks do not pass.

    """
    if np.any(~np.isfinite(jz)):
        msg = "Vertical actions Jz must be finite."
        raise ValueError(msg)

    if np.any(~np.isfinite(thetaz)):
        msg = "Vertical conjugate angles thetaz must be finite."
        raise ValueError(msg)

    if np.any(jz < 0.0):
        msg = "Vertical actions Jz must be non-negative."
        raise ValueError(msg)


# TODO: This function is not correct for non-uniform grids
def _mesh_bin_areas(first: onp.Array2D[np.float64], second: onp.Array2D[np.float64]) -> float:
    """Calculate the area for a 2D bin given their mesh centres.

    Parameters
    ----------
    first : Array2D[f64]
        The first coordinate's mesh.
    second : Array2D[f64]
        The second coordinate's mesh.

    Returns
    -------
    area : float
        The area of a bin.

    """
    if min(first.shape) < 2:
        msg = "Number counts require at least two bin centres along each axis"
        raise ValueError(msg)

    first_is_valid = not (np.all(np.isclose(np.diff(first, axis=0), 0.0)) and np.all(np.isclose(np.diff(first, axis=1), 0.0)))
    second_is_valid = not (np.all(np.isclose(np.diff(second, axis=0), 0.0)) and np.all(np.isclose(np.diff(second, axis=1), 0.0)))
    if first_is_valid and second_is_valid:
        first_width: float = max(abs(first[0, 0] - first[0, 1]), abs(first[0, 0] - first[1, 0]))  # pyright: ignore[reportAny]
        second_width: float = max(abs(second[0, 0] - second[0, 1]), abs(second[0, 0] - second[1, 0]))  # pyright: ignore[reportAny]
        return first_width * second_width

    msg = "Number counts require a rectilinear bin-centre mesh"
    raise ValueError(msg)


def compute_bfe_coeffs(
    jz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    alpha: Literal[0, 1],
    max_radial_mode: int = 19,
    max_azimuthal_mode: int = 2,
    action_scale: float | None = None,
) -> BfeCoeffs:
    """Compute the BFE coefficients for a distribution in the vertical action-angle plane.

    Parameters
    ----------
    jz : Array1D[f64]
        The vertical actions.
    thetaz : Array1D[f64]
        The vertical conjugate angles in radians.
    alpha : 0, 1
        The index of the Laguerre polynomial radial basis.
        0 corresponds to an orthogonal basis while 1 corresponds to the Tavangar et al. 2026 basis.
    max_radial_mode : int
        The maximum radial mode.
    max_azimuthal_mode : int
        The maximum azimuthal mode.
    action_scale : float | None
        The scale of the vertical action Jz which is assumed to have an exponential decay distribution.
        This defaults to the mean of the given Jz if none is given explicitly.

    Returns
    -------
    coeffs : BfeCoeffs
        The computed BFE coefficients.

    """
    num_radial_modes: int = max_radial_mode + 1
    num_azimuthal_modes: int = max_azimuthal_mode + 1
    anm = np.zeros((num_radial_modes, num_azimuthal_modes), dtype=np.complex128)

    scale = action_scale if action_scale is not None else float(np.mean(jz))
    if not np.isfinite(scale) or scale <= 0:
        msg = "Action scale must be finite and positive"
        raise ValueError(msg)
    x = jz / scale

    for n in range(num_radial_modes):
        radial = _radial_basis(n, alpha, x, scale)
        for m in range(num_azimuthal_modes):
            anm[n, m] = np.mean(radial * np.exp(-1j * m * thetaz)) / np.sqrt(2.0 * np.pi)
    return BfeCoeffs(coefficients=anm, alpha=alpha, action_scale=scale, particle_count=jz.size)


def _radial_basis(n: int, alpha: Literal[0, 1], x: onp.Array1D[np.float64], scale: float) -> onp.Array1D[np.float64]:
    """Evaluate the radial basis function with the given parameters and arguments.

    Parameters
    ----------
    n : int
        The degree of the Laguerre polynomial.
    alpha : 0, 1
        The alpha parameter of the Laguerre polynomial.
    x : Array1D[f64]
        The scaled actions `Jz/a`.
    scale : float
        The action scale `a`.

    Returns
    -------
    Array1D[f64]
        The radial basis function evaluated at the given scaled actions.

    """
    prefactor: float = np.sqrt(2 / scale) if alpha == 0 else 1 / (scale * np.sqrt(n + 1))  # pyright: ignore[reportAny]
    return prefactor * np.exp(-x) * special.eval_genlaguerre(n, alpha, 2 * x)


def compute_tavbfe_coefficients(
    jz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    max_radial_mode: int = 19,
    max_azimuthal_mode: int = 2,
    action_scale: float | None = None,
) -> BfeCoeffs:
    """Compute the BFE coefficients for a distribution in the vertical action-angle plane.

    In the basis as published by Tavangar et al. 2026 which uses alpha = 1 Laguerre polynomials as the radial basis.

    Parameters
    ----------
    jz : Array1D[f64]
        The vertical actions.
    thetaz : Array1D[f64]
        The vertical conjugate angles in radians.
    max_radial_mode : int
        The maximum radial mode.
    max_azimuthal_mode : int
        The maximum azimuthal mode.
    action_scale : float | None
        The scale of the vertical action Jz which is assumed to have an exponential decay distribution.
        This defaults to the mean of the given Jz if none is given explicitly.

    Returns
    -------
    coeffs : BfeCoeffs
        The computed BFE coefficients.

    """
    return compute_bfe_coeffs(
        jz, thetaz, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, action_scale=action_scale, alpha=1
    )


def compute_orthogonal_bfe_coefficients(
    jz: onp.Array1D[np.float64],
    thetaz: onp.Array1D[np.float64],
    *,
    max_radial_mode: int = 19,
    max_azimuthal_mode: int = 2,
    action_scale: float | None = None,
) -> BfeCoeffs:
    """Compute the BFE coefficients for a distribution in the vertical action-angle plane.

    Uses a radial basis composed of Laguerre polynomials with alpha = 0 which is orthogonal with respect to dJ dtheta.

    Parameters
    ----------
    jz : Array1D[f64]
        The vertical actions.
    thetaz : Array1D[f64]
        The vertical conjugate angles in radians.
    max_radial_mode : int
        The maximum radial mode.
    max_azimuthal_mode : int
        The maximum azimuthal mode.
    action_scale : float | None
        The scale of the vertical action Jz which is assumed to have an exponential decay distribution.
        This defaults to the mean of the given Jz if none is given explicitly.

    Returns
    -------
    coeffs : BfeCoeffs
        The computed BFE coefficients.

    """
    return compute_bfe_coeffs(
        jz, thetaz, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, action_scale=action_scale, alpha=0
    )
