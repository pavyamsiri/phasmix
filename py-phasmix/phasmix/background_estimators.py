"""Estimators of backgrounds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
from scipy import ndimage, stats

if TYPE_CHECKING:
    from optype import numpy as onp


@runtime_checkable
class BackgroundEstimator(Protocol):
    """The background estimator protocol."""

    def __call__(
        self,
        z: onp.Array1D[np.float64],
        vz: onp.Array1D[np.float64],
        z_edges: onp.Array1D[np.float64],
        vz_edges: onp.Array1D[np.float64],
    ) -> onp.Array2D[np.float64]:
        """Estimate the background given z and vz.

        Parameters
        ----------
        z : Array1D[f64]
            The z coordinate.
        vz : Array1D[f64]
            The Vz velocity.
        z_edges : Array1D[f64]
            The z edges.
        vz_edges : Array1D[f64]
            The vz edges.

        """
        ...


def estimate_kde_background_vz_mirrored(
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
    z_edges: onp.Array1D[np.float64],
    vz_edges: onp.Array1D[np.float64],
) -> onp.Array2D[np.float64]:
    """Generate a symmetric KDE background as the initial background estimate.

    Parameters
    ----------
    z : Array1D[f64]
        The z coordinate of the stars.
    vz : Array1D[f64]
        The Vz velocity of the stars.
    z_edges : Array2D[f64]
        The z edges.
    vz_edges : Array2D[f64]
        The Vz edges.

    Returns
    -------
    background : Array2D[f64]
        Symmetric background in units of counts.

    """
    total_particles = len(z)
    if len(z) != total_particles or len(vz) != total_particles:
        msg = "`z` and `vz` must be equal length arrays."
        raise ValueError(msg)
    if total_particles < 1:
        msg = "There are no particles."
        raise ValueError(msg)

    z_min = z_edges[0]
    z_max = z_edges[-1]
    vz_min = vz_edges[0]
    vz_max = vz_edges[-1]

    num_particles = np.count_nonzero((z >= z_min) & (z <= z_max) & (vz >= vz_min) & (vz <= vz_max))

    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    vz_centres = 0.5 * (vz_edges[:-1] + vz_edges[1:])

    z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)

    if num_particles < 1:
        msg = "There are no particles within the given grid."
        raise ValueError(msg)

    z_mirrored = np.concatenate([z, z])
    vz_mirrored = np.concatenate([vz, -vz])
    kde = stats.gaussian_kde(np.vstack([z_mirrored, vz_mirrored]), bw_method="scott")
    grid_points = np.vstack([z_mesh.ravel(), vz_mesh.ravel()])

    raw_background = kde(grid_points).reshape(z_mesh.shape)

    total = raw_background.sum()

    if not np.isfinite(total) or total < 0:
        msg = "Invalid KDE background."
        raise ValueError(msg)

    if total == 0:
        if num_particles > 0:
            msg = "KDE background has zero density within the grid."
            raise ValueError(msg)
        return np.zeros_like(raw_background)

    return raw_background * num_particles / total


def estimate_gaussian_smooth_background_vz_mirrored(
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
    z_edges: onp.Array1D[np.float64],
    vz_edges: onp.Array1D[np.float64],
) -> onp.Array2D[np.float64]:
    """Generate a symmetric background as the initial background estimate via Gaussian smoothing on a grid.

    Parameters
    ----------
    z : Array1D[f64]
        The z coordinate of the stars.
    vz : Array1D[f64]
        The Vz velocity of the stars.
    z_edges : Array1D[f64]
        The z edges.
    vz_edges : Array1D[f64]
        The vz edges.

    Returns
    -------
    background : Array2D[f64]
        Symmetric background in units of counts.

    """
    total_particles = len(z)
    if len(z) != total_particles or len(vz) != total_particles:
        msg = "`z` and `vz` must be equal length arrays."
        raise ValueError(msg)
    if total_particles < 1:
        msg = "There are no particles."
        raise ValueError(msg)

    dz = np.mean(np.diff(z_edges))
    dvz = np.mean(np.diff(vz_edges))
    z_min = z_edges[0]
    z_max = z_edges[-1]
    vz_min = vz_edges[0]
    vz_max = vz_edges[-1]

    num_particles = np.count_nonzero((z >= z_min) & (z <= z_max) & (vz >= vz_min) & (vz <= vz_max))

    if num_particles < 1:
        msg = "There are no particles within the given grid."
        raise ValueError(msg)

    factor = (2 * num_particles) ** (-1 / 6)

    mean_z = np.mean(z)

    std_z_mirrored = np.sqrt(2 * np.sum((z - mean_z) ** 2) / (2 * num_particles - 1))
    std_vz_mirrored = np.sqrt(2 * np.sum(vz**2) / (2 * num_particles - 1))

    hz = factor * std_z_mirrored / dz
    hvz = factor * std_vz_mirrored / dvz

    density, _, _ = np.histogram2d(vz, z, bins=(vz_edges, z_edges))

    smoothed_density = ndimage.gaussian_filter(density, sigma=(hvz, hz))

    return 0.5 * (smoothed_density + np.flipud(smoothed_density))
