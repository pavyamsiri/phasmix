"""Module containing the representation of on disk data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.npyio import NpzFile

if TYPE_CHECKING:
    from os import PathLike
    from typing import Final

    from optype import numpy as onp


__all__: Final[list[str]] = ["GridData", "ParticleData", "load_data"]


@dataclass(frozen=True)
class ParticleData:
    """Input data in particle format.

    Attributes
    ----------
    z : Array1D[f64]
        The z coordinate or more generally the horizontal axis coordinate.
    vz : Array1D[f64]
        The Vz velocity or more generally the vertical axis coordinate.

    """

    z: onp.Array1D[np.float64]
    vz: onp.Array1D[np.float64]


@dataclass(frozen=True)
class GridData:
    """Input data in grid format.

    Attributes
    ----------
    density : Array2D[f64]
        The number density map in the shape (num_vz_bins, num_z_bins).
    background : Array2D[f64]
        The initial background in the shape (num_vz_bins, num_z_bins).
    z_mesh : Array2D[f64]
        The z mesh in the shape (num_vz_bins, num_z_bins).
    vz_mesh : Array2D[f64]
        The vz mesh in the shape (num_vz_bins, num_z_bins).

    """

    density: onp.Array2D[np.float64]
    background: onp.Array2D[np.float64]
    z_mesh: onp.Array2D[np.float64]
    vz_mesh: onp.Array2D[np.float64]


type FitData = ParticleData | GridData


def load_data(path: PathLike[str]) -> FitData:
    """Load input data from a .npz file.

    Parameters
    ----------
    path : PathLike
        The path to the data.

    Returns
    -------
    data : ParticleData | GridData
        The data in particle format or grid format.

    """
    arch = np.load(path)  # pyright: ignore[reportAny]
    if not isinstance(arch, NpzFile):
        msg = f"Expected the data at {path} to be a npz file but it is {type(arch)} instead."  # pyright: ignore[reportAny]
        raise TypeError(msg)

    # Particle format
    if _check_file_contains(arch, ("z", "vz")):
        z = _verify_array_is_1d(arch["z"].astype(np.float64))
        vz = _verify_array_is_1d(arch["vz"].astype(np.float64))
        return ParticleData(z=z, vz=vz)

    # Grid format
    if _check_file_contains(arch, ("density", "background", "z_mesh", "vz_mesh")):
        density = _verify_array_is_2d(arch["density"].astype(np.float64))
        background = _verify_array_is_2d(arch["background"].astype(np.float64))
        z_mesh = _verify_array_is_2d(arch["z_mesh"].astype(np.float64))
        vz_mesh = _verify_array_is_2d(arch["vz_mesh"].astype(np.float64))
        return GridData(density=density, background=background, z_mesh=z_mesh, vz_mesh=vz_mesh)
    msg = "Expected the data to be in either `ParticleData` format or `GridData` format."
    raise ValueError(msg)


def _check_file_contains(file: NpzFile, keys: tuple[str, ...]) -> bool:
    contains_all: bool = True
    for key in keys:
        contains_all &= key in file
    return contains_all


def _verify_array_is_1d[SCT: np.generic](arr: onp.ArrayND[SCT]) -> onp.Array1D[SCT]:
    if arr.ndim != 1:
        msg = f"Expected a 1D array but got a {arr.ndim}D array instead."
        raise TypeError(msg)
    return arr


def _verify_array_is_2d[SCT: np.generic](arr: onp.ArrayND[SCT]) -> onp.Array2D[SCT]:
    if arr.ndim != 2:
        msg = f"Expected a 2D array but got a {arr.ndim}D array instead."
        raise TypeError(msg)
    return arr
