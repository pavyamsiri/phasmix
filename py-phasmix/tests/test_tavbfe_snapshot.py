"""Snapshot tests for Tavangar et al. 2026's BFE method."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
from phasmock.recipe import MockRecipe

from phasmix.fit.tavangar_bfe import compute_bfe_coeffs

if TYPE_CHECKING:
    from typing import Final, Literal

    from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture


max_azimuthal_mode: Final[int] = 3
max_radial_mode: Final[int] = 3


@pytest.mark.parametrize("alpha", [0, 1])
def test_tavbfe_snapshot(ndarrays_regression: NDArraysRegressionFixture, alpha: Literal[0, 1]) -> None:
    """Compare BFE coefficients and mode strengths against a baseline for each alpha."""
    recipe = MockRecipe.from_yaml(Path(__file__).parent.parent / "recipes/tavbfe_recipe.yaml")
    particles = recipe.generate()
    actions = np.square(particles.x) + np.square(particles.y)
    angles = np.arctan2(particles.y, particles.x)

    coeffs = compute_bfe_coeffs(
        actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )

    ndarrays_regression.check(
        {
            "real": coeffs.coefficients.real,
            "imag": coeffs.coefficients.imag,
            "c0": coeffs.strength(0),
            "c1": coeffs.strength(1),
            "c2": coeffs.strength(2),
        },
        basename=f"bfe_alpha_{alpha}",
        default_tolerance={"rtol": 1e-12, "atol": 1e-12},
    )
