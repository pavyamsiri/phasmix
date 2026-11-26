"""Shared tests for the BFE fitters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from phasmix.fit.tavangar_bfe import compute_orthogonal_bfe_coefficients, compute_tavbfe_coefficients

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp


max_azimuthal_mode: Final[int] = 3
max_radial_mode: Final[int] = 3


class BfeFitResult(Protocol):
    """The coefficient array exposed by either BFE fit."""

    @property
    def coefficients(self) -> onp.Array2D[np.complex128]:
        """Return the radial-by-azimuthal coefficient array."""
        ...


class BfeFitFunc(Protocol):
    """The fitting interface shared by both BFE implementations."""

    def __call__(
        self,
        jz: onp.Array1D[np.float64],
        thetaz: onp.Array1D[np.float64],
        /,
        *,
        max_radial_mode: int,
        max_azimuthal_mode: int,
    ) -> BfeFitResult:
        """Fit the requested modes to the particle sample."""
        ...


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(compute_tavbfe_coefficients, id="tavbfe"),
        pytest.param(compute_orthogonal_bfe_coefficients, id="orthogonal_bfe"),
    ],
)
@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    shift=st.floats(-np.pi, np.pi),
)
def test_angle_shift(fit: BfeFitFunc, particles: list[tuple[float, float]], shift: float) -> None:
    """Test that BFE coefficients are shifted by the correct amount under a global phase transformation."""
    actions, angles = np.asarray(particles, dtype=np.float64).T

    original = fit(actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)
    rotated = fit(actions, angles + shift, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)

    modes = np.arange(max_azimuthal_mode + 1)
    expected = original.coefficients * np.exp(-1j * modes * shift)
    np.testing.assert_allclose(rotated.coefficients, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("fit", "scale_exponent"),
    [
        pytest.param(compute_tavbfe_coefficients, 1.0, id="tavbfe"),
        pytest.param(compute_orthogonal_bfe_coefficients, 0.5, id="orthogonal_bfe"),
    ],
)
@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    scale=st.floats(1e-4, 1e4),
)
def test_action_scale(fit: BfeFitFunc, scale_exponent: float, particles: list[tuple[float, float]], scale: float) -> None:
    """Rescaling actions changes the coefficients by the radial normalization factor."""
    actions, angles = np.asarray(particles, dtype=np.float64).T

    original = fit(actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)
    scaled = fit(scale * actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)

    factor: float = np.power(scale, scale_exponent)  # pyright: ignore[reportAny]

    np.testing.assert_allclose(scaled.coefficients * factor, original.coefficients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("fit", "test_exponent"),
    [
        pytest.param(compute_tavbfe_coefficients, 1.0, id="tavbfe"),
        pytest.param(compute_orthogonal_bfe_coefficients, 0.0, id="orthogonal_bfe"),
    ],
)
@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    repeats=st.integers(1, 5),
)
def test_repeats(fit: BfeFitFunc, test_exponent: float, particles: list[tuple[float, float]], repeats: int) -> None:
    """Repeating particles changes the coefficients by either `1/k` or `1`."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    repeat_actions = np.tile(actions, repeats)
    repeat_angles = np.tile(angles, repeats)

    original = fit(actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)
    repeated = fit(repeat_actions, repeat_angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)

    factor: float = np.power(repeats, test_exponent)  # pyright: ignore[reportAny]

    np.testing.assert_allclose(repeated.coefficients / factor, original.coefficients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(compute_tavbfe_coefficients, id="tavbfe"),
        pytest.param(compute_orthogonal_bfe_coefficients, id="orthogonal_bfe"),
    ],
)
@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    data=st.data(),
)
def test_joint_permutation(fit: BfeFitFunc, particles: list[tuple[float, float]], data: st.DataObject) -> None:
    """Coefficients should be invariant under permutation of the particles."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    order = data.draw(st.permutations(range(len(particles))), label="particle order")

    original = fit(actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)
    permuted = fit(actions[order], angles[order], max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode)

    np.testing.assert_allclose(permuted.coefficients, original.coefficients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "fit",
    [
        pytest.param(compute_tavbfe_coefficients, id="tavbfe"),
        pytest.param(compute_orthogonal_bfe_coefficients, id="orthogonal_bfe"),
    ],
)
@given(
    ring_actions=st.lists(st.floats(0.1, 5.0), min_size=1, max_size=5),
    count=st.integers(2, 8),
    offset=st.floats(-np.pi, np.pi),
)
def test_angular_cancellation(fit: BfeFitFunc, ring_actions: list[float], count: int, offset: float) -> None:
    """Angular modes should cancel if particles are uniformly distributed in angle."""
    ring_angles = offset + 2 * np.pi * np.arange(count) / count
    actions = np.repeat(np.asarray(ring_actions, dtype=np.float64), count)
    angles = np.tile(ring_angles, len(ring_actions))

    coeffs = fit(actions, angles, max_radial_mode=3, max_azimuthal_mode=count).coefficients

    np.testing.assert_allclose(coeffs[:, 1:count], 0.0, rtol=1e-12, atol=1e-12)
    assert abs(coeffs[0, count]) > 1e-12  # pyright: ignore[reportAny]
