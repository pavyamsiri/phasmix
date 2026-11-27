"""Shared tests for the BFE fitters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from phasmix.fit.tavangar_bfe import compute_bfe_coeffs

if TYPE_CHECKING:
    from typing import Final, Literal


max_azimuthal_mode: Final[int] = 3
max_radial_mode: Final[int] = 3


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    shift=st.floats(-np.pi, np.pi),
    alpha=st.integers(0, 1),
)
def test_angle_shift(particles: list[tuple[float, float]], shift: float, alpha: Literal[0, 1]) -> None:
    """Test that BFE coefficients are shifted by the correct amount under a global phase transformation."""
    actions, angles = np.asarray(particles, dtype=np.float64).T

    original = compute_bfe_coeffs(
        actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )
    rotated = compute_bfe_coeffs(
        actions, angles + shift, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )

    modes = np.arange(max_azimuthal_mode + 1)
    expected = original.coefficients * np.exp(-1j * modes * shift)
    np.testing.assert_allclose(rotated.coefficients, expected, rtol=1e-12, atol=1e-12)


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    scale=st.floats(1e-4, 1e4),
    alpha=st.integers(0, 1),
)
def test_action_scale(particles: list[tuple[float, float]], scale: float, alpha: Literal[0, 1]) -> None:
    """Rescaling actions changes the coefficients by the radial normalization factor."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    scale_exponent: float = 0.5 if alpha == 0 else 1.0

    original = compute_bfe_coeffs(
        actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )
    scaled = compute_bfe_coeffs(
        scale * actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )

    factor: float = np.power(scale, scale_exponent)  # pyright: ignore[reportAny]

    np.testing.assert_allclose(scaled.coefficients * factor, original.coefficients, rtol=1e-12, atol=1e-12)


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    repeats=st.integers(1, 5),
    alpha=st.integers(0, 1),
)
def test_repeats(particles: list[tuple[float, float]], repeats: int, alpha: Literal[0, 1]) -> None:
    """Repeating particles should not change the coefficients as they use the mean."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    repeat_actions = np.tile(actions, repeats)
    repeat_angles = np.tile(angles, repeats)

    original = compute_bfe_coeffs(
        actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )
    repeated = compute_bfe_coeffs(
        repeat_actions, repeat_angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )

    np.testing.assert_allclose(repeated.coefficients, original.coefficients, rtol=1e-12, atol=1e-12)


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    alpha=st.integers(0, 1),
    data=st.data(),
)
def test_joint_permutation(particles: list[tuple[float, float]], alpha: Literal[0, 1], data: st.DataObject) -> None:
    """Coefficients should be invariant under permutation of the particles."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    order = data.draw(st.permutations(range(len(particles))), label="particle order")

    original = compute_bfe_coeffs(
        actions, angles, max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )
    permuted = compute_bfe_coeffs(
        actions[order], angles[order], max_radial_mode=max_radial_mode, max_azimuthal_mode=max_azimuthal_mode, alpha=alpha
    )

    np.testing.assert_allclose(permuted.coefficients, original.coefficients, rtol=1e-12, atol=1e-12)


@given(
    ring_actions=st.lists(st.floats(0.1, 5.0), min_size=1, max_size=5),
    count=st.integers(2, 8),
    offset=st.floats(-np.pi, np.pi),
    alpha=st.integers(0, 1),
)
def test_angular_cancellation(ring_actions: list[float], count: int, offset: float, alpha: Literal[0, 1]) -> None:
    """Angular modes should cancel if particles are uniformly distributed in angle."""
    ring_angles = offset + 2 * np.pi * np.arange(count) / count
    actions = np.repeat(np.asarray(ring_actions, dtype=np.float64), count)
    angles = np.tile(ring_angles, len(ring_actions))

    coeffs = compute_bfe_coeffs(actions, angles, max_radial_mode=3, max_azimuthal_mode=count, alpha=alpha).coefficients

    np.testing.assert_allclose(coeffs[:, 1:count], 0.0, rtol=1e-12, atol=1e-12)
    assert abs(coeffs[0, count]) > 1e-12  # pyright: ignore[reportAny]
