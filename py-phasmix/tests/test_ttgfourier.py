"""Property-based tests for Tepper-Garcia et al. 2025's BFE method."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from phasmix.fit.ttg_fourier import compute_ttg_coeffs, compute_ttg_coeffs_grid

if TYPE_CHECKING:
    from typing import Final


max_mode: Final[int] = 3


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    shift=st.floats(-np.pi, np.pi),
)
def test_angle_shift(particles: list[tuple[float, float]], shift: float) -> None:
    """Test that Fourier coefficients are shifted by the correct amount under a global phase transformation."""
    actions, angles = np.asarray(particles, dtype=np.float64).T

    r_edges = np.linspace(0, 5.1, 20)
    original = compute_ttg_coeffs(actions, angles, max_mode=max_mode, r_edges=r_edges)
    rotated = compute_ttg_coeffs(actions, angles + shift, max_mode=max_mode, r_edges=r_edges)

    modes = np.arange(max_mode + 1)
    expected = original.coefficients * np.exp(1j * modes * shift)
    np.testing.assert_allclose(rotated.coefficients, expected, rtol=1e-12, atol=1e-12)


@given(
    particles=st.lists(st.tuples(st.floats(0.1, 5.0), st.floats(-np.pi, np.pi)), min_size=1, max_size=8),
    repeats=st.integers(1, 5),
)
def test_repeats(particles: list[tuple[float, float]], repeats: int) -> None:
    """Repeating particles should change the coefficients as they use the sum."""
    actions, angles = np.asarray(particles, dtype=np.float64).T
    repeat_actions = np.tile(actions, repeats)
    repeat_angles = np.tile(angles, repeats)

    r_edges = np.linspace(0, 5.1, 20)
    original = compute_ttg_coeffs(actions, angles, max_mode=max_mode, r_edges=r_edges)
    repeated = compute_ttg_coeffs(repeat_actions, repeat_angles, max_mode=max_mode, r_edges=r_edges)

    np.testing.assert_allclose(repeated.coefficients, repeats * original.coefficients, rtol=1e-12, atol=1e-12)


@given(
    harmonic=st.integers(1, max_mode),
    shift=st.floats(-np.pi, np.pi),
    base=st.integers(3, 100_000),
)
def test_reconstruction(harmonic: int, shift: float, base: int) -> None:
    """Test that reconstructing a grid of counts is accurate to the integer."""
    r_edges = np.linspace(0, 5.1, 20)
    phi_edges = np.linspace(-np.pi, np.pi, 4 * max_mode + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi_centres = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    r_mesh, phi_mesh = np.meshgrid(r_centres, phi_centres)
    counts = np.rint(base + np.cos(harmonic * phi_mesh + shift))

    coeffs = compute_ttg_coeffs_grid(
        counts, r_mesh, phi_mesh, r_extent=(r_edges[0], r_edges[-1]), num_bins=len(r_edges) - 1, max_mode=max_mode
    )
    reconstructed = np.rint(coeffs.reconstruct(r_mesh, phi_mesh))
    np.testing.assert_allclose(counts, reconstructed, rtol=1e-12, atol=1e-12)
