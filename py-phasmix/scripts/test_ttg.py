"""Test the TTG phase spiral finder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent, GaussianComponent
from phasmock.mock import MockModel

from phasmix.fit.ttg_fourier import compute_ttg_coefficients, detect_ttg_phase_spiral

if TYPE_CHECKING:
    from optype import numpy as onp


@dataclass(frozen=True)
class MockData:
    """Count map, background, and coordinate grids for the benchmark."""

    x_edges: onp.Array1D[np.float64]
    y_edges: onp.Array1D[np.float64]
    x_mesh: onp.Array2D[np.float64]
    y_mesh: onp.Array2D[np.float64]
    x_centres: onp.Array1D[np.float64]
    y_centres: onp.Array1D[np.float64]
    x_samples: onp.Array1D[np.float64]
    y_samples: onp.Array1D[np.float64]


def _generate_mock() -> MockData:
    signal1 = AlinderComponent(
        alpha=0.5,
        b=0.3,
        c=0.0,
        theta0=0.0,
        scale_factor=1.00,
        rho=2.0,
        winding=-1,
    )
    background_comp = GaussianComponent(x_scale=1, y_scale=1.0, amplitude=1, variance=4.0)

    mock_model = MockModel(
        (signal1,),
        (background_comp,),
    )

    num_x_bins = 100
    num_y_bins = 100
    x_edges = np.linspace(-4.0, 4.0, num_x_bins + 1)
    y_edges = np.linspace(-4.0, 4.0, num_y_bins + 1)

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    x_mesh, y_mesh = np.meshgrid(x_centres, y_centres)

    num_particles: int = 100_000
    particles = mock_model.mock_particles(num_particles, x_edges, y_edges)
    x_samples = particles.x
    y_samples = particles.y

    return MockData(
        x_edges=x_edges,
        x_centres=x_centres,
        x_mesh=x_mesh,
        y_edges=y_edges,
        y_centres=y_centres,
        y_mesh=y_mesh,
        x_samples=x_samples,
        y_samples=y_samples,
    )


def main() -> None:
    true_alpha: float = 0.5
    num_r_bins: int = 20

    data = _generate_mock()
    density, _, _ = np.histogram2d(data.x_samples, data.y_samples, bins=(data.x_edges, data.y_edges))

    r = np.hypot(data.x_samples, data.y_samples)

    r_edges = np.linspace(0.0, 4.0, num_r_bins + 1)
    r_centres = 0.5 * (r_edges[:-1] + r_edges[1:])

    theta = np.arctan2(data.y_samples, data.x_samples)
    coeffs = compute_ttg_coefficients(r, theta, r_edges=r_edges, max_mode=6)
    res = detect_ttg_phase_spiral(coeffs)

    amps = res.amps
    phase = res.phase

    points_x = r_centres * np.cos(phase)
    points_y = r_centres * np.sin(phase)

    amp_criterion = res.amp_criterion
    slope_criterion = res.slope_criterion
    scatter_criterion = res.scatter_criterion

    criteria_display = {False: "FAIL", True: "PASS"}

    title_text: list[str] = [
        rf"$A = {res.strength:.3f}, \,\, \nabla(\phi) = {res.slope:.2f}, \,\, \sqrt{{\mathrm{{SSR}}}} = {res.scatter:.2f}$",
        rf"$|\nabla\phi| \geq {res.slope_threshold:.3f}, \,\, \sigma_s/|s| < {res.slope_err_threshold:.2f}, \,\, "
        rf"\sqrt{{\mathrm{{SSR}}}} < {res.scatter_threshold:.2f}$",
        f"Amp: {criteria_display[amp_criterion]}, Slope: {criteria_display[slope_criterion]}, "
        f"Scatter: {criteria_display[scatter_criterion]}",
        f"Slope error: {res.slope_relative_error:.1%} ({criteria_display[res.slope_error_criterion]})",
        rf"$\alpha = {true_alpha:.2f}, \,\, \tilde{{\alpha}} = 2 A = {(2 * res.strength):.3f}$",
    ]

    fig = plt.figure()
    _ = fig.suptitle("\n".join(title_text))
    axes = fig.add_subplot(131)
    amp_axes = fig.add_subplot(132)
    phase_axes = fig.add_subplot(133)

    _ = axes.pcolormesh(data.x_mesh, data.y_mesh, density.T)
    _ = axes.scatter(points_x, points_y, marker="x", c="white")

    for m in range(1, amps.shape[1]):
        _ = amp_axes.plot(r_centres, amps[:, m], label=f"m={m}: $A = {np.median(amps[:, m]):.3f}$")
    _ = amp_axes.legend()

    _ = phase_axes.plot(
        r_centres,
        res.predicted,
        label=rf"Linear fit: $\phi = {res.slope:.2f}r + {res.intercept:.2f}$",
        ls="--",
        color="tab:blue",
    )
    _ = phase_axes.scatter(r_centres[res.fit_mask], phase[res.fit_mask], label="Used in fit", color="tab:orange")
    _ = phase_axes.scatter(r_centres[~res.fit_mask], phase[~res.fit_mask], label="Excluded from fit", color="grey", marker="x")
    _ = phase_axes.legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
