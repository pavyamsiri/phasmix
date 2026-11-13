"""Test the TTG phase spiral finder."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import colors as mplcolors
from matplotlib import pyplot as plt
from phasmock.component import AlinderComponent
from phasmock.recipe import MockRecipe
from rich import print as rich_print
from scipy import ndimage

from phasmix.fit.ttg_fourier import compute_ttg_coefficients, compute_ttg_coefficients_grid, detect_ttg_phase_spiral

if TYPE_CHECKING:
    from optype import numpy as onp


def estimate_scale_factor(
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
) -> float:
    """Estimate the elliptical phase-space scale factor from particles.

    Parameters
    ----------
    z : Array1D[f64]
        The z coordinate.
    vz : Array1D[f64]
        The Vz velocity.

    Returns
    -------
    s : float
        The estimated scale factor.

    """
    return float(np.std(vz) / np.std(z))


def generate_mock() -> tuple[
    onp.Array2D[np.float64],
    onp.Array2D[np.float64],
    onp.Array2D[np.float64],
    onp.Array2D[np.float64],
    onp.Array2D[np.float64],
    onp.Array1D[np.float64],
    onp.Array1D[np.float64],
    float,
    float,
]:
    """Generate the smoke mock and its elliptical phase-space coordinates.

    Returns
    -------
    tuple
        Count map, radius mesh, angle mesh, z mesh, velocity mesh,
        particle z coordinates, particle velocities, outer radial limit,
        and estimated velocity-to-position scale factor. Radii are in kpc,
        velocities in km/s, and angles in radians.

    """
    path = Path(__file__).parent / "../recipes/smoke_recipe.yaml"
    smoke_recipe = MockRecipe.from_yaml(path)
    particles = smoke_recipe.generate()
    z_samples = particles.x
    vz_samples = particles.y

    scale_factor = estimate_scale_factor(z_samples, vz_samples)
    r_mesh = np.hypot(smoke_recipe.x_mesh, smoke_recipe.y_mesh / scale_factor)
    phi_mesh = np.arctan2(smoke_recipe.y_mesh / scale_factor, smoke_recipe.x_mesh)

    density, _, _ = np.histogram2d(vz_samples, z_samples, bins=(smoke_recipe.y_edges, smoke_recipe.x_edges))
    smooth_density = ndimage.gaussian_filter(density, sigma=2)
    smooth_bg = ndimage.gaussian_filter(density, sigma=4)
    contrast = np.zeros_like(density)
    _ = np.divide(smooth_density, smooth_bg, where=smooth_bg > 0, out=contrast)
    contrast -= 1.0
    contrast = density

    # Find the radius which encompasses 95% of stars

    upper: float = 1e-5
    max_trials: int = 100
    fraction: float = 0.0
    for _ in range(max_trials):
        mask = r_mesh <= upper
        fraction = np.sum(density[mask]) / np.sum(density)
        if fraction >= 0.99:
            break
        upper *= 2.0

    return (contrast, r_mesh, phi_mesh, smoke_recipe.x_mesh, smoke_recipe.y_mesh, z_samples, vz_samples, upper, scale_factor)


def main() -> None:
    """Compare measured and Alinder orientations and plot Fourier diagnostics."""
    true_alpha: float = 0.5
    num_r_bins: int = 20

    ps_map, r_mesh, phi_mesh, z_mesh, vz_mesh, z_samples, vz_samples, rmax, scale_factor = generate_mock()
    density = ps_map
    x_mesh = z_mesh
    y_mesh = vz_mesh

    use_particles: bool = True
    if not use_particles:
        coeffs = compute_ttg_coefficients_grid(ps_map, r_mesh, phi_mesh, r_extent=(0.0, rmax), max_mode=6, num_bins=num_r_bins)
    else:
        r_samples = np.hypot(z_samples, vz_samples / scale_factor)
        thetaz_samples = np.arctan2(vz_samples / scale_factor, z_samples)
        coeffs = compute_ttg_coefficients(r_samples, thetaz_samples, r_edges=np.linspace(0, rmax, num_r_bins + 1), max_mode=6)

    r_centres = 0.5 * (coeffs.r_edges[:-1] + coeffs.r_edges[1:])
    res = detect_ttg_phase_spiral(coeffs)
    rich_print(f"m=1 strength = {res.strength:.4f}, m=2 strength = {res.strength_m2:.4f}")
    rich_print(f"orientation = {np.rad2deg(res.orientation)} deg")
    recipe = MockRecipe.from_yaml(Path(__file__).parent / "../recipes/smoke_recipe.yaml")
    for component in recipe.model.signal:
        if isinstance(component, AlinderComponent):
            model_phase = component.winding * component.model_phase(r_test=0.5)
            model_orientation = (model_phase + np.pi) % (2 * np.pi) - np.pi
            phase_difference = (res.orientation - model_orientation + np.pi) % (2 * np.pi) - np.pi
            rich_print(f"Alinder orientation at 500 pc = {np.rad2deg(model_orientation):.2f} deg")
            rich_print(f"Measured - model orientation = {np.rad2deg(phase_difference):.2f} deg")

    recons = coeffs.reconstruct(r_mesh, phi_mesh)
    amps = res.amps
    phase = res.phase

    points_x = r_centres * np.cos(phase)
    points_y = scale_factor * r_centres * np.sin(phase)

    amp_criterion = res.amp_criterion
    slope_criterion = res.slope_criterion

    criteria_display = {False: "FAIL", True: "PASS"}

    title_text: list[str] = [
        rf"$A_1/A_0 = {res.strength:.3f}, \,\, A_2/A_0 = {res.strength_m2:.3f}, \,\, \nabla(\phi) = {res.slope:.2f}$",
        rf"$|\nabla\phi| \geq {res.slope_threshold:.3f}, \,\, \sigma_s/|s| < {res.slope_err_threshold:.2f}$"
        f"Amp: {criteria_display[amp_criterion]}, Slope: {criteria_display[slope_criterion]}, "
        f"Slope error: {res.slope_err:.2f} ({criteria_display[res.slope_error_criterion]})",
        rf"$\alpha = {true_alpha:.2f}, \,\, \tilde{{\alpha}} = 2 A = {(2 * res.strength):.3f}$",
    ]

    norm = mplcolors.LogNorm()
    fig = plt.figure()
    _ = fig.suptitle("\n".join(title_text))
    axes = fig.add_subplot(221)
    recons_axes = fig.add_subplot(223)
    amp_axes = fig.add_subplot(222)
    phase_axes = fig.add_subplot(224)

    _ = axes.pcolormesh(x_mesh, y_mesh, density, norm=norm)
    _ = axes.plot(points_x, points_y, marker="x", c="black")

    _ = recons_axes.pcolormesh(x_mesh, y_mesh, recons, norm=norm)
    _ = recons_axes.plot(points_x, points_y, marker="x", c="black")

    for m in range(1, amps.shape[1]):
        _ = amp_axes.plot(r_centres[res.fit_mask], amps[:, m][res.fit_mask], label=f"m={m}: $A = {np.median(amps[:, m]):.3f}$")
        _ = amp_axes.scatter(r_centres[~res.fit_mask], amps[:, m][~res.fit_mask], marker="x")
    _ = amp_axes.legend()

    _ = phase_axes.plot(
        r_centres,
        res.slope * r_centres + res.intercept,
        label=rf"Linear fit: $\phi = {res.slope:.2f}r + {res.intercept:.2f}$",
        ls="--",
        color="black",
    )
    _ = phase_axes.scatter(r_centres[res.fit_mask], phase[res.fit_mask], color="tab:orange")
    _ = phase_axes.scatter(r_centres[~res.fit_mask], phase[~res.fit_mask], marker="x", color="tab:blue")
    _ = phase_axes.legend()

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
