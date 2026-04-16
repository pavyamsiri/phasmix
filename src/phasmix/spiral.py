from __future__ import annotations

from typing import TYPE_CHECKING

import agama
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mplcolors
from scipy import ndimage
from tqdm import tqdm

if TYPE_CHECKING:
    from optype import numpy as onp


def _get_edges_and_centers_size(
    start: float, end: float, bin_size: float
) -> tuple[onp.Array1D[np.float64], onp.Array1D[np.float64]]:
    edges = np.arange(start, end + bin_size, bin_size)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return (edges, centers)


def generate_multi_bin_spirals_bt21(
    output_file="phase_spiral_bt21.png",
    time_myr=300.0,
    n_samples=200000,
    r_bins=[(7.0, 8.0), (8.0, 9.0)],
):
    """
    Generate phase spirals for multiple radial bins using the model parameters
    from Bland-Hawthorn & Tepper-García (2021) (BT21).

    Each radial bin will contain `n_samples` stars, sampled self-consistently
    from the distribution function.
    """
    dz: float = 0.025
    dvz: float = 2
    max_z: float = 1.2
    max_vz: float = 60
    vz_bins, vz_centres = _get_edges_and_centers_size(-max_vz, max_vz, dvz)
    z_bins, z_centres = _get_edges_and_centers_size(-max_z, max_z, dz)
    z_mesh, vz_mesh = np.meshgrid(z_centres, vz_centres)

    # 1. Set units (Solar mass, kpc, km/s)
    agama.setUnits(mass=1, length=1, velocity=1)

    # 2. Define potential (BT21-like model)
    # Bulge: Hernquist
    pot_bulge = agama.Potential(
        type="Spheroid", mass=1.0e10, scaleRadius=0.6, gamma=1, beta=4
    )  # gamma=1, beta=4 is Hernquist
    # Disk: Miyamoto-Nagai (approximation for the exponential disk in BT21)
    pot_disk = agama.Potential(
        type="MiyamotoNagai", mass=6.0e10, scaleRadius=3.0, scaleHeight=0.3
    )
    # Halo: NFW
    pot_halo = agama.Potential(type="NFW", mass=1.45e12, scaleRadius=15.0)

    pot = agama.Potential(pot_bulge, pot_disk, pot_halo)

    # 3. Define Equilibrium Distribution Function
    # Using a QuasiIsothermal DF for the stellar disk
    df = agama.DistributionFunction(
        type="QuasiIsothermal",
        mass=6.0e10,
        rdisk=3.0,
        hdisk=0.3,
        sigmar0=35.0,  # Slightly higher than before to match standard MW models
        rsigmar=8.0,  # Dispersion scale length from Bennett & Bovy/BT context
        potential=pot,
    )

    # 4. ActionFinder and ActionMapper
    af = agama.ActionFinder(pot, interp=True)
    am = agama.ActionMapper(pot)

    # 5. Create Galaxy Model
    gm = agama.GalaxyModel(pot, df, af)

    time_units = time_myr / 977.79
    delta_vz = 15.0  # km/s kick (Sagittarius-like perturbation effect)

    n_bins = len(r_bins)
    fig, axes = plt.subplots(n_bins, 3, figsize=(18, 5 * n_bins))

    for i, (r_min, r_max) in enumerate(tqdm(r_bins, desc="Radial Bins")):
        # Sample n_samples stars specifically for this radial bin
        needed = n_samples
        bin_samples_list = []
        r_disk = 3.0  # From DF parameters
        # Estimate fraction of stars in this bin to optimize batch size
        fraction = (1 + r_min / r_disk) * np.exp(-r_min / r_disk) - (
            1 + r_max / r_disk
        ) * np.exp(-r_max / r_disk)

        while needed > 0:
            # Sample a batch based on estimated fraction
            batch_size = int(needed / max(fraction, 1e-4) * 1.1) + 1000
            batch_size = min(batch_size, 5_000_000)

            samples_xv, _ = gm.sample(batch_size)
            R = np.sqrt(samples_xv[:, 0] ** 2 + samples_xv[:, 1] ** 2)
            mask = (R >= r_min) & (R < r_max)
            found = samples_xv[mask]

            if len(found) > 0:
                take = min(len(found), needed)
                bin_samples_list.append(found[:take])
                needed -= take

        bin_samples = np.vstack(bin_samples_list)

        # Apply perturbation
        bin_samples_perturbed = bin_samples.copy()
        bin_samples_perturbed[:, 5] += delta_vz

        # Method 1: AA Evolution
        res = af(bin_samples_perturbed, angles=True, frequencies=True)
        actions = res[0]
        angles = res[1]
        frequencies = res[2]

        angles_evolved = (angles + frequencies * time_units) % (2 * np.pi)
        aa_evolved = np.column_stack([actions, angles_evolved])
        xv_aa = am(aa_evolved)

        # Method 2: Orbit Integration
        orbits = agama.orbit(
            potential=pot,
            ic=bin_samples_perturbed,
            time=time_units,
            dtype=object,
            verbose=False,
        )
        xv_orbit = np.array([orb(time_units) for orb in orbits])

        # Plotting
        titles = [
            f"Equilibrium R:[{r_min},{r_max}]",
            "AA Evolution",
            "Orbit Integration",
        ]
        data_z = [bin_samples[:, 2], xv_aa[:, 2], xv_orbit[:, 2]]
        data_vz = [bin_samples[:, 5], xv_aa[:, 5], xv_orbit[:, 5]]

        for j in range(3):
            ax = axes[i, j] if n_bins > 1 else axes[j]
            density, _, _ = np.histogram2d(
                data_vz[j], data_z[j], bins=(vz_bins, z_bins), density=True
            )
            contrast = (
                ndimage.gaussian_filter(density, sigma=2)
                / ndimage.gaussian_filter(density, sigma=4)
                - 1
            )
            img = ax.pcolormesh(
                z_mesh,
                vz_mesh,
                contrast,
                cmap="seismic",
                norm=mplcolors.Normalize(vmin=-1, vmax=1),
            )
            ax.set_xlabel("z [kpc]")
            ax.set_ylabel("Vz [km/s]")
            ax.set_title(f"{titles[j]}: {len(data_vz[j])} stars")
            fig.colorbar(img, ax=ax)

    plt.suptitle(f"Phase Spiral (BT21 Potential) at t={time_myr} Myr", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(output_file, dpi=120)
    print(f"Saved BT21-based multi-bin plot to {output_file}")


if __name__ == "__main__":
    generate_multi_bin_spirals_bt21()
