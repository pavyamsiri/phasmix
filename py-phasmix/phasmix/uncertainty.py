"""Experimental bootstrap and profile-likelihood uncertainty with local refits.

Results condition on the selected winding, component count, and local solution.
Use the original fitter configuration; fitted results do not retain its provenance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._profile_likelihood import ProfileInterval, ProfileLikelihoodResult, ProfilePoint, profile_likelihood
from .background_estimators import BackgroundEstimator, estimate_gaussian_smooth_background_vz_mirrored
from .fit import FitInput, FitSuccess

if TYPE_CHECKING:
    from typing import Final

    from optype import numpy as onp

    from .fit import PSpiralFitter

__all__: list[str] = [
    "BootstrapSamples",
    "ProfileInterval",
    "ProfileLikelihoodResult",
    "ProfilePoint",
    "bootstrap_uncertainty",
    "profile_likelihood",
]


class BootstrapSamples:
    """Bootstrap samples of model parameters.

    Attributes
    ----------
    parameters : Array3D[f64]
        The samples of the model parameters with shape (num_parameters=6, num_components, num_samples).
    model_phases : Array2D[f64]
        The samples of the model phases with shape (num_components, num_samples).

    """

    def __init__(self, *, parameters: onp.Array3D[np.float64], model_phases: onp.Array2D[np.float64]) -> None:
        """Initialize bootstrap samples of parameters.

        Parameters
        ----------
        parameters : Array3D[f64]
            The samples of the model parameters with shape (num_parameters, num_components, num_samples).
        model_phases : Array2D[f64]
            The samples of the model phases with shape (num_components, num_samples).

        """
        num_parameters: int = 6
        num_components: int = parameters.shape[1]
        num_samples: int = parameters.shape[2]
        expected_parameter_shape = (num_parameters, num_components, num_samples)
        if parameters.shape != expected_parameter_shape:
            msg = f"The expected shape of `parameters` is {expected_parameter_shape}"
            raise ValueError(msg)

        expected_model_shape = (num_components, num_samples)
        if model_phases.shape != expected_model_shape:
            msg = f"The expected shape of `model_phases` is {expected_model_shape}"
            raise ValueError(msg)

        self.parameters: Final[onp.Array3D[np.float64]] = parameters
        self.model_phases: Final[onp.Array2D[np.float64]] = model_phases

        # Cached parameter percentiles
        self._medians: onp.Array2D[np.float64] | None = None
        self._lower: onp.Array2D[np.float64] | None = None
        self._upper: onp.Array2D[np.float64] | None = None
        self._errors: onp.Array2D[np.float64] | None = None

        # Cached model phase percentiles
        self._phase_medians: onp.Array2D[np.float64] | None = None
        self._phase_lower: onp.Array2D[np.float64] | None = None
        self._phase_upper: onp.Array2D[np.float64] | None = None
        self._phase_errors: onp.Array2D[np.float64] | None = None

    def errors(self) -> onp.Array2D[np.float64]:
        """Return the estimated standard deviation of the parameters.

        More specifically, it is the half-width of the central 68th quantile.

        Returns
        -------
        errors : Array2D[f64]
            The errors on the parameters in the shape (num_parameters, num_components).

        """
        if self._errors is None:
            lb = self.lower()
            ub = self.upper()
            self._errors = (ub - lb) / 2.0

        return self._errors

    def medians(self) -> onp.Array2D[np.float64]:
        """Return the median of the bootstrap sampled parameters.

        Returns
        -------
        medians : Array2D[f64]
            The medians on the parameters in the shape (num_parameters, num_components).

        """
        if self._medians is None:
            self._medians = np.median(self.parameters, axis=2)
        return self._medians

    def lower(self) -> onp.Array2D[np.float64]:
        """Return the 16th percentile of the bootstrap sampled parameters.

        Returns
        -------
        p16 : Array2D[f64]
            The 16th percentile of the bootstrap sampled parameters in the shape (num_parameters, num_components).

        """
        if self._lower is None:
            self._lower = np.percentile(self.parameters, 16, axis=2)
        return self._lower

    def upper(self) -> onp.Array2D[np.float64]:
        """Return the 84th percentile of the bootstrap sampled parameters.

        Returns
        -------
        p84 : Array2D[f64]
            The 84th percentile of the bootstrap sampled parameters in the shape (num_parameters, num_components).

        """
        if self._upper is None:
            self._upper = np.percentile(self.parameters, 84, axis=2)
        return self._upper

    def phase_errors(self) -> onp.Array2D[np.float64]:
        """Return the estimated standard deviation of the model phases.

        More specifically, it is the half-width of the central 68th quantile.

        Returns
        -------
        errors : Array1D[f64]
            The errors on the model phases in the shape (num_components,).

        """
        if self._phase_errors is None:
            lb = self.phase_lower()
            ub = self.phase_upper()
            self._phase_errors = (ub - lb) / 2.0

        return self._phase_errors

    def phase_medians(self) -> onp.Array2D[np.float64]:
        """Return the median of the bootstrap sampled parameters.

        Returns
        -------
        medians : Array1D[f64]
            The medians of the model phases in the shape (num_components,).

        """
        if self._phase_medians is None:
            self._phase_medians = np.median(self.model_phases, axis=1)
        return self._phase_medians

    def phase_lower(self) -> onp.Array2D[np.float64]:
        """Return the 16th percentile of the bootstrap sampled parameters.

        Returns
        -------
        p16 : Array1D[f64]
            The 16th percentile of the bootstrap sampled model phases in the shape (num_components,).

        """
        if self._phase_lower is None:
            self._phase_lower = np.percentile(self.model_phases, 16, axis=1)
        return self._phase_lower

    def phase_upper(self) -> onp.Array2D[np.float64]:
        """Return the 84th percentile of the bootstrap sampled parameters.

        Returns
        -------
        p16 : Array1D[f64]
            The 84th percentile of the bootstrap sampled model phases in the shape (num_components,).

        """
        if self._phase_upper is None:
            self._phase_upper = np.percentile(self.model_phases, 84, axis=1)
        return self._phase_upper


# TODO(pavyamsiri): How do we couple the fitter and the result together so we don't need
# the user to make sure this is true?
def bootstrap_uncertainty(
    fitter: PSpiralFitter,
    result: FitSuccess,
    z: onp.Array1D[np.float64],
    vz: onp.Array1D[np.float64],
    *,
    z_edges: onp.Array1D[np.float64],
    vz_edges: onp.Array1D[np.float64],
    seed: int | None = None,
    background_estimator: BackgroundEstimator | None = None,
    num_replicates: int = 200,
) -> BootstrapSamples:
    """Perform bootstrap sampling to return model parameter samples.

    Parameters
    ----------
    fitter : PSpiralFitter
        The fitter used and to be used.
    result : FitSuccess
        An already successful fit that was produced by the given fitter.
    z : Array1D[f64]
        The z coordinates of the data that was the input to `result`.
    vz : Array1D[f64]
        The Vz velocities of the data that was the input to `result`.
    z_edges : Array1D[f64]
        The z bin edges used to produce `result`.
    vz_edges : Array1D[f64]
        The Vz bin edges used to produce `result`.
    seed : int | None
        The RNG seed to use or set to `None` to use a non-fixed seed.
    background_estimator : BackgroundEstimator | None
        The background estimator to use if given otherwise the default is Scott's rule Gaussian smoothing.
    num_replicates : int
        The number of replicates/samples.

    Returns
    -------
    samples : BootstrapSamples
        The bootstrap samples.

    """
    total_particles = len(z)
    if len(z) != total_particles or len(vz) != total_particles:
        msg = "`z` and `vz` must be equal length arrays."
        raise ValueError(msg)
    if total_particles < 1:
        msg = "There are no particles."
        raise ValueError(msg)

    num_components = result.result.final_model.num_components
    winding = result.result.final_model.winding
    warm_start = result.result.final_model.to_array()
    z_mesh = result.result.final_model.z_mesh
    vz_mesh = result.result.final_model.vz_mesh

    estimator = background_estimator if background_estimator is not None else estimate_gaussian_smooth_background_vz_mirrored

    rng: np.random.Generator = np.random.default_rng(seed)

    inputs: list[FitInput] = []

    for _ in range(num_replicates):
        bootstrap_indices = rng.integers(0, total_particles, size=total_particles)
        new_z = z[bootstrap_indices]
        new_vz = vz[bootstrap_indices]

        density, _, _ = np.histogram2d(new_z, new_vz, bins=(z_edges, vz_edges))
        density = density.T
        background = estimator(
            new_z,
            new_vz,
            z_edges,
            vz_edges,
        )

        inputs.append(
            FitInput(
                density=density,
                background=background,
                z_mesh=z_mesh,
                vz_mesh=vz_mesh,
                winding=winding,
                num_components=num_components,
                warm_start=warm_start,
                # TODO(pavyamsiri): Get this from the result.
                improve_background=True,
            )
        )

    bootstrap_results = fitter.fit_batch(inputs, progress=True)
    successes = [res for res in bootstrap_results if isinstance(res, FitSuccess)]
    num_successes = len(successes)

    num_parameters = len(warm_start)
    all_parameters = np.zeros((num_parameters, num_successes))
    model_phases = np.zeros((num_components, num_successes))

    for idx, res in enumerate(successes):
        all_parameters[:, idx] = res.result.final_model.to_array()
        for component_idx in range(num_components):
            model_phases[component_idx, idx] = res.result.final_model.components[component_idx].model_phase()

    samples = all_parameters.reshape((6, num_components, num_successes))
    return BootstrapSamples(parameters=samples, model_phases=model_phases)


def round_value_with_error(value: float, error: float, *, num_sig_figs: int = 1) -> tuple[str, str]:
    """Round a value with an associated error to the errors significant figures.

    Parameters
    ----------
    value : float
        The value to round.
    error : float
        The value's error to also round.
    num_sig_figs : int
        The number of significant figures to round the error to.

    Returns
    -------
    rounded_value : float
        The rounded value.
    rounded_error : float
        The rounded error.

    """
    if not np.isfinite(value) or not np.isfinite(error) or error <= 0:
        return f"{value:.6g}", f"{error:.6g}"
    order = np.floor(np.log10(error))
    places = max(int(num_sig_figs - order - 1), 0)
    return (np.format_float_positional(value, precision=places), np.format_float_positional(error, precision=places))
