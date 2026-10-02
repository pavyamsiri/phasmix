"""Conditional local profile intervals for parameters and derived model phases."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import optimize, stats

from ._backends import OptimizationDiagnostics
from .component import PSpiralComponent
from .model import PSpiralModel
from .param_layout import ParameterLayout

if TYPE_CHECKING:
    from collections.abc import Callable

    from optype import numpy as onp

    from ._backends import PSpiralFitResult
    from .fit import PSpiralFitter

_PARAMETERS = ("alpha", "b", "c", "theta0", "scale_factor", "rho")
type _EndpointStatus = Literal["crossing", "bound", "fixed", "failed"]


@dataclass(frozen=True)
class ProfilePoint:
    """A constrained fit, including failures, evaluated along a profile."""

    value: float
    parameters: onp.Array1D[np.float64]
    lnl: float
    likelihood_ratio: float
    diagnostics: OptimizationDiagnostics


@dataclass(frozen=True)
class ProfileInterval:
    """One component's profile and endpoint status; failed endpoints are NaN."""

    component: int
    parameter: str
    estimate: float
    lower: float
    upper: float
    lower_status: _EndpointStatus
    upper_status: _EndpointStatus
    points: tuple[ProfilePoint, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ProfileLikelihoodResult:
    """Fixed-background local profiles in component-major parameter order.

    ``reference`` is the locally polished fit on the original data. ``intervals``
    and ``model_phase_intervals`` have columns lower, estimate, upper; ``errors``
    contain positive lower/upper offsets at ``confidence_level``, not standard
    errors. Bound endpoints are truncated search intervals, not measured
    likelihood crossings. Inspect ``profiles`` and ``warnings`` before use.
    """

    original_reference: onp.Array1D[np.float64]
    reference: onp.Array1D[np.float64]
    lnl: float
    intervals: onp.Array2D[np.float64]
    model_phase_intervals: onp.Array2D[np.float64]
    model_phase_radius: float
    confidence_level: float
    threshold: float
    baseline_diagnostics: OptimizationDiagnostics
    profiles: tuple[ProfileInterval, ...]
    elapsed_seconds: float
    warnings: tuple[str, ...]

    @property
    def errors(self) -> onp.Array2D[np.float64]:
        """Lower and upper parameter error magnitudes at the requested confidence."""
        return np.column_stack((self.intervals[:, 1] - self.intervals[:, 0], self.intervals[:, 2] - self.intervals[:, 1]))

    @property
    def model_phase_errors(self) -> onp.Array2D[np.float64]:
        """Lower and upper phase error magnitudes in radians."""
        values = self.model_phase_intervals
        return np.column_stack((values[:, 1] - values[:, 0], values[:, 2] - values[:, 1]))

    @property
    def nfev(self) -> int:
        """Total objective evaluations, including baseline polishing."""
        return self.baseline_diagnostics.nfev + sum(point.diagnostics.nfev for item in self.profiles for point in item.points)


def _phase(parameters: onp.Array1D[np.float64], component: int, radius: float) -> float:
    # Neither winding nor flattening enters model_phase(r).
    return PSpiralComponent.from_array(parameters.reshape(-1, 6)[component], winding=1, flattening_strength=0.1).model_phase(
        radius
    )


@dataclass(frozen=True)
class _ProfileProblem:
    objective: Callable[[onp.Array1D[np.float64]], float]
    layout: ParameterLayout
    reference: onp.Array1D[np.float64]
    lower: onp.Array1D[np.float64]
    upper: onp.Array1D[np.float64]
    minimum: float
    threshold: float
    radius: float
    maxiter: int
    rtol: float

    def evaluate(self, index: int, *, derived: bool, value: float, start: onp.Array1D[np.float64]) -> ProfilePoint:
        free = self.layout.free_indices
        if not derived:
            free = free[free != index]
        width = self.upper[free] - self.lower[free]
        template = self.reference.copy()
        if not derived:
            template[index] = value

        def unpack(scaled: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]:
            full = template.copy()
            full[free] = self.lower[free] + width * scaled
            return full

        def objective(scaled: onp.Array1D[np.float64]) -> float:
            return self.objective(unpack(scaled)) - self.minimum

        def constraint(scaled: onp.Array1D[np.float64]) -> float:
            return _phase(unpack(scaled), index, self.radius) - value

        if free.size:
            initial = np.clip((start[free] - self.lower[free]) / width, 0.0, 1.0)
            if derived:
                theta_index = index * 6 + 3
                positions = np.flatnonzero(free == theta_index)
                if positions.size:
                    theta = start[theta_index] + value - _phase(start, index, self.radius)
                    position = int(positions[0])
                    initial[position] = np.clip((theta - self.lower[theta_index]) / width[position], 0.0, 1.0)
                fitted = optimize.minimize(
                    objective,
                    initial,
                    method="SLSQP",
                    bounds=optimize.Bounds(np.zeros_like(initial), np.ones_like(initial)),
                    constraints={"type": "eq", "fun": constraint},
                    options={"maxiter": self.maxiter, "ftol": 1e-9},
                )
            else:
                fitted = optimize.minimize(
                    objective,
                    initial,
                    method="L-BFGS-B",
                    bounds=optimize.Bounds(np.zeros_like(initial), np.ones_like(initial)),
                    options={"maxiter": self.maxiter, "ftol": 1e-12, "gtol": 1e-7},
                )
            extra_nfev = 0
            extra_nit = 0
            message = str(fitted.message)
            if not derived and not fitted.success:
                # L-BFGS-B can stop with a line-search failure very near a minimum.
                extra_nfev, extra_nit = int(fitted.nfev), int(fitted.nit)
                retry_start = np.asarray(fitted.x, dtype=np.float64) if np.isfinite(fitted.fun) else initial
                fitted = optimize.minimize(
                    objective,
                    retry_start,
                    method="SLSQP",
                    bounds=optimize.Bounds(np.zeros_like(initial), np.ones_like(initial)),
                    options={"maxiter": self.maxiter, "ftol": 1e-9},
                )
                message = f"L-BFGS-B: {message}; SLSQP fallback: {fitted.message}"
            parameters = unpack(np.asarray(fitted.x, dtype=np.float64))
            cost = float(fitted.fun) + self.minimum
            success = bool(fitted.success) and np.isfinite(cost)
            if derived:
                success = success and abs(_phase(parameters, index, self.radius) - value) <= 1e-6
            diagnostics = OptimizationDiagnostics(
                success=success, message=message, nfev=int(fitted.nfev) + extra_nfev, nit=int(fitted.nit) + extra_nit
            )
        else:
            parameters = template
            cost = self.objective(parameters)
            diagnostics = OptimizationDiagnostics(
                success=np.isfinite(cost).item(), message="No nuisance parameters.", nfev=1, nit=0
            )
        return ProfilePoint(value, parameters, -cost, 2 * (cost - self.minimum), diagnostics)

    def interval(self, index: int, *, derived: bool) -> ProfileInterval:
        component = index if derived else index // 6
        parameter = "model_phase" if derived else _PARAMETERS[index % 6]
        estimate = _phase(self.reference, index, self.radius) if derived else float(self.reference[index])
        if derived:
            # Phase decreases monotonically with the nonnegative b/c magnitudes.
            lo = self.upper.copy()
            hi = self.lower.copy()
            lo[6 * index + 3] = self.lower[6 * index + 3]
            hi[6 * index + 3] = self.upper[6 * index + 3]
            lower, upper = _phase(lo, index, self.radius), _phase(hi, index, self.radius)
        else:
            lower, upper = float(self.lower[index]), float(self.upper[index])
        if lower == upper:
            return ProfileInterval(component, parameter, estimate, estimate, estimate, "fixed", "fixed", (), ())
        points: dict[float, ProfilePoint] = {}
        warnings: list[str] = []

        def evaluate(value: float) -> float:
            if value == estimate:
                return -self.threshold
            if value not in points:
                successful = [point for point in points.values() if point.diagnostics.success]
                start = min(successful, key=lambda point: abs(point.value - value)).parameters if successful else self.reference
                point = self.evaluate(index, derived=derived, value=value, start=start)
                points[value] = point
            point = points[value]
            if not point.diagnostics.success:
                msg = f"Constrained optimizer failed at {value:g}: {point.diagnostics.message}"
                raise RuntimeError(msg)
            if point.likelihood_ratio < -1e-3:
                msg = "A constrained fit improved the baseline; the reference local solution is inadequate."
                raise RuntimeError(msg)
            return point.likelihood_ratio - self.threshold

        def endpoint(limit: float) -> tuple[float, _EndpointStatus]:
            if limit == estimate:
                return limit, "bound"
            previous = 0.0
            fraction = 1 / 64
            # Bracket outwards before root-finding the first detected crossing.
            while True:
                value = estimate + fraction * (limit - estimate)
                residual = evaluate(value)
                if residual >= 0:
                    root = optimize.brentq(
                        lambda position: evaluate(estimate + position * (limit - estimate)),
                        previous,
                        fraction,
                        xtol=self.rtol,
                        rtol=self.rtol,
                    )
                    return estimate + root * (limit - estimate), "crossing"
                if fraction == 1:
                    return limit, "bound"
                previous, fraction = fraction, min(1.0, fraction * 2)

        endpoints: list[float] = []
        statuses: list[_EndpointStatus] = []
        for side, limit in (("lower", lower), ("upper", upper)):
            try:
                value, status = endpoint(limit)
                if status == "bound":
                    warnings.append(f"{side} interval reaches the search bound without a likelihood crossing.")
            except (RuntimeError, ValueError) as exc:
                value, status = float("nan"), "failed"
                warnings.append(f"{side} interval failed: {exc}")
            endpoints.append(value)
            statuses.append(status)
        if any(point.likelihood_ratio < -1e-3 for point in points.values() if point.diagnostics.success):
            endpoints = [float("nan"), float("nan")]
            statuses = ["failed", "failed"]
        return ProfileInterval(
            component,
            parameter,
            estimate,
            endpoints[0],
            endpoints[1],
            statuses[0],
            statuses[1],
            tuple(points.values()),
            tuple(warnings),
        )


def _validate_options(confidence_level: float, model_phase_radius: float, maxiter: int, workers: int, rtol: float) -> None:
    for name, value in (("maxiter", maxiter), ("workers", workers)):
        if type(value) is not int or value < 1:
            msg = f"{name} must be a positive integer."
            raise ValueError(msg)
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        msg = "confidence_level must be between zero and one."
        raise ValueError(msg)
    if not np.isfinite(model_phase_radius) or model_phase_radius < 0:
        msg = "model_phase_radius must be finite and nonnegative."
        raise ValueError(msg)
    if not np.isfinite(rtol) or not 1e-12 <= rtol < 1:
        msg = "rtol must be finite and in [1e-12, 1)."
        raise ValueError(msg)


def profile_likelihood(
    fitter: PSpiralFitter,
    result: PSpiralFitResult,
    *,
    confidence_level: float = 0.95,
    model_phase_radius: float = 0.5,
    maxiter: int = 500,
    workers: int = 1,
    rtol: float = 1e-5,
) -> ProfileLikelihoodResult:
    """Profile all parameters and each component's model phase on original data.

    Supply the original fitter configuration (Python or Rust). This implementation
    uses SciPy local optimization for both: scaled L-BFGS-B parameter profiles and
    SLSQP phase constraints (also a fallback for failed L-BFGS-B searches).
    Objective evaluation follows the supplied backend, including Python count
    normalization and Rust native model evaluation. It holds the final background, winding, and component
    count fixed, and first polishes the reference on that same objective.
    Full-period theta0 bounds use a continuous branch around the original fit.
    ``workers`` parallelizes independent profiles, not individual optimizer steps.

    The cutoff is chi2.ppf(confidence_level, 1) for twice the score drop. The current
    weighted residual score is not a full count likelihood, so nominal coverage is
    unvalidated. Intervals describe the connected local solution; no global audit
    or background uncertainty is included. Failed endpoints are NaN, with retained
    optimizer diagnostics. A failed baseline raises RuntimeError. ``rtol`` controls
    root precision as a fraction of each search range.
    """
    from .uncertainty import _local_bounds  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    start_time = perf_counter()
    _validate_options(confidence_level, model_phase_radius, maxiter, workers, rtol)
    backend = fitter._backend  # noqa: SLF001  # pyright: ignore[reportPrivateUsage] -- package-internal configuration.
    model = result.final_model
    original = model.parameters.flatten().copy()
    bounds = backend.component_bounds(model.num_components)
    original_layout = ParameterLayout.from_bounds(bounds)
    if not np.allclose(original_layout.unpack(original_layout.pack(original)), original, rtol=1e-10, atol=1e-10):
        msg = "The fitter bounds are incompatible with the reference parameters."
        raise ValueError(msg)
    layout = ParameterLayout.from_bounds(
        tuple(_local_bounds(bound, float(original[6 * i + 3])) for i, bound in enumerate(bounds))
    )
    lower, upper = layout.template.copy(), layout.template.copy()
    lower[layout.free_indices], upper[layout.free_indices] = layout.lower, layout.upper
    mask = backend.fitting_mask(model.z_mesh, model.vz_mesh)
    grids = (result.data, model.background, model.z_mesh, model.vz_mesh, mask)
    if any(grid.ndim != 2 or grid.shape != result.data.shape or not np.all(np.isfinite(grid)) for grid in grids):
        msg = "Profile grids and fitting mask must be finite and share a 2D shape."
        raise ValueError(msg)
    if np.sum(result.data) <= 0 or np.sum(model.background) <= 0:
        msg = "Profile counts and background must have positive totals."
        raise ValueError(msg)
    if np.any(mask < 0) or not np.any(mask > 0) or np.any(result.data < 0) or np.any(model.background < 0):
        msg = "Profile counts, background, and mask must be nonnegative with positive mask weights."
        raise ValueError(msg)

    def objective(parameters: onp.Array1D[np.float64]) -> float:
        candidate = PSpiralModel(
            parameters.reshape(-1, 6),
            model.z_mesh,
            model.vz_mesh,
            model.background,
            winding=model.winding,
            flattening_strength=model.flattening_strength,
        )
        return -backend.model_score(candidate, result.data, mask)

    threshold = float(stats.chi2.ppf(confidence_level, df=1))
    initial_cost = objective(original)
    if not np.isfinite(initial_cost):
        msg = "Reference prediction must have a finite profile objective."
        raise ValueError(msg)
    initial = _ProfileProblem(
        objective, layout, original, lower, upper, initial_cost, threshold, model_phase_radius, maxiter, rtol
    )
    # index=-1 removes no parameters: this is the unconstrained baseline polish.
    baseline = initial.evaluate(-1, derived=False, value=original[-1], start=original)
    if not baseline.diagnostics.success or baseline.lnl < -initial_cost - 1e-6:
        msg = f"Profile baseline optimization failed: {baseline.diagnostics.message}"
        raise RuntimeError(msg)
    problem = _ProfileProblem(
        objective, layout, baseline.parameters, lower, upper, -baseline.lnl, threshold, model_phase_radius, maxiter, rtol
    )
    tasks = [(i, False) for i in range(original.size)] + [(i, True) for i in range(model.num_components)]

    def run(task: tuple[int, bool]) -> ProfileInterval:
        return problem.interval(task[0], derived=task[1])

    if workers == 1:
        profiles = tuple(map(run, tasks))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            profiles = tuple(executor.map(run, tasks))
    intervals = np.array([[item.lower, item.estimate, item.upper] for item in profiles], dtype=np.float64)
    warnings = [
        "Experimental local profiles conditional on the fitted background, component count, and winding.",
        "Chi-square thresholds applied to the weighted residual score; nominal interval coverage is unvalidated.",
    ]
    if abs(initial_cost + result.lnl) > 1e-5:
        warnings.append("Stored fit score differs from the recomputed objective; profiles use the recomputed, polished baseline.")
    warnings.extend(
        f"Component {item.component + 1} {item.parameter}: {warning}" for item in profiles for warning in item.warnings
    )
    if model.num_components == 2 and str(bounds[0]) == str(bounds[1]):
        warnings.append(
            "Exchangeable components retain their local labels; ambiguous assignments can invalidate individual profiles."
        )
    return ProfileLikelihoodResult(
        original,
        baseline.parameters,
        baseline.lnl,
        intervals[: original.size],
        intervals[original.size :],
        model_phase_radius,
        confidence_level,
        threshold,
        baseline.diagnostics,
        profiles,
        perf_counter() - start_time,
        tuple(warnings),
    )
