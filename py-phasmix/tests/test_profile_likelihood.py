"""Numerical and integration tests of conditional profile intervals."""

# ruff: noqa: D103
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import numpy as np
import pytest
from phasmix import profile_likelihood
from phasmix._profile_likelihood import _ProfileProblem  # pyright: ignore[reportPrivateUsage]
from phasmix.bounds import ParameterBounds
from phasmix.fit import FitSuccess, PSpiralFitter
from phasmix.model import PSpiralModel
from phasmix.param_layout import ParameterLayout

if TYPE_CHECKING:
    from optype import numpy as onp


def test_correlated_quadratic_profile_refits_nuisance_parameter() -> None:
    reference = np.array([0.4, 0.05, 0.002, 0.3, 40.0, 0.09])
    layout = ParameterLayout.from_bounds(
        (
            ParameterBounds(
                alpha=(0.0, 1.0),
                b=0.05,
                c=0.002,
                theta0=(-2, 2),
                scale_factor=40,
                rho=0.09,
            ),
        )
    )
    lower, upper = layout.template.copy(), layout.template.copy()
    lower[layout.free_indices], upper[layout.free_indices] = layout.lower, layout.upper
    covariance = np.array([[0.01, 0.016], [0.016, 0.04]])
    precision = np.linalg.inv(covariance)

    def objective(values: onp.Array1D[np.float64]) -> float:
        residual = (values - reference)[[0, 3]]
        return float(0.5 * residual @ precision @ residual)

    problem = _ProfileProblem(objective, layout, reference, lower, upper, 0.0, 1.0, 0.5, 500, 1e-7)
    interval = problem.interval(3, derived=False)
    np.testing.assert_allclose([interval.lower, interval.upper], [0.1, 0.5], atol=1e-6)
    assert interval.lower_status == interval.upper_status == "crossing"
    phase = problem.interval(0, derived=True)
    offset = reference[3] - phase.estimate
    np.testing.assert_allclose([phase.lower + offset, phase.upper + offset], [0.1, 0.5], atol=1e-6)
    assert all(point.diagnostics.success for point in phase.points)
    assert abs(interval.points[0].parameters[0] - reference[0]) > 1e-3


def test_derived_phase_with_fixed_theta0_and_free_b() -> None:
    reference = np.array([0.4, 0.05, 0.0, 0.3, 40.0, 0.09])
    layout = ParameterLayout.from_bounds(
        (
            ParameterBounds(
                alpha=0.4,
                b=(0.03, 0.07),
                c=0,
                theta0=0.3,
                scale_factor=40,
                rho=0.09,
            ),
        )
    )
    lower, upper = layout.template.copy(), layout.template.copy()
    lower[layout.free_indices], upper[layout.free_indices] = layout.lower, layout.upper

    def objective(values: onp.Array1D[np.float64]) -> float:
        return float(0.5 * ((values[1] - 0.05) / 0.005) ** 2)

    problem = _ProfileProblem(objective, layout, reference, lower, upper, 0.0, 1.0, 0.5, 500, 1e-7)
    phase = problem.interval(0, derived=True)
    np.testing.assert_allclose([phase.lower, phase.upper], [0.5 / 0.055 + 0.3, 0.5 / 0.045 + 0.3], atol=1e-5)
    assert phase.lower_status == phase.upper_status == "crossing"
    assert all(abs(0.5 / point.parameters[1] + point.parameters[3] - point.value) < 1e-6 for point in phase.points)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_api_profiles_original_data_without_mutation(backend: Literal["python", "rust"]) -> None:
    z, vz = np.meshgrid(np.linspace(-0.8, 0.8, 14), np.linspace(-35, 35, 15))
    background = 100 * np.exp(-(z**2) - (vz / 40) ** 2)
    parameters = np.array([[0.4, 0.05, 0.002, 3.1, 40.0, 0.09]])
    model = PSpiralModel(parameters, z, vz, background)
    counts = np.random.default_rng(8).poisson(model.prediction()).astype(np.float64)
    bounds = ParameterBounds(alpha=(0, 0.9), b=0.05, c=0.002, theta0=3.1, scale_factor=40, rho=0.09)
    fitter = PSpiralFitter(backend=backend, bounds=bounds, mask_func=lambda z, _vz: np.ones_like(z))
    fit = fitter.fit_spiral_with_background(
        counts,
        background,
        z,
        vz,
        num_components=1,
        winding=1,
        improve_background=False,
        rng=np.random.default_rng(9),
    )
    assert isinstance(fit, FitSuccess)
    original = fit.result.final_model.parameters.copy()
    original_background = fit.result.final_model.background.copy()
    with patch("phasmix._python_backend.optimize.differential_evolution", side_effect=AssertionError("global search")):
        serial = profile_likelihood(fitter, fit.result)
        parallel = profile_likelihood(fitter, fit.result, workers=2)
    np.testing.assert_allclose(serial.intervals, parallel.intervals)
    np.testing.assert_array_equal(original, fit.result.final_model.parameters)
    np.testing.assert_array_equal(original_background, fit.result.final_model.background)
    assert serial.intervals.shape == (6, 3)
    assert serial.errors.shape == (6, 2)
    assert serial.profiles[0].lower_status == serial.profiles[0].upper_status == "crossing"
    np.testing.assert_allclose(serial.errors[1:], 0)
    np.testing.assert_allclose(serial.model_phase_errors, 0)
    assert serial.nfev > 0
    assert serial.elapsed_seconds > 0
    assert (
        serial.lnl
        >= fitter._backend.model_score(  # noqa: SLF001  # pyright: ignore[reportPrivateUsage]
            fit.result.final_model, counts, np.ones_like(counts)
        )
        - 1e-6
    )
    assert serial.threshold == pytest.approx(3.841458820694124)


def test_bound_and_optimizer_failure_are_distinguished() -> None:
    reference = np.array([0.4, 0.05, 0.002, 0.3, 40.0, 0.09])
    layout = ParameterLayout.from_bounds(
        (
            ParameterBounds(
                alpha=(0.3, 0.5),
                b=0.05,
                c=0.002,
                theta0=0.3,
                scale_factor=40,
                rho=0.09,
            ),
        )
    )
    lower, upper = layout.template.copy(), layout.template.copy()
    lower[layout.free_indices], upper[layout.free_indices] = layout.lower, layout.upper

    def objective(values: onp.Array1D[np.float64]) -> float:
        return float(0.5 * (values[0] - 0.4) ** 2)

    problem = _ProfileProblem(objective, layout, reference, lower, upper, 0.0, 1.0, 0.5, 500, 1e-5)
    bounded = problem.interval(0, derived=False)
    assert bounded.lower_status == bounded.upper_status == "bound"
    assert bounded.lower == 0.3
    assert bounded.upper == 0.5
    with patch.object(_ProfileProblem, "evaluate", side_effect=RuntimeError("optimizer failure")):
        failed = problem.interval(0, derived=False)
    assert failed.lower_status == failed.upper_status == "failed"
    assert np.isnan(failed.lower)
    assert np.isnan(failed.upper)
    assert any("optimizer failure" in warning for warning in failed.warnings)


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("confidence_level", 0.0),
        ("confidence_level", 1.0),
        ("confidence_level", np.nan),
        ("model_phase_radius", -0.5),
        ("model_phase_radius", np.inf),
        ("workers", 0),
        ("workers", True),
        ("maxiter", 0),
        ("rtol", 0.0),
    ],
)
def test_invalid_profile_options(option: str, value: float) -> None:
    from phasmix._profile_likelihood import _validate_options  # noqa: PLC0415  # pyright: ignore[reportPrivateUsage]

    options = {"confidence_level": 0.95, "model_phase_radius": 0.5, "maxiter": 500, "workers": 1, "rtol": 1e-5}
    options[option] = value
    with pytest.raises(ValueError, match=option):
        _validate_options(**options)  # pyright: ignore[reportArgumentType] -- intentionally invalid numeric options.


def test_profile_detects_an_inadequate_baseline() -> None:
    reference = np.array([0.4, 0.05, 0.002, 0.3, 40.0, 0.09])
    layout = ParameterLayout.from_bounds(
        (
            ParameterBounds(
                alpha=(0.3, 0.5),
                b=0.05,
                c=0.002,
                theta0=0.3,
                scale_factor=40,
                rho=0.09,
            ),
        )
    )
    lower, upper = layout.template.copy(), layout.template.copy()
    lower[layout.free_indices], upper[layout.free_indices] = layout.lower, layout.upper

    def objective(values: onp.Array1D[np.float64]) -> float:
        return float(0.5 * (values[0] - 0.4) ** 2)

    # A baseline with a deliberately worse score must not yield finite intervals.
    problem = _ProfileProblem(objective, layout, reference, lower, upper, 1.0, 1.0, 0.5, 500, 1e-5)
    profile = problem.interval(0, derived=False)
    assert profile.lower_status == profile.upper_status == "failed"
    assert np.isnan(profile.lower)
    assert np.isnan(profile.upper)
    assert any("baseline" in warning for warning in profile.warnings)


def test_two_component_profiles_keep_component_order_and_fixed_values() -> None:
    from phasmix.fit import FitTerminationReason, PSpiralFitResult  # noqa: PLC0415

    parameters = np.array([[0.4, 0.05, 0.002, 0.3, 40, 0.09], [0.6, 0.07, 0.003, -1, 50, 0.12]])
    z, vz = np.meshgrid(np.linspace(-0.8, 0.8, 14), np.linspace(-35, 35, 15))
    model = PSpiralModel(parameters, z, vz, np.full_like(z, 100.0))
    data = model.prediction()
    bounds = tuple(
        ParameterBounds(
            alpha=(0.1, 0.9),
            b=float(row[1]),
            c=float(row[2]),
            theta0=float(row[3]),
            scale_factor=float(row[4]),
            rho=float(row[5]),
        )
        for row in parameters
    )
    fitter = PSpiralFitter(bounds=bounds)
    result = PSpiralFitResult(
        initial_model=model,
        final_model=model,
        data=data,
        num_iterations=0,
        max_iterations=0,
        lnl=0.0,
        reason=FitTerminationReason.FIXED_BACKGROUND,
    )
    summary = profile_likelihood(fitter, result, model_phase_radius=0.7)
    assert summary.intervals.shape == (12, 3)
    assert summary.model_phase_intervals.shape == (2, 3)
    assert summary.profiles[0].component == 0
    assert summary.profiles[6].component == 1
    assert summary.profiles[12].parameter == "model_phase"
    assert summary.profiles[13].component == 1
    np.testing.assert_array_equal(summary.reference, parameters.flatten())
    np.testing.assert_allclose(summary.model_phase_errors, 0)
    for i, item in enumerate(summary.profiles[:12]):
        if i % 6:
            assert item.lower_status == item.upper_status == "fixed"
    assert np.all(np.isfinite(summary.intervals))
