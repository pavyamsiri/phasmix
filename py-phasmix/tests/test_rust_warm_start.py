"""Warm starts reach native optimizers through every fitting path."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from phasmix import _internal
from phasmix._rust_backend import RustFitBackend
from phasmix.fit import FitInput, FitSuccess, ParameterBounds, PSpiralFitter
from phasmix.optimizers import NelderMeadConfig, TikTakConfig


@pytest.mark.parametrize("optimizer", ["tiktak", "differential_evolution", "nelder_mead"])
@pytest.mark.parametrize("num_components", [1, 2])
def test_warm_start_paths(optimizer: Literal["tiktak", "differential_evolution", "nelder_mead"], num_components: int) -> None:
    """Full vectors with fixed parameters work for single, event, and batch fits."""
    parameters = np.tile([0.0, 0.05, 0.0, 0.0, 40.0, 0.09], num_components)
    grid = np.ones((2, 3))
    mesh = np.zeros_like(grid)
    fitter = PSpiralFitter(
        backend="rust",
        optimizer=optimizer,
        max_iterations=1,
        bounds=ParameterBounds(alpha=0.0, b=0.05, c=0.0, theta0=0.0, scale_factor=40.0, rho=0.09),
    )
    single = fitter.fit_spiral_with_background(
        grid,
        grid,
        mesh,
        mesh,
        num_components=num_components,
        winding=1,
        improve_background=False,
        warm_start=parameters,
    )
    assert isinstance(single, FitSuccess)
    np.testing.assert_array_equal(single.result.final_model.to_array(), parameters)
    events = list(
        fitter.fit_spiral_with_background_gen(
            grid,
            grid,
            mesh,
            mesh,
            num_components=num_components,
            winding=1,
            improve_background=False,
            warm_start=parameters,
        )
    )
    assert isinstance(events[-1], FitSuccess)
    item = FitInput(
        density=grid,
        background=grid,
        z_mesh=mesh,
        vz_mesh=mesh,
        num_components=num_components,
        winding=1,
        improve_background=False,
        warm_start=parameters,
    )
    for result in fitter.fit_batch([item, item], workers=2):
        assert isinstance(result, FitSuccess)
        np.testing.assert_array_equal(result.result.final_model.to_array(), parameters)
        assert result.diagnostics.nfev == single.diagnostics.nfev


@pytest.mark.parametrize("point", [[], [0.0] * 12, [float("nan")] * 6, [2.0] * 6])
def test_native_warm_start_validation(point: list[float]) -> None:
    """Invalid guesses fail before native objective evaluation."""
    fitter = _internal.PSpiralFitter(
        max_iterations=1,
        optimizer=TikTakConfig(),
        bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
    )
    grid = np.ones(4)
    with pytest.raises(ValueError, match=r"length|bound"):
        _ = fitter.fit_spiral_with_background(grid, grid, grid, grid, grid, (2, 2), num_components=1, warm_start=point)


def test_native_warm_start_options() -> None:
    """Native callers must specify component count and align batch starts."""
    fitter = _internal.PSpiralFitter(
        max_iterations=1,
        optimizer=TikTakConfig(),
        bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
    )
    grid = np.ones(4)
    parameters = [0.0, 0.05, 0.0, 0.0, 40.0, 0.09]
    with pytest.raises(ValueError, match="component count"):
        _ = fitter.fit_spiral_with_background(grid, grid, grid, grid, grid, (2, 2), warm_start=parameters)
    with pytest.raises(ValueError, match="number of inputs"):
        _ = fitter.fit_batch([], warm_starts=[parameters])


def test_local_optimizer_requires_start_for_each_fit_path() -> None:
    """Local fits cannot silently choose a global or arbitrary starting point."""
    fitter = PSpiralFitter(backend="rust", optimizer="nelder_mead")
    grid = np.ones((2, 2))
    with pytest.raises(ValueError, match="requires a warm_start"):
        _ = fitter.fit_spiral_with_background(grid, grid, grid, grid, num_components=1)
    with pytest.raises(ValueError, match="requires a warm_start"):
        _ = list(fitter.fit_spiral_with_background_gen(grid, grid, grid, grid, num_components=1))
    item = FitInput(density=grid, background=grid, z_mesh=grid, vz_mesh=grid, num_components=1)
    with pytest.raises(ValueError, match="requires a warm_start"):
        _ = fitter.fit_batch([item])


@pytest.mark.parametrize("maxiter", [0, -1])
def test_local_iteration_limit_validation(maxiter: int) -> None:
    """The optimizer budget must be a positive integer."""
    with pytest.raises((ValueError, TypeError), match=r"positive|max_iter"):
        _ = PSpiralFitter(backend="rust", optimizer=NelderMeadConfig(max_iter=maxiter))


def test_backend_local_optimizer_copy() -> None:
    """Local copies preserve native bounds and leave the original backend usable."""
    from phasmix._backends import FitRequest  # noqa: PLC0415 -- only used in this integration test.
    from phasmix._rust_backend import RustFitBackend  # noqa: PLC0415

    backend = RustFitBackend(
        max_iterations=1,
        atol=0.0,
        rtol=0.0,
        smoothing_func=None,
        mask_func=None,
        bounds=ParameterBounds(alpha=0.0, b=0.05, c=0.0, theta0=0.0, scale_factor=40.0, rho=0.09),
    )
    local = backend.with_local_optimizer(maxiter=1)
    grid = np.ones((2, 2))
    request = FitRequest(
        initial_density=grid,
        initial_background=grid,
        z_mesh=grid,
        vz_mesh=grid,
        num_components=1,
        winding=1,
        improve_background=False,
        warm_start=np.array([0.0, 0.05, 0.0, 0.0, 40.0, 0.09]),
    )
    result = local.fit(request)
    assert isinstance(result, FitSuccess)
    assert result.diagnostics.nfev == 1
    assert "local Nelder-Mead" in result.diagnostics.message
    global_result = backend.fit(request)
    assert isinstance(global_result, FitSuccess)
    assert global_result.diagnostics.nfev > result.diagnostics.nfev
    with pytest.raises(ValueError, match="positive integer"):
        _ = backend.with_local_optimizer(maxiter=0)


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_update_bounds_after_initialization(backend: Literal["python", "rust"]) -> None:
    """Public bounds updates reach each component and invalid updates are atomic."""
    initial = ParameterBounds(alpha=0.0, b=0.05, c=0.0, theta0=0.0, scale_factor=40.0, rho=0.09)
    fitter = PSpiralFitter(backend=backend, bounds=initial, max_iterations=1)
    first = ParameterBounds(alpha=0.2, b=0.04, c=0.001, theta0=0.4, scale_factor=35.0, rho=0.08)
    second = ParameterBounds(alpha=0.5, b=0.07, c=0.002, theta0=-1.0, scale_factor=50.0, rho=0.12)
    fitter.update_bounds((first, second))
    with pytest.raises(ValueError, match="nonempty"):
        fitter.update_bounds(())
    parameters = np.array([0.2, 0.04, 0.001, 0.4, 35.0, 0.08, 0.5, 0.07, 0.002, -1.0, 50.0, 0.12])
    grid = np.ones((2, 3))
    mesh = np.zeros_like(grid)
    result = fitter.fit_spiral_with_background(
        grid,
        grid,
        mesh,
        mesh,
        num_components=2,
        winding=1,
        improve_background=False,
        warm_start=parameters,
    )
    assert isinstance(result, FitSuccess)
    np.testing.assert_array_equal(result.result.final_model.to_array(), parameters)


def test_native_bounds_update_is_atomic() -> None:
    """Rejecting native constraints preserves the last valid bounds."""
    fitter = _internal.PSpiralFitter(
        optimizer=NelderMeadConfig(),
        bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
    )
    parameters = [0.0, 0.05, 0.0, 0.0, 40.0, 0.09]
    fitter.update_bounds([[(value, value) for value in parameters]])
    with pytest.raises(ValueError, match="reversed"):
        fitter.update_bounds([[(1.0, 0.0)] * 6])
    grid = np.ones(4)
    result = fitter.fit_spiral_with_background(
        grid,
        grid,
        grid,
        grid,
        grid,
        (2, 2),
        num_components=1,
        winding=1,
        improve_background=False,
        warm_start=parameters,
    )
    assert result.optimizer_success
    assert result.nfev == 1
    assert result.nit == 0
