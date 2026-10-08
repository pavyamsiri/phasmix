"""Optimizer dataclasses reach native fitting and honor optimizer budgets."""

import numpy as np
import pytest

from phasmix import _internal
from phasmix._rust_backend import RustFitBackend
from phasmix.bounds import ParameterBounds
from phasmix.fit import FitSuccess, PSpiralFitter
from phasmix.optimizers import DifferentialEvolutionConfig, NelderMeadConfig, TikTakConfig


@pytest.mark.parametrize("num_components", [1, 2])
@pytest.mark.parametrize(
    "config",
    [
        DifferentialEvolutionConfig(max_iter=0, max_local_iter=0, mutation="rand2", initializer="random", boundary="resample"),
        TikTakConfig(log_num_samples=1, keep_ratio=0.5, min_weight=0.2, max_weight=0.8),
        NelderMeadConfig(max_iter=1),
    ],
)
def test_configured_native_fit(
    config: DifferentialEvolutionConfig | TikTakConfig | NelderMeadConfig,
    num_components: int,
) -> None:
    """Nondefault configs execute for both component counts with free parameters."""
    fitter = PSpiralFitter(backend="rust", optimizer=config)
    grid = np.ones((2, 3))
    mesh = np.zeros_like(grid)
    result = fitter.fit_spiral_with_background(
        grid,
        grid,
        mesh,
        mesh,
        winding=1,
        num_components=num_components,
        improve_background=False,
        warm_start=np.tile([0.0, 0.05, 0.0, 0.0, 40.0, 0.09], num_components),
    )
    assert isinstance(result, FitSuccess)
    if isinstance(config, DifferentialEvolutionConfig):
        assert result.diagnostics.nit == 0
    if isinstance(config, NelderMeadConfig):
        assert result.diagnostics.nit <= config.max_iter
        assert fitter._backend.local_optimizer_maxiter == config.max_iter  # noqa: SLF001


def test_native_config_dispatch_preserves_extraction_errors() -> None:
    """An invalid DE config must not fall back to the smaller Nelder-Mead shape."""
    config = DifferentialEvolutionConfig(pop_size_factor=-1)
    with pytest.raises(TypeError, match="pop_size_factor"):
        _internal.PSpiralFitter(
            optimizer=config,
            bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
        )


def test_unsupported_crossover_rejected() -> None:
    """Unsupported strategies fail at construction instead of being ignored."""
    with pytest.raises(ValueError, match="Exponential crossover"):
        PSpiralFitter(backend="rust", optimizer=DifferentialEvolutionConfig(crossover="exponential"))


@pytest.mark.parametrize("optimizer", [None, "tiktak"])
def test_native_requires_config(optimizer: object) -> None:
    """Native initialization accepts concrete dataclasses only."""
    with pytest.raises(TypeError, match="configuration"):
        _internal.PSpiralFitter(
            bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
            optimizer=optimizer,  # pyright: ignore[reportArgumentType] -- exercise runtime rejection.
        )


def test_native_requires_bounds() -> None:
    """Missing or None bounds cannot request native defaults."""
    with pytest.raises(TypeError):
        _internal.PSpiralFitter(optimizer=TikTakConfig())  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        _internal.PSpiralFitter(bounds=None, optimizer=TikTakConfig())  # pyright: ignore[reportArgumentType]


def test_native_requires_optimizer() -> None:
    """A native caller cannot request an implicit optimizer default."""
    with pytest.raises(TypeError):
        _internal.PSpiralFitter(  # pyright: ignore[reportCallIssue]
            bounds=[RustFitBackend._rust_bounds_for_component(ParameterBounds())],  # noqa: SLF001
        )
