from collections.abc import Sequence

import numpy as np
from optype import numpy as onp

from .optimizers import DifferentialEvolutionConfig, NelderMeadConfig, TikTakConfig

def ln_likelihood_f64(
    data: onp.Array1D[np.float64], prediction: onp.Array1D[np.float64], mask: onp.Array1D[np.float64]
) -> float: ...

class PSpiralComponent:
    def __init__(
        self,
        alpha: float,
        b: float,
        c: float,
        theta0: float,
        scale_factor: float,
        rho: float,
        winding: int,
    ) -> None: ...
    @property
    def alpha(self) -> float: ...
    @property
    def b(self) -> float: ...
    @property
    def c(self) -> float: ...
    @property
    def theta0(self) -> float: ...
    @property
    def scale_factor(self) -> float: ...
    @property
    def rho(self) -> float: ...
    @property
    def winding(self) -> float: ...
    def perturbation(self, z: onp.Array1D[np.float64], vz: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...

class PSpiralModel:
    def __init__(self, components: Sequence[PSpiralComponent]) -> None: ...
    @property
    def components(self) -> Sequence[PSpiralComponent]: ...
    def perturbation(self, z: onp.Array1D[np.float64], vz: onp.Array1D[np.float64]) -> onp.Array1D[np.float64]: ...

class PSpiralFitter:
    def __init__(
        self,
        max_iterations: int | None = 50,
        atol: float = 0.0,
        rtol: float = 0.0,
        sigma_z: float = 2.0,
        sigma_vz: float = 2.0,
        *,
        bounds: Sequence[Sequence[tuple[float, float]]],
        optimizer: DifferentialEvolutionConfig | TikTakConfig | NelderMeadConfig,
    ) -> None: ...
    def update_bounds(self, bounds: Sequence[Sequence[tuple[float, float]]]) -> None: ...
    def with_local_optimizer(self, *, maxiter: int = 1500) -> PSpiralFitter: ...
    def fit_batch(
        self,
        inputs: Sequence[
            tuple[
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                onp.Array1D[np.float64],
                tuple[int, int],
            ]
        ],
        *,
        seeds: Sequence[int],
        workers: int | None = None,
        options: Sequence[tuple[int | None, int | None, bool]] | None = None,
        warm_starts: Sequence[Sequence[float] | None] | None = None,
    ) -> list[PSpiralFitResult]: ...
    def fit_spiral_with_background(
        self,
        initial_density: onp.Array1D[np.float64],
        initial_background: onp.Array1D[np.float64],
        mask: onp.Array1D[np.float64],
        mesh_x: onp.Array1D[np.float64],
        mesh_y: onp.Array1D[np.float64],
        shape: tuple[int, int],
        *,
        seed: int,
        num_components: int | None = None,
        winding: int | None = None,
        improve_background: bool = True,
        warm_start: Sequence[float] | None = None,
    ) -> PSpiralFitResult: ...
    def fit_spiral_with_background_events(
        self,
        initial_density: onp.Array1D[np.float64],
        initial_background: onp.Array1D[np.float64],
        mask: onp.Array1D[np.float64],
        mesh_x: onp.Array1D[np.float64],
        mesh_y: onp.Array1D[np.float64],
        shape: tuple[int, int],
        *,
        seed: int,
        num_components: int | None = None,
        winding: int | None = None,
        improve_background: bool = True,
        warm_start: Sequence[float] | None = None,
    ) -> PSpiralFitIterator: ...

class PSpiralFitIterator:
    def __iter__(self) -> PSpiralFitIterator: ...
    def __next__(self) -> PSpiralFitResult: ...

class PSpiralFitResult:
    @property
    def initial_model(self) -> PSpiralModel: ...
    @property
    def final_model(self) -> PSpiralModel: ...
    @property
    def data(self) -> onp.Array1D[np.float64]: ...
    @property
    def initial_background(self) -> onp.Array1D[np.float64]: ...
    @property
    def final_background(self) -> onp.Array1D[np.float64]: ...
    @property
    def num_iterations(self) -> int: ...
    @property
    def max_iterations(self) -> int | None: ...
    @property
    def converged(self) -> bool: ...
    @property
    def lnl(self) -> float: ...
    @property
    def initial_pvalue(self) -> float: ...
    @property
    def final_pvalue(self) -> float: ...
    @property
    def nfev(self) -> int: ...
    @property
    def nit(self) -> int: ...
    @property
    def optimizer_success(self) -> bool: ...
    @property
    def optimizer_message(self) -> str: ...
    @property
    def terminal(self) -> bool: ...
