"""Configuration dataclasses for the global optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type MutationStrategy = Literal[
    "best1",
    "best2",
    "rand1",
    "rand2",
    "randtobest1",
    "currenttobest1",
]
type CrossoverStrategy = Literal["binomial", "exponential"]
type BoundaryStrategy = Literal["resample", "reflect"]
type InitializerStrategy = Literal["latinhypercube", "sobol", "random"]


@dataclass
class DifferentialEvolutionConfig:
    """Configuration for the differential evolution configuration."""

    pop_size_factor: int = 6
    max_iter: int = 100
    max_local_iter: int = 1500
    mutation_factor: float = 0.5
    crossover_rate: float = 0.7
    atol: float = 0.0
    rtol: float = 0.01
    mutation: MutationStrategy = "best1"
    crossover: CrossoverStrategy = "binomial"
    boundary: BoundaryStrategy = "reflect"
    initializer: InitializerStrategy = "latinhypercube"

    def __post_init__(self) -> None:
        """Validate configuration."""
        # TODO: Actually validate


@dataclass
class TikTakConfig:
    """Configuration for the TikTak configuration."""

    log_num_samples: int = 12
    keep_ratio: float = 0.0078125
    min_weight: float = 0.1
    max_weight: float = 0.995

    def __post_init__(self) -> None:
        """Validate configuration."""
        # TODO: Actually validate


@dataclass
class NelderMeadConfig:
    """Configuration for the Nelder-Mead configuration."""

    max_iter: int = 1500

    def __post_init__(self) -> None:
        """Validate configuration."""
        # TODO: Actually validate
