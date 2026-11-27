"""Generate test data for Tavangar et al. 2026's BFE method."""

from __future__ import annotations


import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from matplotlib import pyplot as plt
from phasmock.mock import MockParticles
from phasmock.recipe import MockRecipe
from rich.console import Console
from rich.logging import RichHandler

from phasmix import fit
from phasmix._likelihood_utils import ln_likelihood
from phasmix.background_estimators import estimate_gaussian_smooth_background_vz_mirrored
from phasmix.fit import FitSuccess, PSpiralFitter
from phasmix.optimizers import DifferentialEvolutionConfig
from phasmix.uncertainty import bootstrap_uncertainty

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Final

    from optype import numpy as onp


FORMAT: Final[str] = "%(message)s"
log: Final[logging.Logger] = logging.getLogger(__name__)


def setup_logging() -> Iterable[logging.Handler]:
    """Set up logging.

    Returns
    -------
    handlers : Iterable[logging.Handler]
        The logging handlers.

    """
    console = Console()
    console_handler = RichHandler(console=console, show_time=False, markup=True)
    handlers: list[logging.Handler] = [console_handler]
    logging.basicConfig(
        level="NOTSET",
        format=FORMAT,
        datefmt="[%X]",
        handlers=handlers,
        encoding="utf-8",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    return handlers


def _generate_mock() -> MockParticles:
    smoke_recipe = MockRecipe.from_yaml(Path(__name__).parent / "recipes/tavbfe_recipe.yaml")
    return smoke_recipe.generate()


def _main() -> None:
    particles = _generate_mock()
    print(particles.x)
    print(np.mean(particles.x))


if __name__ == "__main__":
    _ = setup_logging()
    _main()
