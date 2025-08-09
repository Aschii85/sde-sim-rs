from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl


def simulate(
    processes_equations: Sequence[str],
    time_steps: Sequence[float],
    scenarios: int,
    initial_values: Mapping[str, float],
    rng_method: Literal["pseudo", "sobol"] = "pseudo",
    scheme: Literal["euler", "runge-kutta"] = "euler",
) -> pl.DataFrame: 
    """
    Simulate stochastic differential equations (SDEs) using the specified methods.
    """
    ...