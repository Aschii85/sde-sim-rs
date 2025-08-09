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
    Simulates stochastic differential equations (SDEs) using the specified methods.

    This function simulates a set of SDEs using specified numerical schemes and
    random number generation methods. The core simulation logic is implemented
    in a high-performance Rust backend for speed and efficiency.

    Args:
        processes_equations: A list of strings, where each string represents an SDE.
            The equation must follow a specific format:
            `d{ProcessName} = ({expression})*d{Incrementor} + ...`.
            For example, a Geometric Brownian Motion can be represented as:
            `dX = (0.5 * X) * dt + (0.2 * X) * dW1`.
            Supported incrementors are `dt` (for the drift term) and `dW` (for
            Wiener processes, e.g., `dW1`, `dW2`).

        time_steps: A sequence of time points at which to calculate the process
            values. Must be in increasing order.

        scenarios: The number of simulation paths to generate.

        initial_values: A dictionary mapping process names (as strings) to their
            initial numerical values.

        rng_method: The random number generation method to use. Can be **"pseudo"** for
            pseudorandom numbers or **"sobol"** for Sobol sequences (quasi-random).
            Defaults to "pseudo".

        scheme: The numerical integration scheme to use. Can be **"euler"** for the
            Euler-Maruyama method or **"runge-kutta"** for a higher-order Runge-Kutta
            method. Defaults to "euler".

    Returns:
        A Polars DataFrame containing the simulated values. The DataFrame has a
        `time` column and columns for each process, with each row corresponding
        to a specific time step and scenario.

    Raises:
        ValueError: If the process equations are malformed or if initial values
            are missing for any process.
    """
    ...