from collections.abc import Mapping, Sequence
import polars as pl

def simulate(
    processes_equations: Sequence[str],
    time_steps: Sequence[float],
    scenarios: int,
    initial_values: Mapping[str, float]
) -> pl.DataFrame: ...