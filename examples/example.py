import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl

print(dir(sde_sim_rs))


def main():
    initial_values = {"X1": 100.0, "X2": 0.0}
    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            "dX1 = ( sin(t) ) * dt + (0.01 * X1) * dW1",
            "X2 = max(X1 - 100.0, 0.0)",
        ],
        time_steps=list(np.arange(0.0, 100.0, 0.1)),
        scenarios=10_000,
        initial_values=initial_values,
        rng_method="pseudo",
        scheme="euler",
    )
    print(df)
    for i in range(1, len(initial_values) + 1):
        fig = px.line(
            df.filter(pl.col("process_name") == f"X{i}"),
            x="time",
            y="value",
            color="scenario",
            line_dash="process_name",
            title="Simulated SDE Process",
        )
        fig.show()
    for i in range(1, len(initial_values) + 1):
        fig = px.line(
            df.filter(pl.col("process_name") == f"X{i}")
            .group_by("time")
            .agg(pl.col("value").mean())
            .sort("time"),
            x="time",
            y="value",
            title=f"Mean Simulated SDE Process for X{i}",
        )
        fig.show()


if __name__ == "__main__":
    main()
