import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl


def main():
    initial_values = {"X0": 0.5, "X1": 100.0, "X2": 0.0}
    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            "dX0 = ( 2.0 * (0.5 - X0) ) * dt + ( 0.1 ) * dN1(X0)",
            "dX1 = ( 0.05 * X1 ) * dt + ( 0.2 * X1 ) * dW1 + ( 0.5 ) * dN1(X0)",
            "X2 = max(X1 - 100.0, 0.0)",
        ],
        time_steps=list(np.arange(0.0, 10.0, 0.01)),
        scenarios=10_000,
        initial_values=initial_values,
        rng_method="pseudo",
        scheme="euler",
    )
    print(df)
    for i in range(0, len(initial_values)):
        fig = px.line(
            df.filter(pl.col("process_name") == f"X{i}"),
            x="time",
            y="value",
            color="scenario",
            line_dash="process_name",
            title="Simulated SDE Process",
        )
        fig.show()
    for i in range(0, len(initial_values)):
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
