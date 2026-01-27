import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl

print(dir(sde_sim_rs))


def main():
    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            "dX1 = ( 0.01 ) * dW1",
            "dX2 = ( (0.01 - 0.05 * 0.01^2) * X2 ) * dt + ( 0.15 * X2 ) * dW2 + ( dW1 * X2 ) * dJ1(0.05)",
            # "dX2 = ( 0.1 * (10.0 - X2) ) * dt + ( 0.01 * X1 * X2) * dJ1(0.1)",
            # "dX3 = ( 0.001 * X3 ) * dt + ( 0.01 * X2 * X3 ) * dW2",
        ],
        time_steps=list(np.arange(0.0, 100.0, 0.1)),
        scenarios=1000,
        initial_values={"X1": 0.0, "X2": 100.0},
        rng_method="pseudo",
        scheme="runge-kutta",
    )
    print(df)
    fig = px.line(
        df.filter(pl.col("process_name") == "X2"),
        x="time",
        y="value",
        color="scenario",
        line_dash="process_name",
        title="Simulated SDE Process",
    )
    fig.show()
    fig = px.histogram(
        df.filter((pl.col("process_name") == "X2") & (pl.col("time") == 99.0)),
        x="value",
        nbins=100,
        title="Histogram of X2 at time 100.0",
    )
    fig.show()
    fig = px.histogram(
        df.filter((pl.col("process_name") == "X2"))["value"].diff(),
        x="value",
        nbins=100,
        title="Histogram of X2 at time 100.0",
    )
    fig.show()


if __name__ == "__main__":
    main()
