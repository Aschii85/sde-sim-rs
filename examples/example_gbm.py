import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl


def main():
    mu = 0.05
    sigma = 0.1
    start_value = 1.0
    expectation = lambda t: start_value * np.exp(mu * t)

    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            f"dX1 = ( {mu} * X1 ) * dt + ( {sigma} * X1) * dW1",
        ],
        time_steps=list(np.arange(0.0, 10.0, 0.1)),
        scenarios=10000,
        initial_values={"X1": start_value},
        rng_method="sobol",
        scheme="runge-kutta",
    )
    print(df)
    fig = px.line(
        df,
        x="time",
        y="value",
        color="scenario",
        line_dash="process_name",
        title="Simulated SDE Process",
    )
    fig.show()
    mean_df = (
        df.group_by(["time", "process_name"])
        .mean()
        .sort(["time", "process_name"])["time", "process_name", "value"]
    )
    print(mean_df)
    fig = px.line(
        mean_df,
        x="time",
        y="value",
        color="process_name",
        title="Simulated SDE Process",
    )
    fig.add_scatter(
        x=mean_df["time"],
        y=expectation(mean_df["time"]),
        mode="lines",
        name="Expectation",
        line=dict(dash="dash"),
    )
    fig.show()


if __name__ == "__main__":
    main()
