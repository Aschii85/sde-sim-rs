import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl

print(dir(sde_sim_rs))


def main():
    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            "dX1 = ( 0.1 - 0.01 * X1 ) * dt + ( 0.2 * X1) * dW1",
            "dX2 = ( 0.01 * X2 ) * dt + ( 0.01 * X2 ) * dW1 + ( 0.02 * X2 ) * dW1",
        ],
        time_steps=list(np.arange(0.0, 100.0, 0.1)),
        scenarios=1000,
        initial_values={"X1": 1.0, "X2": 0.5},
        rng_method="sobol",
        scheme="euler",
    )
    print(df)
    # fig = px.line(
    #     df,
    #     x="time",
    #     y="value",
    #     color="scenario",
    #     line_dash="process_name",
    #     title="Simulated SDE Process",
    # )
    # fig.show()
    mean_df = df.group_by(["time", "process_name"]).mean().sort(["time", "process_name"])["time", "process_name", "value"]
    print(mean_df)
    fig = px.line(
        mean_df,
        x="time",
        y="value",
        color="process_name",
        title="Simulated SDE Process",
    )
    fig.show()

if __name__ == "__main__":
    main()
