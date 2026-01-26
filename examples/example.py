import sde_sim_rs
import numpy as np
import plotly.express as px
import polars as pl

print(dir(sde_sim_rs))


def main():
    df: pl.DataFrame = sde_sim_rs.simulate(
        processes_equations=[
            "dX1 = ( 10.0 ) * dW1",
            "dX2 = ( 0.1 * (10.0 - X2) ) * dt + ( 0.01 * X1 * X2) * dJ1(0.1)",
            "dX3 = ( 0.001 * X3 ) * dt + ( 0.01 * X2 * X3 ) * dW2",
        ],
        time_steps=list(np.arange(0.0, 100.0, 0.1)),
        scenarios=1000,
        initial_values={"X1": 0.0, "X2": 0.5, "X3": 100.0},
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


if __name__ == "__main__":
    main()
