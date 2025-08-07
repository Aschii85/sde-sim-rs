import sde_sim_rs
import numpy as np
import plotly.express as px

print(dir(sde_sim_rs))


def main():
    df = sde_sim_rs.simulate(
        processes_equations=[
            "dX1 = ( 0.01 * X1 ) * dt + ( 0.2 * X1 ) * dW1",
            "dX2 = ( 0.01 * X1 ) * dt + ( 0.2 * X2 ) * dW1",
        ],
        time_steps=list(np.arange(0.0, 100.0, 0.1)),
        scenarios=1000,
        initial_values={"X1": 1.0, "X2": 0.5},
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
