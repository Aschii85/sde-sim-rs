# sde-sim-rs: Flexible stochastic differential equation simulation library written in Rust

`sde-sim-rs` is a high-performance library for simulating stochastic differential equations (SDEs), which are foundational in fields like quantitative finance, physics, and biology. By leveraging the speed and memory safety of Rust, the project provides a fast and flexible core while offering seamless bindings for use in Python. This project is ideal for researchers, data scientists, and developers who need to run complex SDE simulations with remarkable efficiency and reliability. The architecture is designed to bridge the gap between high-performance compiled languages and the scientific computing ecosystem of Python.

## Features

High Performance: The implementation in Rust provides bare-metal performance, which is critical for time-sensitive and computationally intensive simulations. The language's zero-cost abstractions and memory-safe concurrency models allow for the efficient handling of large datasets and provide the potential for parallelizing simulation tasks, offering a significant speed advantage over purely interpreted solutions.

Python Integration: A user-friendly and comprehensive Python interface via maturin allows you to utilize the Rust core without leaving your Python environment. This means data scientists and researchers can leverage the speed of a compiled language for the most demanding parts of their code, with bindings designed for seamless function calls and data exchange between the two languages.

Flexibility: The library's design and modular architecture allows for the creation and integration of custom SDE models to suit specialized research or application needs.

## Setup

### Python

Install the latest `sde-sim-rs` version with:

```
pip install polars
```

Requires Python version >=3.11.

To build the package locally for, you'll first need to compile the Rust package for local development. The project is set up to use `maturin` and `uv`. This command builds the Rust library and creates a Python wheel that can be used directly in your environment.

```
maturin develop --uv --release
```

After the compilation is complete, you can run the example to see how the library works. This command uses `uv` to execute the Python example script.

```
uv run python/sde_simulators/example.py
```

### Rust

You can take latest release from crates.io, or if you want to use the latest features / performance improvements point to the main branch of this repo.

```
polars = { git = "https://github.com/pola-rs/polars", rev = "<optional git tag>" }
```

## Contributing

We welcome contributions from the community! If you'd like to contribute, please feel free to open an issue to discuss a feature or a bug fix, or submit a pull request with your changes.
