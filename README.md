# PySpectral Quantum Solver 

**PySpectral Quantum Solver** is an advanced Python library designed for the numerical resolution of the time-independent Schrödinger Equation in one-dimensional potential wells. Utilizing the **Galerkin Spectral Method** with a sine function basis, the library enables researchers to find energy eigenvalues and eigenfunctions for complex quantum systems with high precision.

---

## Key Features

* **Generalized Eigenvalue Problem Solver:** Solves equations of the type $-\\frac{d^{2}}{dx^{2}}+f(x))\\psi(x)=E~g(x)\\psi(x)$.

* **Domain Optimization:** Includes a native algorithm to find the optimal box length ($L\_{opt}$) to minimize truncation errors.

* **Dynamic Analysis:** Calculates time evolution of wave packets and probability densities.

* **Heisenberg Uncertainty:** Built-in methods to calculate and verify the uncertainty principle ($\\sigma\_x \\cdot \\sigma\_p \\ge \\hbar/2$) in real-time.

* **Scientific Visualization:**
* 2D and 3D static and animated plots of wave functions.
* "Cartoon" representations (heatmaps) of probability density.
* Data export to structured formats (.txt) for external analysis.

---

## Requirements \& Installation

The library relies on the following scientific Python stack:

* `numpy >= 1.24`
* `scipy >= 1.11`
* `sympy >= 1.12`
* `matplotlib >= 3.7`
* `pandas >= 2.0`


Install the dependencies using pip:

```bash

pip install -r requirements.txt

```

---

## Quick Start

The library is structured around the `SpectralMethod` class. Below is a basic example solving a \*\*Harmonic Oscillator\*\*:

```python

import numpy as np
from PySpectral import SpectralMethod


# 1. Define the potential f(x) and the weighting function g(x)
def f(x): return x\*\*2  # Harmonic potential
def g(x): return 1     # Standard Schrödinger equation

# 2. Instantiate the solver
# num\_levels: number of basis functions | length: box size L

solver = SpectralMethod(num\_levels=50, length=10.0, f\_function=f, g\_function=g, label="Harmonic\_Osc")

# 3. Solve the system
solver.is\_solved()

# 4. Retrieve results
energies = solver.its\_eigenvalues()
print(f"First 3 energy levels: {energies\[:3]}")

# 5. Visualize eigenfunctions
solver.plot\_eigenfunctions(num\_levels=3)

```

---

## Mathematical Background

The solver approximates the wave function $\\psi\_p(x)$ as an expansion of $N$ sine basis functions:

$$\\psi\_{p}(x)\\approx\\sqrt{\\frac{2}{L}}\\sum\_{n=1}^{N}A\_{n}^{(p)}sin\\left(\\frac{n\\pi x}{L}\\right)$$

This transforms the differential equation into a matrix eigenvalue problem:

$$\\Delta A^{(p)} = E\_p \\Gamma A^{(p)}$$

Where $\\Delta$ represents the kinetic and potential energy terms, and $\\Gamma$ handles the overlap integrals.

---

## Project Structure

To maintain a professional and organized environment, the repository follows this structure:

* **`pyspectral/`**: Core package directory.
    * `__init__.py`: Package initialization (exposes the main class).
    * `solver.py`: The main library engine (formerly `PySpectral.py`).
* **`docs/`**: Technical manuals and documentation (including the reference PDF).
* **`examples/`**: Demonstration scripts and Jupyter notebooks to get started.
* **`Data/`, `Figures/`, `Evolution/`**: Automatically generated directories for exported results, plots, and animations.
* **`setup.py`**: Installation script for the library.
* **`requirements.txt`**: List of Python dependencies.

---

## Author \& Contact

**Vagner Jandre Monteiro** (mailto: vagner.jandre@iprj.uerj.br)

IPRJ/UERJ - PPGMC/DO

---

## Acknowledgments

Thanks **CAPES** for financial support.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---
