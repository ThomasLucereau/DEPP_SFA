# Depp_sfa: Stochastic Frontier Analysis (SFA)

[![Documentation Status](https://readthedocs.org/projects/depp-sfa/badge/?version=latest)](https://depp-sfa.readthedocs.io/en/latest/?badge=latest)

**Depp_sfa** is a Python library dedicated to the estimation of Stochastic Frontier Analysis (SFA) models. It is specifically designed to provide high robustness against numerical convergence issues—such as "boundary effects"—frequently encountered in applied econometrics.

---

## Key Features

* **Hybrid Inference**: Primarily relies on Maximum Likelihood Estimation (MLE), featuring an **automatic fallback to Bayesian estimation** (MCMC via PyMC) if the optimization fails to converge.
* **Advanced Panel Data**: Implements the dynamic **Battese & Coelli (1992)** model for time-varying inefficiency and **Greene (2005)** True Fixed/Random Effects.
* **Functional Flexibility**: Supports Linear, **Cobb-Douglas**, and **Translog** specifications with automatic handling of log-transformations, interaction terms, and dummy variables.
* **Built-in Standardization**: Includes an internal preprocessing option (`standardize=True`) to center and scale continuous variables, ensuring stable solver convergence.



---

## Installation

Install the library directly from the repository:

'''
pip install depp_sfa
'''
---

## Quick Start

The library can automatically handle data scaling and functional form transformations (logs, interactions, and squares).

# --- PYTHON START ---
import pandas as pd
import numpy as np
from depp_sfa import SFA, FUN_COST

# 1. Load your data
df = pd.read_csv("utility_data.csv")

# 2. Instantiate the model
# Using 'standardize=True' is highly recommended for Translog forms
model = SFA(
    y=df["total_cost"].values,
    x=df[["output", "labor_price", "capital_price"]].values,
    fun=FUN_COST,
    form='translog',
    standardize=True,
    inference_method='mle' # Falls back to 'pymc' automatically if MLE fails
)

# 3. Estimate and display results
model.summary()

# 4. Retrieve Efficiency Scores (TE)
te_scores = model.get_technical_efficiency()
# --- PYTHON END ---

---

## API Reference: SFA Class

### Constructor Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| y | array | Dependent variable (Output for production, Cost for cost frontier). |
| x | array | Independent variables (Inputs or Price/Output mix). |
| z | array | (Optional) Determinants of inefficiency (BC95 model). |
| id_var | array | Individual identifiers (Required for panel models). |
| time_var | array | Time/Year variable (Required for BC92 model). |
| fun | int | FUN_PROD (0) or FUN_COST (1). |
| form | str | Functional form: 'linear', 'cobb_douglas', or 'translog'. |
| standardize| bool | If True, centers and scales continuous variables automatically. |
| draws | int | Number of MCMC draws if using PyMC (default: 2000). |

### Core Methods

Calling these methods will automatically trigger the optimize() routine if the model hasn't been estimated yet.

* summary(): Prints a detailed results table including Coefficients, Std. Errors, z-values, and P-values.
* get_technical_efficiency(): Returns technical efficiency scores (bounded between 0 and 1).
* get_beta(): Returns the estimated frontier coefficients.
* get_residuals(): Returns the composite error terms (v_i +/- u_i).
* get_lambda(): Returns the signal-to-noise ratio (sigma_u / sigma_v).

---

## MLE vs. PyMC

* **MLE (Maximum Likelihood)**: Fast and standard. Best for large, balanced cross-sections. It may suffer from "boundary failures" (where inefficiency variance collapses to zero) on small or unbalanced panels.
* **PyMC (Bayesian Inference)**: Highly robust. By using prior distributions, it prevents variance collapse and successfully separates noise from inefficiency even in datasets with many singletons (unbalanced panels). It provides full posterior distributions for all parameters.

---

## Credits & License

Distributed under the **MIT License**.
* Base architecture and likelihood derivations inspired by Sheng Dai (2023).
* Bayesian integration, numerical stabilization, and panel model extensions developed for the Depp project.
