# depp_SFA: Stochastic Frontier Analysis

depp_SFA is a Python library dedicated to the estimation of Stochastic Frontier Analysis (SFA) models. 

It is designed to provide high robustness against the numerical convergence issues frequently encountered in applied econometrics. For cross-sectional data, the library relies primarily on Maximum Likelihood Estimation (MLE), featuring an automatic fallback to Bayesian estimation (MCMC via PyMC) in the event of optimization failure. For panel data, it implements a strictly Bayesian estimation of the dynamic Battese and Coelli (1992) model.

## Main Features

* **Production and Cost Frontiers:** Supports both orientations.
* **Functional Forms:** Linear, Cobb-Douglas, and Translog specifications. Includes support for dummy variables.
* **Cross-Sectional Data:** Standard estimation with optional inclusion of inefficiency determinants (BC95 model).
* **Panel Data (Time-Varying):** Implementation of the Battese & Coelli (1992) model, capturing the temporal evolution of technical inefficiency.
* **Efficiency Decomposition Methods:** Jondrow et al. (1982), Battese and Coelli (1988), and a modified approach.

## Installation

The library can be installed directly from its Git repository:

```bash
pip install -U git+https://github.com/ThomasLucereau/depp_SFA.git
```

## Usage and Quick Start

It is strongly recommended to standardize (center and scale) continuous variables prior to estimation to ensure the convergence of the optimization algorithms.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from depp_SFA import SFA, FUN_COST, TE_teJ

# 1. Data preparation
df = pd.read_csv("data.csv")
df = df[(df["cost"] > 0) & (df["output"] > 0)]

# Standardization
vars_to_scale = ["density", "quality_index"]
scaler = StandardScaler()
df[vars_to_scale] = scaler.fit_transform(df[vars_to_scale])

# 2. Vector extraction
y = df["cost"].to_numpy(dtype=float)
X = df[["output"] + vars_to_scale].to_numpy(dtype=float)
firm_ids = df["firm_id"].to_numpy()
years = df["year"].to_numpy(dtype=float)

# 3. Model instantiation and execution (Panel)
model = SFA(
    y=y, 
    x=X, 
    id_var=firm_ids, 
    time_var=years, 
    fun=FUN_COST, 
    method=TE_teJ, 
    form='cobb_douglas',
    dummy_indices=[]
)

# Display results
model.summary()

# Retrieve efficiency scores
efficiency_scores = model.get_technical_efficiency()
```

## API Documentation (SFA Class)

### Instantiation: `SFA(...)`

The class constructor configures the model parameters and transforms the data according to the specified functional form.

**Parameters:**
* `y` (array-like): Dependent variable (output for production frontier, cost for cost frontier).
* `x` (array-like, 2D): Independent variables (inputs or prices/outputs).
* `z` (array-like, 2D, optional): Explanatory variables for inefficiency (cross-sectional data only).
* `id_var` (array-like, optional): Individual identifiers for panel data.
* `time_var` (array-like, optional): Time variable for panel data.
* `fun` (constant): Frontier type. Use `SFA.FUN_PROD` (default) or `SFA.FUN_COST`.
* `intercept` (bool): Include an intercept in the model (default: True).
* `lamda0` (float): Initial value of lambda for MLE optimization (default: 1).
* `method` (constant): Method for computing technical efficiency. Choose among `SFA.TE_teJ` (Jondrow et al.), `SFA.TE_te` (Battese & Coelli), or `SFA.TE_teMod`.
* `form` (str): Functional form. Choose among `'linear'`, `'cobb_douglas'`, or `'translog'`.
* `dummy_indices` (list): List of column indices in `x` that are indicator variables (0/1) and should not be log-transformed.

### Public Methods

Once the model is instantiated, the following methods are available. Calling any of these methods will automatically trigger the estimation process (`optimize()`) if it has not already been executed.

* **`optimize()`**
  Triggers the estimation algorithm. Automatically routes to panel estimation (MCMC) or cross-sectional estimation (MLE, with an MCMC fallback in case of convergence failure).

* **`summary()`**
  Computes and prints a summary table of the results to the console, including estimated coefficients, standard errors, t-values, p-values, associated significance levels, and the log-likelihood value (for MLE).

* **`get_beta()`**
  Returns a Numpy array containing the estimated coefficients ($\beta$) for the frontier variables (including the intercept if `intercept=True`).

* **`get_residuals()`**
  Returns a Numpy array containing the model residuals ($\epsilon = y - X\beta$).

* **`get_sigma2()`**
  Returns the estimated total variance of the composite error ($\sigma^2 = \sigma_u^2 + \sigma_v^2$).

* **`get_lambda()`**
  Returns the ratio of standard deviations ($\lambda = \sigma_u / \sigma_v$), which measures the relative importance of inefficiency compared to statistical noise.

* **`get_technical_efficiency()`**
  Returns a Numpy array containing the technical efficiency scores (bounded between 0 and 1) computed for each observation in the sample, using the method specified during instantiation.

## Credits and Licenses

This software is distributed under the MIT License.

* The base architecture, matrix processing, and classic log-likelihood derivations are inspired by and adapted from the work of Sheng Dai (Copyright (c) 2023, MIT License).
* The integration of Bayesian estimators (Markov Chain Monte Carlo via PyMC), the numerical stabilization of panel models (Battese & Coelli 1992), and the algorithmic exception handling were developed specifically for this project, with the assistance of language models (Google Gemini).
