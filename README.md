# Depp_sfa: stochastic frontier analysis

Depp_sfa is a python library dedicated to the estimation of stochastic frontier analysis (sfa) models. 

It is designed to provide high robustness against the numerical convergence issues frequently encountered in applied econometrics. For cross-sectional data, the library relies primarily on maximum likelihood estimation (mle), featuring an automatic fallback to bayesian estimation (mcmc via pymc) in the event of optimization failure. For panel data, it implements a strictly bayesian estimation of the dynamic battese and coelli (1992) model.

## Main features

* Production and cost frontiers: supports both orientations.
* Functional forms: linear, cobb-douglas, and translog specifications. Includes support for dummy variables.
* Cross-sectional data: standard estimation with optional inclusion of inefficiency determinants (bc95 model).
* Panel data (time-varying): implementation of the battese and coelli (1992) model, capturing the temporal evolution of technical inefficiency.
* True effects (greene 2005): integration of unobserved heterogeneity separation.
* Efficiency decomposition methods: jondrow et al. (1982), battese and coelli (1988), and a modified approach.

## Mle versus pymc

The library offers two distinct inference methods to adapt to your data structure and overcome traditional solver limitations.

Mle (maximum likelihood estimation) is the standard frequentist approach. It is computationally fast and works well on large, balanced cross-sectional datasets. However, it frequently suffers from boundary failures (where the inefficiency variance collapses to zero) on unbalanced panels or when dealing with high ratios of singletons.

Pymc (bayesian inference) is the robust alternative. By integrating prior distributions, it prevents boundary collapses and successfully separates signal from noise even when the majority of the panel consists of single observations. It provides full posterior distributions but requires more computation time.

## Installation

The library can be installed directly from its git repository:

```bash
pip install -U git+https://github.com/ThomasLucereau/depp_SFA.git
```

## Usage and data preparation

It is strongly recommended to standardize (center and scale) continuous variables prior to estimation to ensure the convergence of the optimization algorithms.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from depp_SFA import SFA, FUN_COST

# 1. data preparation
df = pd.read_csv("data.csv")
df = df[(df["cost"] > 0) & (df["output"] > 0)]

# standardization
vars_to_scale = ["density", "quality_index"]
scaler = StandardScaler()
df[vars_to_scale] = scaler.fit_transform(df[vars_to_scale])

# 2. vector extraction
y = df["cost"].to_numpy(dtype=float)
X = df[["output"] + vars_to_scale].to_numpy(dtype=float)
firm_ids = df["firm_id"].to_numpy()
years = df["year"].to_numpy(dtype=float)
Z_vars = df[["contract_type", "vandalism_risk"]].to_numpy(dtype=float)
```

## Implementation guide

### 1. Panel data with time evolution (bc92)

The battese and coelli (1992) model captures how inefficiency evolves over time using a decay parameter. It is best estimated using pymc for short or unbalanced panels.

```python
model_bc92 = SFA(
    y=y, 
    x=X, 
    id_var=firm_ids, 
    time_var=years, 
    fun=FUN_COST, 
    panel_model='bc92',
    inference_method='pymc', # or mle
    draws=2000
)

# display results
model_bc92.summary()

# retrieve efficiency scores
efficiency_scores = model_bc92.get_technical_efficiency()
```

### 2. Inefficiency effects model (bc95)

The battese and coelli (1995) model uses environmental variables (z) to explain why certain units are less efficient, rather than treating these variables as direct cost drivers.

```python
model_bc95 = SFA(
    y=y, 
    x=X, 
    z=Z_vars,
    fun=FUN_COST,      
    inference_method='pymc', #or mle
    draws=2000
)

# display results
model_bc95.summary()
```

### 3. True random effects (greene 2005)

This specification separates unobserved structural heterogeneity (specific to each firm) from pure managerial inefficiency. It prevents the model from penalizing firms for structural geographical disadvantages.

```python
model_greene = SFA(
    y=y, 
    x=X, 
    id_var=firm_ids,
    time_var=years,
    fun=FUN_COST,
    panel_model='greene',
    inference_method='pymc', # or mle
    draws=2000
)

# display results
model_greene.summary()
```

## Api documentation (sfa class)

### Instantiation: sfa(...)

The class constructor configures the model parameters and transforms the data according to the specified functional form.

**Parameters:**
* y (array-like): dependent variable (output for production frontier, cost for cost frontier).
* x (array-like, 2d): independent variables (inputs or prices/outputs).
* z (array-like, 2d, optional): explanatory variables for inefficiency (cross-sectional data only).
* id_var (array-like, optional): individual identifiers for panel data.
* time_var (array-like, optional): time variable for panel data.
* fun (constant): frontier type. Use sfa.fun_prod (default) or sfa.fun_cost.
* intercept (bool): include an intercept in the model (default: true).
* lamda0 (float): initial value of lambda for mle optimization (default: 1).
* method (constant): method for computing technical efficiency. Choose among sfa.te_tej (jondrow et al.), sfa.te_te (battese and coelli), or sfa.te_temod.
* form (str): functional form. Choose among 'linear', 'cobb_douglas', or 'translog'.
* dummy_indices (list): list of column indices in x that are indicator variables (0/1) and should not be log-transformed.

### Public methods

Once the model is instantiated, the following methods are available. Calling any of these methods will automatically trigger the estimation process (optimize()) if it has not already been executed.

* optimize(): triggers the estimation algorithm. Automatically routes to panel estimation (mcmc) or cross-sectional estimation (mle, with an mcmc fallback in case of convergence failure).
* summary(): computes and prints a summary table of the results to the console, including estimated coefficients, standard errors, t-values, p-values, associated significance levels, and the log-likelihood value (for mle).
* get_beta(): returns a numpy array containing the estimated coefficients for the frontier variables (including the intercept if intercept=true).
* get_residuals(): returns a numpy array containing the model residuals.
* get_sigma2(): returns the estimated total variance of the composite error.
* get_lambda(): returns the ratio of standard deviations, which measures the relative importance of inefficiency compared to statistical noise.
* get_technical_efficiency(): returns a numpy array containing the technical efficiency scores (bounded between 0 and 1) computed for each observation in the sample, using the method specified during instantiation.

## Credits and licenses

This software is distributed under the mit license.

* The base architecture, matrix processing, and classic log-likelihood derivations are inspired by and adapted from the work of sheng dai (copyright (c) 2023, mit license).
* The integration of bayesian estimators (markov chain monte carlo via pymc), the numerical stabilization of panel models (battese and coelli 1992), and the algorithmic exception handling were developed specifically for this project.
