import math
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.linear_model import LinearRegression
import pymc as pm

from . import constant

# Bind constants from the external module
FUN_COST = constant.FUN_COST
FUN_PROD = constant.FUN_PROD
TE_teJ = constant.TE_teJ
TE_te = constant.TE_te
TE_teMod = constant.TE_teMod

# Mute PyMC internal logging to keep the terminal clean
logging.getLogger("pymc").setLevel(logging.ERROR)


class SFA:
    """
    Stochastic Frontier Analysis (SFA).
    Supports Cross-sectional (ALS77, BC95) and Panel Data (BC92 Time-varying).
    """

    FUN_PROD = FUN_PROD
    FUN_COST = FUN_COST
    TE_teJ = TE_teJ
    TE_te = TE_te
    TE_teMod = TE_teMod

    def __init__(
        self, y, x, z=None, id_var=None, time_var=None,
        fun=FUN_PROD, intercept=True, lamda0=1, method=TE_teJ,
        form='linear', dummy_indices=None, inference_method='mle'
    ):
        self.fun = fun
        self.intercept = intercept
        self.lamda0 = lamda0
        self.method = method
        self.form = form
        self.dummy_indices = dummy_indices if dummy_indices is not None else []
        self.inference_method = inference_method.lower()

        # Set orientation of the inefficiency term
        # Production: y = x'b + v - u (sign = 1)
        # Cost:       y = x'b + v + u (sign = -1)
        self.sign = 1 if self.fun == self.FUN_PROD else -1

        self.is_panel = (id_var is not None) and (time_var is not None)
        self.has_z = z is not None

        if self.is_panel and self.has_z:
            raise ValueError(
                "This class does not currently support combining "
                "Z variables (BC95) with Panel data (BC92)."
            )

        # Ensure proper input data types and transform functional form
        y_arr = np.array(y, dtype=float)
        x_arr = np.array(x, dtype=float)
        self.y, self.x, self.x_names = self.__transform_data(
            y_arr, x_arr, self.form, self.dummy_indices
        )

        # Setup Panel Data structures
        if self.is_panel:
            self._setup_panel(id_var, time_var)
        # Setup inefficiency determinants structures
        elif self.has_z:
            self._setup_z(z)
        else:
            self.z = None
            self.z_names = []

        # Internal state variables
        self.is_fitted = False
        self.estimation_method = None
        self._params = None
        self._std_err = None
        self._llf = np.nan
        self.pymc_trace = None

    def _setup_panel(self, id_var, time_var):
        """Initialize panel indices and time variables."""
        id_var = np.array(id_var)
        self.time_array = np.array(time_var, dtype=float)

        unique_ids = np.unique(id_var)
        self.num_firms = len(unique_ids)
        id_map = {val: i for i, val in enumerate(unique_ids)}
        self.firm_idx = np.array([id_map[val] for val in id_var])

        self.T_max_per_firm = np.zeros(self.num_firms)
        for i in range(self.num_firms):
            mask = self.firm_idx == i
            self.T_max_per_firm[i] = np.max(self.time_array[mask])

    def _setup_z(self, z):
        """Initialize Z variables for BC95 models."""
        z_array = np.array(z, dtype=float)
        if z_array.ndim == 1:
            z_array = np.atleast_2d(z_array)
        if z_array.shape[0] != len(self.y):
            z_array = z_array.T
            
        self.z = np.hstack((np.ones((z_array.shape[0], 1)), z_array))
        self.z_names = ['delta_0'] + [
            f"z{i+1}" for i in range(z_array.shape[1])
        ]

    def __transform_data(self, y, x, form, dummy_indices):
        """Transform raw data into the specified functional form."""
        x_2d = np.atleast_2d(x) if x.ndim == 1 else x
        n_obs, n_vars = x_2d.shape
        base_names = [f"x{i+1}" for i in range(n_vars)]

        if form == 'linear':
            return y, x_2d, base_names

        cont_idx = [i for i in range(n_vars) if i not in dummy_indices]
        x_cont = x_2d[:, cont_idx]

        if dummy_indices:
            x_dummies = x_2d[:, dummy_indices]
        else:
            x_dummies = np.empty((n_obs, 0))

        if np.any(y <= 0) or np.any(x_cont <= 0):
            raise ValueError(
                "Continuous variables must be strictly positive "
                "for log transformation."
            )

        log_y = np.log(y)
        log_x_cont = np.log(x_cont)
        cont_names = [f"ln_{base_names[i]}" for i in cont_idx]
        dummy_names = [f"d_{base_names[i]}" for i in dummy_indices]

        if form == 'cobb_douglas':
            if dummy_indices:
                final_x = np.hstack((log_x_cont, x_dummies))
            else:
                final_x = log_x_cont
            return log_y, final_x, cont_names + dummy_names

        elif form == 'translog':
            new_x_cols = [log_x_cont]
            final_names = list(cont_names)
            n_cont = len(cont_idx)

            for i in range(n_cont):
                for j in range(i, n_cont):
                    if i == j:
                        col = 0.5 * (log_x_cont[:, i] ** 2)
                        new_x_cols.append(col.reshape(-1, 1))
                        final_names.append(f"0.5*{cont_names[i]}^2")
                    else:
                        col = log_x_cont[:, i] * log_x_cont[:, j]
                        new_x_cols.append(col.reshape(-1, 1))
                        final_names.append(f"{cont_names[i]}*{cont_names[j]}")

            if dummy_indices:
                new_x_cols.append(x_dummies)
                final_names.extend(dummy_names)

            return log_y, np.hstack(new_x_cols), final_names

    def optimize(self):
        """Main estimation router."""
        if self.is_fitted:
            return

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if self.inference_method == 'mle':
                if self.is_panel:
                    self.__optimize_mle_panel()
                    self.estimation_method = 'MLE (Panel BC92)'
                else:
                    self.__optimize_mle()
                    self.estimation_method = 'MLE (Cross)'
            else:
                if self.is_panel:
                    self.__optimize_pymc_panel()
                    self.estimation_method = 'PyMC (Panel BC92)'
                else:
                    self.__optimize_pymc_cross()
                    self.estimation_method = 'PyMC (Cross)'

        self.is_fitted = True

    def __optimize_mle(self):
        """Maximum Likelihood Estimation for Cross-sectional & Z-models."""
        reg = LinearRegression(fit_intercept=self.intercept).fit(X=self.x, y=self.y)

        if self.intercept:
            beta_init = np.concatenate(([reg.intercept_], reg.coef_))
        else:
            beta_init = reg.coef_

        if self.has_z:
            delta_init = np.zeros(self.z.shape[1])
            parm = np.concatenate(
                (beta_init, delta_init, [np.var(reg.resid_), 0.5])
            )
        else:
            parm = np.concatenate((beta_init, [self.lamda0]))

        def __loglik(p):
            N = len(self.x)
            K = len(self.x[0]) + (1 if self.intercept else 0)

            if not self.has_z:
                beta0, lamda0 = p[0:K], p[K]
                if self.intercept:
                    y_pred = beta0[0] + np.dot(self.x, beta0[1:])
                else:
                    y_pred = np.dot(self.x, beta0)

                res = self.y - y_pred
                sig2 = np.sum(res**2) / N
                
                # Stabilized with log_ndtr
                ll = (
                    -0.5 * N * np.log(2 * math.pi) 
                    - 0.5 * N * np.log(sig2)
                    + np.sum(log_ndtr(-self.sign * res * lamda0 / math.sqrt(sig2)))
                    - 0.5 * np.sum(res**2) / sig2
                )
                return -ll
            else:
                beta = p[0:K]
                delta = p[K : K + self.z.shape[1]]
                sigma2, gamma = p[-2], p[-1]

                if sigma2 <= 0 or gamma <= 0 or gamma >= 1:
                    return 1e15

                if self.intercept:
                    y_pred = beta[0] + np.dot(self.x, beta[1:])
                else:
                    y_pred = np.dot(self.x, beta)

                eps = self.sign * (self.y - y_pred)
                mu = np.dot(self.z, delta)

                sigma_star = np.sqrt(gamma * (1 - gamma) * sigma2)
                mu_star = (1 - gamma) * mu - gamma * eps

                # Stabilized with log_ndtr
                ll = (
                    -0.5 * np.log(sigma2)
                    - 0.5 * ((eps + mu)**2 / sigma2)
                    + log_ndtr(mu_star / sigma_star)
                    - log_ndtr(mu / np.sqrt(gamma * sigma2))
                )
                return -np.sum(ll)

        method = 'L-BFGS-B' if self.has_z else 'BFGS'

        if self.has_z:
            bounds = (
                [(None, None)] * len(beta_init)
                + [(None, None)] * self.z.shape[1]
                + [(1e-6, None), (1e-6, 0.9999)]
            )
        else:
            bounds = None

        res = minimize(__loglik, parm, method=method, bounds=bounds)
        
        self._params = res.x
        self._llf = -res.fun

        try:
            hessian = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
            self._std_err = np.sqrt(np.diag(hessian))
        except (AttributeError, ValueError):
            self._std_err = np.full_like(res.x, np.nan)

    def __optimize_mle_panel(self):
        """Frequentist MLE for BC92."""
        if self.intercept:
            x_mat = np.hstack([np.ones((self.x.shape[0], 1)), self.x])
        else:
            x_mat = self.x

        # Initial OLS to get starting values
        reg = LinearRegression(fit_intercept=False).fit(x_mat, self.y)
        y_pred = reg.predict(x_mat)
        resid_var = np.var(self.y - y_pred)
        beta_ols = reg.coef_

        def bc92_ll(params):
            num_k = x_mat.shape[1]
            beta = params[:num_k]
            eta = params[num_k]
            sig2 = params[num_k + 1]
            gamma = params[num_k + 2]
            
            mu = 0.0 # Standard BC92

            if sig2 <= 0 or gamma <= 0 or gamma >= 0.999:
                return 1e15

            epsilon = self.y - np.dot(x_mat, beta)
            t_diff = self.time_array - self.T_max_per_firm[self.firm_idx]
            f_it = np.exp(-eta * t_diff)

            sig_u2 = gamma * sig2
            sig_v2 = (1 - gamma) * sig2
            total_ll = 0

            for i in range(self.num_firms):
                mask = self.firm_idx == i
                eps_i, f_i = epsilon[mask], f_it[mask]
                num_ti = len(eps_i)

                sum_f2 = np.sum(f_i**2)
                sum_f_eps = np.sum(f_i * eps_i)

                di_term = 1 + (sig_u2 / sig_v2) * sum_f2
                
                # Correctly handle sign for cost vs production
                mu_star = (
                    (mu * sig_v2 - self.sign * sig_u2 * sum_f_eps) /
                    (sig_v2 + sig_u2 * sum_f2)
                )
                sig_star = np.sqrt(
                    (sig_u2 * sig_v2) / (sig_v2 + sig_u2 * sum_f2)
                )

                # Stabilized likelihood calculation
                ll_i = (
                    -0.5 * num_ti * np.log(2 * np.pi * sig_v2)
                    - 0.5 * (np.sum(eps_i**2) / sig_v2)
                    - 0.5 * (mu**2 / sig_u2)
                    + 0.5 * (mu_star**2 / sig_star**2)
                    + log_ndtr(mu_star / sig_star)
                    - log_ndtr(mu / np.sqrt(sig_u2))
                    - 0.5 * np.log(di_term)
                )
                total_ll += ll_i

            return -total_ll

        # Grid Search for optimal starting values
        best_ll = float('inf')
        best_start = None

        for g_test in np.linspace(0.1, 0.9, 9):
            for e_test in [-0.05, 0.0, 0.05]:
                test_params = np.concatenate([
                    beta_ols, [e_test, resid_var, g_test]
                ])

                current_ll = bc92_ll(test_params)
                if current_ll < best_ll:
                    best_ll = current_ll
                    best_start = test_params

        bounds = (
            [(None, None)] * x_mat.shape[1] +
            [(None, None), (1e-6, None), (1e-6, 0.999)]
        )

        res = minimize(
            bc92_ll, best_start, method='L-BFGS-B', 
            bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-8}
        )

        self._params = res.x
        self._llf = -res.fun

        try:
            hessian = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
            self._std_err = np.sqrt(np.diag(hessian))
        except (AttributeError, ValueError):
            self._std_err = np.full_like(res.x, np.nan)

    def __optimize_pymc_cross(self):
        """Bayesian Estimation for Cross-sectional."""
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sigma=10, shape=len(self.x[0]))

            if self.intercept:
                beta0 = pm.Normal('beta0', mu=0, sigma=10)
                mu_y = beta0 + pm.math.dot(self.x, beta)
            else:
                mu_y = pm.math.dot(self.x, beta)

            sigma_v = pm.HalfNormal('sigma_v', sigma=5)
            sigma_u = pm.HalfNormal('sigma_u', sigma=5)

            if self.has_z:
                delta = pm.Normal('delta', mu=0, sigma=10, shape=self.z.shape[1])
                mu_u = pm.math.dot(self.z, delta)
                U = pm.TruncatedNormal(
                    'U', mu=mu_u, sigma=sigma_u, lower=0, shape=self.x.shape[0]
                )
            else:
                U = pm.HalfNormal('U', sigma=sigma_u, shape=self.x.shape[0])

            pm.Deterministic('TE', pm.math.exp(-U))

            mu_final = mu_y - U if self.sign == 1 else mu_y + U
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            trace = pm.sample(
                draws=1000, tune=1000, 
                progressbar=False, return_inferencedata=True
            )
            self.__extract_pymc_params(trace, model_type='cross')

    def __optimize_pymc_panel(self):
        """Bayesian Estimation for Panel Data (BC92)."""
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sigma=3, shape=len(self.x[0]))

            if self.intercept:
                beta0 = pm.Normal('beta0', mu=0, sigma=5)
                mu_y = beta0 + pm.math.dot(self.x, beta)
            else:
                mu_y = pm.math.dot(self.x, beta)

            sigma_v = pm.HalfNormal('sigma_v', sigma=1)
            sigma_u = pm.HalfNormal('sigma_u', sigma=1)

            mu = pm.Normal('mu', mu=0, sigma=1)
            eta = pm.Normal('eta', mu=0, sigma=0.2)

            U_i = pm.TruncatedNormal(
                'U_i', mu=mu, sigma=sigma_u, lower=0, shape=self.num_firms
            )

            time_diff = self.time_array - self.T_max_per_firm[self.firm_idx]
            decay = pm.math.exp(-eta * time_diff)

            U_it = pm.Deterministic('U_it', U_i[self.firm_idx] * decay)
            pm.Deterministic('TE', pm.math.exp(-U_it))

            mu_final = mu_y - U_it if self.sign == 1 else mu_y + U_it
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            trace = pm.sample(
                draws=2000, tune=2000, target_accept=0.99,
                progressbar=False, return_inferencedata=True
            )
            self.__extract_pymc_params(trace, model_type='panel')

    def __extract_pymc_params(self, trace, model_type):
        """Standardize PyMC output into standard numpy arrays."""
        self.pymc_trace = trace
        post = trace.posterior

        # Extract Betas
        beta_m = post['beta'].mean(dim=['chain', 'draw']).values
        beta_s = post['beta'].std(dim=['chain', 'draw']).values

        if self.intercept:
            betas = np.concatenate(([post['beta0'].mean().values], beta_m))
            betas_se = np.concatenate(([post['beta0'].std().values], beta_s))
        else:
            betas, betas_se = beta_m, beta_s

        sigma_u_post = post['sigma_u']
        sigma_v_post = post['sigma_v']

        sigma2_post = sigma_u_post**2 + sigma_v_post**2
        gamma_post = (sigma_u_post**2) / sigma2_post

        sigma2_m = sigma2_post.mean().values
        sigma2_s = sigma2_post.std().values
        gamma_m = gamma_post.mean().values
        gamma_s = gamma_post.std().values

        if model_type == 'panel':
            eta_m = post['eta'].mean().values
            eta_s = post['eta'].std().values
            mu_m = post['mu'].mean().values
            mu_s = post['mu'].std().values

            self._params = np.concatenate((betas, [mu_m, eta_m, sigma2_m, gamma_m]))
            self._std_err = np.concatenate((betas_se, [mu_s, eta_s, sigma2_s, gamma_s]))

        elif self.has_z:
            delta_m = post['delta'].mean(dim=['chain', 'draw']).values
            delta_s = post['delta'].std(dim=['chain', 'draw']).values

            self._params = np.concatenate((betas, delta_m, [sigma2_m, gamma_m]))
            self._std_err = np.concatenate((betas_se, delta_s, [sigma2_s, gamma_s]))

        else:
            lam_post = sigma_u_post / sigma_v_post
            self._params = np.concatenate((betas, [lam_post.mean().values]))
            self._std_err = np.concatenate((betas_se, [lam_post.std().values]))

        self._llf = np.nan

    def get_beta(self):
        self.optimize()
        K = len(self.x[0]) + (1 if self.intercept else 0)
        return self._params[0:K]

    def get_residuals(self):
        self.optimize()
        beta = self.get_beta()
        if self.intercept:
            return self.y - beta[0] - np.dot(self.x, beta[1:])
        return self.y - np.dot(self.x, beta)

    def get_lambda(self):
        self.optimize()
        if self.has_z or self.is_panel:
            gamma = self._params[-1]
            return np.sqrt(gamma / (1 - gamma))
        return self._params[-1]

    def get_sigma2(self):
        self.optimize()
        if self.has_z or self.is_panel:
            return self._params[-2]
        return np.sum(self.get_residuals()**2) / len(self.x)

    def __teJ(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        self.sstar = (lam / (1 + lam**2)) * math.sqrt(self.get_sigma2())
        ratio = self.ustar / self.sstar
        
        # Use log_ndtr to prevent division by zero in normal cdf
        log_term = norm.logpdf(ratio) - log_ndtr(ratio)
        return np.exp(-self.ustar - self.sstar * np.exp(log_term))

    def __te(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        self.sstar = (lam / (1 + lam**2)) * math.sqrt(self.get_sigma2())
        ratio = self.ustar / self.sstar
        
        log_term = log_ndtr(ratio - self.sstar) - log_ndtr(ratio)
        return np.exp(log_term + (self.sstar**2 / 2) - self.ustar)

    def __teMod(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        return np.exp(np.minimum(0, -self.ustar))

    def get_technical_efficiency(self):
        """Returns technical efficiency based on the selected method."""
        self.optimize()
        if self.estimation_method and 'PyMC' in self.estimation_method:
            return self.pymc_trace.posterior['TE'].mean(
                dim=['chain', 'draw']
            ).values
            
        if self.method == self.TE_teJ:
            return self.__teJ()
        elif self.method == self.TE_te:
            return self.__te()
        elif self.method == self.TE_teMod:
            return self.__teMod()
        else:
            raise ValueError("Undefined decomposition technique.")

    def summary(self):
        """Print the summary of the SFA model with significance stars."""
        self.optimize()

        # Build dynamic names based on model type
        if self.intercept:
            names = ['(Intercept)'] + list(self.x_names)
        else:
            names = list(self.x_names)

        if self.is_panel:
            missing_count = len(self._params) - len(names)
            # MLE estimates 3 extra params (eta, sig2, gamma)
            # PyMC estimates 4 extra params (mu, eta, sig2, gamma)
            if missing_count == 4:
                names += ['mu', 'eta', 'sigma2', 'gamma']
            else:
                names += ['eta', 'sigma2', 'gamma']
        elif self.has_z:
            names += self.z_names + ['sigma2', 'gamma']
        else:
            names += ['lambda']

        # Failsafe alignment to avoid pandas KeyError
        params = self._params[:len(names)]
        std_err = self._std_err[:len(names)]

        # Calculate z-values and p-values safely
        with np.errstate(divide='ignore', invalid='ignore'):
            z_values = params / std_err
            p_values = 2 * norm.sf(np.abs(z_values))

        # Significance mapping
        stars = []
        for p in p_values:
            if np.isnan(p):
                stars.append('')
            elif p < 0.01:
                stars.append('***')
            elif p < 0.05:
                stars.append('**')
            elif p < 0.10:
                stars.append('*')
            else:
                stars.append('')

        res_table = pd.DataFrame(
            {
                'Estimate': np.round(params, 5),
                'Std. Error': np.round(std_err, 6),
                'z value': np.round(z_values, 3),
                'Pr(>|z|)': np.round(p_values, 4),
                'Sig.': stars,
            },
            index=names
        )

        print(f"\nStochastic Frontier Analysis ({self.estimation_method})")
        print("=" * 75)
        print(res_table.to_string(na_rep='NaN'))
        print("-" * 75)
        print("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")

        if not np.isnan(self._llf):
            print(f"Log-Likelihood:  {self._llf:.5f}")
        print("=" * 75)