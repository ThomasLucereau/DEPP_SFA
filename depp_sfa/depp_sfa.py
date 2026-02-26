import math

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import pymc as pm

from . import constant


FUN_COST = constant.FUN_COST
FUN_PROD = constant.FUN_PROD
TE_teJ = constant.TE_teJ
TE_te = constant.TE_te
TE_teMod = constant.TE_teMod


class SFA:
    """Stochastic Frontier Analysis (SFA)
    Supports Cross-sectional (ALS77, BC95) and Panel Data (BC92 Time-varying).
    """

    # Bind constants to class attributes for easier access
    FUN_PROD = FUN_PROD
    FUN_COST = FUN_COST
    TE_teJ = TE_teJ
    TE_te = TE_te
    TE_teMod = TE_teMod

    def __init__(
        self, y, x, z=None, id_var=None, time_var=None,
        fun=FUN_PROD, intercept=True, lamda0=1, method=TE_teJ,
        form='linear', dummy_indices=None
    ):
        self.fun = fun
        self.intercept = intercept
        self.lamda0 = lamda0
        self.method = method
        self.form = form
        self.dummy_indices = dummy_indices if dummy_indices is not None else []
        self.sign = -1 if self.fun == self.FUN_COST else 1

        # Ensure proper input data types
        y = np.array(y, dtype=float)
        x = np.array(x, dtype=float)

        # Transform production data (X and Y)
        self.y, self.x, self.x_names = self.__transform_data(
            y, x, self.form, self.dummy_indices
        )

        # Determine model type based on inputs
        self.is_panel = (id_var is not None) and (time_var is not None)
        self.has_z = z is not None

        if self.is_panel and self.has_z:
            raise ValueError(
                "This class does not currently support combining "
                "Z variables (BC95) with Panel data (BC92). Choose one."
            )

        # Setup Panel Data (BC92)
        if self.is_panel:
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

        # Setup inefficiency determinants (BC95)
        elif self.has_z:
            z_array = np.array(z, dtype=float)
            if z_array.ndim == 1:
                z_array = np.atleast_2d(z_array)
            self.z = np.hstack((np.ones((z_array.shape[0], 1)), z_array))
            self.z_names = ['delta_0'] + [
                f"z{i+1}" for i in range(z_array.shape[1])
            ]
        else:
            self.z = None
            self.z_names = []

        # State variables
        self.is_fitted = False
        self.estimation_method = None
        self._params = None
        self._std_err = None
        self._llf = None
        self.pymc_trace = None

    def __transform_data(self, y, x, form, dummy_indices):
        """Transform raw data into specified functional form."""
        y = np.array(y, dtype=float)
        x = np.array(x, dtype=float)

        x_2d = np.atleast_2d(x) if x.ndim == 1 else x
        n_obs, n_vars = x_2d.shape
        base_names = [f"x{i+1}" for i in range(n_vars)]

        if form == 'linear':
            return y, x_2d, base_names

        cont_indices = [i for i in range(n_vars) if i not in dummy_indices]
        x_cont = x_2d[:, cont_indices]

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
        cont_names = [f"ln_{base_names[i]}" for i in cont_indices]
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
            n_cont = len(cont_indices)

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

        import warnings
        import logging

        # Temporarily suppress standard warnings during estimation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Temporarily suppress PyMC logging
            logger = logging.getLogger("pymc")
            old_level = logger.level
            logger.setLevel(logging.ERROR)

            try:
                if self.is_panel:
                    self.__optimize_pymc_panel()
                    self.estimation_method = 'PyMC (Panel BC92)'
                else:
                    try:
                        self.__optimize_mle()
                        self.estimation_method = 'MLE'
                    except Exception as e:
                        print(
                            f"[Warning] MLE convergence failed ({str(e)}). "
                            "Fallback to PyMC sampling..."
                        )
                        self.__optimize_pymc_cross()
                        self.estimation_method = 'PyMC (Cross)'
            finally:
                # Restore original logging level after estimation
                logger.setLevel(old_level)

        self.is_fitted = True

    def __optimize_mle(self):
        """Maximum Likelihood Estimation logic (Cross-sectional only)."""
        fit_inter = self.intercept
        reg = LinearRegression(fit_intercept=fit_inter).fit(X=self.x, y=self.y)

        if fit_inter:
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

        def __loglik(parm):
            N = len(self.x)
            K = len(self.x[0]) + (1 if self.intercept else 0)

            if not self.has_z:
                beta0, lamda0 = parm[0:K], parm[K]
                if self.intercept:
                    y_pred = beta0[0] + np.dot(self.x, beta0[1:])
                else:
                    y_pred = np.dot(self.x, beta0)

                res = self.y - y_pred
                sig2 = np.sum(res**2) / N
                z_val = -lamda0 * self.sign * res / math.sqrt(sig2)
                pz = np.maximum(norm.cdf(z_val), 1e-323)

                ll = (
                    (N / 2) * math.log(math.pi / 2)
                    + (N / 2) * math.log(sig2)
                    - np.sum(np.log(pz))
                    + N / 2.0
                )
                return ll
            else:
                beta = parm[0:K]
                delta = parm[K: K + self.z.shape[1]]
                sigma2, gamma = parm[-2], parm[-1]

                if sigma2 <= 0 or gamma <= 0 or gamma >= 1:
                    return 1e10

                if self.intercept:
                    y_pred = beta[0] + np.dot(self.x, beta[1:])
                else:
                    y_pred = np.dot(self.x, beta)

                eps = self.sign * (self.y - y_pred)
                mu = np.dot(self.z, delta)

                sigma_star = np.sqrt(gamma * (1 - gamma) * sigma2)
                mu_star = (1 - gamma) * mu - gamma * eps
                ratio = np.clip(mu_star / sigma_star, -10, 10)

                ll = (
                    -0.5 * np.log(sigma2)
                    - 0.5 * ((eps + mu)**2 / sigma2)
                    + np.log(np.maximum(norm.cdf(ratio), 1e-323))
                    - np.log(
                        np.maximum(
                            norm.cdf(mu / np.sqrt(gamma * sigma2)), 1e-323
                        )
                    )
                )
                return -np.sum(ll)

        method = 'L-BFGS-B' if self.has_z else 'BFGS'

        if self.has_z:
            bounds = (
                [(None, None)] * len(beta_init)
                + [(None, None)] * self.z.shape[1]
                + [(1e-5, None), (1e-5, 0.9999)]
            )
        else:
            bounds = None

        res = minimize(__loglik, parm, method=method, bounds=bounds)
        if not res.success or np.isnan(res.fun):
            raise ValueError(res.message)

        self._params = res.x

        if hasattr(res.hess_inv, 'todense'):
            hessian = res.hess_inv.todense()
        else:
            hessian = res.hess_inv

        self._std_err = np.sqrt(np.diag(hessian))
        self._llf = -res.fun

    def __optimize_pymc_cross(self):
        """Bayesian Fallback for Cross-sectional."""
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sigma=10, shape=len(self.x[0]))

            if self.intercept:
                beta0 = pm.Normal('beta0', 0, 10)
                mu_y = beta0 + pm.math.dot(self.x, beta)
            else:
                mu_y = pm.math.dot(self.x, beta)

            sigma_v = pm.HalfNormal('sigma_v', sigma=5)
            sigma_u = pm.HalfNormal('sigma_u', sigma=5)

            if self.has_z:
                delta = pm.Normal(
                    'delta', mu=0, sigma=10, shape=self.z.shape[1]
                )
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
            # Priors
            beta = pm.Normal('beta', mu=0, sigma=3, shape=len(self.x[0]))

            if self.intercept:
                beta0 = pm.Normal('beta0', mu=0, sigma=5)
                mu_y = beta0 + pm.math.dot(self.x, beta)
            else:
                mu_y = pm.math.dot(self.x, beta)

            sigma_v = pm.HalfNormal('sigma_v', sigma=1)
            sigma_u = pm.HalfNormal('sigma_u', sigma=1)

            # Panel Parameters
            mu = pm.Normal('mu', mu=0, sigma=1)
            eta = pm.Normal('eta', mu=0, sigma=0.2)

            # Inefficiency
            U_i = pm.TruncatedNormal(
                'U_i', mu=mu, sigma=sigma_u, lower=0, shape=self.num_firms
            )

            time_diff = self.time_array - self.T_max_per_firm[self.firm_idx]
            decay = pm.math.exp(-eta * time_diff)

            U_it = pm.Deterministic('U_it', U_i[self.firm_idx] * decay)
            pm.Deterministic('TE', pm.math.exp(-U_it))

            mu_final = mu_y - U_it if self.sign == 1 else mu_y + U_it
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            # Sampling
            trace = pm.sample(
                draws=2000,
                tune=2000,
                target_accept=0.99,
                progressbar=True,
                return_inferencedata=True
            )
            self.__extract_pymc_params(trace, model_type='panel')

    def __extract_pymc_params(self, trace, model_type):
        """Standardize PyMC output into standard numpy arrays."""
        self.pymc_trace = trace
        post = trace.posterior

        beta_m = post['beta'].mean(dim=['chain', 'draw']).values
        beta_s = post['beta'].std(dim=['chain', 'draw']).values

        if self.intercept:
            betas = np.concatenate(([post['beta0'].mean().values], beta_m))
            betas_se = np.concatenate(([post['beta0'].std().values], beta_s))
        else:
            betas, betas_se = beta_m, beta_s

        su_m = post['sigma_u'].mean().values
        sv_m = post['sigma_v'].mean().values

        if model_type == 'panel':
            eta_m, mu_m = post['eta'].mean().values, post['mu'].mean().values
            eta_s, mu_s = post['eta'].std().values, post['mu'].std().values
            gamma = (su_m**2) / (su_m**2 + sv_m**2)

            self._params = np.concatenate(
                (betas, [mu_m, eta_m, su_m**2 + sv_m**2, gamma])
            )
            self._std_err = np.concatenate(
                (betas_se, [mu_s, eta_s, np.nan, np.nan])
            )

        elif self.has_z:
            delta_m = post['delta'].mean(dim=['chain', 'draw']).values
            delta_s = post['delta'].std(dim=['chain', 'draw']).values
            gamma = (su_m**2) / (su_m**2 + sv_m**2)

            self._params = np.concatenate(
                (betas, delta_m, [su_m**2 + sv_m**2, gamma])
            )
            self._std_err = np.concatenate(
                (betas_se, delta_s, [np.nan, np.nan])
            )

        else:
            self._params = np.concatenate((betas, [su_m / sv_m]))
            self._std_err = np.concatenate((betas_se, [np.nan]))

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
        return np.exp(-self.ustar - self.sstar * (
            norm.pdf(ratio) / norm.cdf(ratio)
        ))

    def __te(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        self.sstar = (lam / (1 + lam**2)) * math.sqrt(self.get_sigma2())
        ratio = self.ustar / self.sstar
        return (norm.cdf(ratio - self.sstar) / norm.cdf(ratio)) * np.exp(
            self.sstar**2 / 2 - self.ustar
        )

    def __teMod(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        return np.exp(np.minimum(0, -self.ustar))

    def get_technical_efficiency(self):
        """Returns technical efficiency based on the estimation method."""
        self.optimize()
        if (
            self.estimation_method
            and self.estimation_method.startswith('PyMC')
        ):
            return self.pymc_trace.posterior['TE'].mean(
                dim=['chain', 'draw']
            ).values
        else:
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

        # Prepare variable names
        if self.intercept:
            names = ['(Intercept)'] + self.x_names
        else:
            names = self.x_names

        if self.is_panel:
            names += ['mu', 'eta', 'sigma2', 'gamma']
        elif self.has_z:
            names += self.z_names + ['sigma2', 'gamma']
        else:
            names += ['lambda']

        params = self._params
        std_err = self._std_err

        # Calculate t-values and p-values
        with np.errstate(divide='ignore', invalid='ignore'):
            t_values = params / std_err
            p_values = 2 * norm.sf(np.abs(t_values))

        # Assign significance stars
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

        # Create presentation table (Pandas DataFrame)
        re = pd.DataFrame({
            'Estimate': np.round(params, 5),
            'Std. Error': np.round(std_err, 6),
            't value': np.round(t_values, 3),
            'Pr(>|t|)': np.round(p_values, 4),
            'Sig.': stars
        }, index=names)

        # Final output display
        print(f"\nStochastic Frontier Analysis ({self.estimation_method})")
        print("=" * 75)
        print(re.to_string(na_rep='NaN'))
        print("-" * 75)
        print("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")
        if not np.isnan(self._llf):
            print(f"Log-Likelihood:  {self._llf:.5f}")
        print("=" * 75)
