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
        form='linear', dummy_indices=None, inference_method='bayesian'
    ):
        self.fun = fun
        self.intercept = intercept
        self.lamda0 = lamda0
        self.method = method
        self.form = form
        self.dummy_indices = dummy_indices if dummy_indices is not None else []
        self.sign = -1 if self.fun == self.FUN_COST else 1
        self.inference_method = inference_method.lower()
        self.sign = -1 if self.fun == self.FUN_COST else 1

        # Internal flag for model type
        self.is_panel = (id_var is not None) and (time_var is not None)

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
        """
        Main estimation router: Switches between MLE and Bayesian inference.
        Includes warning suppression for cleaner console output during
        optimization.
        """
        if self.is_fitted:
            return

        import warnings
        import logging

        # Mute RuntimeWarnings during optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Mute PyMC internal logging to keep the terminal clean
            logger = logging.getLogger("pymc")
            old_level = logger.level
            logger.setLevel(logging.ERROR)

            try:
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

            finally:
                # Restore logging level after estimation
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

    def __optimize_mle_panel(self):
        """
        Frequentist MLE for BC92.
        Replicates R 'frontier' package logic (Battese & Coelli 1992).
        """
        from scipy.special import ndtr

        # Prepare data matrix
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
            sig2, gamma = params[num_k + 1], params[num_k + 2]

            mu = 0.0

            # Hard constraints for stability
            if sig2 <= 0 or gamma <= 0 or gamma >= 1:
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
                mu_star = (
                    (mu * sig_v2 - self.sign * sig_u2 * sum_f_eps) /
                    (sig_v2 + sig_u2 * sum_f2)
                )
                sig_star = np.sqrt(
                    (sig_u2 * sig_v2) / (sig_v2 + sig_u2 * sum_f2)
                )

                ll_i = (
                    -0.5 * num_ti * np.log(2 * np.pi * sig_v2)
                    - 0.5 * (np.sum(eps_i**2) / sig_v2)
                    - 0.5 * (mu**2 / sig_u2)
                    + 0.5 * (mu_star**2 / sig_star**2)
                    + np.log(np.maximum(ndtr(mu_star / sig_star), 1e-20))
                    - np.log(np.maximum(ndtr(mu / np.sqrt(sig_u2)), 1e-20))
                    - 0.5 * np.log(di_term)
                )
                total_ll += ll_i

            return -total_ll

        # R-Style Grid Search (L'arme secrète pour trouver le même optimum)
        best_ll = float('inf')
        best_start = None

        for g_test in np.linspace(0.1, 0.9, 9):

            for e_test in [-0.05, 0.0, 0.05]:
                test_params = np.concatenate([beta_ols,
                                              [e_test, resid_var, g_test]])

                current_ll = bc92_ll(test_params)

                if current_ll < best_ll:
                    best_ll = current_ll
                    best_start = test_params

        # Optimization depuis le MEILLEUR point de départ trouvé
        bounds = (
            [(None, None)] * x_mat.shape[1] +
            [(None, None), (1e-6, None), (1e-6, 0.9999)]
        )

        res = minimize(
            bc92_ll,
            best_start,
            method='L-BFGS-B',
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-8}
        )

        self._params = res.x
        self._llf = -res.fun

        # Standard Errors (from Hessian)
        try:
            self._std_err = np.sqrt(np.diag(res.hess_inv.todense()))
        except (AttributeError, ValueError):
            self._std_err = np.full_like(res.x, np.nan)

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
        """
        Standardize PyMC output into standard numpy arrays.
        Calculates standard errors for derived parameters eg. sigma2 and gamma.
        """
        self.pymc_trace = trace
        post = trace.posterior

        # Extract Betas
        beta_m = post['beta'].mean(dim=['chain', 'draw']).values
        beta_s = post['beta'].std(dim=['chain', 'draw']).values

        if self.intercept:
            betas = np.concatenate(
                ([post['beta0'].mean().values], beta_m)
            )
            betas_se = np.concatenate(
                ([post['beta0'].std().values], beta_s)
            )
        else:
            betas, betas_se = beta_m, beta_s

        # 1. Extract full posterior distributions for sigmas
        sigma_u_post = post['sigma_u']
        sigma_v_post = post['sigma_v']

        # 2. Derive posterior distributions for sigma2 and gamma
        sigma2_post = sigma_u_post**2 + sigma_v_post**2
        gamma_post = (sigma_u_post**2) / sigma2_post

        # 3. Calculate Means and Standard Errors (Std Dev of posterior)
        sigma2_m = sigma2_post.mean().values
        sigma2_s = sigma2_post.std().values
        gamma_m = gamma_post.mean().values
        gamma_s = gamma_post.std().values

        if model_type == 'panel':
            eta_m = post['eta'].mean().values
            eta_s = post['eta'].std().values
            mu_m = post['mu'].mean().values
            mu_s = post['mu'].std().values

            self._params = np.concatenate(
                (betas, [mu_m, eta_m, sigma2_m, gamma_m])
            )
            self._std_err = np.concatenate(
                (betas_se, [mu_s, eta_s, sigma2_s, gamma_s])
            )

        elif self.has_z:
            delta_m = post['delta'].mean(dim=['chain', 'draw']).values
            delta_s = post['delta'].std(dim=['chain', 'draw']).values

            self._params = np.concatenate(
                (betas, delta_m, [sigma2_m, gamma_m])
            )
            self._std_err = np.concatenate(
                (betas_se, delta_s, [sigma2_s, gamma_s])
            )

        else:
            # Cross-sectional basic model
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
        """
        Print the summary of the SFA model with significance stars.
        Outputs a clean, R-style regression table.
        """
        self.optimize()

        # Base parameter names (Betas)
        if self.intercept:
            names = ['(Intercept)'] + list(self.x_names)
        else:
            names = list(self.x_names)

        # Dynamically add the exact number of missing variance/panel names
        num_betas = len(names)
        num_params = len(self._params)
        missing_count = num_params - num_betas

        if self.is_panel:
            if missing_count == 3:
                names += ['eta', 'sigma2', 'gamma']
            else:
                names += ['mu', 'eta', 'sigma2', 'gamma']
        elif self.has_z:
            names += self.z_names + ['sigma2', 'gamma']
        else:
            # Cross-sectional model
            if missing_count == 2:
                names += ['sigma2', 'gamma']
            elif missing_count == 1:
                names += ['lambda']
            else:
                names += [f'var_{i}' for i in range(missing_count)]

        # Extract parameters safely
        params = self._params
        std_err = self._std_err

        # Calculate z-values and p-values
        with np.errstate(divide='ignore', invalid='ignore'):
            z_values = params / std_err
            p_values = 2 * norm.sf(np.abs(z_values))

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

        # Final output display
        print(f"\nStochastic Frontier Analysis ({self.estimation_method})")
        print("=" * 75)
        print(res_table.to_string(na_rep='NaN'))
        print("-" * 75)
        print("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")

        # Print Log-Likelihood if available (MLE mode)
        if not np.isnan(self._llf):
            print(f"Log-Likelihood:  {self._llf:.5f}")
        print("=" * 75)
