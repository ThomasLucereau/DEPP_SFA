from inspect import trace
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
    Supports Cross-sectional (ALS77, BC95), Panel Data (BC92) and True Effects (Greene 2005).
    """

    FUN_PROD = FUN_PROD
    FUN_COST = FUN_COST
    TE_teJ = TE_teJ
    TE_te = TE_te
    TE_teMod = TE_teMod

    def __init__(
        self, y, x, z=None, id_var=None, time_var=None,
        fun=FUN_PROD, intercept=True, lamda0=1, method=TE_teJ,
        form='linear', dummy_indices=None, inference_method='mle',
        panel_model='bc92', draws=1500, tune=1500
    ):
        self.fun = fun
        self.intercept = intercept
        self.lamda0 = lamda0
        self.method = method
        self.form = form
        self.dummy_indices = dummy_indices if dummy_indices is not None else []
        self.inference_method = inference_method.lower()
        self.panel_model = panel_model.lower()
        self.draws = draws
        self.tune = tune

        self.sign = 1 if self.fun == self.FUN_PROD else -1

        self.is_panel = (id_var is not None) and (time_var is not None)
        self.has_z = z is not None

        if self.is_panel and self.has_z:
            raise ValueError(
                "This class does not currently support combining "
                "Z variables (BC95) with Panel data."
            )

        y_arr = np.array(y, dtype=float)
        x_arr = np.array(x, dtype=float)
        self.y, self.x, self.x_names = self.__transform_data(
            y_arr, x_arr, self.form, self.dummy_indices
        )

        if self.is_panel:
            self._setup_panel(id_var, time_var)
        elif self.has_z:
            self._setup_z(z)
        else:
            self.z = None
            self.z_names = []

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

        # === GREENE 2005 (TFE) ===
        if self.panel_model == 'greene' and self.inference_method == 'mle':
            firm_dummies = pd.get_dummies(id_var, drop_first=True).astype(float).values
            self.x = np.hstack((self.x, firm_dummies))
            dummy_names = [f"Firme_{uid}" for uid in np.unique(id_var)[1:]]
            self.x_names.extend(dummy_names)

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
        if self.is_fitted: return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # === BRANCHE MLE ===
            if self.inference_method == 'mle':
                if self.is_panel:
                    if self.panel_model == 'greene':
                        self.__optimize_mle()
                        self.estimation_method = 'MLE (Greene 2005 TFE)'
                    else:
                        self.__optimize_mle_panel()
                        self.estimation_method = 'MLE (Panel BC92)'
                else:
                    self.__optimize_mle()
                    self.estimation_method = 'MLE (BC95 Inefficiency Effects)' if self.has_z else 'MLE (Cross-sectional)'

            # === BRANCHE PyMC ===
            elif self.inference_method == 'pymc':
                if self.is_panel:
                    if self.panel_model == 'greene':
                        self.__optimize_pymc_greene_tre()
                        self.estimation_method = 'PyMC (Greene 2005 TRE)'
                    else:
                        self.__optimize_pymc_panel()
                        self.estimation_method = 'PyMC (Panel BC92)'
                else:
                    self.__optimize_pymc_cross()
                    self.estimation_method = 'PyMC (BC95 Inefficiency Effects)' if self.has_z else 'PyMC (Cross-sectional)'

        self.is_fitted = True

    def __optimize_mle(self):
        """Maximum Likelihood Estimation for Cross-sectional, Z-models & TFE."""
        reg = LinearRegression(fit_intercept=self.intercept).fit(X=self.x, y=self.y)

        if self.intercept:
            beta_init = np.concatenate(([reg.intercept_], reg.coef_))
        else:
            beta_init = reg.coef_

        K = len(beta_init)
        N = len(self.x)

        y_pred_init = reg.predict(self.x)
        resid_var = np.var(self.y - y_pred_init)

        if self.has_z:
            delta_init = np.zeros(self.z.shape[1])
            parm = np.concatenate((beta_init, delta_init, [resid_var, 0.5]))
        else:
            lam0 = self.lamda0
            gamma_init = (lam0**2) / (1 + lam0**2)
            parm = np.concatenate((beta_init, [resid_var, gamma_init]))

        def __loglik(p):
            if not self.has_z:
                beta = p[0:K]
                sigma2, gamma = p[-2], p[-1]

                if sigma2 <= 0 or gamma <= 0 or gamma >= 0.9999:
                    return 1e15

                y_pred = beta[0] + np.dot(self.x, beta[1:]) if self.intercept else np.dot(self.x, beta)
                res = self.y - y_pred
                lam = np.sqrt(gamma / (1 - gamma))
                
                ll = (
                    -0.5 * N * np.log(2 * math.pi) 
                    - 0.5 * N * np.log(sigma2)
                    + np.sum(log_ndtr(-self.sign * res * lam / math.sqrt(sigma2)))
                    - 0.5 * np.sum(res**2) / sigma2
                )
                return -ll
            else:
                beta = p[0:K]
                delta = p[K : K + self.z.shape[1]]
                sigma2, gamma = p[-2], p[-1]

                if sigma2 <= 0 or gamma <= 0 or gamma >= 0.9999:
                    return 1e15

                y_pred = beta[0] + np.dot(self.x, beta[1:]) if self.intercept else np.dot(self.x, beta)
                eps = self.sign * (self.y - y_pred)
                mu = np.dot(self.z, delta)

                sigma_star = np.sqrt(gamma * (1 - gamma) * sigma2)
                mu_star = (1 - gamma) * mu - gamma * eps

                ll = (
                    -0.5 * np.log(sigma2)
                    - 0.5 * ((eps + mu)**2 / sigma2)
                    + log_ndtr(mu_star / sigma_star)
                    - log_ndtr(mu / np.sqrt(gamma * sigma2))
                )
                return -np.sum(ll)

        method = 'L-BFGS-B'

        if self.has_z:
            bounds = ([(None, None)] * K + [(None, None)] * self.z.shape[1] + [(1e-6, None), (1e-6, 0.9999)])
        else:
            bounds = ([(None, None)] * K + [(1e-6, None), (1e-6, 0.9999)])

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
        x_mat = np.hstack([np.ones((self.x.shape[0], 1)), self.x]) if self.intercept else self.x

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

            mu = 0.0  

            if sig2 <= 0 or gamma <= 0 or gamma >= 0.9999:
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

                mu_star = ((mu * sig_v2 - self.sign * sig_u2 * sum_f_eps) / (sig_v2 + sig_u2 * sum_f2))
                sig_star = np.sqrt((sig_u2 * sig_v2) / (sig_v2 + sig_u2 * sum_f2))

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

        best_ll = float('inf')
        best_start = None

        for g_test in np.linspace(0.1, 0.9, 9):
            for e_test in [-0.05, 0.0, 0.05]:
                test_params = np.concatenate([beta_ols, [e_test, resid_var, g_test]])
                current_ll = bc92_ll(test_params)
                if current_ll < best_ll:
                    best_ll = current_ll
                    best_start = test_params

        bounds = ([(None, None)] * x_mat.shape[1] + [(None, None), (1e-6, None), (1e-6, 0.9999)])
        res = minimize(bc92_ll, best_start, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-8})

        self._params = res.x
        self._llf = -res.fun

        try:
            hessian = res.hess_inv.todense() if hasattr(res.hess_inv, 'todense') else res.hess_inv
            self._std_err = np.sqrt(np.diag(hessian))
        except (AttributeError, ValueError):
            self._std_err = np.full_like(res.x, np.nan)

    def __optimize_pymc_cross(self):
        """Bayesian Estimation for Cross-sectional & BC95."""
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
                U = pm.TruncatedNormal('U', mu=mu_u, sigma=sigma_u, lower=0, shape=self.x.shape[0])
            else:
                U = pm.HalfNormal('U', sigma=sigma_u, shape=self.x.shape[0])

            pm.Deterministic('TE', pm.math.exp(-U))

            mu_final = mu_y - U if self.sign == 1 else mu_y + U
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            trace = pm.sample(draws=self.draws, tune=self.tune, progressbar=False, return_inferencedata=True)
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

            U_i = pm.TruncatedNormal('U_i', mu=mu, sigma=sigma_u, lower=0, shape=self.num_firms)

            time_diff = self.time_array - self.T_max_per_firm[self.firm_idx]
            decay = pm.math.exp(-eta * time_diff)

            U_it = pm.Deterministic('U_it', U_i[self.firm_idx] * decay)
            pm.Deterministic('TE', pm.math.exp(-U_it))

            mu_final = mu_y - U_it if self.sign == 1 else mu_y + U_it
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            trace = pm.sample(draws=self.draws, tune=self.tune, target_accept=0.999, progressbar=False, return_inferencedata=True)
            self.__extract_pymc_params(trace, model_type='panel')

    def __optimize_pymc_greene_tre(self):
        with pm.Model() as model:
            beta = pm.Normal('beta', mu=0, sigma=5, shape=len(self.x[0]))
            
            mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=5)
            sigma_alpha = pm.HalfNormal('sigma_alpha', sigma=2)
            alpha_offset = pm.Normal('alpha_offset', mu=0, sigma=1, shape=self.num_firms)
            alpha_i = pm.Deterministic('alpha_i', mu_alpha + alpha_offset * sigma_alpha)

            mu_y = alpha_i[self.firm_idx] + pm.math.dot(self.x, beta)

            sigma_u = pm.HalfNormal('sigma_u', sigma=2)
            U_it = pm.HalfNormal('U_it', sigma=sigma_u, shape=len(self.y))
            sigma_v = pm.HalfNormal('sigma_v', sigma=2)

            pm.Deterministic('TE', pm.math.exp(-U_it))

            mu_final = mu_y - U_it if self.sign == 1 else mu_y + U_it
            pm.Normal('Y_obs', mu=mu_final, sigma=sigma_v, observed=self.y)

            trace = pm.sample(
                draws=self.draws, 
                tune=self.tune, 
                target_accept=0.999, 
                progressbar=True, 
                return_inferencedata=True
            )
            self.__extract_pymc_params(trace, model_type='tre')

    def __extract_pymc_params(self, trace, model_type):
        """Standardize PyMC output to perfectly match MLE structure."""
        self.pymc_trace = trace
        post = trace.posterior

        beta_m = post['beta'].mean(dim=['chain', 'draw']).values
        beta_s = post['beta'].std(dim=['chain', 'draw']).values

        if self.intercept and model_type != 'tre': 
            betas = np.concatenate(([post['beta0'].mean().values], beta_m))
            betas_se = np.concatenate(([post['beta0'].std().values], beta_s))
        else:
            betas, betas_se = beta_m, beta_s

        if model_type == 'tre':
            mu_alpha_m, mu_alpha_s = post['mu_alpha'].mean().values, post['mu_alpha'].std().values
            s_alpha_m, s_alpha_s = post['sigma_alpha'].mean().values, post['sigma_alpha'].std().values
            su_m, su_s = post['sigma_u'].mean().values, post['sigma_u'].std().values
            sv_m, sv_s = post['sigma_v'].mean().values, post['sigma_v'].std().values
            self._params = np.concatenate((betas, [mu_alpha_m, s_alpha_m, su_m, sv_m]))
            self._std_err = np.concatenate((betas_se, [mu_alpha_s, s_alpha_s, su_s, sv_s]))
        else:
            sigma_u_post = post['sigma_u']
            sigma_v_post = post['sigma_v']
            sigma2_post = sigma_u_post**2 + sigma_v_post**2
            gamma_post = (sigma_u_post**2) / sigma2_post

            sigma2_m, sigma2_s = sigma2_post.mean().values, sigma2_post.std().values
            gamma_m, gamma_s = gamma_post.mean().values, gamma_post.std().values

            if model_type == 'panel':
                eta_m, eta_s = post['eta'].mean().values, post['eta'].std().values
                mu_m, mu_s = post['mu'].mean().values, post['mu'].std().values
                self._params = np.concatenate((betas, [mu_m, eta_m, sigma2_m, gamma_m]))
                self._std_err = np.concatenate((betas_se, [mu_s, eta_s, sigma2_s, gamma_s]))
            elif self.has_z:
                delta_m = post['delta'].mean(dim=['chain', 'draw']).values
                delta_s = post['delta'].std(dim=['chain', 'draw']).values
                self._params = np.concatenate((betas, delta_m, [sigma2_m, gamma_m]))
                self._std_err = np.concatenate((betas_se, delta_s, [sigma2_s, gamma_s]))
            else:
                self._params = np.concatenate((betas, [sigma2_m, gamma_m]))
                self._std_err = np.concatenate((betas_se, [sigma2_s, gamma_s]))

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
        gamma = self._params[-1]
        return np.sqrt(gamma / (1 - gamma))

    def get_sigma2(self):
        self.optimize()
        return self._params[-2]

    def __teJ(self):
        lam = self.get_lambda()
        self.ustar = -self.sign * self.get_residuals() * (lam**2 / (1+lam**2))
        self.sstar = (lam / (1 + lam**2)) * math.sqrt(self.get_sigma2())
        ratio = self.ustar / self.sstar
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
            return self.pymc_trace.posterior['TE'].mean(dim=['chain', 'draw']).values
            
        if self.method == self.TE_teJ: return self.__teJ()
        elif self.method == self.TE_te: return self.__te()
        elif self.method == self.TE_teMod: return self.__teMod()
        else: raise ValueError("Undefined decomposition technique.")

    def summary(self):
        self.optimize()

        ess_map = {}
        if self.inference_method == 'pymc' and self.pymc_trace is not None:
            import arviz as az
            diag = az.summary(self.pymc_trace)
            for idx in diag.index:
                ess_map[idx] = diag.loc[idx, 'ess_bulk']
            
            su_ess = ess_map.get('sigma_u', 9999)
            sv_ess = ess_map.get('sigma_v', 9999)
            if su_ess < 400 or sv_ess < 400:
                ess_map['sigma2'], ess_map['gamma'] = 0, 0

        names = (['(Intercept)'] + list(self.x_names)) if self.intercept else list(self.x_names)
        
        if self.is_panel and self.panel_model == 'greene':
            if 'pymc' in self.inference_method:
                names = list(self.x_names) + ['mu_alpha', 'sigma_alpha', 'sigma_u', 'sigma_v', 'sigma2', 'gamma']
            else:
                names += ['sigma_u', 'sigma_v', 'sigma2', 'gamma']
        elif self.is_panel:
            if 'pymc' in self.inference_method:
                names += ['mu', 'eta', 'sigma_u', 'sigma_v', 'sigma2', 'gamma']
            else:
                names += ['eta', 'sigma_u', 'sigma_v', 'sigma2', 'gamma']
        elif self.has_z:
            names += self.z_names + ['sigma_u', 'sigma_v', 'sigma2', 'gamma']
        else:
            names += ['sigma_u', 'sigma_v', 'sigma2', 'gamma']

        s2 = self._params[-2]
        gam = self._params[-1]
        s2_se = self._std_err[-2]
        gam_se = self._std_err[-1]
        
        su_val = np.sqrt(gam * s2) if gam * s2 > 0 else 0
        sv_val = np.sqrt((1 - gam) * s2) if (1 - gam) * s2 > 0 else 0

        display_params = []
        display_std = []

        param_dict = dict(zip(self._names_all, self._params))
        std_dict = dict(zip(self._names_all, self._std_err))

        for name in names:
            if name in param_dict and name not in ['sigma_u', 'sigma_v', 'sigma2', 'gamma']:
                display_params.append(param_dict[name])
                display_std.append(std_dict[name])
            elif name == 'sigma_u':
                val = param_dict.get('sigma_u', su_val)
                std = std_dict.get('sigma_u', np.nan)
                display_params.append(val); display_std.append(std)
            elif name == 'sigma_v':
                val = param_dict.get('sigma_v', sv_val)
                std = std_dict.get('sigma_v', np.nan)
                display_params.append(val); display_std.append(std)
            elif name == 'sigma2':
                display_params.append(s2); display_std.append(s2_se)
            elif name == 'gamma':
                display_params.append(gam); display_std.append(gam_se)
            else:
                display_params.append(np.nan); display_std.append(np.nan)

        display_params = np.array(display_params)
        display_std = np.array(display_std)

        with np.errstate(divide='ignore', invalid='ignore'):
            z_values = display_params / display_std
            p_values = 2 * norm.sf(np.abs(z_values))

        rows = []
        for i, name in enumerate(names):
            current_ess = 9999
            if self.inference_method == 'pymc':
                if name == '(Intercept)': key = 'beta0'
                elif name in self.x_names: key = f"beta[{list(self.x_names).index(name)}]"
                else: key = name
                current_ess = ess_map.get(key, 9999)

            if current_ess < 400:
                rows.append({
                    'Estimate': np.round(display_params[i], 5),
                    'Std. Error': 'Low ESS',
                    'z value': '---',
                    'Pr(>|z|)': '---',
                    'Sig.': ''
                })
            else:
                p = p_values[i]
                std = display_std[i]
                z = z_values[i]
                star = ('***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else '') if not np.isnan(p) else ''
                
                rows.append({
                    'Estimate': np.round(display_params[i], 5),
                    'Std. Error': np.round(std, 6) if not np.isnan(std) else '---',
                    'z value': np.round(z, 3) if not np.isnan(z) else '---',
                    'Pr(>|z|)': np.round(p, 4) if not np.isnan(p) else '---',
                    'Sig.': star
                })

        res_table = pd.DataFrame(rows, index=names)
        print(f"\nStochastic Frontier Analysis ({self.estimation_method})")
        print("=" * 75)
        print(res_table.to_string(na_rep='NaN'))
        print("-" * 75)
        print("Signif. codes:  0 '***' 0.01 '**' 0.05 '*' 0.1 ' ' 1")
        if not np.isnan(self._llf):
            print(f"Log-Likelihood:  {self._llf:.5f}")
        print("=" * 75)