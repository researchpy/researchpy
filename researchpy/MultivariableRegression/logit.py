import numpy as np
import scipy.stats
import scipy.optimize
import patsy
import pandas as pd
from scipy.special import expit, logit

from researchpy.model import general_model
from researchpy.utility import *
from researchpy.predict import predict
from researchpy.objective_functions import likelihood


class logit(general_model):
    """
    Logistic regression using scipy.optimize for MLE.

    Falls back to Newton-Raphson if scipy optimization fails.
    Includes regularization support and better numerical stability.
    """

    def __init__(self, formula_like, data=None,
                 solver_method="mle", solver_options=None,
                 initial_betas=None, initial_betas_method="ols"):

        self.__name__ = "researchpy.Logit"

        if data is None:
            data = {}

        if solver_options is None:
            solver_options = {
                "algorithm": "BFGS",
                "tol": 1e-7,
                "max_iter": 1000,
                "display": True,
                "regularization": None,  # None, 'l1', or 'l2'
                "alpha": 0.0  # Regularization strength
                }

        base_solver_options = {"algorithm": "BFGS", "tol": 1e-7, "max_iter": 1000,  "display": True,
                               "regularization": None,  # None, 'l1', or 'l2'
                               "alpha": 0.0  # Regularization strength
                               }
        solver_options = base_solver_options | solver_options

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.model_data = {}

        self.n, self.k = self.IV.shape

        # Initialize betas with better starting values
        self._initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

        # Fit the model
        self._fit_model()

        # Compute standard errors and statistics
        self._compute_statistics()

    def _initialize_betas(self, initial_betas=None, initial_betas_method=None):

        if initial_betas is not None and initial_betas_method is not None:
            return("Warning: Both initial_betas and initial_betas_method were provided. Ignoring initial_betas_method and using provided initial_betas.")

        if initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.model_data["betas"] = initial_betas
            else:
                raise ValueError(f"initial_betas must be a numpy array of shape ({self.k}, 1), but got {initial_betas.shape}")

        elif initial_betas_method.lower() == "ols":
            try:
                self.model_data["betas"] = np.linalg.inv(self.IV.T @ self.IV) @ self.IV.T @ self.DV
            except:
                self.model_data["betas"] = np.linalg.pinv(self.IV.T @ self.IV) @ self.IV.T @ self.DV

        else:
            """Initialize betas with smarter starting values."""
            # Start with zeros (better than OLS for binary outcomes)
            betas = np.ones((self.n, self.k))

            # Set intercept to log odds of outcome proportion
            y_mean = np.mean(self.DV)
            if 0 < y_mean < 1:
                betas[0] = np.log(y_mean / (1 - y_mean))

            self.model_data["betas"] = betas.reshape(-1, 1)

    def _neg_log_likelihood(self, params):
        """Negative log-likelihood function for scipy.optimize."""
        params = params.reshape(-1, 1)  # Ensure params is a column vector
        linear_pred = self.IV @ params
        p = expit(linear_pred)  # Numerically stable sigmoid

        # Clip to avoid log(0)
        eps = 1e-15
        p = np.clip(p, eps, 1 - eps)

        ll = -np.sum(self.DV * np.log(p) + (1 - self.DV) * np.log(1 - p))

        # Add regularization if specified
        if self.solver_options.get("regularization") == "l2":
            alpha = self.solver_options.get("alpha", 0.0)
            # Don't regularize intercept (first coefficient)
            ll += alpha * np.sum(params[1:] ** 2)
        elif self.solver_options.get("regularization") == "l1":
            alpha = self.solver_options.get("alpha", 0.0)
            ll += alpha * np.sum(np.abs(params[1:]))

        return ll

    def _gradient_neg_log_likelihood(self, params):
        """Gradient of negative log-likelihood."""
        params = params.reshape(-1, 1)  # Ensure params is a column vector
        linear_pred = self.IV @ params
        p = expit(linear_pred)

        grad = -self.IV.T @ (self.DV - p)

        # Add regularization gradient if specified
        if self.solver_options.get("regularization") == "l2":
            alpha = self.solver_options.get("alpha", 0.0)
            reg_grad = np.zeros_like(params)
            reg_grad[1:] = 2 * alpha * params[1:]  # Don't regularize intercept
            grad += reg_grad
        elif self.solver_options.get("regularization") == "l1":
            alpha = self.solver_options.get("alpha", 0.0)
            reg_grad = np.zeros_like(params)
            reg_grad[1:] = alpha * np.sign(params[1:])
            grad += reg_grad

        return grad.flatten()  # Return flattened gradient for scipy.optimize

    def _fit_model(self):
        """Fit the model using scipy.optimize with fallback."""
        self.logL = []
        it = 0
        converged = False

        # Try scipy.optimize first
        try:
            if self.solver_options["display"]:
                print(f"Starting optimization with {self.solver_options['algorithm']}...")

            result = scipy.optimize.minimize(
                fun=self._neg_log_likelihood,
                x0=self.model_data["betas"].flatten(),
                jac=self._gradient_neg_log_likelihood,
                method=self.solver_options["algorithm"],
                options={
                    'maxiter': self.solver_options["max_iter"],
                    'disp': self.solver_options["display"],
                    'gtol': self.solver_options["tol"]
                }
            )

            if result.success:
                self.model_data["betas"] = result.x.reshape(-1, 1)
                self.logL.append(-result.fun)
                converged = True

                if self.solver_options["display"]:
                    print(f"Optimization converged in {result.nit} iterations")
                    print(f"Final log-likelihood: {-result.fun:.4f}")
            else:
                if self.solver_options["display"]:
                    print(f"Warning: scipy optimization did not converge ({result.message})")
                    print("Falling back to Newton-Raphson...")

        except Exception as e:
            if self.solver_options["display"]:
                print(f"scipy.optimize failed: {e}")
                print("Falling back to Newton-Raphson...")

        # Fallback to Newton-Raphson if scipy failed
        if not converged:
            self._fit_newton_raphson()

    def _fit_newton_raphson(self):
        """Fallback Newton-Raphson implementation."""
        it = 0
        error = np.ones_like(self.model_data["betas"])
        tol = self.solver_options["tol"]
        max_iter = self.solver_options["max_iter"]

        while np.any(error > tol) and it < max_iter:
            linear_pred = self.IV @ self.model_data["betas"]
            p = expit(linear_pred)  # Use expit for stability

            w = p * (1 - p).reshape(-1, 1)
            H = -(self.IV.T @ (self.IV * w))
            G = self.IV.T @ (self.DV - p)

            try:
                betas_new = self.model_data["betas"] - np.linalg.inv(H) @ G
            except np.linalg.LinAlgError:
                betas_new = self.model_data["betas"] - np.linalg.pinv(H) @ G

            error = np.abs(betas_new - self.model_data["betas"])
            self.model_data["betas"] = betas_new

            ll = np.sum(self.DV * np.log(p + 1e-12) + (1 - self.DV) * np.log(1 - p + 1e-12))
            self.logL.append(ll)

            it += 1
            if self.solver_options["display"]:
                print(f"NR Iteration {it}: Log-likelihood = {ll:.4f}")

        if self.solver_options["display"]:
            print(f"Newton-Raphson completed in {it} iterations")

    def _compute_statistics(self):
        """Compute standard errors, test statistics, and p-values."""
        linear_pred = self.IV @ self.model_data["betas"]
        p = expit(linear_pred)
        w = p * (1 - p).reshape(-1, 1)
        X_w = self.IV * w

        # Covariance matrix (inverse of Fisher information)
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ X_w)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ X_w)

        self.model_data["standard_errors"] = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)

        # Wald z-statistics and p-values
        self.model_data["test_stat"] = self.model_data["betas"] / self.model_data["standard_errors"]
        self.model_data["test_stat_p_values"] = 2 * scipy.stats.norm.sf(np.abs(self.model_data["test_stat"]))

        # Compute confidence intervals
        conf_int_lower = []
        conf_int_upper = []

        for beta, se in zip(self.model_data["betas"], self.model_data["standard_errors"]):

            try:
                lower, upper = scipy.stats.norm.interval(self._CI_LEVEL, loc=beta, scale=se)
                conf_int_lower.append(float(lower))
                conf_int_upper.append(float(upper))

            except TypeError:
                try:
                    conf_int_lower.append(lower.item())
                    conf_int_upper.append(upper.item())

                except:
                    conf_int_lower.append(np.nan)
                    conf_int_upper.append(np.nan)

        self.model_data["conf_int_lower"] = np.array(conf_int_lower)
        self.model_data["conf_int_upper"] = np.array(conf_int_upper)

    def predict(self, estimate=None):
        """Predict probabilities."""
        linear_pred = self.IV @ self.model_data["betas"]
        return expit(linear_pred)

    def results(self, report="or", return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        if report.lower() in ["or", "odds ratio"]:
            self.model_data["betas"] = np.exp(self.model_data["betas"])
            self.model_data["conf_int_lower"] = np.exp(self.model_data["conf_int_lower"])
            self.model_data["conf_int_upper"] = np.exp(self.model_data["conf_int_upper"])

            table = self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                   decimals=decimals)
            return table.rename(columns={"Coef.": "Odds Ratio"})

        else:
            return self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                  decimals=decimals)