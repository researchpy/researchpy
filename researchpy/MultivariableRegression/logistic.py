
import numpy as np
import scipy.stats
import patsy
import pandas as pd

from researchpy.model import general_model
from researchpy.utility import *
from researchpy.predict import predict
from researchpy.objective_functions import likelihood




class logistic(general_model):
    """

    Logistic regression using Newton-Raphson for MLE.

    """



    def __init__(self, formula_like, data=None,
                 solver_method="mle",
                 solver_options=None):

        self.__name__ = "researchpy.Logistic"

        if data is None:
            data = {}

        if solver_options is None:
            solver_options = {"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.model_data = {}

        self.n, self.k = self.IV.shape

        # Initialize betas with OLS
        #self.model_data["betas"] = np.zeros((self.n, self.k))
        try:
            self.model_data["betas"] = np.linalg.inv(self.IV.T @ self.IV) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = np.linalg.pinv(self.IV.T @ self.IV) @ self.IV.T @ self.DV


        it = 0
        error = np.ones_like(self.model_data["betas"])
        self.logL = []

        # Newton-Raphson iteration
        while np.any(error > self.solver_options["tol"]) and it < self.solver_options["max_iter"]:
            # Predicted probabilities
            linear_pred = self.IV @ self.model_data["betas"]
            p = 1 / (1 + np.exp(-linear_pred))

            # Hessian (H) and Gradient (G)
            w = p * (1 - p).reshape(-1, 1)
            H = -(self.IV.T @ (self.IV * w))    # This is equivalent to -X.T @ W @ X without building W
            G = self.IV.T @ (self.DV - p)       # Shape: (k,)

            # Update betas
            try:
                betas_new = self.model_data["betas"] - np.linalg.inv(H) @ G
            except np.linalg.LinAlgError:
                betas_new = self.model_data["betas"] - np.linalg.pinv(H) @ G

            error = np.abs(betas_new - self.model_data["betas"])
            self.model_data["betas"] = betas_new

            # Bernoulli Log-likelihood
            ll = np.sum(self.DV * np.log(p + 1e-12) + (1 - self.DV) * np.log(1 - p + 1e-12))
            self.logL.append(ll)

            it += 1
            if self.solver_options["display"]:
                print(f"Iteration {it}: Bernoulli Log-likelihood = {ll}")


        # Standard errors
        linear_pred = self.IV @ self.model_data["betas"]
        p = 1 / (1 + np.exp(-linear_pred))
        w = p * (1 - p).reshape(-1, 1)
        X_w = self.IV * w

        # Covariance matrix
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ X_w)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ X_w)

        self.model_data["standard_errors"] = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)

        # Wald z-statistics and p-values
        self.model_data["test_stat"] = self.model_data["betas"] / self.model_data["standard_errors"]
        self.model_data["test_stat_p_values"] = 2 * scipy.stats.norm.sf(np.abs(self.model_data["test_stat"]))

        # Confidence intervals
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
        linear_pred = self.IV @ self.model_data["betas"]
        return 1 / (1 + np.exp(-linear_pred))


    def results(self, report="or", return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        if report.lower() in ["or", "odds ratio"]:
            self.model_data["betas"] = np.exp(self.model_data["betas"])
            self.model_data["conf_int_lower"] = np.exp(self.model_data["conf_int_lower"])
            self.model_data["conf_int_upper"] = np.exp(self.model_data["conf_int_upper"])

            table = self._table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)
            return table.rename(columns={"Coef.": "Odds Ratio"})

        else:
            return self._table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)
