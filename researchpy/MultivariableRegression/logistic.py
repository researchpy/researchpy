
import numpy as np
import scipy.stats
import patsy
import pandas

from researchpy.model import model, general_model
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

        if data is None:
            data = {}

        if solver_options is None:
            solver_options = {"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.__name__ = "researchpy.Logistic"

        self.model_data = {}

        # Initialize betas with OLS
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

            # Hessian and Gradient
            """
            Building W as a full n x n diagonal matrix (np.diagflat(p * (1 - p))) makes Newton-Raphson iterations O(n^2) 
            memory/time and will not scale. You can compute the Hessian as -(X.T * w) @ X (where w = p*(1-p) broadcast 
            across columns) without materializing W.
            """
            W = np.diagflat(p * (1 - p))
            H = -(self.IV.T @ W @ self.IV)
            G = self.IV.T @ (self.DV - p)

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

        self.n, self.k = self.IV.shape

        #self.model_data["betas"] = np.array(self.model_data["betas"])

        # Standard errors
        linear_pred = self.IV @ self.model_data["betas"]
        p = 1 / (1 + np.exp(-linear_pred))
        W = np.diagflat(p * (1 - p))
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ W @ self.IV)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ W @ self.IV)
        """
        The same dense-diagonal approach is repeated when computing the covariance matrix (`W = np.diagflat(...)`). 
        This has the same O(n^2) scaling issue; consider reusing the vector-weighted formulation instead of building `W`.
        
        Suggestion:
        w = (p * (1 - p)).reshape(-1, 1)
        X_w = self.IV * w
        try:
            cov_matrix = np.linalg.inv(self.IV.T @ X_w)
        except np.linalg.LinAlgError:
            cov_matrix = np.linalg.pinv(self.IV.T @ X_w)
        """

        self.model_data["standard_errors"] = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)

        # Wald z-statistics and p-values
        self.model_data["test_stat"] = self.model_data["betas"] / self.model_data["standard_errors"]
        self.model_data["test_stat_p_values"] = 2 * scipy.stats.norm.sf(np.abs(self.model_data["test_stat"]))

        # Confidence intervals
        self.model_data["conf_int_lower"] = self.model_data["betas"] - 1.96 * self.model_data["standard_errors"]
        self.model_data["conf_int_upper"] = self.model_data["betas"] + 1.96 * self.model_data["standard_errors"]



    def predict(self, estimate=None):
        linear_pred = self.IV @ self.model_data["betas"]
        return 1 / (1 + np.exp(-linear_pred))


    def results(self, return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        return self._table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

