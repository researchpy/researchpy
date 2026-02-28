
import numpy as np
import scipy.stats
import patsy
import pandas

from researchpy.model import model
from researchpy.utility import *
from researchpy.predict import predict
from researchpy.objective_functions import likelihood




class LogisticRegression(model):
    """
    Logistic regression using Newton-Raphson for MLE.
    """



    def __init__(self, formula_like, data={},
                 solver_method="mle",
                 solver_options={"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}):

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.__name__ = "researchpy.LogisticRegression"

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

        self.model_data["standard_errors"] = np.sqrt(np.diag(cov_matrix)).reshape(-1, 1)
        #self.model_data["standard_errors"] = np.array(self.model_data["standard_errors"])

        # Wald z-statistics and p-values
        self.model_data["test_stat"] = self.model_data["betas"] / self.model_data["standard_errors"]
        self.model_data["test_stat_p_values"] = 2 * scipy.stats.norm.sf(np.abs(self.model_data["test_stat"]))

        # Confidence intervals
        self.model_data["conf_int_lower"] = self.model_data["betas"] - 1.96 * self.model_data["standard_errors"]
        self.model_data["conf_int_upper"] = self.model_data["betas"] + 1.96 * self.model_data["standard_errors"]



        # Creating Variable Table Information
        """
            regression_info = {
            self._DV_design_info.term_names[0]: [],
            "Coef.": [],
            "Std. Err.": [],
            "z": [],
            "p-value": [],
            f"{int(self._CI_LEVEL * 100)}% Conf. Interval": []
        }
        """



    def predict(self, estimate=None):
        linear_pred = self.IV @ self.model_data["betas"]
        return 1 / (1 + np.exp(-linear_pred))


    def results(self, decimals=4):

        # Results table
        columns = self._IV_design_info.column_names
        results = pd.DataFrame({
            "Coef.": self.model_data["betas"].flatten(),
            "Std. Err.": self.model_data["standard_errors"].flatten(),
            "z": self.model_data["test_stat"].flatten(),
            "p-value": self.model_data["test_stat_p_values"].flatten(),
            "95% Conf. Interval Lower": self.model_data["conf_int_lower"].flatten(),
            "95% Conf. Interval Upper": self.model_data["conf_int_upper"].flatten()
        }, index=columns).round(decimals)

        return results