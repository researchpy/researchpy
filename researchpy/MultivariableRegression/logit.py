from scipy.stats import norm
from scipy.special import expit
from researchpy.optimize.iterative_algorithms import scipy_minimize, newton_raphson

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

        if data is None:
            data = {}

        if solver_options is None:
            solver_options = {}

        base_solver_options = {"algorithm": "BFGS", # 'BFGS' (Broyden-Fletcher-Goldfarb-Shanno algorithm, 'nr' (Newton-Raphson)
                               "tol": 1e-7,
                               "max_iter": 1000,
                               "display": True,
                               "regularization": None,  # None, 'l1', or 'l2'
                               "alpha": 0.0  # Regularization strength
                               }
        solver_options = base_solver_options | solver_options

        self._test_stat_name = "z"

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.__name__ = "researchpy.Logit"

        self.model_data = {}


        # Initializing betas
        self._general_model__initialize_betas(initial_betas=initial_betas, initial_betas_method=initial_betas_method)

        # Fit the model
        self._general_model__fit_model()

        # Compute standard errors and statistics
        self._compute_statistics()



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
        self.model_data["test_stat_p_values"] = 2 * norm.sf(np.abs(self.model_data["test_stat"]))

        # Compute confidence intervals
        self._core_model__compute_confidence_intervals(add_to_model_data=True)



    def predict(self, estimate=None):
        """Predict probabilities."""
        linear_pred = self.IV @ self.model_data["betas"]
        return expit(linear_pred)



    def results(self, report="or", return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        if report.lower() in ["or", "odds ratio", "odds_ratio"]:
            if self._beta_type == "coef":
                self.model_data["betas"] = np.exp(self.model_data["betas"])
                self.model_data["conf_int_lower"] = np.exp(self.model_data["conf_int_lower"])
                self.model_data["conf_int_upper"] = np.exp(self.model_data["conf_int_upper"])
                self._beta_type = "odds ratio"


            if return_type == "Dataframe":
                table = self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                       decimals=decimals)
                if self._beta_type == "odds ratio":
                    return table.rename(columns={"Coef.": "Odds Ratio"})
                else:
                    return table

            else:
                dct = self._table_regression_results(return_type=return_type, pretty_format=pretty_format,
                                                     decimals=decimals)
                if self._beta_type == "odds ratio":
                    dct = {('Odds Ratio' if k == 'Coef.' else k): v for k, v in dct.items()}
                    return dct
                else:
                    return dct