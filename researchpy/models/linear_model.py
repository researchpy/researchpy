import patsy
import pandas as pd

from researchpy.models.base import CoreModel
from researchpy.utility import *


class LinearModel(CoreModel):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95, display_summary=True,
                 family="gaussian", link="normal", solver_method="ols", obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True},
                 table_decimals=None):

        self.__name__ = "Researchpy.LinearModel"

        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_method=solver_method, obj_function=obj_function,
                         table_decimals=table_decimals)

        # Creating the J matrix
        J = self._j_matrix(add_to_model_data=False)

        # Creating the identify matrix
        I = self._identity_matrix(add_to_model_data=False)

        # Calculation of the Hat matrix
        self._hat_matrix(add_to_model_data=True)

        # OLS fit to compute the coefficients (betas)
        self._CoreModel__ols_fit(add_to_model_data=True)

        # Compute the model sum of square, degrees of freedom, mean squares, F-value, p-value, and effect size measures
        self._model_sum_of_square_stats(add_to_model_data=True)

        # Compute standard errors and confidence intervals
        self.__compute_beta_se_and_stats(add_to_model_data=True)



    # Compute standard errors and statistics
    def __compute_beta_se_and_stats(self, add_to_model_data=True):
        variance_covariance_beta_matrix = self._variance_covariance_beta_matrix(method="standard", to_return=True, add_to_self=False, add_to_model_data=False)

        ## Standard Errors
        self.model_data["standard_errors"] = (np.array(np.sqrt(variance_covariance_beta_matrix.diagonal()))).T

        ## T-statistics
        self.model_data["test_stat"] = self.model_data["betas"] * (1 / self.model_data["standard_errors"])

        ## Two-sided p-value
        self.model_data["test_stat_p_values"] = np.array([
            float((scipy.stats.t.sf(np.abs(t), self.model_data["degrees_of_freedom_residual"]) * 2).item())
            for t in self.model_data["test_stat"]
        ])

        self._CoreModel__compute_confidence_intervals(add_to_model_data=True)


    def _variance_covariance_residual_matrix(self, method="standard", to_return=True, add_to_self=False, add_to_model_data=False):

        if method == "standard":
            variance_covariance_residual_matrix = np.matrix(self.model_data["mse"] * (self.model_data["I"] - self.model_data["H"]))

        if add_to_self:
            self.variance_covariance_residual_matrix = variance_covariance_residual_matrix

        if add_to_model_data:
            self.model_data["variance_covariance_residual_matrix"] = variance_covariance_residual_matrix

        if to_return:
            return variance_covariance_residual_matrix


    def _variance_covariance_beta_matrix(self, method="standard", to_return=True, add_to_self=False, add_to_model_data=False):

        if method == "standard":
            try:
                variance_covariance_beta_matrix = np.matrix(self.model_data["mse"] * np.linalg.inv(self.IV.T @ self.IV))
            except np.linalg.LinAlgError:
                variance_covariance_beta_matrix = np.matrix(self.model_data["mse"] * np.linalg.pinv(self.IV.T @ self.IV))

        if add_to_self:
            self.variance_covariance_beta_matrix = variance_covariance_beta_matrix

        if add_to_model_data:
            self.model_data["variance_covariance_beta_matrix"] = variance_covariance_beta_matrix

        if to_return:
            return variance_covariance_beta_matrix




    def _model_sum_of_square_stats(self, to_return=False, add_to_self=False, add_to_model_data=True):

        predicted_y = self.IV @ self.model_data["betas"]    # predicted y values
        residuals = self.DV - predicted_y                   # Calculation of residuals (error)

        J = self._j_matrix(add_to_model_data=False)         # Creating the J matrix
        I = self._identity_matrix(add_to_model_data=False)  # Creating the identify matrix


        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = float((self.DV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = float((self.model_data["betas"].T @ self.IV.T @ self.DV - (1/self.n) * self.DV.T @ J @ self.DV).item())

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = float((residuals.T @ residuals).item())

        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = np.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.n - np.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.n - 1

        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        #Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])

        ## Root Mean Square Error
        self.model_data["root_mse"] = float(np.sqrt(self.model_data["mse"]))


        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf( self.model_data["f_value_model"], self.model_data["degrees_of_freedom_model"], self.model_data["degrees_of_freedom_residual"])


        ### Effect Size Measures
        # Model
        self.model_data["r squared"] = (self.model_data["sum_of_square_model"] / self.model_data["sum_of_square_total"])
        self.model_data["r squared adj."] = 1 - (self.model_data["degrees_of_freedom_total"] / self.model_data["degrees_of_freedom_residual"]) * (
            self.model_data["sum_of_square_residual"] / self.model_data["sum_of_square_total"])
        self.model_data["Eta squared"] = self.model_data["r squared"]

        self.model_data["Epsilon squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"])

        self.model_data["Omega squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"] + self.model_data["mse"])


    def _regression_base_table(self):

        self._CoreModel__regression_base_table()


    def _table_regression_results(self, return_type="Dataframe", decimals={"Coef.": 2,
                                                                          "Std. Err.": 4,
                                                                          "test_stat": 4,
                                                                          "test_stat_p": 4,
                                                                          "CI": 2,
                                                                          "Root MSE": 4,
                                                                          "R-squared": 4,
                                                                          "Adj R-squared": 4,
                                                                          "Sum of Squares": 4,
                                                                          'Degrees of Freedom': 1,
                                                                          'Mean Squares': 4,
                                                                          'Effect size': 4
                                                                          },
                                 pretty_format=True, *args):

        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

        if pretty_format == True:

            descriptives = {
                "Number of obs": self.n,
                "Root MSE": round(self.model_data["root_mse"], decimals.get("Root MSE", 4)),
                "R-squared": round(self.model_data["r squared"], decimals.get("R-squared", 4)),
                "Adj R-squared": round(self.model_data["r squared adj."], decimals.get("Adj R-squared", 4))
                }

            top = {
                "Source": ["Model", ''],
                "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals.get("Sum of Squares", 4)), ''],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals.get("Degrees of Freedom", 4)), ''],
                "Mean Squares": [round(self.model_data["msr"], decimals.get("Mean Squares", 4)), ''],
                "F value": [round(self.model_data["f_value_model"], decimals.get("test_stat", 4)), ''],
                "p-value": [round(self.model_data["f_p_value_model"], decimals.get("test_stat_p", 4)), ''],
                "Eta squared": [round(self.model_data["Eta squared"], decimals.get("Effect size", 4)), ''],
                "Epsilon squared": [round(self.model_data["Epsilon squared"], decimals.get("Effect size", 4)), ''],
                "Omega squared": [round(self.model_data["Omega squared"], decimals.get("Effect size", 4)), '']
            }

            bottom = {
                "Source": ["Residual", "Total"],
                "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals.get("Sum of Squares", 4)),
                                   round(self.model_data["sum_of_square_total"], decimals.get("Sum of Squares", 4))],
                "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals.get("Degrees of Freedom", 4)),
                                       round(self.model_data["degrees_of_freedom_total"], decimals.get("Degrees of Freedom", 4))],
                "Mean Squares": [round(self.model_data["mse"], decimals.get("Mean Squares", 4)),
                                 round(self.model_data["mst"], decimals.get("Mean Squares", 4))],
                "F value": ['', ''],
                "p-value": ['', ''],
                "Eta squared": ['', ''],
                "Epsilon squared" : ['', ''],
                "Omega squared": ['', '']
            }

            model_results = {
                "Source": top["Source"] + bottom["Source"],
                "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
                "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
                "F value": top["F value"] + bottom["F value"],
                "p-value": top["p-value"] + bottom["p-value"],
                "Eta squared": top["Eta squared"] + bottom["Eta squared"],
                "Epsilon squared" : top["Epsilon squared"] + bottom["Epsilon squared"],
                "Omega squared": top["Omega squared"] + bottom["Omega squared"]
            }

        else:

            descriptives = {
                "Number of obs": self.n,
                "Root MSE": round(return_numeric(self.model_data["root_mse"]), decimals.get("Root MSE", 4)),
                "R-squared": round(return_numeric(self.model_data["r squared"]), decimals.get("R-squared", 4)),
                "Adj R-squared": round(return_numeric(self.model_data["r squared adj."]), decimals.get("Adj R-squared", 4))
            }

            top = {
                "Source": ["Model"],
                "Sum of Squares": [round(return_numeric(self.model_data["sum_of_square_model"]), decimals.get("Sum of Squares", 4))],
                "Degrees of Freedom": [round(return_numeric(self.model_data["degrees_of_freedom_model"]), decimals.get("Degrees of Freedom", 4))],
                "Mean Squares": [round(return_numeric(self.model_data["msr"]), decimals.get("Mean Squares", 4))],
                "F value": [round(return_numeric(self.model_data["f_value_model"]), decimals.get("test_stat", 4))],
                "p-value": [round(return_numeric(self.model_data["f_p_value_model"]), decimals.get("test_stat_p", 4))],
                "Eta squared": [round(return_numeric(self.model_data["Eta squared"]), decimals.get("Effect size", 4))],
                "Epsilon squared": [round(return_numeric(self.model_data["Epsilon squared"]), decimals.get("Effect size", 4))],
                "Omega squared": [round(return_numeric(self.model_data["Omega squared"]), decimals.get("Effect size", 4))]
            }

            bottom = {
                "Source": ["Residual", "Total"],
                "Sum of Squares": [round(return_numeric(self.model_data["sum_of_square_residual"]), decimals.get("Sum of Squares", 4)),
                                   round(return_numeric(self.model_data["sum_of_square_total"]), decimals.get("Sum of Squares", 4))],
                "Degrees of Freedom": [round(return_numeric(self.model_data["degrees_of_freedom_residual"]), decimals.get("Degrees of Freedom", 4)),
                                       round(return_numeric(self.model_data["degrees_of_freedom_total"]), decimals.get("Degrees of Freedom", 4))],
                "Mean Squares": [round(return_numeric(self.model_data["mse"]), decimals.get("Mean Squares", 4)),
                                 round(return_numeric(self.model_data["mst"]), decimals.get("Mean Squares", 4))],
                "F value": [np.nan, np.nan],
                "p-value": [np.nan, np.nan],
                "Eta squared": [np.nan, np.nan],
                "Epsilon squared" : [np.nan, np.nan],
                "Omega squared": [np.nan, np.nan]
            }

            model_results = {
                "Source": top["Source"] + bottom["Source"],
                "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
                "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
                "F value": top["F value"] + bottom["F value"],
                "p-value": top["p-value"] + bottom["p-value"],
                "Eta squared": top["Eta squared"] + bottom["Eta squared"],
                "Epsilon squared" : top["Epsilon squared"] + bottom["Epsilon squared"],
                "Omega squared": top["Omega squared"] + bottom["Omega squared"]
            }


        if return_type == "Dataframe":
            descriptives = pd.DataFrame.from_dict(descriptives, orient="index")
            model_results = pd.DataFrame.from_dict(model_results)
            return (descriptives.T, model_results, self._CoreModel__regression_base_table())

        elif return_type == "Dictionary":
            return (descriptives, model_results, self.regression_table_info)

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")