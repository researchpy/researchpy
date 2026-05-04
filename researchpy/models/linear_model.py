import patsy
import pandas as pd

from researchpy.models.core_model import CoreModel
from researchpy.utility import *


class LinearModel(CoreModel):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols",
                 obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "Researchpy.LinearModel"

        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                            family=family, link=link, solver_method=solver_method, obj_function=obj_function)


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
                "Number of obs": self.nobs,
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
                "Number of obs": self.nobs,
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