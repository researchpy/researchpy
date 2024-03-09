
# Used
import numpy as np
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .utility import *


class model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "researchpy.model"


        self.CI_LEVEL = conf_level
        self.conf_level = conf_level
        @property
        def CI_LEVEL(self):
            return self._CI_LEVEL
        @CI_LEVEL.setter
        def CI_LEVEL(self, conf_level):
            self._CI_LEVEL = float(conf_level)

        @property
        def conf_level(self):
            return self._CI_LEVEL
        @conf_level.setter
        def conf_level(self, conf_level):
            self._CI_LEVEL = float(conf_level)


        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        self.nobs = self.IV.shape[0]
        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        self._family = family
        self._link = link
        self._CI_LEVEL = conf_level
        self._solver = {"method": solver_method,
                        "tol": solver_options["tol"],
                        "max_iter": solver_options["max_iter"],
                        "display": solver_options["display"]}

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

    def table_regression_results(self, return_type="Dataframe", decimals={
                                     "Coef.": 2,
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



        ## Creating variable table information
        regression_info = {self._DV_design_info.term_names[0]: [],
                           "Coef.": [],
                           "Std. Err.": [],
                           f"{self._test_stat_name}": [],
                           "p-value": [],
                           f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}

        for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                          self.model_data["betas"], self.model_data["standard_errors"],
                                                          self.model_data["test_stat"], self.model_data["test_stat_p_values"],
                                                          self.model_data["conf_int_lower"], self.model_data["conf_int_upper"]):

            regression_info[self._DV_design_info.term_names[0]].append(column)
            regression_info["Coef."].append(round(beta[0], decimals["Coef."]))
            regression_info["Std. Err."].append(round(stderr[0], decimals["Std. Err."]))
            regression_info[f"{self._test_stat_name}"].append(round(t[0], decimals["test_stat"]))
            regression_info["p-value"].append(round(p, decimals["test_stat_p"]))
            regression_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci, decimals["CI"]),
                                                                                   round(u_ci, decimals["CI"])])

        regression_info = base_table(self._patsy_factor_information, self._mapping,
                                     self._rp_factor_information, pandas.DataFrame.from_dict(regression_info))

        if pretty_format==True:

            descriptives = {

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals.get("Root MSE", 4)),
                    "R-squared = ": round(self.model_data["r squared"], decimals.get("R-squared", 4)),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals.get("Adj R-squared", 4))

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

            results = {

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

                    "Number of obs = ": self.nobs,
                    "Root MSE = ": round(self.model_data["root_mse"], decimals),
                    "R-squared = ": round(self.model_data["r squared"], decimals),
                    "Adj R-squared = ": round(self.model_data["r squared adj."], decimals)

                }

            top = {

                    "Source": ["Model"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals)],
                    "Mean Squares": [round(self.model_data["msr"], decimals)],
                    "F value": [round(self.model_data["f_value_model"], decimals)],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals)]

                    }

            bottom = {

                    "Source": ["Residual", "Total"],
                    "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals), round(self.model_data["sum_of_square_total"], decimals)],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals), round(self.model_data["degrees_of_freedom_total"], decimals)],
                    "Mean Squares": [round(self.model_data["mse"], decimals), round(self.model_data["mst"], decimals)],
                    "F value": [numpy.nan, numpy.nan],
                    "p-value": [numpy.nan, numpy.nan]

                    }

            results = {

                    "Source": top["Source"] + bottom["Source"],
                    "Sum of Squares": top["Sum of Squares"] + bottom["Sum of Squares"],
                    "Degrees of Freedom": top["Degrees of Freedom"] + bottom["Degrees of Freedom"],
                    "Mean Squares": top["Mean Squares"] + bottom["Mean Squares"],
                    "F value": top["F value"] + bottom["F value"],
                    "p-value": top["p-value"] + bottom["p-value"]

                    }

        if return_type == "Dataframe":

            return (pandas.DataFrame.from_dict(descriptives, orient="index"),
                    pandas.DataFrame.from_dict(results),
                    pandas.DataFrame.from_dict(regression_info))

        elif return_type == "Dictionary":

            return (descriptives, results, regression_info)

        else:

            print(
                "Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")





class general_model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """


    def __init__(self, formula_like, data = {}, matrix_type = 1, family="gaussian", link="normal", tol=1e-7, max_iter=300, display=True):
        # matrix_type = 1 includes intercept
        # matrix_type = 0 does not include the intercept

        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        self.nobs = self.IV.shape[0]
        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        self._family = family
        self._link = link
        self.tol = tol
        self.max_iter = max_iter
        self.display = display

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information  = variable_information(self.IV.design_info.term_names, self.IV.design_info.column_names, data)


