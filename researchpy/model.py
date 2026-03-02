
# Used
import numpy as np
import scipy.stats
import patsy
import pandas as pd

from .summary import summarize
from .utility import *
from .predict import predict
from .objective_functions import likelihood




# Base model class for regression models. This class is not meant to be used directly, but rather to be inherited by
# specific regression model classes (e.g., OLS, Logistic, etc.). It contains common functionality and attributes that
# are shared across different types of regression models.
class core_model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

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

    @property
    def obj_function(self):
        return self._obj_function

    @obj_function.setter
    def obj_function(self, obj_function):
        self._obj_function = obj_function

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols",
                 obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "researchpy.core_model"

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        # Dictionary for storing model results and information. This will be populated by specific regression model
        # classes that inherit from this base class.
        self.model_data = {}

        self.solver_options = solver_options
        self.obj_function = obj_function
        self.CI_LEVEL = conf_level
        self.conf_level = conf_level

        self.nobs = self.IV.shape[0]
        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        self._family = family
        self._link = link
        if not hasattr(self, "_test_stat_name"):
            self._test_stat_name = "t" if family == "gaussian" else "z"
        self._CI_LEVEL = conf_level

        self._solver = {"method": solver_method,
                        "obj_function": obj_function,
                        "algorithm": solver_options.get("algorithm", None),
                        "tol": solver_options.get("tol", None),
                        "max_iter": solver_options.get("max_iter", None),
                        "display": solver_options.get("display", True)}

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(
            self.IV.design_info.term_names,
            self.IV.design_info.column_names,
            data)

        ## Creating dictionary for regression table information
        self.regression_table_info = {self.DV_name: [],
                                      "Coef.": [],
                                      "Std. Err.": [],
                                      f"{self._test_stat_name}": [],
                                      "p-value": [],
                                      f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}


    def _regression_base_table(self):

        dv = list(self.regression_table_info)[0]

        # Creating the first table #
        terms = (pd.DataFrame.from_dict(self._patsy_factor_information, orient="index")).reset_index()
        terms.columns = ["term", "term_cleaned"]
        terms["intx"] = [1 if ":" in t else 0 for t in list(self._patsy_factor_information.keys())]
        terms["factor"] = [1 if "C(" in t else 0 for t in list(self._patsy_factor_information.keys())]

        # Creating the second table #
        term_levels = {"term_cleaned": [],
                       "term_level_cleaned": []}

        for key in self._rp_factor_information.keys():

            count = 1

            if key == 'Intercept' or terms[terms.term_cleaned == key].factor.item() == 0:
                term_levels["term_cleaned"].append(key)
                term_levels["term_level_cleaned"].append(self._rp_factor_information[key])

            else:
                for value in self._rp_factor_information[key]:

                    term_levels["term_cleaned"].append(key)

                    if count == 1:

                        term_levels["term_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(value)

                        count += 1
                    else:
                        term_levels["term_level_cleaned"].append(value)

        # Creating the third table #
        current_terms = (pd.DataFrame.from_dict(self._mapping, orient="index")).reset_index()
        current_terms.columns = [dv, "term_level_cleaned"]
        current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in self._mapping.keys()]

        # Joining the tables together #
        table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                         how="left", on="term_cleaned")

        table = pd.merge(table, current_terms,
                         how="left", on=["term_cleaned", "term_level_cleaned"])

        table = pd.merge(table, pd.DataFrame.from_dict(self.regression_table_info),
                         how="left", on=dv)

        # Cleaning up final table #
        table[dv] = table["term_level_cleaned"]
        table["Coef."] = table["Coef."].astype(object)

        for idx in table.index:
            if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(self._rp_factor_information.keys())[
                1:]:
                table.iloc[idx, 6] = "(reference)"
                table.iloc[idx, 7:] = np.nan

            else:
                if table.iloc[idx][dv] in list(self._rp_factor_information.keys())[1:] and pd.isnull(
                        table.iloc[idx, 6]):
                    table.iloc[idx, 6:] = np.nan

        table = table[(table.intx == 0) |
                      ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

        return table.iloc[:, 5:]


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

        try:
            self.regression_table_info[self._DV_design_info.term_names[0]] = self._IV_design_info.column_names
            self.regression_table_info["Coef."] = np.round(self.model_data["betas"].flatten(),
                                                           decimals["Coef."]).tolist()
            self.regression_table_info["Std. Err."] = np.round(self.model_data["standard_errors"].flatten(),
                                                               decimals["Std. Err."]).tolist()
            self.regression_table_info[f"{self._test_stat_name}"] = np.round(self.model_data["test_stat"].flatten(),
                                                                             decimals["test_stat"]).tolist()
            self.regression_table_info["p-value"] = np.round(self.model_data["test_stat_p_values"].flatten(),
                                                             decimals["test_stat_p"]).tolist()
            self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"] = [list(x) for x in np.round(
                np.hstack((self.model_data["conf_int_lower"].flatten().reshape(-1, 1),
                           self.model_data["conf_int_upper"].flatten().reshape(-1, 1))), decimals["CI"]).tolist()]

        except:
            try:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"],
                                                                  self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"],
                                                                  self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"],
                                                                  self.model_data["conf_int_upper"]):
                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append(
                        [round(l_ci.item(), decimals["CI"]),
                         round(u_ci.item(), decimals["CI"])])
            except AttributeError:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"],
                                                                  self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"],
                                                                  self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"],
                                                                  self.model_data["conf_int_upper"]):
                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append(
                        [round(l_ci, decimals["CI"]),
                         round(u_ci, decimals["CI"])])

        self.regression_table_info = self._regression_base_table()




class model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

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

    @property
    def obj_function(self):
        return self._obj_function
    @obj_function.setter
    def obj_function(self, obj_function):
        self._obj_function = obj_function

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols",
                 obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "researchpy.model"

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        base_solver_options = {"tol": 1e-7, "max_iter": 300, "display": True}
        self.solver_options = base_solver_options | solver_options

        self.obj_function = obj_function

        self.CI_LEVEL = conf_level
        self.conf_level = conf_level

        self.nobs = self.IV.shape[0]
        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        if not hasattr(self, "_test_stat_name"):
            self._test_stat_name = "t" if family == "gaussian" else "z"
        self._family = family
        self._link = link
        self._CI_LEVEL = conf_level
        self._solver = {"method": solver_method,
                        "obj_function": obj_function,
                        "algorithm": solver_options.get("algorithm", None),
                        "tol": solver_options.get("tol", None),
                        "max_iter": solver_options.get("max_iter", None),
                        "display": solver_options.get("display", True)}

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        ## Creating variable table information
        self.regression_table_info = {self.DV_name: [],
                                      "Coef.": [],
                                      "Std. Err.": [],
                                      f"{self._test_stat_name}": [],
                                      "p-value": [],
                                      f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}

    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def objective_function(self):

        if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:

            y_e = predict(estimate="predict_y")
            objective = likelihood.log_likelihood(y_e)

        return objective


    def _regression_base_table(self):

        dv = list(self.regression_table_info)[0]

        # Creating the first table #
        terms = (pd.DataFrame.from_dict(self._patsy_factor_information, orient="index")).reset_index()
        terms.columns = ["term", "term_cleaned"]
        terms["intx"] = [1 if ":" in t else 0 for t in list(self._patsy_factor_information.keys())]
        terms["factor"] = [1 if "C(" in t else 0 for t in list(self._patsy_factor_information.keys())]

        # Creating the second table #
        term_levels = {"term_cleaned"      : [],
                       "term_level_cleaned": []}

        for key in self._rp_factor_information.keys():

            count = 1

            if key == 'Intercept' or terms[terms.term_cleaned == key].factor.item() == 0:
                term_levels["term_cleaned"].append(key)
                term_levels["term_level_cleaned"].append(self._rp_factor_information[key])

            else:
                for value in self._rp_factor_information[key]:

                    term_levels["term_cleaned"].append(key)

                    if count == 1:

                        term_levels["term_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(value)

                        count += 1
                    else:
                        term_levels["term_level_cleaned"].append(value)

        # Creating the third table #
        current_terms = (pd.DataFrame.from_dict(self._mapping, orient="index")).reset_index()
        current_terms.columns = [dv, "term_level_cleaned"]
        current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in self._mapping.keys()]

        # Joining the tables together #
        table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                         how="left", on="term_cleaned")

        table = pd.merge(table, current_terms,
                         how="left", on=["term_cleaned", "term_level_cleaned"])

        table = pd.merge(table, pd.DataFrame.from_dict(self.regression_table_info),
                         how="left", on=dv)

        # Cleaning up final table #
        table[dv] = table["term_level_cleaned"]
        table["Coef."] = table["Coef."].astype(object)

        for idx in table.index:
            if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(self._rp_factor_information.keys())[1:]:
                table.iloc[idx, 6] = "(reference)"
                table.iloc[idx, 7:] = np.nan

            else:
                if table.iloc[idx][dv] in list(self._rp_factor_information.keys())[1:] and pd.isnull(table.iloc[idx, 6]):
                    table.iloc[idx, 6:] = np.nan

        table = table[(table.intx == 0) |
                      ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

        return table.iloc[:, 5:]


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

        base_decimals = {"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                         "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                         'Degrees of Freedom': 1,  'Mean Squares': 4, 'Effect size': 4}
        decimals = base_decimals | decimals

        try:
            self.regression_table_info[self._DV_design_info.term_names[0]] = self._IV_design_info.column_names
            self.regression_table_info["Coef."] = np.round(self.model_data["betas"].flatten(), decimals["Coef."]).tolist()
            self.regression_table_info["Std. Err."] = np.round(self.model_data["standard_errors"].flatten(), decimals["Std. Err."]).tolist()
            self.regression_table_info[f"{self._test_stat_name}"] = np.round(self.model_data["test_stat"].flatten(), decimals["test_stat"]).tolist()
            self.regression_table_info["p-value"] = np.round(self.model_data["test_stat_p_values"].flatten(), decimals["test_stat_p"]).tolist()
            self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"] = [list(x) for x in np.round(np.hstack((self.model_data["conf_int_lower"].flatten().reshape(-1, 1),
                                                                                                                              self.model_data["conf_int_upper"].flatten().reshape(-1, 1))), decimals["CI"]).tolist()]

        except:
            try:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"], self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"], self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"], self.model_data["conf_int_upper"]):

                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci.item(), decimals["CI"]),
                                                                                                      round(u_ci.item(), decimals["CI"])])
            except AttributeError:
                for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                                  self.model_data["betas"],
                                                                  self.model_data["standard_errors"],
                                                                  self.model_data["test_stat"],
                                                                  self.model_data["test_stat_p_values"],
                                                                  self.model_data["conf_int_lower"],
                                                                  self.model_data["conf_int_upper"]):

                    self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                    self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                    self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                    self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                    self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                    self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci, decimals["CI"]),
                                                                                                      round(u_ci, decimals["CI"])])

        #self.regression_table_info = self._regression_base_table()



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
                "Number of obs = ": self.nobs,
                "Root MSE = ": round(self.model_data["root_mse"], decimals.get("Root MSE", 4)),
                "R-squared = ": round(self.model_data["r squared"], decimals.get("R-squared", 4)),
                "Adj R-squared = ": round(self.model_data["r squared adj."], decimals.get("Adj R-squared", 4))
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
                    "Sum of Squares": [round(self.model_data["sum_of_square_residual"], decimals.get("Sum of Squares", 4)),
                                       round(self.model_data["sum_of_square_total"], decimals.get("Sum of Squares", 4))],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_residual"], decimals.get("Degrees of Freedom", 4)),
                                           round(self.model_data["degrees_of_freedom_total"], decimals.get("Degrees of Freedom", 4))],
                    "Mean Squares": [round(self.model_data["mse"], decimals.get("Mean Squares", 4)),
                                     round(self.model_data["mst"], decimals.get("Mean Squares", 4))],
                    "F value": [np.nan, np.nan],
                    "p-value": [np.nan, np.nan]
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
            descriptives = pd.DataFrame.from_dict(descriptives, orient="index")
            model_results = pd.DataFrame.from_dict(model_results)
            return (descriptives.T, model_results, self._regression_base_table())

        elif return_type == "Dictionary":
            return (descriptives, results, self.regression_table_info)

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")





class general_model():
    """

    This is the base -model- object for Researchpy. By default, missing
    observations are dropped from the data. -matrix_type- parameter determines
    which design matrix will be returned; value of 1 will return a design matrix
    with the intercept, while a value of 0 will not.

    """

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

    @property
    def obj_function(self):
        return self._obj_function
    @obj_function.setter
    def obj_function(self, obj_function):
        self._obj_function = obj_function

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="mle",
                 obj_function="log-likelihood",
                 solver_options={"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "researchpy.general_model"

        # matrix_type = 1 includes intercept; matrix_type = 0 does not include the intercept
        if matrix_type == 1:
            self.DV, self.IV = patsy.dmatrices(formula_like, data, 1)
        if matrix_type == 0:
            self.DV, self.IV = patsy.dmatrices(formula_like + "- 1", data, 1)

        # Dictionary for storing model results and information. This will be populated by specific regression model
        # classes that inherit from this base class.
        self.model_data = {}

        self.solver_options = solver_options
        self.obj_function = obj_function
        self.CI_LEVEL = conf_level
        self.conf_level = conf_level
        self.nobs = self.IV.shape[0]
        self.n, self.k = self.IV.shape

        # Model design information
        self.formula = formula_like
        self._DV_design_info = self.DV.design_info
        self._IV_design_info = self.IV.design_info
        if not hasattr(self, "_test_stat_name"):
            self._test_stat_name = "t" if family == "gaussian" else "z"
        self._family = family
        self._link = link
        self._CI_LEVEL = conf_level
        self._solver = {"method": solver_method,
                        "obj_function": obj_function,
                        "algorithm": solver_options.get("algorithm", None),
                        "tol": solver_options.get("tol", None),
                        "max_iter": solver_options.get("max_iter", None),
                        "display": solver_options.get("display", True)}

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]
        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        ## Creating variable table information
        self.regression_table_info = {self.DV_name: [],
                                      "Coef.": [],
                                      "Std. Err.": [],
                                      f"{self._test_stat_name}": [],
                                      "p-value": [],
                                      f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}


    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def objective_function(self):
        if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:
            y_e = predict(estimate="predict_y")
            objective = likelihood.log_likelihood(y_e)

        return objective


    def _regression_base_table(self):
        dv = list(self.regression_table_info)[0]

        # Creating the first table #
        terms = (pd.DataFrame.from_dict(self._patsy_factor_information, orient="index")).reset_index()
        terms.columns = ["term", "term_cleaned"]
        terms["intx"] = [1 if ":" in t else 0 for t in list(self._patsy_factor_information.keys())]
        terms["factor"] = [1 if "C(" in t else 0 for t in list(self._patsy_factor_information.keys())]

        # Creating the second table #
        term_levels = {"term_cleaned"      : [],
                       "term_level_cleaned": []}

        for key in self._rp_factor_information.keys():

            count = 1

            if key == 'Intercept' or terms[terms.term_cleaned == key].factor.item() == 0:
                term_levels["term_cleaned"].append(key)
                term_levels["term_level_cleaned"].append(self._rp_factor_information[key])

            else:
                for value in self._rp_factor_information[key]:

                    term_levels["term_cleaned"].append(key)

                    if count == 1:

                        term_levels["term_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(key)
                        term_levels["term_level_cleaned"].append(value)

                        count += 1
                    else:
                        term_levels["term_level_cleaned"].append(value)

        # Creating the third table #
        current_terms = (pd.DataFrame.from_dict(self._mapping, orient="index")).reset_index()
        current_terms.columns = [dv, "term_level_cleaned"]
        current_terms["term_cleaned"] = [patsy_term_cleaner(key) for key in self._mapping.keys()]

        # Joining the tables together #
        table = pd.merge(terms, pd.DataFrame.from_dict(term_levels),
                         how="left", on="term_cleaned")

        table = pd.merge(table, current_terms,
                         how="left", on=["term_cleaned", "term_level_cleaned"])

        table = pd.merge(table, pd.DataFrame.from_dict(self.regression_table_info),
                         how="left", on=dv)

        # Cleaning up final table #
        table[dv] = table["term_level_cleaned"]
        table["Coef."] = table["Coef."].astype(object)

        for idx in table.index:
            if pd.isnull(table.iloc[idx, 6]) and table.iloc[idx][dv] not in list(self._rp_factor_information.keys())[1:]:
                table.iloc[idx, 6] = "(reference)"
                table.iloc[idx, 7:] = np.nan

            else:
                if table.iloc[idx][dv] in list(self._rp_factor_information.keys())[1:] and pd.isnull(table.iloc[idx, 6]):
                    table.iloc[idx, 6:] = np.nan

        table = table[(table.intx == 0) |
                      ((table.intx == 1) & (table.iloc[:, 6] != "(reference)"))]

        return table.iloc[:, 5:]


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

        base_decimals = {"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                         "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                         'Degrees of Freedom': 1,  'Mean Squares': 4, 'Effect size': 4}
        decimals = base_decimals | decimals

        try:
            self.regression_table_info[self._DV_design_info.term_names[0]] = self._IV_design_info.column_names
            self.regression_table_info["Coef."] = np.round(self.model_data["betas"].flatten(), decimals["Coef."]).tolist()
            self.regression_table_info["Std. Err."] = np.round(self.model_data["standard_errors"].flatten(), decimals["Std. Err."]).tolist()
            self.regression_table_info[f"{self._test_stat_name}"] = np.round(self.model_data["test_stat"].flatten(), decimals["test_stat"]).tolist()
            self.regression_table_info["p-value"] = np.round(self.model_data["test_stat_p_values"].flatten(), decimals["test_stat_p"]).tolist()
            self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"] = [list(x) for x in np.round(np.hstack((self.model_data["conf_int_lower"].flatten().reshape(-1, 1),
                                                                                                                              self.model_data["conf_int_upper"].flatten().reshape(-1, 1))), decimals["CI"]).tolist()]

        except:
            for column, beta, stderr, t, p, l_ci, u_ci in zip(self._IV_design_info.column_names,
                                                              self.model_data["betas"], self.model_data["standard_errors"],
                                                              self.model_data["test_stat"], self.model_data["test_stat_p_values"],
                                                              self.model_data["conf_int_lower"], self.model_data["conf_int_upper"]):

                self.regression_table_info[self._DV_design_info.term_names[0]].append(column)
                self.regression_table_info["Coef."].append(round(beta.item(), decimals["Coef."]))
                self.regression_table_info["Std. Err."].append(round(stderr.item(), decimals["Std. Err."]))
                self.regression_table_info[f"{self._test_stat_name}"].append(round(t.item(), decimals["test_stat"]))
                self.regression_table_info["p-value"].append(round(p.item(), decimals["test_stat_p"]))
                self.regression_table_info[f"{int(self.CI_LEVEL * 100)}% Conf. Interval"].append([round(l_ci.item(), decimals["CI"]),
                                                                                                  round(u_ci.item(), decimals["CI"])])

        if return_type == "Dataframe":
            return self._regression_base_table()

        elif return_type == "Dictionary":
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")