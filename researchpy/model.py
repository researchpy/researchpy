# Used
import numpy as np
import scipy.stats
from scipy.special import expit, logit
import patsy
import pandas as pd

from .utility import *
from .predict import predict
from .objective_functions import likelihood
from researchpy.objective_functions.likelihood import neg_log_likelihood, gradient_neg_log_likelihood

from researchpy.optimize.iterative_algorithms import scipy_minimize, newton_raphson
from researchpy.optimize.trackers import OptimizationTracker


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

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols", obj_function="numeric",
                 solver_options=None):

        self.__name__ = "researchpy.core_model"

        self._beta_type = "coef"

        if data is None:
            data = {}
        if solver_options is None:
            solver_options = {}

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

        # Initialize an optimization tracker instance for this model. This tracker can be used by optimization
        # algorithms to store and monitor the optimization process.
        self._OptimizationTracker = OptimizationTracker()

        ## My design information ##
        self.DV_name = self.DV.design_info.term_names[0]

        self._patsy_factor_information, self._mapping, self._rp_factor_information = variable_information(self.IV.design_info.term_names,
                                                                                                          self.IV.design_info.column_names,
                                                                                                          data)

        # Dictionary for storing model results and information. This will be populated by specific regression model
        # classes that inherit from this base class.
        if not hasattr(self, "model_data"):
            self.model_data = {}

        ## Creating variable table information
        if not hasattr(self, "regression_table_info"):
            self.regression_table_info = {self.DV_name: [],
                                          "Coef.": [],
                                          "Std. Err.": [],
                                          f"{self._test_stat_name}": [],
                                          "p-value": [],
                                          f"{int(self.CI_LEVEL * 100)}% Conf. Interval": []}


    def _hat_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            try:
                self.model_data["H"] = np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
            except:
                self.model_data["H"] = np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
                #print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")
        else:
            try:
                H = np.asarray(self.IV) @ np.linalg.inv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
            except:
                H = np.asarray(self.IV) @ np.linalg.pinv(np.asarray(self.IV.T) @ np.asarray(self.IV)) @ np.asarray(self.IV.T)
                #print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")
            return H


    def _j_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            self.model_data["J"] = np.ones((self.nobs, self.nobs))
        else:
            J = np.ones((self.nobs, self.nobs))
            return J

    def _identity_matrix(self, add_to_model_data=True):
        if add_to_model_data:
            self.model_data["I"] = np.identity(self.nobs)
        else:
            I = np.identity(self.nobs)
            return I


    def __ols_fit(self, add_to_model_data=True):
        # Eigenvalues
        self.eigvals = np.linalg.eigvals(np.asarray(self.IV.T) @ np.asarray(self.IV))


        # Estimation of betas
        if add_to_model_data:
            try:
                self.model_data["betas"] = np.linalg.inv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            except:
                self.model_data["betas"] = np.linalg.pinv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
        else:
            try:
                betas = np.linalg.inv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            except:
                betas = np.linalg.pinv((np.asarray(self.IV.T) @ np.asarray(self.IV))) @ np.asarray(self.IV.T) @ np.asarray(self.DV)
            return betas



    def __compute_confidence_intervals(self, add_to_model_data=True):
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

        if add_to_model_data:
            self.model_data["conf_int_lower"] = np.array(conf_int_lower)
            self.model_data["conf_int_upper"] = np.array(conf_int_upper)
        else:
            return np.array(conf_int_lower), np.array(conf_int_upper)


    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def __regression_base_table(self):

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


    def __table_regression_results(self, return_type="Dataframe", pretty_format=True,
                                  decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4,
                                            "CI": 2, "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4,
                                            "Sum of Squares": 4, 'Degrees of Freedom': 1,
                                            'Mean Squares': 4, 'Effect size': 4},
                                 *args):

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




class model():
    """

    This is the deprecated base -model- object for Researchpy, replaced by core_model, and will be maintained for
    the next few versions for legacy support. Ample time will be provided as well as notifications. The change should
    have no impact on user's existing code and use cases.

    By default, missing observations are dropped from the data. -matrix_type- parameter determines
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
        # Copilot always wants to delete this LoL
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
                    "Sum of Squares": [round(self.model_data["sum_of_square_model"], decimals.get("Sum of Squares", 4))],
                    "Degrees of Freedom": [round(self.model_data["degrees_of_freedom_model"], decimals.get("Degrees of Freedom", 4))],
                    "Mean Squares": [round(self.model_data["msr"], decimals.get("Mean Squares", 4))],
                    "F value": [round(self.model_data["f_value_model"], decimals.get("test_stat", 4))],
                    "p-value": [round(self.model_data["f_p_value_model"], decimals.get("test_stat_p", 4))]
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




class linear_model(core_model):
    """

    This is a subclass of core_model for linear statistical models that use ordinary least squares.

    """

    def __init__(self, formula_like, data={}, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="ols",
                 obj_function="numeric",
                 solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        self.__name__ = "researchpy.linear_model"

        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                            family=family, link=link, solver_method=solver_method, obj_function=obj_function)


    def _regression_base_table(self):

        self._core_model__regression_base_table()


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

        self._core_model__table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

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
            return (descriptives.T, model_results, self._core_model__regression_base_table())

        elif return_type == "Dictionary":
            return (descriptives, model_results, self.regression_table_info)

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")




class general_model(core_model):
    """

    This is a subclass of core_model for generalized statistical models such as logistic, poisson, and etc.

    """

    def __init__(self, formula_like, data=None, matrix_type=1, conf_level=0.95,
                 family="gaussian", link="normal",
                 solver_method="mle",
                 obj_function="log-likelihood",
                 solver_options=None):

        self.__name__ = "researchpy.general_model"

        if data is None:
            data = {}
        if solver_options is None:
            solver_options = {"algorithm": "newton-raphson", "tol": 1e-7, "max_iter": 300, "display": True}

        #base_solver_options = {"tol": 1e-7, "max_iter": 300, "display": True}
        #self.solver_options = base_solver_options | solver_options


        super().__init__(formula_like=formula_like, data=data, matrix_type=matrix_type, conf_level=conf_level,
                         family=family, link=link, solver_options=solver_options, solver_method=solver_method, obj_function=obj_function)



    def __initialize_betas(self, initial_betas=None, initial_betas_method=None):

        if initial_betas is not None and initial_betas_method is not None:
            return("Warning: Both initial_betas and initial_betas_method were provided. Ignoring initial_betas_method and using provided initial_betas.")

        if initial_betas is not None:
            if isinstance(initial_betas, np.ndarray) and initial_betas.shape == (self.k, 1):
                self.model_data["betas"] = initial_betas
            else:
                raise ValueError(f"initial_betas must be a numpy array of shape ({self.k}, 1), but got {initial_betas.shape}")

        elif initial_betas_method.lower() == "ols":
            self._core_model__ols_fit()

        # Default initialization based on model type
        else:
            if self.__name__ in ["researchpy.Logistic", "researchpy.Logit"]:
                """Initialize betas with smarter starting values."""
                # Start with zeros (better than OLS for binary outcomes)
                betas = np.ones((self.n, self.k))

                # Set intercept to log odds of outcome proportion
                y_mean = np.mean(self.DV)
                if 0 < y_mean < 1:
                    betas[0] = np.log(y_mean / (1 - y_mean))

                self.model_data["betas"] = betas.reshape(-1, 1)



    def _neg_log_likelihood(self, params, *args, **kwargs):

        # Initialize an optimization tracker
        #tracker = LikelihoodTracker()

        return neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link,
            tracker=self._OptimizationTracker
        )

    def _gradient_neg_log_likelihood(self, params):
        """Gradient of negative log-likelihood."""
        return gradient_neg_log_likelihood(
            params=params,
            IV=self.IV,
            DV=self.DV,
            solver_options=self.solver_options,
            distribution_family=self._family,
            link_function=self._link
        )


    def __fit_model(self):
        """Fit the model using scipy.optimize with fallback."""
        self.logL = []
        converged = False

        # Try scipy.optimize first
        try:
            if self.solver_options["display"]:
                print(f"Starting optimization with {self.solver_options['algorithm']}...")

            result = scipy_minimize(
                fun=self._neg_log_likelihood,
                x0=self.model_data["betas"].flatten(),
                jac=self._gradient_neg_log_likelihood,
                method=self.solver_options["algorithm"],
                options={
                    'maxiter': self.solver_options["max_iter"],
                    'gtol': self.solver_options["tol"]
                },
                callback=None
            )

            if result.success:
                self.model_data["betas"] = result.x.reshape(-1, 1)
                self.logL.append(-result.fun)
                converged = True

                if self.solver_options["display"]:
                    print("")
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
            self.model_data["betas"], self.logL = newton_raphson(
                IV=self.IV,
                DV=self.DV,
                betas=self.model_data["betas"],
                tol=self.solver_options["tol"],
                max_iter=self.solver_options["max_iter"],
                display=self.solver_options["display"]
            )

        # Print log-likelihood values for each iteration
        #if self.solver_options["display"]:
        #    for i, log_likelihood in enumerate(self.logL, start=1):
        #        print(f"Iteration {i}: Log-Likelihood = {log_likelihood:.4f}")




    def predict(self, estimate=None, trans=None):
        return predict(self, estimate=estimate, trans=trans)


    def objective_function(self):
        if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:
            y_e = predict(estimate="predict_y")
            objective = likelihood.log_likelihood(y_e)

        return objective


    def _regression_base_table(self):

        self._core_model__regression_base_table()


    def _table_regression_results(self, report=None, return_type="Dataframe",
                                  decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4,
                                            "CI": 2, "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4,
                                            "Sum of Squares": 4, 'Degrees of Freedom': 1,
                                            'Mean Squares': 4, 'Effect size': 4  },
                                 pretty_format=True, *args):

        self._core_model__table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

        if return_type == "Dataframe":
            return self._core_model__regression_base_table()

        elif return_type == "Dictionary":
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")
