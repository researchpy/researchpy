# Used
import numpy as np
import scipy.stats
from scipy.special import expit, logit
import patsy
import pandas as pd


from researchpy.utility import *
from researchpy.predict import predict

from researchpy.optimize.trackers import OptimizationTracker





# Base model class for regression models. This class is not meant to be used directly, but rather to be inherited by
# specific regression model classes (e.g., OLS, Logistic, etc.). It contains common functionality and attributes that
# are shared across different types of regression models.
class CoreModel():
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

        self.__name__ = "Researchpy.CoreModel"

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
                                  decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 2, "test_stat_p": 4,
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
    def summary(self, return_string=False):
        """
        Print a formatted summary of the model results to the terminal.

        This method displays the model results in a format with:
        - Model header information (model type, number of observations, fit statistics)
        - Coefficient table with standard errors, test statistics, p-values, and confidence intervals

        Parameters
        ----------
        return_string : bool, optional
            If True, returns the formatted string instead of printing.
            Default is False (prints to terminal).

        Returns
        -------
        str or None
            If return_string is True, returns the formatted summary string.
            Otherwise, prints to terminal and returns None.

        Examples
        --------
        >>> model = rp.OLS("y ~ x1 + x2", data=df)
        >>> model.summary()  # prints to terminal
        >>> s = model.summary(return_string=True)  # returns string
        """
        # Get the results - this calls the subclass's results() method
        results = self.results(return_type="Dictionary")

        # Build the summary string
        output_lines = []

        # Determine total width for the output
        total_width = 78

        # === HEADER SECTION ===
        output_lines.append(self._summary_header(total_width))

        # === COEFFICIENT TABLE ===
        output_lines.append(self._summary_coef_table(results, total_width))

        # Join all sections
        summary_str = "\n".join(output_lines)

        if return_string:
            return summary_str
        else:
            print(summary_str)
            return None


    def _summary_header(self, width=78):
        """
        Build the header section of the summary output.

        Subclasses can override this to customize the header for their model type.
        """
        lines = []

        # Get model type name - clean up the __name__ attribute
        model_name = getattr(self, '__name__', 'Model').replace('Researchpy.', '').replace('researchpy.', '')

        # Map internal names to display names
        model_display_names = {
            'OLS': 'Linear Regression (OLS)',
            'LinearRegression': 'Linear Regression (OLS)',
            'LM': 'Linear Regression (OLS)',
            'Anova': 'Analysis of Variance',
            'LogisticRegression': 'Logistic Regression',
            'Logistic': 'Logistic Regression',
            'GeneralModel': 'Generalized Linear Model',
            'CoreModel': 'Model'
        }
        model_display = model_display_names.get(model_name, model_name)

        # Build header with model info on left, statistics on right
        # Left side info
        left_info = [model_display]

        # Add log likelihood if available (for MLE models)
        if hasattr(self, 'logL') and self.logL:
            if isinstance(self.logL, list):
                ll_val = self.logL[-1] if self.logL else None
            else:
                ll_val = self.logL
            if ll_val is not None:
                left_info.append(f"Log likelihood = {ll_val:.4f}")

        # Right side info
        right_info = []
        right_info.append(f"Number of obs = {self.nobs:>8}")

        # Add model-specific statistics
        if hasattr(self, 'model_data'):
            # For OLS models
            if 'r squared' in self.model_data:
                right_info.append(f"R-squared     = {self.model_data['r squared']:>8.4f}")
            if 'r squared adj.' in self.model_data:
                right_info.append(f"Adj R-squared = {self.model_data['r squared adj.']:>8.4f}")
            if 'f_value_model' in self.model_data:
                df_model = self.model_data.get('degrees_of_freedom_model', '?')
                right_info.append(f"F({df_model}, {self.model_data.get('degrees_of_freedom_residual', '?')}) = {self.model_data['f_value_model']:>8.4f}")
                right_info.append(f"Prob > F      = {self.model_data.get('f_p_value_model', 0):>8.4f}")
            if 'root_mse' in self.model_data:
                right_info.append(f"Root MSE      = {self.model_data['root_mse']:>8.4f}")

        # For MLE models (Logistic, etc.)
        if hasattr(self, 'LR_chi2'):
            df = getattr(self, 'model_df', '?')
            right_info.append(f"LR chi2({df})    = {self.LR_chi2:>8.4f}")
        if hasattr(self, 'model_p_value'):
            right_info.append(f"Prob > chi2   = {self.model_p_value:>8.4f}")

        # Build the header lines with left and right alignment
        max_lines = max(len(left_info), len(right_info))
        left_width = width // 2

        for i in range(max_lines):
            left_text = left_info[i] if i < len(left_info) else ""
            right_text = right_info[i] if i < len(right_info) else ""
            line = f"{left_text:<{left_width}}{right_text}"
            lines.append(line)

        return "\n".join(lines)


    def _summary_coef_table(self, results, width=78):
        """
        Build the coefficient table section of the summary output.

        Parameters
        ----------
        results : tuple or dict
            The results from self.results(return_type="Dictionary")
        width : int
            Total width of the output

        Returns
        -------
        str
            Formatted coefficient table
        """
        lines = []

        # Get coefficient data - handle different result structures
        if isinstance(results, tuple):
            # OLS returns (descriptives, model_results, coef_table)
            # Logistic returns (model_meta, model_description, coef_table)
            coef_data = results[-1]  # Coefficient table is always last
        else:
            coef_data = results

        # Determine column names and data
        dv_name = self.DV_name

        # Get the beta column name (could be "Coef." or "Odds Ratio")
        beta_col = "Coef."
        if isinstance(coef_data, dict):
            if "Odds Ratio" in coef_data:
                beta_col = "Odds Ratio"

        # Build column headers
        ci_label = f"[{int(self.CI_LEVEL * 100)}% Conf. Interval]"
        test_stat_label = self._test_stat_name
        p_label = f"P>|{test_stat_label}|"

        # Define column widths
        col_widths = {
            'var': 12,
            'coef': 11,
            'stderr': 10,
            'tstat': 8,
            'pval': 7,
            'ci_low': 11,
            'ci_high': 11
        }

        # Header separator
        sep_line = "-" * width
        mid_sep = "-" * col_widths['var'] + "+" + "-" * (width - col_widths['var'] - 1)

        lines.append(sep_line)

        # Column header line
        header = (f"{dv_name:>{col_widths['var']}} | "
                  f"{beta_col:>{col_widths['coef']}} "
                  f"{'Std. Err.':>{col_widths['stderr']}} "
                  f"{test_stat_label:>{col_widths['tstat']}} "
                  f"{p_label:>{col_widths['pval']}} "
                  f"{ci_label:>{col_widths['ci_low'] + col_widths['ci_high'] + 1}}")
        lines.append(header)
        lines.append(mid_sep)

        # Get data rows
        if isinstance(coef_data, dict):
            # Dictionary format
            var_names = coef_data.get(dv_name, [])
            coefs = coef_data.get(beta_col, coef_data.get("Coef.", []))
            std_errs = coef_data.get("Std. Err.", [])
            test_stats = coef_data.get(self._test_stat_name, [])
            p_values = coef_data.get("p-value", [])
            ci_col = coef_data.get(f"{int(self.CI_LEVEL * 100)}% Conf. Interval", [])

            for i, var in enumerate(var_names):
                coef = coefs[i] if i < len(coefs) else ""
                se = std_errs[i] if i < len(std_errs) else ""
                ts = test_stats[i] if i < len(test_stats) else ""
                pv = p_values[i] if i < len(p_values) else ""

                # Handle confidence intervals (stored as [lower, upper] pairs)
                if i < len(ci_col) and isinstance(ci_col[i], (list, tuple)):
                    ci_low, ci_high = ci_col[i]
                else:
                    ci_low, ci_high = "", ""

                # Format the row
                row = self._format_coef_row(var, coef, se, ts, pv, ci_low, ci_high, col_widths)
                lines.append(row)

        lines.append(sep_line)

        return "\n".join(lines)


    def _format_coef_row(self, var_name, coef, stderr, tstat, pval, ci_low, ci_high, col_widths):
        """
        Format a single row of the coefficient table.

        Parameters
        ----------
        var_name : str
            Variable name
        coef, stderr, tstat, pval, ci_low, ci_high : numeric or str
            Statistics for the row
        col_widths : dict
            Dictionary of column widths

        Returns
        -------
        str
            Formatted row string
        """
        # Format numeric values, handle missing/empty
        def fmt_num(val, width, decimals=4):
            if val == "" or val is None or (isinstance(val, float) and np.isnan(val)):
                return " " * width
            try:
                return f"{float(val):>{width}.{decimals}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{width}}"

        # Truncate variable name if too long
        var_display = var_name[:col_widths['var']] if len(str(var_name)) > col_widths['var'] else var_name

        row = (f"{var_display:>{col_widths['var']}} | "
               f"{fmt_num(coef, col_widths['coef'], 4)} "
               f"{fmt_num(stderr, col_widths['stderr'], 4)} "
               f"{fmt_num(tstat, col_widths['tstat'], 2)} "
               f"{fmt_num(pval, col_widths['pval'], 3)} "
               f"{fmt_num(ci_low, col_widths['ci_low'], 4)} "
               f"{fmt_num(ci_high, col_widths['ci_high'], 4)}")

        return row

