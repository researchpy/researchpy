# -*- coding: utf-8 -*-
"""
Ordinary Least Squares (OLS) Regression

This module provides the OLS class for fitting linear regression models
using the ordinary least squares method.
"""

import numpy as np
import scipy.stats
import pandas as pd

from researchpy.models.core_model import CoreModel
#from researchpy.models.linear_model import LinearModel
from researchpy.utility import *
from researchpy.predict import predict


class OLS(CoreModel):
    """
    Ordinary Least Squares (OLS) regression for continuous outcomes.

    This class provides linear regression analysis using the ordinary least squares
    method, including:
    - Coefficient estimation
    - Standard errors and t-statistics
    - R-squared and adjusted R-squared
    - F-test for overall model significance
    - Effect size measures (Eta squared, Epsilon squared, Omega squared)
    - Confidence intervals
    - Variance-covariance matrices

    Parameters
    ----------
    formula_like : str
        A string representing a valid Patsy formula (e.g., "y ~ x1 + x2").
        See https://patsy.readthedocs.io/en/latest/ for formula syntax.
    data : dict or DataFrame, optional
        Data containing the variables referenced in the formula.
    conf_level : float, optional
        Confidence level for confidence intervals. Default is 0.95.

    Attributes
    ----------
    model_data : dict
        Dictionary containing model results including:
        - 'betas': Estimated coefficients
        - 'standard_errors': Standard errors of coefficients
        - 'test_stat': t-statistics
        - 'test_stat_p_values': p-values for t-tests
        - 'conf_int_lower', 'conf_int_upper': Confidence interval bounds
        - 'r squared', 'r squared adj.': R-squared values
        - 'sum_of_square_total', 'sum_of_square_model', 'sum_of_square_residual'
        - 'degrees_of_freedom_model', 'degrees_of_freedom_residual', 'degrees_of_freedom_total'
        - 'msr', 'mse', 'mst': Mean squares
        - 'root_mse': Root mean square error
        - 'f_value_model', 'f_p_value_model': F-statistic and p-value
        - 'Eta squared', 'Epsilon squared', 'Omega squared': Effect sizes

    Examples
    --------
    >>> import researchpy as rp
    >>> import pandas as pd
    >>> df = pd.DataFrame({'y': [1, 2, 3, 4, 5], 'x': [1, 2, 3, 4, 5]})
    >>> model = rp.OLS("y ~ x", data=df)
    >>> model.results()

    See Also
    --------
    LinearRegression : Alias for OLS
    LM : Alias for OLS (R-style shorthand)
    Anova : Analysis of Variance (inherits from OLS)
    """

    def __init__(self, formula_like, data=None, conf_level=0.95):

        if data is None:
            data = {}

        self._test_stat_name = "t"
        self._CI_LEVEL = conf_level

        super().__init__(formula_like, data, matrix_type=1, conf_level=conf_level,
                         solver_method="ols", family="gaussian", link="normal",
                         obj_function="numeric")

        self.__name__ = "Researchpy.OLS"

        # Dictionary for storing model results and information. This will be populated by specific regression model
        # classes that inherit from this base class.
        if not hasattr(self, "model_data"):
            self.model_data = {}

        # J matrix of ones based on y
        self.model_data["J"] = np.ones((self.nobs, self.nobs))

        # Identity matrix (I) based on x
        self.model_data["I"] = np.identity(self.nobs)

        # Eigenvalues
        self.eigvals = np.linalg.eigvals(self.IV.T @ self.IV)

        # Hat matrix
        try:
            self.model_data["H"] = self.IV @ np.linalg.inv(
                self.IV.T @ self.IV) @ self.IV.T
        except np.linalg.LinAlgError:
            self.model_data["H"] = self.IV @ np.linalg.pinv(
                self.IV.T @ self.IV) @ self.IV.T

        # Estimation of betas
        try:
            self.model_data["betas"] = np.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except np.linalg.LinAlgError:
            self.model_data["betas"] = np.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        predicted_y = self.IV @ self.model_data["betas"]

        # Calculation of residuals (error)
        residuals = self.DV - predicted_y

        ### Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = float(
            (self.DV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["J"] @ self.DV).item()
        )

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = float(
            (self.model_data["betas"].T @ self.IV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["J"] @ self.DV).item()
        )

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = float((residuals.T @ residuals).item())

        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = np.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.nobs - np.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.nobs - 1

        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        # Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])

        ## Root Mean Square Error
        self.model_data["root_mse"] = float(np.sqrt(self.model_data["mse"]))

        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf(
            self.model_data["f_value_model"],
            self.model_data["degrees_of_freedom_model"],
            self.model_data["degrees_of_freedom_residual"]
        )

        ### Effect Size Measures
        # Model
        self.model_data["r squared"] = (self.model_data["sum_of_square_model"] / self.model_data["sum_of_square_total"])
        self.model_data["r squared adj."] = 1 - (self.model_data["degrees_of_freedom_total"] / self.model_data["degrees_of_freedom_residual"]) * (
            self.model_data["sum_of_square_residual"] / self.model_data["sum_of_square_total"])
        self.model_data["Eta squared"] = self.model_data["r squared"]

        self.model_data["Epsilon squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"])

        self.model_data["Omega squared"] = (self.model_data["degrees_of_freedom_model"] * (self.model_data["msr"] - self.model_data["mse"])) / (self.model_data["sum_of_square_total"] + self.model_data["mse"])

        ### Variance-covariance matrices
        # Non-robust - from Applied Linear Statistical Models, pg. 203
        self.variance_covariance_residual_matrix = np.matrix(self.model_data["mse"] * (self.model_data["I"] - self.model_data["H"]))

        try:
            self.variance_covariance_beta_matrix = np.matrix(
                self.model_data["mse"] * np.linalg.inv(self.IV.T @ self.IV))
        except np.linalg.LinAlgError:
            self.variance_covariance_beta_matrix = np.matrix(
                self.model_data["mse"] * np.linalg.pinv(self.IV.T @ self.IV))

        ### Standard Errors
        self.model_data["standard_errors"] = (np.array(np.sqrt(self.variance_covariance_beta_matrix.diagonal()))).T

        ### Confidence Intervals
        conf_int_lower = []
        conf_int_upper = []

        for beta, se in zip(self.model_data["betas"], self.model_data["standard_errors"]):
            try:
                lower, upper = scipy.stats.t.interval(
                    self._CI_LEVEL,
                    self.model_data["degrees_of_freedom_residual"],
                    loc=beta,
                    scale=se
                )
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

        ### T-statistics
        self.model_data["test_stat"] = self.model_data["betas"] * (1 / self.model_data["standard_errors"])

        # Two-sided p-value
        self.model_data["test_stat_p_values"] = np.array([
            float((scipy.stats.t.sf(np.abs(t), self.model_data["degrees_of_freedom_residual"]) * 2).item())
            for t in self.model_data["test_stat"]
        ])


    def _table_regression_results(self, return_type="Dataframe", pretty_format=True,
                                  decimals=None, *args):
        """
        Build and return the regression results tables.

        This method constructs the coefficient table using the parent class method,
        then builds the model summary and ANOVA table specific to OLS.
        """
        if decimals is None:
            decimals = {
                "Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4
            }

        # Build the coefficient table using parent's private method
        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

        if pretty_format:
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
                "Epsilon squared": ['', ''],
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
                "Epsilon squared": top["Epsilon squared"] + bottom["Epsilon squared"],
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
                "Epsilon squared": [np.nan, np.nan],
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
                "Epsilon squared": top["Epsilon squared"] + bottom["Epsilon squared"],
                "Omega squared": top["Omega squared"] + bottom["Omega squared"]
            }

        if return_type == "Dataframe":
            descriptives_df = pd.DataFrame.from_dict(descriptives, orient="index")
            model_results_df = pd.DataFrame.from_dict(model_results)
            return (descriptives_df.T, model_results_df, self._CoreModel__regression_base_table())

        elif return_type == "Dictionary":
            return (descriptives, model_results, self.regression_table_info)

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")


    def results(self, return_type="Dataframe", pretty_format=True,
                decimals=None, *args):
        """
        Return the regression results.

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        tuple
            If return_type is "Dataframe": (descriptives_df, model_results_df, coefficients_df)
            If return_type is "Dictionary": (descriptives_dict, model_results_dict, coefficients_dict)
        """
        if decimals is None:
            decimals = {
                "Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4
            }

        return self._table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)


    def predict(self, estimate=None):
        """
        Generate predictions from the fitted model.

        Parameters
        ----------
        estimate : str, optional
            Type of estimate to return.

        Returns
        -------
        ndarray
            Predicted values.
        """
        return predict(self, estimate=estimate)


    def _summary_header_left(self, width=78, model_summary_df=None):
        """
        Build the left side of the OLS summary header with an ANOVA source table.

        Shows model name followed by a Source/SS/df/MS table with Model, Residual,
        and Total rows.

        Parameters
        ----------
        width : int
            Total character width of the output.
        model_summary_df : DataFrame or None
            Model summary DataFrame from ``self.results()``.  When None the
            ANOVA mini-table is skipped (used by Anova subclass).

        Returns
        -------
        list of str
            Lines for the left side of the header.
        """
        if model_summary_df is None:
            return [self._get_model_display_name()]

        lines = []
        model_display = self._get_model_display_name()

        # Define column widths for ANOVA table
        col_w = {'source': 10, 'ss': 12, 'df': 4, 'ms': 12}
        table_w = col_w['source'] + col_w['ss'] + col_w['df'] + col_w['ms'] + 6

        def fmt_num(val, w, d=4):
            try:
                return f"{float(val):>{w}.{d}f}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        def fmt_int(val, w):
            try:
                return f"{int(val):>{w}}"
            except (ValueError, TypeError):
                return f"{str(val):>{w}}"

        sep = "-" * col_w['source'] + "+" + "-" * (table_w - col_w['source'] - 1)

        lines.append(f"{model_display:<{table_w}}")
        lines.append(
            f"{'Source':>{col_w['source']}} | "
            f"{'SS':>{col_w['ss']}} "
            f"{'df':>{col_w['df']}} "
            f"{'MS':>{col_w['ms']}}"
        )
        lines.append(sep)
        lines.append(
            f"{'Model':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_model', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_model', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('msr', 0), col_w['ms'])}"
        )
        lines.append(
            f"{'Residual':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_residual', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_residual', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('mse', 0), col_w['ms'])}"
        )
        lines.append(sep)
        lines.append(
            f"{'Total':>{col_w['source']}} | "
            f"{fmt_num(self.model_data.get('sum_of_square_total', 0), col_w['ss'])} "
            f"{fmt_int(self.model_data.get('degrees_of_freedom_total', 0), col_w['df'])} "
            f"{fmt_num(self.model_data.get('mst', 0), col_w['ms'])}"
        )

        return lines


    def _splice_f_stat_lines(self):
        """
        Return the F-statistic and Prob > F lines for the header right side.

        These are not stored in the descriptives DataFrame, so they are
        built from ``self.model_data`` and spliced into the header.
        Shared by OLS and Anova header_right methods.

        Returns
        -------
        list of str
        """
        df_model = self.model_data.get("degrees_of_freedom_model", 0)
        df_resid = self.model_data.get("degrees_of_freedom_residual", 0)
        return [
            f"F({df_model}, {df_resid}) = "
            f"{self.model_data.get('f_value_model', 0):>8.2f}",
            f"Prob > F      = "
            f"{self.model_data.get('f_p_value_model', 0):>8.4f}",
        ]


    def _summary_header_right(self, width=78, descriptives_df=None):
        """
        Build the right side of the OLS summary header with fit statistics.

        When *descriptives_df* is provided the values are read from that
        DataFrame via ``to_string(header=False)`` so the summary is driven
        entirely by the DataFrames returned from ``self.results()``.

        Parameters
        ----------
        width : int
            Available character width.
        descriptives_df : DataFrame or None
            Descriptives DataFrame from ``self.results()`` (transposed,
            single-row with stat names as columns).

        Returns
        -------
        list of str
            Lines for the right side of the header.
        """
        if descriptives_df is not None:
            # OLS descriptives_df is transposed: single row, columns = stat names.
            # Convert to index-oriented for to_string.
            desc_lines = descriptives_df.T.to_string(header=False).split("\n")
            return [desc_lines[0]] + self._splice_f_stat_lines() + desc_lines[1:]

        # Fallback: build from self.model_data directly
        return [
            f"Number of obs = {self.nobs:>8}",
        ] + self._splice_f_stat_lines() + [
            f"R-squared     = {self.model_data.get('r squared', 0):>8.4f}",
            f"Adj R-squared = {self.model_data.get('r squared adj.', 0):>8.4f}",
            f"Root MSE      = {self.model_data.get('root_mse', 0):>8.4f}",
        ]


    def _get_summary_parts(self):
        """
        Return (model_description_df, coef_df) for the OLS summary.

        Calls ``self.results()`` with stdout suppressed (to avoid side-effect
        prints) and unpacks the 3-tuple into the two DataFrames that
        ``CoreModel.summary()`` needs.

        Returns
        -------
        tuple of (model_description_df, coef_df)
        """
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            model_description_df, model_summary_df, coef_df = self.results(
                return_type="Dataframe", pretty_format=True
            )
        return model_summary_df, model_description_df, coef_df



# Convenience aliases
LinearRegression = OLS
LM = OLS

