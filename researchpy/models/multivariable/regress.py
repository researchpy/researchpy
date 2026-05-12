# -*- coding: utf-8 -*-
"""
Ordinary Least Squares (OLS) Regression

This module provides the OLS class for fitting linear regression models
using the ordinary least squares method.
"""

import numpy as np
import scipy.stats
import pandas as pd

#from researchpy.models.base import CoreModel
from researchpy.models.linear_model import LinearModel
from researchpy.utility import *
from researchpy.predict import predict


class Regress(LinearModel):
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
    >>> model = rp.Regress("y ~ x", data=df)
    >>> model.results()

    See Also
    --------
    LinearRegression : Alias for Regress
    LM : Alias for Regress (R-style shorthand)
    Anova : Analysis of Variance (inherits from OLS)
    """

    def __init__(self, formula_like, data=None, conf_level=0.95, display_summary=True,
                 table_decimals=None):

        self._test_stat_name = "t"
        self._CI_LEVEL = conf_level

        super().__init__(formula_like, data, matrix_type=1, conf_level=conf_level,
                         solver_method="ols", family="gaussian", link="normal", obj_function="numeric",
                         table_decimals=table_decimals)

        self.__name__ = "Researchpy.Regress"


        # Display the model results summary
        #if display_summary:
        #    self.summary()


    def _table_regression_results(self, return_type="Dataframe", pretty_format=True,
                                  table_decimals=None, *args):
        """
        Build and return the regression results tables.

        This method constructs the coefficient table using the parent class method,
        then builds the model summary and ANOVA table specific to OLS.
        """

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        # Build the coefficient table using parent's private method
        self._CoreModel__table_regression_results(return_type=return_type, pretty_format=pretty_format, table_decimals=self._table_decimals)


        if return_type == "Dataframe":
            return self._CoreModel__regression_base_table()

        elif return_type == "Dictionary":
            return self.regression_table_info

        else:
            print("Not a valid return type option, please use either 'Dataframe' or 'Dictionary'.")


    def results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):
        """
        Return the regression results.

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        pretty_format : bool, optional
            Whether to format the output for display. Default is False.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        tuple
            If return_type is "Dataframe": (descriptives_df, model_results_df, coefficients_df)
            If return_type is "Dictionary": (descriptives_dict, model_results_dict, coefficients_dict)
        """

        return self._get_results(return_type=return_type, pretty_format=pretty_format, table_decimals=table_decimals)



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
            f"Number of obs = {self.n:>8}",
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
LinearRegression = Regress
LM = Regress

