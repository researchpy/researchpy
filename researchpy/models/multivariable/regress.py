# -*- coding: utf-8 -*-
"""
Ordinary Least Squares (OLS) Regression

This module provides the OLS class for fitting linear regression models
using the ordinary least squares method.
"""

import numpy as np
import scipy.stats
import pandas as pd

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
    ModelEffects : ModelEffects
        Dataclass containing model-level results including:
        - sum_of_square_total, sum_of_square_model, sum_of_square_residual
        - degrees_of_freedom_model, degrees_of_freedom_residual, degrees_of_freedom_total
        - msr, mse, mst: Mean squares
        - root_mse: Root mean square error
        - model_test_stat, model_test_pval: F-statistic and p-value
        - r_squared, r_squared_adj: R-squared values
        - eta_squared, epsilon_squared, omega_squared: Effect sizes
    CoefResults : CoefResults
        Dataclass containing per-coefficient results including:
        - betas: Estimated coefficients
        - std_error: Standard errors of coefficients
        - test_stat: t-statistics
        - p_value: p-values for t-tests
        - conf_int_lower, conf_int_upper: Confidence interval bounds

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


        self.ModelFit.model = self.__name__
        self.ModelFit.model_display_name = self._get_model_display_name()

        # Build ModelResults (results() sets self.ModelResults internally)
        self.results(return_type="Dataframe", pretty_format=True, table_decimals=table_decimals)

        # Display the model results summary
        if display_summary:
            self.summary()



    def results(self, return_type="Dataframe", pretty_format=True, table_decimals=None, *args):
        """
        Return the regression results as a ``ModelResults`` dataclass.

        Parameters
        ----------
        return_type : str, optional
            Format of the returned results. Either "Dataframe" or "Dictionary".
            Default is "Dataframe".
        pretty_format : bool, optional
            Whether to format the output for display. Default is True.
        table_decimals : dict, optional
            Dictionary specifying decimal places for different statistics.

        Returns
        -------
        ModelResults
            A dataclass with fields:
            - ``model_name``: ``"Linear Regression (OLS)"``
            - ``fit_statistics``: Descriptive fit statistics (DataFrame or dict)
            - ``model_table``: ANOVA decomposition table (DataFrame or dict)
            - ``coefficients``: Coefficient table (DataFrame or dict)
            - ``details``: ``None``

            Supports tuple unpacking::

                name, fit_stats, model_table, coefs, details = model.results()

            Or attribute access::

                result = model.results()
                result.fit_statistics
                result.coefficients
        """

        if table_decimals is not None:
            self._table_decimals = self._table_decimals | table_decimals

        return self._get_results(return_type=return_type,
                                 pretty_format=pretty_format,
                                 table_decimals=self._table_decimals)



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




# Convenience aliases for users who prefer different naming conventions
LinearRegression = Regress
LM = Regress

