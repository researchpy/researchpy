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
        if display_summary:
            self.summary()


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




# Convenience aliases for users who prefer different naming conventions
LinearRegression = Regress
LM = Regress

