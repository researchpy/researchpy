import numpy as np
import scipy.stats
import patsy
import pandas as pd

from researchpy.models.linear_model import LinearModel
from researchpy.utility import *
from researchpy.predict import predict


class lm(LinearModel):
    """

    Parameters
    ----------
    formula_like: string
        A string which represents a valid Patsy formula; https://patsy.readthedocs.io/en/latest/

    data : array_like
        Array like data object.

    Returns
    -------
    Ordinary Least Squares regression object with assessible methods and stored class data. The class data
    which is stored is the following:


        self.model_data: dictionary object
            The following data is stored with the dictionary key ("Key"):
                J matrix ('J')
                Identify matrix ('I')
                Hat matrix ('H')
                Coeffeicients ('betas')
                Total Sum of Squares ('sum_of_square_total')
                Model Sum of Squares ('sum_of_square_model')
                Residual Sum of Squares ('sum_of_square_residual')
                Model Degrees of Freedom ('degrees_of_freedom_model')
                Residual Degrees of Freedom ('degrees_of_freedom_residual')
                Total Degrees of Freedom ('degrees_of_freedom_total')
                Model Mean Squares ('msr')
                Error Mean Squares ('mse')
                Total Mean Squares ('mst')
                Root Mean Square Error ('root_mse')
                Model F-value ('f_value_model')
                Model p-value ('f_p_value_model')
                R-sqaured ('r squared')
                Adjusted R-squared ('r squared adj.')
                Eta squared ('Eta squared')
                Epsilon squared ('Epsilon squared')
                Omega squared ('Omega squared')
=][]
    """


    def __init__(self, formula_like, data=None, conf_level=0.95, display_summary=True,
                 table_decimals=None):

        self._test_stat_name = "t"

        super().__init__(formula_like, data=data, conf_level=conf_level, matrix_type=1, solver_method="ols",
                         family="gaussian", link="normal", obj_function="numeric",
                         table_decimals=table_decimals)

        #self.__name__ = "rsearchpy.lm"
        self.__name__ = "rsearchpy.LM"

        if display_summary == True:
            self.summary(table_decimals=table_decimals)



    def results(self, return_type="Dataframe", pretty_format=True,
                decimals={"Coef.": 2, "Std. Err.": 4, "test_stat": 4, "test_stat_p": 4, "CI": 2,
                          "Root MSE": 4, "R-squared": 4, "Adj R-squared": 4, "Sum of Squares": 4,
                          'Degrees of Freedom': 1, 'Mean Squares': 4, 'Effect size': 4},
                *args):

        return self._table_regression_results(return_type=return_type, pretty_format=pretty_format, decimals=decimals)

    def predict(self, estimate=None):
        
        return predict(self, estimate= estimate)

