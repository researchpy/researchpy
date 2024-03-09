


import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .model import model
from .utility import *
from .predict import predict


class lm(model):
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


    def __init__(self, formula_like, data={}, conf_level=0.95):
        super().__init__(formula_like, data, matrix_type=1, conf_level=conf_level)
        self.__name__ = "researchpy.lm"

        @property
        def CI_LEVEL(self, conf_level):
            return super().CI_LEVEL(conf_level)
        @property
        def conf_level(self, conf_level):
            return super().conf_level(conf_level)

        self.model_data = {}

        ###########

        # J matrix of ones based on y
        self.model_data["J"] = numpy.ones((self.nobs, self.nobs))

        # identity matrix (I) based on x
        self.model_data["I"] = numpy.identity(self.nobs)

        # Eigenvalues
        self.eigvals = numpy.linalg.eigvals(self.IV.T @ self.IV)

        # Hat matrix
        try:
            self.model_data["H"] = self.IV @ numpy.linalg.inv(
                self.IV.T @ self.IV) @ self.IV.T
        except:
            self.model_data["H"] = self.IV @ numpy.linalg.pinv(
                self.IV.T @ self.IV) @ self.IV.T
            #print(f"NOTE: Using pseudo-inverse, smallest eigenvalue is {} ")

        # Estimation of betas
        try:
            self.model_data["betas"] = numpy.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = numpy.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        predicted_y = self.IV @ self.model_data["betas"]

        # Calculation of residuals (error)
        residuals = self.DV - predicted_y

        ###  Sum of Squares
        # Total sum of squares (SSTO)
        self.model_data["sum_of_square_total"] = float(self.DV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["J"] @ self.DV)

        # Model sum of squares (SSR)
        self.model_data["sum_of_square_model"] = float(self.model_data["betas"].T @ self.IV.T @ self.DV - (1/self.nobs) * self.DV.T @ self.model_data["J"] @ self.DV)

        # Error sum of squares (SSE)
        self.model_data["sum_of_square_residual"] = float(residuals.T @ residuals)

        ### Degrees of freedom
        # Model
        self.model_data["degrees_of_freedom_model"] = numpy.linalg.matrix_rank(self.IV) - 1

        # Error
        self.model_data["degrees_of_freedom_residual"] = self.nobs - numpy.linalg.matrix_rank(self.IV)

        # Total
        self.model_data["degrees_of_freedom_total"] = self.nobs - 1

        ### Mean Square
        # Model (MSR)
        self.model_data["msr"] = self.model_data["sum_of_square_model"] * (1/self.model_data["degrees_of_freedom_model"])

        # Residual (error; MSE)
        self.model_data["mse"] = self.model_data["sum_of_square_residual"] * (1/self.model_data["degrees_of_freedom_residual"])

        #Total (MST)
        self.model_data["mst"] = self.model_data["sum_of_square_total"] * (1/self.model_data["degrees_of_freedom_total"])

        ## Root Mean Square Error
        self.model_data["root_mse"] = float(numpy.sqrt(self.model_data["mse"]))

        ### F-values
        # Model
        self.model_data["f_value_model"] = float(self.model_data["msr"] / self.model_data["mse"])
        self.model_data["f_p_value_model"] = scipy.stats.f.sf( self.model_data["f_value_model"], self.model_data["degrees_of_freedom_model"], self.model_data["degrees_of_freedom_residual"])

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
        self.variance_covariance_residual_matrix = numpy.matrix(self.model_data["mse"] * (self.model_data["I"] - self.model_data["H"]))

        try:
            self.variance_covariance_beta_matrix = numpy.matrix(
                self.model_data["mse"] * numpy.linalg.inv(self.IV.T @ self.IV))
        except:
            self.variance_covariance_beta_matrix = numpy.matrix(
                self.model_data["mse"] * numpy.linalg.pinv(self.IV.T @ self.IV))

        #if robust_standard_errors == False:
            #self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))
        #elif robust_standard_errors == True:
            ## Not implemented yet so it's same as non-robust
         #   self.beta_variance_covariance_matrix = np.matrix(self.mse * np.linalg.inv(self.x.T @ self.x))


        ### Standard Errors
        self.model_data["standard_errors"] = (numpy.array(numpy.sqrt(self.variance_covariance_beta_matrix.diagonal()))).T

        ### Confidence Intrvals
        self.model_data["conf_int_lower"] = []
        self.model_data["conf_int_upper"] = []

        for beta, se in zip(self.model_data["betas"], self.model_data["standard_errors"]):

            try:
                lower, upper = scipy.stats.t.interval(self.CI_LEVEL, self.model_data["degrees_of_freedom_residual"],
                                                      loc=beta, scale=se)

                self.model_data["conf_int_lower"].append(float(lower))
                self.model_data["conf_int_upper"].append(float(upper))

            except:

                self.model_data["conf_int_lower"].append(numpy.nan)
                self.model_data["conf_int_upper"].append(numpy.nan)


        ### T-stastics
        self._test_stat_name = "t"
        self.model_data["test_stat"] = self.model_data["betas"] * (1 / self.model_data["standard_errors"])
        # Two-sided p-value
        self.model_data["test_stat_p_values"] = numpy.array(
                [float(scipy.stats.t.sf(numpy.abs(t), self.model_data["degrees_of_freedom_residual"]) * 2) for t in self.model_data["test_stat"]]
                )



    def results(self, return_type="Dataframe", pretty_format=True):

        return super().table_regression_results(return_type=return_type, pretty_format=pretty_format)[2]


    def predict(self, estimate=None):
        
        return predict(self, estimate= estimate)

