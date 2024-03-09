
import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .model import general_model
from .utility import *
from .predict import predict







class logistic(general_model):

    __name__ = "researchpy.logit"
    def __init__(self, formula_like, data={}, family="binomial", link="logit", tol=1e-7, max_iter=300, display=True):
        super().__init__(formula_like, data, matrix_type=1, family="binomial", link="logit", tol=1e-7, max_iter=300, display=True)

        self.model_data = {}
        self.n, self.k = self.IV.shape

        ###########

        ## Initializing values betas with OLS
        try:
            self.model_data["betas"] = numpy.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = numpy.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        self.predicted_y = numpy.exp(self.IV @ self.model_data["betas"]) / \
                           (1 + numpy.exp(self.IV @ self.model_data["betas"]))

        def y_e():
            return numpy.exp(self.IV @ self.model_data["betas"]) / \
                           (1 + numpy.exp(self.IV @ self.model_data["betas"]))

        self.predicted_y = y_e()
        ## NOTE: log(y_e()) = logit transformation of the probability (logit response function)

        def logL():
            #ll = np.sum(self.DV * (self.IV @ self.model_data["betas"])) - np.sum(np.log((1 + y_e())))
            ll = numpy.sum(y_e()) - numpy.sum(numpy.log((1 + y_e())))

            return ll


        # Iteration until convergence
        it = 0
        error = 100
        self.logL = []

        while numpy.any(error > self.tol) and it < self.max_iter:

            u = y_e()
            H = -(self.IV.T @ (u * self.IV))
            G = self.IV.T @ (self.DV - u)

            betas_new = self.model_data["betas"] - (numpy.linalg.inv(H) @ G)

            error = numpy.abs(betas_new - self.model_data["betas"])
            self.model_data["betas"] = betas_new

            likelihood = logL()
            self.logL.append(likelihood)

            it += 1

            # Print iterations
            if display:
                iteration_update = f'Iteration {it}: Log-likelihood = {likelihood}'
                print(iteration_update)


