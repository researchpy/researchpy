
import numpy as np
import scipy.stats
import patsy
import pandas

from researchpy.model import model
from researchpy.utility import *
from researchpy.predict import predict
from researchpy.objective_functions import likelihood




class logistic(model):

    __name__ = "researchpy.logit"

    def __init__(self, formula_like, data={},
                 solver_method="mle", solver_options={"tol": 1e-7, "max_iter": 300, "display": True}):

        super().__init__(formula_like, data, matrix_type=1,
                         solver_method=solver_method, solver_options=solver_options,
                         family="binomial", link="logit", obj_function="log-likelihood")

        self.family = "binomial"
        self.link = "logit"

        self.model_data = {}
        self.n, self.k = self.IV.shape

        ###########

        ## Initializing values betas with OLS
        try:
            self.model_data["betas"] = np.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = np.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV



        def objective_function(self):

            if self.obj_func.lower() in ["log-likelihood", "log likelihood", "ll"]:
                y_e = predict(estimate="y")
                objective = likelihood.log_likelihood(y_e)

            return objective

        # Iteration until convergence
        it = 0
        error = 100
        self.logL = []

        while np.any(error > self.solver_options["tol"]) and it < self.solver_options["max_iter"]:

            u = super().predict(estimate="y", trans=np.exp)

            H = -(self.IV.T @ (u * self.IV))
            G = self.IV.T @ (self.DV - u)

            betas_new = self.model_data["betas"] - (np.linalg.inv(H) @ G)

            error = np.abs(betas_new - self.model_data["betas"])
            self.model_data["betas"] = betas_new

            ll = likelihood.log_likelihood(super().predict(estimate="y", trans=np.exp))
            self.logL.append(ll)

            it += 1

            # Print iterations
            if self.solver_options["display"]:
                iteration_update = f'Iteration {it}: Log-likelihood = {ll}'
                print(iteration_update)

"""
class logistic_working(general_model):

    __name__ = "researchpy.logit"
    def __init__(self, formula_like, data={}, family="binomial", link="logit", tol=1e-7, max_iter=300, display=True):
        super().__init__(formula_like, data, matrix_type=1, family="binomial", link="logit", tol=1e-7, max_iter=300, display=True)

        self.model_data = {}
        self.n, self.k = self.IV.shape

        ###########

        ## Initializing values betas with OLS
        try:
            self.model_data["betas"] = np.linalg.inv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = np.linalg.pinv((self.IV.T @ self.IV)) @ self.IV.T @ self.DV

        # Predicted y values
        self.predicted_y = np.exp(self.IV @ self.model_data["betas"]) / \
                           (1 + np.exp(self.IV @ self.model_data["betas"]))

        def y_e():
            return np.exp(self.IV @ self.model_data["betas"]) / \
                           (1 + np.exp(self.IV @ self.model_data["betas"]))

        self.predicted_y = y_e()
        ## NOTE: log(y_e()) = logit transformation of the probability (logit response function)

        def logL():
            #ll = np.sum(self.DV * (self.IV @ self.model_data["betas"])) - np.sum(np.log((1 + y_e())))
            ll = np.sum(y_e()) - np.sum(np.log((1 + y_e())))

            return ll


        # Iteration until convergence
        it = 0
        error = 100
        self.logL = []

        while np.any(error > self.tol) and it < self.max_iter:

            u = y_e()
            H = -(self.IV.T @ (u * self.IV))
            G = self.IV.T @ (self.DV - u)

            betas_new = self.model_data["betas"] - (np.linalg.inv(H) @ G)

            error = np.abs(betas_new - self.model_data["betas"])
            self.model_data["betas"] = betas_new

            likelihood = logL()
            self.logL.append(likelihood)

            it += 1

            # Print iterations
            if display:
                iteration_update = f'Iteration {it}: Log-likelihood = {likelihood}'
                print(iteration_update)

"""
