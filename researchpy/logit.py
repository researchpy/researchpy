
import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .model import model
from .utility import *
from .predict import predict







class logit(model):

    def __init__(self, formula_like, data={}, iterations=100):
        super().__init__(formula_like, data, matrix_type=1)

        self.model_data = {}

        # Estimation of betas
        try:
            self.model_data["betas"] = numpy.linalg.inv(
                (self.IV.T @ self.IV)) @ self.IV.T @ self.DV
        except:
            self.model_data["betas"] = numpy.linalg.pinv(
                (self.IV.T @ self.IV)) @ self.IV.T @ self.DV


        def sigmoid(z):
            return 1 / (1 + numpy.exp(-z))


        for i in range(iterations):

            # Predicted y values
            predicted_y = sigmoid(self.IV @ self.model_data["betas"])
