
import numpy
import scipy.stats
import patsy
import pandas

from .summary import summarize
from .model import model
from .utility import *
from .predict import predict


class logit(model):


    def __init__(self, formula_like, data={}):
        super().__init__(formula_like, data, matrix_type=1)

        self.model_data = {}
