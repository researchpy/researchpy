# -*- coding: utf-8 -*-

from .version import __version__
from .utility import *
from .model import *
from .ttest import ttest
from .summary import *
from .correlation import *
from .crosstab import *
from .difference_test import *
from .basic_stats import *
from .signrank import *
from .predict import *

# New modular structure imports
from .models import CoreModel, GeneralModel
from .models.multivariable import Regress, LinearRegression, LM, Anova, LogisticRegression, Logistic


# Legacy MultivariableRegression imports - commented out during refactoring
# These will be removed once refactoring is complete as they were never published
from .ols import ols
from .anova import anova
#from .MultivariableRegression.logistic import logistic
from .MultivariableRegression.logit import logit
