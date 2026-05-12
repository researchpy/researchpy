# -*- coding: utf-8 -*-
"""
Researchpy Multivariable Regression Models

This submodule contains implementations of multivariable regression models including:

- OLS: Ordinary Least Squares regression (aliases: LinearRegression, LM)
- Anova: Analysis of Variance (inherits from OLS)
- LogisticRegression: Logistic regression for binary outcomes using MLE
  (alias: Logistic)

Future models to be added:
- PoissonRegression: Poisson regression for count data (alias: Poisson)

Usage:
    from researchpy.models.multivariable import OLS, Anova, LogisticRegression
    # or use the aliases
    from researchpy.models.multivariable import LinearRegression, LM, Logistic

    # OLS regression
    ols_model = OLS("y ~ x1 + x2", data=df)
    ols_results = ols_model.results()

    # ANOVA
    anova_model = Anova("y ~ C(group)", data=df)
    anova_results = anova_model.results()

    # Logistic regression
    logit_model = LogisticRegression("outcome ~ predictor1 + predictor2", data=df)
    logit_results = logit_model.results()

"""

from researchpy.models.multivariable.regress import Regress, LinearRegression, LM
#from researchpy.models.multivariable.ols import OLS
from researchpy.models.multivariable.anova import Anova
from researchpy.models.multivariable.logistic import LogisticRegression, Logistic

# Define what gets exported with "from researchpy.models.multivariable import *"
__all__ = [
    # Regress and aliases
    "Regress",
    "LinearRegression",
    "LM",
    # ANOVA
    "Anova",
    # Logistic and alias
    "LogisticRegression",
    "Logistic",
]
