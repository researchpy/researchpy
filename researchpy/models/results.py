# -*- coding: utf-8 -*-
"""
Result Containers

Standardized dataclass containers for model and test results returned by
researchpy model classes and postestimation utilities.

``ModelResults`` is returned by model ``.results()`` methods (e.g., Regress,
Anova, LogisticRegression).  ``TestResults`` is returned by postestimation
test ``.results()`` methods (e.g., LikelihoodRatioTest, future Wald/contrast
tests).

Both support iteration for tuple-style unpacking::

    # ModelResults
    stats, table, coefs, details = model.results()

    # TestResults
    name, stats, details = lr_test.results()

Or attribute access::

    result = model.results()
    result.fit_statistics
    result.coefficients
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Union, Optional

import pandas as pd


@dataclass
class ModelResults:
    """
    Standardized container for model results.

    Returned by ``Regress.results()``, ``Anova.results()``,
    ``LogisticRegression.results()``, and future model classes.

    Parameters
    ----------
    model_name : str
        Display name of the model (e.g., "Linear Regression (OLS)",
        "Analysis of Variance", "Logistic Regression").
    fit_statistics : DataFrame or dict
        Model fit statistics such as N, R², Root MSE, log-likelihood, etc.
    model_table : DataFrame, dict, or None
        Model-level summary table. For OLS this is the SS/df/MS/F ANOVA
        decomposition; for ANOVA it is the full ANOVA table with factor
        rows and effect sizes. ``None`` for MLE-based models that lack
        a sum-of-squares decomposition (e.g., logistic, Poisson).
    coefficients : DataFrame, dict, or None
        Coefficient / parameter estimate table with columns for estimate,
        standard error, test statistic, p-value, and confidence interval.
        ``None`` when the model's primary output is not a coefficient table
        (e.g., ANOVA where the main table is ``model_table``).
    details : dict or None
        Any additional model-specific information (e.g., type of standard
        errors, odds ratios, convergence info for MLE models). Default is
        ``None``.

    Examples
    --------
    Tuple unpacking (all 5 fields):

    >>> result = model.results()
    >>> name, stats, table, coefs, details = result

    Attribute access:

    >>> result.fit_statistics
    >>> result.coefficients
    """

    model_name: str
    fit_statistics: Union[pd.DataFrame, dict]
    model_table: Optional[Union[pd.DataFrame, dict]]
    coefficients: Optional[Union[pd.DataFrame, dict]]
    details: Optional[dict] = None


    def to_dict(self):
        """Convert the ModelResults dataclass to a dictionary."""
        return {
            'model_name': self.model_name,
            'fit_statistics': self.fit_statistics.to_dict() if isinstance(self.fit_statistics, pd.DataFrame) else self.fit_statistics,
            'model_table': self.model_table.to_dict() if isinstance(self.model_table, pd.DataFrame) else self.model_table,
            'coefficients': self.coefficients.to_dict() if isinstance(self.coefficients, pd.DataFrame) else self.coefficients,
            'details': self.details,
        }


    def _get_summary(self):
        fit_statistics = pd.DataFrame.from_dict(self.fit_statistics) if isinstance(self.fit_statistics, dict) else self.fit_statistics
        model_table = pd.DataFrame.from_dict(self.model_table) if isinstance(self.model_table, dict) else self.model_table
        coefficients = pd.DataFrame.from_dict(self.coefficients) if isinstance(self.coefficients, dict) else self.coefficients
        details = pd.DataFrame.from_dict(self.details) if isinstance(self.details, dict) else self.details

        print(fit_statistics.to_string(index=False),
              model_table.to_string(index=False),
              coefficients.to_string(index=False), sep="\n"*2)

        if details is not None:
            for key, value in details.items():
                print(key, value)



    def __iter__(self):
        """Yield fields in order for tuple-style unpacking."""
        for f in fields(self):
            yield getattr(self, f.name)

    def __repr__(self):
        parts = [f"ModelResults(model_name={self.model_name!r}"]
        for name, val in [("fit_statistics", self.fit_statistics),
                          ("model_table", self.model_table),
                          ("coefficients", self.coefficients),
                          ("details", self.details)]:
            if val is None:
                parts.append(f"  {name}=None")
            elif isinstance(val, pd.DataFrame):
                parts.append(f"  {name}=DataFrame({val.shape[0]}x{val.shape[1]})")
            elif isinstance(val, dict):
                parts.append(f"  {name}=dict({len(val)} keys)")
            else:
                parts.append(f"  {name}={type(val).__name__}")
        return "\n".join(parts) + "\n)"


    def __str__(self):
        """Custom string representation."""
        return f"ModelResults({self.model_name})"






@dataclass
class TestResults:
    """
    Standardized container for postestimation test results.

    Returned by ``LikelihoodRatioTest.results()`` and future postestimation
    test classes (Wald test, contrast tables, etc.).

    Parameters
    ----------
    test_name : str
        Display name of the test (e.g., "Likelihood Ratio Test").
    statistics : DataFrame or dict
        Test statistics including test statistic value, degrees of freedom,
        and p-value.
    details : dict or None
        Any additional test-specific information (e.g., null model
        coefficients, convergence info). Default is ``None``.

    Examples
    --------
    Tuple unpacking:

    >>> name, stats, details = lr_test.results()

    Attribute access:

    >>> result = lr_test.results()
    >>> result.statistics
    """

    test_name: str
    statistics: Union[pd.DataFrame, dict]
    details: Optional[dict] = None

    def __iter__(self):
        """Yield fields in order for tuple-style unpacking."""
        for f in fields(self):
            yield getattr(self, f.name)

    def __repr__(self):
        parts = [f"TestResults(test_name={self.test_name!r}"]
        for name, val in [("statistics", self.statistics),
                          ("details", self.details)]:
            if val is None:
                parts.append(f"  {name}=None")
            elif isinstance(val, pd.DataFrame):
                parts.append(f"  {name}=DataFrame({val.shape[0]}x{val.shape[1]})")
            elif isinstance(val, dict):
                parts.append(f"  {name}=dict({len(val)} keys)")
            else:
                parts.append(f"  {name}={type(val).__name__}")
        return "\n".join(parts) + "\n)"
