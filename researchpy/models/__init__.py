# -*- coding: utf-8 -*-
"""
Researchpy Models Module

This module provides the core model classes for regression analysis in Researchpy.
The module is organized into submodules for different types of models:

- Core model classes (CoreModel, GeneralModel) - base classes for building regression models
- Multivariable regression models (Logistic, etc.) - specific regression implementations
- Postestimation methods (likelihood ratio tests, etc.) - tools for model evaluation

Usage:
    from researchpy.models import CoreModel, GeneralModel
    from researchpy.models.multivariable import Logistic

"""

from researchpy.models.base import CoreModel
#from researchpy.models.core_model import CoreModel
from researchpy.models.general_model import GeneralModel

# Define what gets exported with "from researchpy.models import *"
__all__ = [
    "CoreModel",
    "GeneralModel",
    #"ModelResults",
    #"TestResults",
]

