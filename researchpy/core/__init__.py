# -*- coding: utf-8 -*-
"""
ResearchPy Code Module

This module provides the core classes for ResearchPy, including result containers, base model classes, and
helper/utility functions.

Usage:
    >>> import researchpy.core
    >>> from researchpy.core import *

"""

from researchpy.core.containerclasses import ModelFit, ModelEffects, CoefResults, FactorEffects, FitStatistics, ModelResults, TestResults, Term, ModelTerms

# Define what gets exported with "from researchpy.core import *"
__all__ = [
    #'CoreModel',
    #'GeneralModel',
    'ModelFit',
    'ModelEffects',
    'CoefResults',
    'FactorEffects',
    'FitStatistics',
    'ModelResults',
    'TestResults',
    'Term',
    'ModelTerms'
]