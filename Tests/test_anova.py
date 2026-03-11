import pytest
import pandas as pd
from researchpy.MultivariableRegression.anova import anova

def test_anova_initialization():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the anova object
    model = anova("Y ~ X", data)

    # Assertions
    assert model.__name__ == "researchpy.anova"
    assert model.formula == "Y ~ X"

def test_anova_sum_of_squares():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the anova object
    model = anova("Y ~ X", data, sum_of_squares=1)

    # Assertions
    assert "Sum of Squares" in model.model_data
