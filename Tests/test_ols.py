import pytest
import pandas as pd
from researchpy.MultivariableRegression.ols import ols

def test_ols_initialization():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the ols object
    model = ols("Y ~ X", data)

    # Assertions
    assert model.__name__ == "researchpy.ols"
    assert model.formula == "Y ~ X"

def test_ols_model_data():
    # Create sample data
    data = pd.DataFrame({
        "X": [1, 2, 3, 4, 5],
        "Y": [2, 4, 6, 8, 10]
    })

    # Initialize the ols object
    model = ols("Y ~ X", data)

    # Assertions
    assert "betas" in model.model_data
    assert "H" in model.model_data
    assert "J" in model.model_data
