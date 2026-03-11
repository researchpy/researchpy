# %% Loading libraries
import researchpy, pandas, numpy
import pytest
import pandas as pd
from researchpy.crosstab import crosstab

# %% Setting seed and creating data
numpy.random.seed(123)

df = pandas.DataFrame(numpy.random.randint(3, size= (101, 3)),
                  columns= ['disease', 'severity', 'alive'])

df.head()

def test_crosstab_basic():
    # Create sample data
    group1 = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
    group2 = pd.Series(['X', 'Y', 'X', 'Y', 'X', 'Y'])

    # Call the function
    result = crosstab(group1, group2)

    # Assertions
    assert result.loc['All', 'All'] == 6
    assert result.loc['A', 'X'] == 1
    assert result.loc['B', 'Y'] == 1

def test_crosstab_with_test():
    # Create sample data
    group1 = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
    group2 = pd.Series(['X', 'Y', 'X', 'Y', 'X', 'Y'])

    # Call the function with chi-square test
    ct, table = crosstab(group1, group2, test="chi-square")

    # Assertions
    assert ct.loc['All', 'All'] == 6
    assert "Pearson Chi-square" in table.columns[0]
    assert table.iloc[0, 1] >= 0  # Chi-square value should be non-negative

def test_crosstab_with_prop():
    # Create sample data
    group1 = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
    group2 = pd.Series(['X', 'Y', 'X', 'Y', 'X', 'Y'])

    # Call the function with row proportions
    result = crosstab(group1, group2, prop='row')

    # Assertions
    assert result.loc['A', 'X'] == 50.0  # 50% of group A is in column X
    assert result.loc['B', 'Y'] == 50.0  # 50% of group B is in column Y
