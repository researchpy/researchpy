import patsy
from patsy import EvalEnvironment
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'researchpy', 'core'))
from containerclasses import Term, ModelTerms

np.random.seed(42)
n = 60
data = pd.DataFrame({
    'y': np.random.randn(n),
    'drug': np.random.choice([1, 2, 3, 4], n),
    'disease': np.random.randn(n),
    'group': np.random.choice(['A', 'B', 'C'], n)
})

env = EvalEnvironment([{}])

# Test 1: Factor + continuous + interaction
print("=" * 70)
print("TEST 1: C(drug, Treatment(2)) + disease + C(drug):disease")
print("=" * 70)
_, IV = patsy.dmatrices('y ~ C(drug, Treatment(2)) + disease + C(drug, Treatment(2)):disease', data, eval_env=env)
mt = ModelTerms.from_design_info(IV.design_info)
print(repr(mt))
print()
for t in mt:
    print(f"  {t.name}:")
    print(f"    levels    = {t.levels}")
    print(f"    reference = {t.reference}")
    print(f"    columns_cleaned = {t.columns_cleaned}")
    print()

print()
print("term_map:", mt.term_map)
print()
print("column_map:", mt.column_map)

# Test 2: Two categorical factor interaction
print()
print("=" * 70)
print("TEST 2: C(drug, Treatment(2)) * C(group)")
print("=" * 70)
_, IV2 = patsy.dmatrices('y ~ C(drug, Treatment(2)) * C(group)', data, eval_env=env)
mt2 = ModelTerms.from_design_info(IV2.design_info)
print(repr(mt2))
print()
for t in mt2:
    print(f"  {t.name}:")
    print(f"    levels    = {t.levels}")
    print(f"    reference = {t.reference}")
    print(f"    columns_cleaned = {t.columns_cleaned}")
    print()
