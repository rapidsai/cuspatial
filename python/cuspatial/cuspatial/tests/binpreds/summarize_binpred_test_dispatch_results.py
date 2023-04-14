import pandas as pd

pp = pd.read_csv('predicate_passes.csv')
pf = pd.read_csv('predicate_fails.csv')
fp = pd.read_csv('feature_passes.csv')
ff = pd.read_csv('feature_fails.csv')

print(pp)
print(pf)
print(fp)
print(ff)
