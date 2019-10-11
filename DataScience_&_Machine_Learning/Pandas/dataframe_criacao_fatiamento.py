import numpy as np
import pandas as pd

np.random.seed(101)
df = pd.DataFrame(np.random.randn(5, 4), index='A B C D E'.split(), columns='W X Y Z'.split())

print(df)

print('\n\n')

print(df['W'])

print('\n\n')

print(df[['W', 'Z']])

print('\n\n')

print(df.W)

print('\n\n')

df['new'] = df['W'] + df['X']

print(df)

print('\n\n')

df.drop('new', axis=1)

print('\n\n')

df.drop('new', axis=1, inplace=True)

print('\n\n')

print(df.loc['A', 'W'])

df.loc[['A', 'B'], ['X', 'Y', 'Z']]

df.iloc[1:4, 2:]