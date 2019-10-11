import numpy as np
import pandas as pd

labels = ['a', 'b', 'c']

minha_lista = [10, 20, 30]
arr = np.array([10, 20, 30])
d = {'a':10, 'b':20, 'c':30}

print(pd.Series(data=minha_lista, index=labels))

series = pd.Series(data=minha_lista, index=labels)

print('\n\n')

print(series['b'])

print('\n\n')

print(pd.Series(minha_lista, labels))
print(pd.Series(labels, minha_lista))

print('\n\n')

ser1 = pd.Series([1, 2, 3], index=['EUA', 'Alemanha', 'URSS'])

print(ser1)

ser2 = pd.Series([1, 2, 3], index=['EUA', 'Alemanha', 'Italia'])

print('\n\n')

print(ser1 + ser2)