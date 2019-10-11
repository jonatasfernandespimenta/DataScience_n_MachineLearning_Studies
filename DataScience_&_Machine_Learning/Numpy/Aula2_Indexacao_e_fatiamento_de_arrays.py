import numpy as np

# IMPORTANTE: TAMANHO DO RESHAPE TEM Q SER A MULTIPLICACAO DO SEU ARRAY
# Exemplo: array.linspace(0, 100, 30)
# Exemplo: array.reshape(3, 10) 3 vezes 10 e 30 assim sendo possivel fzr o reshape

arr = np.arange(0, 30, 3)
print(arr)
# Output: [ 0  3  6  9 12 15 18 21 24 27]

print(' \nPuxa o elemento 9 do array\n')
arr[3]
print(arr[3])

print('\n Puxa multiplos elementos do array\n')
print(arr[2:5]) # Output: [ 6  9 12 ]

print(' \nPuxa desde o primeiro elemento ate o determinado\n')
print(arr[:5]) # Output: [ 0  3  6  9 12 ]

print(' \nPuxa de um determinado elemento ate o ultimo\n')
print(arr[2:]) # Out: [ 6  9 12 15 18 21 24 27 ]

print(' \nAltera dados do array\n')
arr[2:] = 100 # Altera todos os valores do 3 elemento ate o ultimo para 100
print(arr) # Out: [  0   3 100 100 100 100 100 100 100 100 ]

arr = np.arange(50).reshape((5, 10)) # Cria matriz de 0 ate 50 e define para ter 5 colunas e 10 linhas
print(arr) # Out: [[ 0  1  2  3  4  5  6  7  8  9]
           #      [10 11 12 13 14 15 16 17 18 19]
           #      [20 21 22 23 24 25 26 27 28 29]
           #      [30 31 32 33 34 35 36 37 38 39]
           #      [40 41 42 43 44 45 46 47 48 49]]

print(' \nFatiamento\n')
arr[:4][:] # Aqui e possivel visualizar da quarta linha ate o resto, primeiro colchete e a linha e o segundo a coluna
#Para puxar as 3 primeiras linhas e as 4 primeiras colunas, utilizaria [:3, :4]

print(' \nPuxando os elementos maior que 20\n')
bol = arr > 20
print(arr[bol]) # Out: [21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
                #       45 46 47 48 49]
