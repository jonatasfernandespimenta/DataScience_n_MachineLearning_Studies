import numpy as np

arr = np.arange(0, 16)
array = np.arange(0, 10)

print(arr)# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
print(array)# [ 0  1  2  3  4  5  6  7  8  9]

# Somando Arrays (devem ter os mesmo tamanho)
print('Soma')
soma = arr + arr
print(soma) # [ 0  2  4  6  8 10 12 14 16 18 20 22 24 26 28 30]

print('Subtracao')
subtracao = arr - arr
print(subtracao) # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

print('Multiplicacao')
multiplicacao = arr * arr
print(multiplicacao) # [  0   1   4   9  16  25  36  49  64  81 100 121 144 169 196 225]

mult = arr * 2
print(mult)

print('Divisao')
divisao = arr / arr
print(divisao) # [0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]

# Somando valores especificos de arrays
print('Soma de valor especifico de um array com valor especifico de outro array')
z = arr[10] + array[1]
print(z)

print('Soma de valor especifico de um array com valor especifico do mesmo')
y = arr[10] + arr[11]
print(y)
