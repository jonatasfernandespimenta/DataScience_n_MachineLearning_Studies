import numpy as np

minha_lista = [1, 2, 3]
np.array(minha_lista)

minha_matriz = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
np.array(minha_matriz)

# Cria um array de 0 a 9
np.arrange(0, 10)

# Cria um array que vai de 2 em 2
np.arange(0, 10, 2)

# Cria arrays apenas com números 0
np.zeros(3)

# Cria matriz apenas com números 0
np.zeros((5, 5))

# Cria matriz apenas com números 1
np.ones((3, 3))

# Cria matriz de identidade
np.eye(4)

# O número final (2) especifica quantos números terão
np.linspace(0, 10, 2)
# Saida:
# array([0., 10.])

np.linspace(0, 10, 3)
# Saida:
# array([0., 5,. 10.])

# Cria array com numeros aleatorios
np.random.rand(5)

# Cria matriz com numeros aleatorios
np.random.rand(5, 4)

# Cria array com numeros aleatorios não uniformes
np.random.randn(4)

# Cria array com numeros INTEIROS aleatorios
# 0 = Valor inicial, 100 = valor maximo desejado, 10 = Quantidade
np.random.randint(0, 100, 10)
