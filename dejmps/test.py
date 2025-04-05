import numpy as np

ket_0 = [[1], [0]]
ket_1 = [[0], [1]]

def ketbra(ket):
    bra = np.transpose(ket)
    return np.outer(ket, bra)

bell_00 = ketbra(ket_0)

print(bell_00)