import numpy as np

data = np.load("./dejmps/data/f=0.3_g=1_dejmps.npz")
print(data.files)

def bell_state():
    ket_0 = [[1.+0.j], [0.j]]
    ket_1 = [[0.j], [1.+0.j]]
    
    ket_00 = np.kron(ket_0, ket_0)
    ket_11 = np.kron(ket_1, ket_1)

    return (ket_00 + ket_11) / np.sqrt(2)

def fidelity(state, dm):
    state = state.reshape(-1, 1)
    bra = state.conj().T
    result = np.dot(bra, np.dot(dm, state))
    return np.real_if_close(result.item())

def th_psucc(f):
    return f**2 + 2 * f * (1 - f) / 3 + 5 * ((1 - f) / 3)**2

def th_fidelity(f):
    return (f**2 + ((1-f)/3)**2) / th_psucc(f)

matrices = data['matrices']
successes = data['successes']
dens_out = np.average(matrices, axis=0)

phi_00 = bell_state()
f = data['fidelity']

print(fidelity(phi_00, dens_out))
print(th_fidelity(f))