import numpy as np

from dejmps import dejmps_protocol_alice
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state

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

def main(app_config=None):

    socket = Socket("alice", "bob", log_config=app_config.log_config)

    epr_socket = EPRSocket("bob")

    alice = NetQASMConnection("alice", log_config=app_config.log_config, epr_sockets=[epr_socket])

    with alice:
        epr_1, epr_2 = epr_socket.create(number=2)

        succ = dejmps_protocol_alice(epr_1, epr_2, alice, socket)

    if succ:
        print("Alice succeeded :-)")
    else:
        print("Alice did not succceed ;-(")

if __name__ == "__main__":
    main()
