import numpy as np

from dejmps import dejmps_protocol_bob
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state

# def bell_state():
#     ket_0 = [[1], [0]]
#     ket_1 = [[0], [1]]

#     def ketbra(ket, bra):
#         return np.outer(ket, bra)
    
#     return (ketbra(ket_0, np.transpose(ket_0)) + ketbra(ket_1, np.transpose(ket_1))) / 2

def bell_state():
    ket_0 = [[1], [0]]
    ket_1 = [[0], [1]]
    
    ket_00 = np.kron(ket_0, ket_0)
    ket_11 = np.kron(ket_1, ket_1)

    return (ket_00 + ket_11) / np.sqrt(2)

def fidelity(state, dm):
    bra = np.transpose(state)
    return np.dot(bra, np.dot(dm, state))

def main(app_config=None):
    socket = Socket("bob", "alice", log_config=app_config.log_config)

    epr_socket = EPRSocket("alice", min_fidelity=0)

    bob = NetQASMConnection("bob", log_config=app_config.log_config, epr_sockets=[epr_socket])

    with bob:
        epr_1, epr_2 = epr_socket.recv(number=2)

        succ = dejmps_protocol_bob(epr_1, epr_2, bob, socket)
        dens_matrix = get_qubit_state(epr_1, reduced_dm=False)

        phi_00 = bell_state()
        f = fidelity(phi_00, dens_matrix)
    
    if succ:    
        print("Bob succeeded :-)")
        print("The fidelity of the final state is:", f)
    else:
        print("Bob did not succceed ;-(")


if __name__ == "__main__":
    main()
