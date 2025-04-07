import numpy as np

from epl import epl_protocol_bob
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state

def bell_state():
    ket_0 = [[1.+0.j], [0.j]]
    ket_1 = [[0.j], [1.+0.j]]
    
    ket_01 = np.kron(ket_0, ket_1)
    ket_10 = np.kron(ket_1, ket_0)

    return (ket_01 + ket_10) / np.sqrt(2)

def fidelity(state, dm):
    state = state.reshape(-1, 1)
    bra = state.conj().T
    result = np.dot(bra, np.dot(dm, state))
    return np.real_if_close(result.item())


def main(app_config=None):
    socket = Socket("bob", "alice", log_config=app_config.log_config)

    epr_socket = EPRSocket("alice")

    bob = NetQASMConnection("bob", log_config=app_config.log_config, epr_sockets=[epr_socket])

    phi_10 = bell_state()

    with bob:
        epr_1, epr_2 = epr_socket.recv(number=2)

        succ = epl_protocol_bob(epr_1, epr_2, bob, socket)

        dens_out = get_qubit_state(epr_1, reduced_dm=False)

        f_out = fidelity(phi_10, dens_out)
    
    if succ:    
        print("Bob succeeded :-)")
        print("The fidelity of the final state is:", f_out)
    else:
        print("Bob did not succceed ;-(")


if __name__ == "__main__":
    main()
