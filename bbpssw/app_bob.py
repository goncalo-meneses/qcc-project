import numpy as np

from bbpssw import bbpssw_protocol_bob
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

    socket = Socket("bob", "alice", log_config=app_config.log_config)

    epr_socket = EPRSocket("alice")

    bob = NetQASMConnection("bob", log_config=app_config.log_config, epr_sockets=[epr_socket])

    phi_00 = bell_state()

    with bob:
        epr1, epr2 = epr_socket.recv(number=2)

        succ = bbpssw_protocol_bob(epr1, epr2, bob, socket)

        dens_out = get_qubit_state(epr1, reduced_dm=False)

        f_out = fidelity(phi_00, dens_out)

    if succ:    
        print("Bob succeeded :-)")
        print("The fidelity of the final state is:", f_out)
    else:
        print("Bob did not succceed ;-(")

if __name__ == "__main__":
    main()
