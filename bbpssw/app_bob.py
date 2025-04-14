import numpy as np

from bbpssw import bbpssw_protocol_bob
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state
import yaml
import os

def get_link_fidelity():
    """Read the link fidelity from network.yaml"""
    try:
        # Assuming network.yaml is in the same directory as the script
        with open('network.yaml', 'r') as file:
            network_config = yaml.safe_load(file)
            
        # Get the first link's fidelity (you may need to adjust if you have multiple links)
        for link in network_config.get('links', []):
            return link.get('fidelity', 0.0)
            
        return 0.0  # Default if no links found
    except Exception as e:
        print(f"Error reading network.yaml: {e}")
        return 0.0  # Default on error

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

def main(app_config=None):

    socket = Socket("bob", "alice", log_config=app_config.log_config)

    epr_socket = EPRSocket("alice")

    bob = NetQASMConnection("bob", log_config=app_config.log_config, epr_sockets=[epr_socket])

    phi_00 = bell_state()

    print("Bell state norm:", np.linalg.norm(phi_00))

    with bob:
        epr1, epr2 = epr_socket.recv(number=2)

        succ = bbpssw_protocol_bob(epr1, epr2, bob, socket)

        dens_out = get_qubit_state(epr1, reduced_dm=False)

        print("Density matrix trace:", np.trace(dens_out))
        print("Density matrix:", dens_out)

        f_out = fidelity(phi_00, dens_out)

    f_in = get_link_fidelity()
    print("The expected fidelity of the final stats is:", th_fidelity(f_in))

    if succ:    
        print("Bob succeeded :-)")
        print("The fidelity of the final state is:", f_out)
    else:
        print("Bob did not succceed ;-(")

if __name__ == "__main__":
    main()
