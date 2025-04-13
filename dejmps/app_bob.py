import numpy as np
import os
import yaml

from dejmps import dejmps_protocol_bob
from netqasm.sdk import EPRSocket
from netqasm.sdk.external import NetQASMConnection, Socket, get_qubit_state

def get_fidelity(dm):
    '''
    Computes the fidelity of a pure state and a density matrix
    returns computed fidelity
    '''
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    fid = np.dot(state.conj().T, np.dot(dm, state))
    return np.real_if_close(fid.item())

def read_simulation_parameters(yaml_path="network.yaml"):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    fidelity = config["links"][0]["fidelity"]
    gate_fidelity = config["nodes"][0]["gate_fidelity"]  # assumes gate fidelity is the same for alice and bob

    return fidelity, gate_fidelity


def main(app_config=None):
    socket = Socket("bob", "alice", log_config=app_config.log_config)

    epr_socket = EPRSocket("alice")

    bob = NetQASMConnection("bob", log_config=app_config.log_config, epr_sockets=[epr_socket])

    with bob:
        epr_1, epr_2 = epr_socket.recv(number=2)

        succ = dejmps_protocol_bob(epr_1, epr_2, bob, socket)

        dens_out = get_qubit_state(epr_1, reduced_dm=False)

    fidelity, gate_fidelity = read_simulation_parameters()
    print(get_fidelity(dens_out))

    FILENAME = f'./data/f={fidelity}_g={gate_fidelity}_dejmps.npz'

    if os.path.exists(FILENAME):
        # Load existing
        data = np.load(FILENAME, allow_pickle=False)
        matrices = data["matrices"]
        matrices = np.concatenate((matrices, [dens_out]), axis=0)
        successes = data["successes"]
        successes = np.concatenate((successes, [succ]), axis=0)
    else:
        # Start new array
        matrices = np.array([dens_out])
        successes = np.array([succ])

    np.savez(FILENAME,
             matrices=matrices,
             successes=successes,
             protocol='dejmps',
             fidelity=fidelity,
             gate_fidelity=gate_fidelity,)


if __name__ == "__main__":
    main()
