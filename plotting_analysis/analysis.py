"""Code to read data from disk and compute statistics (with bootstrap error)"""

import numpy as np
import os
from collections import defaultdict


# === Reading data ===

# Define the top-level folder names
protocols = {
    'dejmps': 'data',
    'bbpssw': 'data',
    'epl': 'data',
    'epl_local_change': 'data_local_change'
}

protocol_data = {name: defaultdict(dict) for name in protocols}

base_dir = '../'

for name, subfolder in protocols.items():
    data_dir = os.path.join(base_dir, name if name != 'epl_local_change' else 'epl', subfolder)

    for filename in os.listdir(data_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(data_dir, filename)
            with np.load(filepath) as data:
                fidelity = float(data['fidelity'].item())
                gate_fidelity = float(data['gate_fidelity'].item())
                matrices = data['matrices']
                successes = data['successes']
                protocol = str(data['protocol'].item())  # read protocol from file

                protocol_data[name][(fidelity, gate_fidelity)] = {
                    'matrices': matrices,
                    'successes': successes,
                    'protocol': protocol
                }


# === Variable definitions to be used in analysis notebooks ===

# Each data variable is indexed by an (f, g) combo and gives you matrices and successes for that combo
dejmps_data = protocol_data['dejmps']
bbpssw_data = protocol_data['bbpssw']
epl_data    = protocol_data['epl']
epl_local_change_data = protocol_data['epl_local_change']

phi_00 = np.array([[1, 0, 0, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 1]]) / 2

phi_01 = np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]]) / 2


# === Analysis code ===

#  Helper to select data in some link/gate fidelity ranges. Leave range as None to accept all data for that parameter.
def filter_keys_by_range(protocol_data, fidelity_range=None, gate_fidelity_range=None):
    filtered_keys = []
    for (fidelity, gate_fidelity) in protocol_data.keys():
        in_fid_range = True if fidelity_range is None else fidelity_range[0] <= fidelity <= fidelity_range[1]
        in_gate_range = True if gate_fidelity_range is None else gate_fidelity_range[0] <= gate_fidelity <= gate_fidelity_range[1]
        if in_fid_range and in_gate_range:
            filtered_keys.append((fidelity, gate_fidelity))
    return sorted(filtered_keys)


# Helper to see quantity of available data in some range
def print_protocol_summary(protocol_data, fidelity_range=None, gate_fidelity_range=None):
    print(f"Summary for protocol")
    print("-" * 50)
    
    f_gs = filter_keys_by_range(protocol_data, fidelity_range, gate_fidelity_range)
    if not f_gs:
        print("No data available in the specified range.\n")
        return
    
    for (fidelity, gate_fidelity) in f_gs:
        entry = protocol_data[(fidelity, gate_fidelity)]
        num_simulations = entry['matrices'].shape[0]
        print(f"Fidelity: {fidelity:.4f}, Gate Fidelity: {gate_fidelity:.4f} -> Simulations: {num_simulations}")
    print("\n")


# Helper for below function to compute statistics and errors at a fixed link/gate fidelity combo
# Computes output fidelity with respect to given Bell state and estimates error with bootstrap
# Computes success probability and estimates error with known stderr formula for binomial RVs
def analyze_single_combo(matrices, successes, target_bell_state, fidelity_in, n_bootstrap=1000, seed=None):
    rng = np.random.default_rng(seed)
    successes = np.asarray(successes, dtype=bool)
    total = len(successes)
    num_success = np.count_nonzero(successes)

    if total == 0:
        return {
            'avg_fidelity': None,
            'avg_fidelity_std': None,
            'success_probability': None,
            'success_probability_std': None,
        }

    success_probability = num_success / total

    if num_success == 0:
        return {
            'avg_fidelity': None,
            'avg_fidelity_std': None,
            'success_probability': success_probability,
            'success_probability_std': 0.0,
        }

    # Extract successful matrices
    successful_matrices = matrices[successes]

    # shape of matrix product is (N, 4, 4), so axis1 and axis2 should point to 1 and 2 for the matrix dimensions
    fidelities = np.trace(matrices @ target_bell_state, axis1=1, axis2=2)
    fidelities = np.real_if_close(fidelities)

    # Bootstrap the fidelity mean
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(fidelities, size=len(fidelities), replace=True)
        bootstrap_means.append(sample.mean())

    fid_mean = fidelities.mean()
    fid_err = np.std(bootstrap_means)

    return {
        'avg_fidelity': fid_mean,
        'avg_fidelity_err': fid_err,
        'delta_fidelity': fid_mean - fidelity_in,  # same error/spread as regular fidelities
        'success_probability': success_probability,
        'success_probability_err': np.sqrt(success_probability * (1 - success_probability) / total)
    }


# Main function that computes statistics over a range of link/gate fidelities.
# Returns data in the same format as the input data was read in above: a dictionary indexed with (f, g) combos, returning average fidelity and success probability for each, their estimated errors, and some metadata
# Computes output fidelity with respect to Phi_00 for all protocols except EPL with an initial correction, in which case Phi_01 is used. Can also manually input a Bell state to override default behavior.
def analyze_multiple_combos(protocol_data, 
                            target_bell_state=None, 
                            fidelity_range=None, gate_fidelity_range=None, 
                            n_bootstrap=1000, seed=None):
    result = {}
    f_gs = filter_keys_by_range(protocol_data, fidelity_range, gate_fidelity_range)

    for (f, g) in f_gs:
        entry = protocol_data[f, g]
        protocol = entry['protocol']
        if target_bell_state is None:
            target_bell_state = phi_01 if protocol == 'epl_local_change' else phi_00

        stats = analyze_single_combo(
            entry['matrices'],
            entry['successes'],
            target_bell_state,
            fidelity_in=f,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        result[f, g] = stats

    return result

