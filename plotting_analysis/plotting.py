import numpy as np
import matplotlib.pyplot as plt


def protocol_string(protocol):
    if protocol == 'dejmps':
        return 'DEJMPS'
    if protocol == 'bbpssw':
        return 'BBPSSW'
    if protocol == 'epl':
        return 'EPL'
    if protocol == 'epl_local_change':
        return r'EPL with $X_B$ correction applied'
    

# hacky thing to get pretty protocol name
def protocol_string_from_dict(dict):
    first_data = next(iter(dict.values()))
    protocol = first_data['protocol']
    return protocol_string(protocol)


# Given a result dict, tries to extract 1D data. This means that the input must have multiple values for either fidelity or gate_fidelity but only one value for the other parameter.
# Delta determines whether the actual value of the output fidelity or the difference in fidelity from input to output is plotted (maybe want to change this to a percentage in the future)
def extract_1d(results, delta):
    # Auto infer sweep param
    unique_fs = {f for f, _ in results}
    unique_gs = {g for _, g in results}

    if len(unique_fs) > 1 and len(unique_gs) > 1:
        raise ValueError("Data is not 1D: both fidelity and gate_fidelity vary.")
    elif len(unique_fs) > 1:
        sweep_param = 'fidelity'
    elif len(unique_gs) > 1:
        sweep_param = 'gate_fidelity'
    else:
        raise ValueError("Data is constant — neither fidelity nor gate_fidelity varies.")
    
    grouped = {}
    for (f, g), stats in results.items():
        key = f if sweep_param == 'fidelity' else g
        grouped[key] = stats

    xs = sorted(grouped.keys())

    to_plot = 'delta_fidelity' if delta else 'avg_fidelity'
    avg_fids = [grouped[x][to_plot] for x in xs]
    avg_fid_errs = [grouped[x]['avg_fidelity_err'] for x in xs]
    succ_probs = [grouped[x]['success_probability'] for x in xs]
    succ_prob_errs = [grouped[x]['success_probability_err'] for x in xs]

    return xs, avg_fids, avg_fid_errs, succ_probs, succ_prob_errs, sweep_param


def plot_fidelity(results_list, title=None, delta=False):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, avg_fids, avg_fid_errs, _, _, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, avg_fids, yerr=avg_fid_errs,
                         fmt='-o', label=protocol,
                         capsize=3)
    
    ax1.set_xlabel(sweep_param.replace('_', ' ').title())
    ax1.set_ylabel('Δ Fidelity' if delta else 'Average Fidelity')
    fig.tight_layout()
    if title is None:
        title = f"Sweep of {sweep_param.replace('_', ' ').title()}"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_success_prob(results_list, title=None, delta=False):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, _, _, success_probs, success_prob_errs, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, success_probs, yerr=success_prob_errs,
                         fmt='-o', label=protocol,
                         capsize=3)
    
    ax1.set_xlabel(sweep_param.replace('_', ' ').title())
    ax1.set_ylabel('Success probability')
    fig.tight_layout()
    if title is None:
        title = f"Sweep of {sweep_param.replace('_', ' ').title()}"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_2d_heatmaps(results, title_prefix=""):
    
    # todo: crash if given results aren't 2d

    keys = sorted(results.keys())
    fidelities = sorted(set(f for f, _ in keys))
    gate_fidelities = sorted(set(g for _, g in keys))

    fidelity_idx = {f: i for i, f in enumerate(fidelities)}
    gate_fid_idx = {g: i for i, g in enumerate(gate_fidelities)}

    avg_fid_grid = np.full((len(fidelities), len(gate_fidelities)), np.nan)
    succ_prob_grid = np.full((len(fidelities), len(gate_fidelities)), np.nan)

    for (f, g), stats in results.items():
        i, j = fidelity_idx[f], gate_fid_idx[g]
        avg_fid_grid[i, j] = stats['avg_fidelity']
        succ_prob_grid[i, j] = stats['success_probability']

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(avg_fid_grid, origin='lower', aspect='auto', cmap='viridis',
                        extent=[min(gate_fidelities), max(gate_fidelities), min(fidelities), max(fidelities)])
    axs[0].set_title(f'{title_prefix} Avg Fidelity')
    axs[0].set_xlabel('Gate Fidelity')
    axs[0].set_ylabel('Fidelity')
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(succ_prob_grid, origin='lower', aspect='auto', cmap='plasma',
                        extent=[min(gate_fidelities), max(gate_fidelities), min(fidelities), max(fidelities)])
    axs[1].set_title(f'{title_prefix} Success Probability')
    axs[1].set_xlabel('Gate Fidelity')
    axs[1].set_ylabel('Fidelity')
    fig.colorbar(im2, ax=axs[1])

    plt.suptitle(f"{title_prefix} Results")
    plt.tight_layout()
    plt.show()

# todo: maybe 3d barplot or something

# todo: fits for 1d plots?