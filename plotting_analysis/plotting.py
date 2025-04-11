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


def plot_1d_analysis(results, sweep_param='fidelity', title=None):
    assert sweep_param in ('fidelity', 'gate_fidelity'), "sweep_param must be 'fidelity' or 'gate_fidelity'"
    
    # todo: crash if given results aren't 1d
    # also automatically choose sweep parameter -- shouldnt have to specify it manually since only one should be more than length 1

    grouped = {}
    for (f, g), stats in results.items():
        key = f if sweep_param == 'fidelity' else g
        grouped[key] = stats

    xs = sorted(grouped.keys())
    avg_fids = [grouped[x]['avg_fidelity'] for x in xs]
    avg_fid_errs = [grouped[x]['avg_fidelity_err'] for x in xs]
    succ_probs = [grouped[x]['success_probability'] for x in xs]
    succ_prob_errs = [grouped[x]['success_probability_err'] for x in xs]

    fig, ax1 = plt.subplots()

    ax1.errorbar(xs, avg_fids, yerr=avg_fid_errs, label='Average Fidelity', fmt='-o', color='tab:blue')
    ax1.set_ylabel('Average Fidelity', color='tab:blue')
    ax1.set_xlabel(sweep_param.replace('_', ' ').title())
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.errorbar(xs, succ_probs, yerr=succ_prob_errs, label='Success Probability', fmt='-s', color='tab:green')
    ax2.set_ylabel('Success Probability', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')

    fig.tight_layout()
    if title is None:
        title = f'Sweep of {sweep_param.replace("_", " ").title()}'
    plt.title(title)
    plt.grid(True)
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