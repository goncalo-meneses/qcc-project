import IPython
import numpy as np
import matplotlib.pyplot as plt
from analysis import *

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


def plot_fidelity(results_list, title=None, delta=False, theoretical_f=False):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, avg_fids, avg_fid_errs, _, _, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, avg_fids, yerr=avg_fid_errs,
                     fmt='-o', label=protocol,
                     capsize=3)

    if theoretical_f:
        F = np.linspace(0.45, 1, 500)
        psucc = F**2 + 2 * F * (1 - F)/3 + 5 * ((1 - F)/3)**2
        Fout = (F**2 + ((1 - F)/3)**2) / psucc
        ax1.plot(F, Fout, '-', label=r'$F_{out}$ (BBPSSW)', color='red', linewidth=2)

    if theoretical_f:
        F = np.linspace(0.45, 1, 500)
        psucc = (1 + F**2) / 2
        Fout = 5 / 4 + (F - 2) / (4 * psucc)
        ax1.plot(F, Fout, '-', label=r'$F_{out}$ (DEJMPS)', color='black', linewidth=2)

    ax1.set_xlabel(sweep_param.replace('_', ' ').title())
    ax1.set_ylabel('Δ Fidelity' if delta else 'Average Fidelity')
    fig.tight_layout()
    if title is None:
        title = f"Sweep of {sweep_param.replace('_', ' ').title()}"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_success_prob(results_list, title=None, delta=False, theoretical_p=False):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, _, _, success_probs, success_prob_errs, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, success_probs, yerr=success_prob_errs,
                         fmt='-o', label=protocol,
                         capsize=3)
        
    if theoretical_p:
        F = np.linspace(0.45, 1, 500)
        psucc = F**2 + 2 * F * (1 - F)/3 + 5 * ((1 - F)/3)**2
        ax1.plot(F, psucc, '-', label=r'$p_{succ}$ (BBPSSW)', color='red', linewidth=2)

    if theoretical_p:
        F = np.linspace(0.45, 1, 500)
        psucc = (1 + F**2) / 2
        ax1.plot(F, psucc, '-', label=r'$p_{succ}$ (DEJMPS)', color='black', linewidth=2)
    
    ax1.set_xlabel(sweep_param.replace('_', ' ').title())
    ax1.set_ylabel('Success probability')
    fig.tight_layout()
    if title is None:
        title = f"Sweep of {sweep_param.replace('_', ' ').title()}"
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_combo_scatter(results_dict, protocol_data=None, 
                       color_by='avg_fidelity', deltas=False, 
                       cbar_range=None, title=None, cmap='viridis'):
    """
    Scatter plot of (fidelity, gate_fidelity) combinations colored by a quantity.

    Args:
        results_dict: Dict with (fidelity, gate_fidelity) keys and stats as values.
        protocol_data: Required if color_by='num_runs' or deltas=True.
        color_by: 'avg_fidelity' or 'num_runs'. What to color points by.
        deltas: If True, color by output fidelity minus input fidelity.
        cbar_range: Optional (vmin, vmax) for colorbar. Useful for comparison between protocols.
        title: Optional plot title.
        cmap: Colormap.
    """
    assert color_by in ('avg_fidelity', 'num_runs'), "color_by must be 'avg_fidelity' or 'num_runs'"
    if deltas and color_by != 'avg_fidelity':
        raise ValueError("deltas=True is only valid when color_by='avg_fidelity'")
    if color_by == 'num_runs' and protocol_data is None:
        raise ValueError("protocol_data must be provided when deltas=True")

    fids = []
    gates = []
    values = []

    for (f, g), stats in results_dict.items():
        if color_by == 'avg_fidelity':
            fid_out = stats['avg_fidelity']
            if fid_out is None:
                continue
            val = stats['delta_fidelity'] if deltas else fid_out
        elif color_by == 'num_runs':
            val = len(protocol_data[(f, g)]['matrices'])
        fids.append(f)
        gates.append(g)
        values.append(val)

    fig, ax = plt.subplots()
    sc = ax.scatter(gates, fids, c=values, cmap=cmap, s=60, edgecolor='k', vmin=None if cbar_range is None else cbar_range[0], vmax=None if cbar_range is None else cbar_range[1])
    cbar = plt.colorbar(sc, ax=ax)

    if color_by == 'avg_fidelity':
        cbar.set_label('Δ Fidelity (Output - Input)' if deltas else 'Average Output Fidelity')
    else:
        cbar.set_label('Number of Simulations')

    ax.set_xlabel('Gate Fidelity')
    ax.set_ylabel('Input Fidelity')
    ax.set_title(title or f'Simulation Coverage Colored by {"Δ Fidelity" if deltas else color_by.replace("_", " ").title()}')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# not sure yet if this is correct
# also maybe want to add success prob option
def plot_combo_heatmap(results_dict, deltas=False, cbar_range=None, title=None, cmap='viridis', bins=20, highlight_positive_deltas=False):
    """
    Plots a heatmap of average output fidelity or fidelity delta over (fidelity, gate_fidelity) combos.
    Uses histogram2d to handle uneven grid spacing robustly.
    Can optionally highlight cells with positive fidelity delta.
    """
    # Extract data
    f_vals = []
    g_vals = []
    weights = []

    for (f, g), stats in results_dict.items():
        fid = stats.get('avg_fidelity')
        if fid is None:
            continue
        f_vals.append(f)
        g_vals.append(g)
        weights.append(fid - f if deltas else fid)

    f_vals = np.array(f_vals)
    g_vals = np.array(g_vals)
    weights = np.array(weights)

    # Bin the data using histogram2d
    H_sum, f_edges, g_edges = np.histogram2d(
        f_vals, g_vals, bins=bins, weights=weights
    )
    H_count, _, _ = np.histogram2d(
        f_vals, g_vals, bins=[f_edges, g_edges]
    )

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = np.divide(H_sum, H_count)
        H_avg = np.where(H_count == 0, np.nan, H_avg)

    # Create the plot
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(g_edges, f_edges, H_avg, cmap=cmap,
                         vmin=cbar_range[0] if cbar_range else None,
                         vmax=cbar_range[1] if cbar_range else None,
                         shading='auto')

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Δ Fidelity (Output - Input)' if deltas else 'Average Output Fidelity')

    ax.set_xlabel('Gate Fidelity')
    ax.set_ylabel('Input Fidelity')
    ax.set_title(title or ('Δ Fidelity Heatmap' if deltas else 'Output Fidelity Heatmap'))

    # Overlay markers for positive deltas
    if highlight_positive_deltas and deltas:
        f_centers = 0.5 * (f_edges[:-1] + f_edges[1:])
        g_centers = 0.5 * (g_edges[:-1] + g_edges[1:])
        for i in range(len(f_centers)):
            for j in range(len(g_centers)):
                if not np.isnan(H_avg[i, j]) and H_avg[i, j] > 0:
                    ax.plot(g_centers[j], f_centers[i], marker='o', color='red', markersize=3)

    plt.tight_layout()
    plt.show()



# todo: maybe 3d barplot or something

# todo: fits for 1d plots?
