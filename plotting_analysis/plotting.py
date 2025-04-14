import IPython
import numpy as np
import matplotlib.pyplot as plt
from analysis import *
import matplotlib.patches as patches

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


def plot_fidelity(results_list, title=None, delta=False, theoretical_f=False, fr=(0.45, 0.55)):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, avg_fids, avg_fid_errs, _, _, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, avg_fids, yerr=avg_fid_errs,
                     fmt='-o', label=protocol,
                     capsize=3)

    if theoretical_f:
        F = np.linspace(*fr, 500)
        psucc = F**2 + 2 * F * (1 - F)/3 + 5 * ((1 - F)/3)**2
        Fout = (F**2 + ((1 - F)/3)**2) / psucc
        ax1.plot(F, Fout, '-', label=r'$F_{out}$ (BBPSSW)', color='red', linewidth=2)

    if theoretical_f:
        F = np.linspace(*fr, 500)
        psucc = (1 + ((4 * F - 1)/3)**2) / 2
        Fout = 5 / 4 + (4 * F - 7) / (12 * psucc)
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


def plot_success_prob(results_list, title=None, delta=False, theoretical_p=False, fr=(0.45, 0.55)):
    fig, ax1 = plt.subplots()

    for protocol, result in results_list:
        xs, _, _, success_probs, success_prob_errs, sweep_param = extract_1d(result, delta)
        ax1.errorbar(xs, success_probs, yerr=success_prob_errs,
                         fmt='-o', label=protocol,
                         capsize=3)
        
    if theoretical_p:
        F = np.linspace(*fr, 500)
        psucc = F**2 + 2 * F * (1 - F)/3 + 5 * ((1 - F)/3)**2
        ax1.plot(F, psucc, '-', label=r'$p_{succ}$ (BBPSSW)', color='red', linewidth=2)

    if theoretical_p:
        F = np.linspace(*fr, 500)
        psucc = (1 + ((4 * F - 1)/3)**2) / 2
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
def plot_fidelity_heatmap(results_dict,
                          deltas=False,
                          cbar_range=None,
                          title=None,
                          cmap='viridis',
                          bins=20,
                          highlight_positive_deltas=False,
                          scatter_overlay=False):
    """
    Heatmap of output fidelity or Δ fidelity over (input fidelity, gate fidelity).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    f_vals = []
    g_vals = []
    weights = []

    for (f, g), stats in results_dict.items():
        fid = stats.get('avg_fidelity')
        val = fid - f if (deltas and fid is not None) else fid
        if val is None:
            continue
        f_vals.append(f)
        g_vals.append(g)
        weights.append(val)

    f_vals = np.array(f_vals)
    g_vals = np.array(g_vals)
    weights = np.array(weights)

    H_sum, f_edges, g_edges = np.histogram2d(f_vals, g_vals, bins=bins, weights=weights)
    H_count, _, _ = np.histogram2d(f_vals, g_vals, bins=[f_edges, g_edges])

    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = np.divide(H_sum, H_count)
        H_avg = np.where(H_count == 0, np.nan, H_avg)

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

    if highlight_positive_deltas and deltas:
        for i in range(len(f_edges) - 1):
            for j in range(len(g_edges) - 1):
                if not np.isnan(H_avg[i, j]) and H_avg[i, j] > 0:
                    rect = patches.Rectangle(
                        (g_edges[j], f_edges[i]),
                        g_edges[j + 1] - g_edges[j],
                        f_edges[i + 1] - f_edges[i],
                        linewidth=1.5, edgecolor='red', facecolor='none'
                    )
                    ax.add_patch(rect)

    if scatter_overlay:
        ax.scatter(g_vals, f_vals, color='black', s=10, alpha=0.5, label='Sampled Combos')
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


def plot_success_probability_heatmap(results_dict,
                                     cbar_range=None,
                                     title=None,
                                     cmap='viridis',
                                     bins=20,
                                     scatter_overlay=False):
    """
    Heatmap of success probability over (input fidelity, gate fidelity).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    f_vals = []
    g_vals = []
    weights = []

    for (f, g), stats in results_dict.items():
        prob = stats.get('success_probability')
        if prob is None:
            continue
        f_vals.append(f)
        g_vals.append(g)
        weights.append(prob)

    f_vals = np.array(f_vals)
    g_vals = np.array(g_vals)
    weights = np.array(weights)

    H_sum, f_edges, g_edges = np.histogram2d(f_vals, g_vals, bins=bins, weights=weights)
    H_count, _, _ = np.histogram2d(f_vals, g_vals, bins=[f_edges, g_edges])

    with np.errstate(divide='ignore', invalid='ignore'):
        H_avg = np.divide(H_sum, H_count)
        H_avg = np.where(H_count == 0, np.nan, H_avg)

    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(g_edges, f_edges, H_avg, cmap=cmap,
                         vmin=cbar_range[0] if cbar_range else None,
                         vmax=cbar_range[1] if cbar_range else None,
                         shading='auto')

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label('Success Probability')

    ax.set_xlabel('Gate Fidelity')
    ax.set_ylabel('Input Fidelity')
    ax.set_title(title or 'Success Probability Heatmap')

    if scatter_overlay:
        ax.scatter(g_vals, f_vals, color='black', s=10, alpha=0.5, label='Sampled Combos')
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

def plot_fidelity_vs_success(results_list, gate_fidelity=1.0, title=None, delta=False, markers=None, colors=None):
    """
    Plot output fidelity versus success probability for a specific gate fidelity.
    
    Args:
        results_list: List of (protocol_name, results_dict) tuples
        gate_fidelity: Gate fidelity value to filter by (default: 1.0)
        title: Optional plot title
        delta: If True, plot delta fidelity (output - input) instead of absolute fidelity
        markers: Optional list of markers to use for each protocol
        colors: Optional list of colors to use for each protocol
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if markers is None:
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*', 'h']
    
    if colors is None:
        colors = plt.cm.tab10.colors
    
    for i, (protocol, results) in enumerate(results_list):
        # Filter results for the specific gate fidelity
        filtered_results = {(f, g): stats for (f, g), stats in results.items() 
                           if abs(g - gate_fidelity) < 1e-6}
        
        if not filtered_results:
            print(f"No data found for protocol {protocol} with gate fidelity {gate_fidelity}")
            continue
        
        # Extract data points
        fids = []
        success_probs = []
        fid_errs = []
        success_prob_errs = []
        input_fids = []
        
        for (f, g), stats in filtered_results.items():
            if stats['avg_fidelity'] is None or stats['success_probability'] is None:
                continue
            
            input_fids.append(f)
            fids.append(stats['delta_fidelity'] if delta else stats['avg_fidelity'])
            success_probs.append(stats['success_probability'])
            fid_errs.append(stats['avg_fidelity_err'])
            success_prob_errs.append(stats['success_probability_err'])
        
        # Sort by input fidelity to connect dots in order
        if input_fids:
            sort_idx = np.argsort(input_fids)
            input_fids = np.array(input_fids)[sort_idx]
            fids = np.array(fids)[sort_idx]
            success_probs = np.array(success_probs)[sort_idx]
            fid_errs = np.array(fid_errs)[sort_idx]
            success_prob_errs = np.array(success_prob_errs)[sort_idx]
            
            marker = markers[i % len(markers)]
            color = colors[i % len(colors)]
            
            # Plot with error bars
            ax.errorbar(
                success_probs, 
                fids, 
                xerr=success_prob_errs, 
                yerr=fid_errs,
                fmt=f'-{marker}', 
                label=f"{protocol} (Input F: {min(input_fids):.2f}-{max(input_fids):.2f})",
                capsize=3, 
                color=color,
                markersize=8
            )
            
            # Add input fidelity annotations to points
            for j, (sp, fid, in_fid) in enumerate(zip(success_probs, fids, input_fids)):
                if j % 2 == 0:  # Skip some annotations to avoid overcrowding
                    ax.annotate(
                        f"{in_fid:.2f}", 
                        (sp, fid),
                        textcoords="offset points", 
                        xytext=(0, 10),
                        ha='center',
                        fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
                    )
    
    ax.set_xlabel('Success Probability')
    ax.set_ylabel('Δ Fidelity (Output - Input)' if delta else 'Output Fidelity')
    
    if title is None:
        title = f"Output Fidelity vs Success Probability (Gate Fidelity = {gate_fidelity})"
    ax.set_title(title)
    
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    # Add a diagonal y=x line for reference
    if not delta:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        overall_min = min(x_min, y_min)
        overall_max = max(x_max, y_max)
        ax.plot([overall_min, overall_max], [overall_min, overall_max], 'k--', alpha=0.3)
    else:
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

# todo: maybe 3d barplot or something

# todo: fits for 1d plots?
