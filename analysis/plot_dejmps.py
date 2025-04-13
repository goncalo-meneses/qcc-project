import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate theoretical fidelity for DEJMPS
def th_psucc(f):
    return f**2 + 2 * f * (1 - f) / 3 + 5 * ((1 - f) / 3)**2

def th_fidelity(f):
    return (f**2 + ((1-f)/3)**2) / th_psucc(f)

# Bell state for fidelity calculation
def bell_state():
    ket_0 = np.array([[1.+0.j], [0.j]])
    ket_1 = np.array([[0.j], [1.+0.j]])
    
    ket_00 = np.kron(ket_0, ket_0)
    ket_11 = np.kron(ket_1, ket_1)

    return (ket_00 + ket_11) / np.sqrt(2)

def fidelity(state, dm):
    state = state.reshape(-1, 1)
    bra = state.conj().T
    result = np.dot(bra, np.dot(dm, state))
    return np.real_if_close(result.item())

# Find all DEJMPS data files with gate_fidelity = 1 (both g=1.0 and g=1)
data_files = glob.glob('dejmps/data/f=*_g=1_dejmps.npz')
data_files.extend(glob.glob('dejmps/data/f=*_g=1.0_dejmps.npz'))

print(f"Found {len(data_files)} files with gate_fidelity = 1")

# Data collection for plotting
input_fidelities = []
output_fidelities = []
success_rates = []

# Process each data file
for file_path in data_files:
    try:
        data = np.load(file_path, allow_pickle=False)
        
        # Extract data
        input_fid = data['fidelity'].item()
        matrices = data['matrices']
        successes = data['successes']
        
        # Filter for successful runs
        if np.any(successes):
            successful_matrices = matrices[successes]
            
            if len(successful_matrices) > 0:
                # Average density matrix
                avg_density = np.mean(successful_matrices, axis=0)
                
                # Calculate output fidelity with Bell state
                phi_plus = bell_state()
                output_fid = fidelity(phi_plus, avg_density)
                
                # Success rate
                success_rate = np.mean(successes)
                
                # Store data
                input_fidelities.append(input_fid)
                output_fidelities.append(output_fid)
                success_rates.append(success_rate)
                
                print(f"File: {os.path.basename(file_path)}")
                print(f"  Input fidelity: {input_fid:.4f}")
                print(f"  Output fidelity: {output_fid:.4f}")
                print(f"  Success rate: {success_rate:.4f}")
            else:
                print(f"No successful runs in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Sort data points by input fidelity
if input_fidelities:
    sorted_indices = np.argsort(input_fidelities)
    input_fidelities = np.array(input_fidelities)[sorted_indices]
    output_fidelities = np.array(output_fidelities)[sorted_indices]
    success_rates = np.array(success_rates)[sorted_indices]

# Generate theoretical curve
x_theory = np.linspace(0.5, 1.0, 100)
y_theory = [th_fidelity(f) for f in x_theory]

# Create the plot
plt.figure(figsize=(10, 6))

# Plot theoretical curve
plt.plot(x_theory, y_theory, 'r-', linewidth=2, label='Theoretical DEJMPS')

# Plot simulation data points
if len(input_fidelities) > 0:
    plt.scatter(input_fidelities, output_fidelities, c='b', marker='o', s=80, label='Simulation (g=1)')

# No improvement line (y=x)
plt.plot([0.5, 1], [0.5, 1], 'k--', alpha=0.5, label='No improvement')

# Labels and title
plt.xlabel('Input Fidelity', fontsize=14)
plt.ylabel('Output Fidelity', fontsize=14)
plt.title('DEJMPS Protocol: Output vs. Input Fidelity (gate_fidelity = 1)', fontsize=16)

# Limits and grid
plt.xlim(0.5, 1.0)
plt.ylim(0.5, 1.0)
plt.grid(True, alpha=0.3)

# Legend
plt.legend(fontsize=12)

# Save the figure
output_dir = 'analysis'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'dejmps_fidelity_plot.png'), dpi=300)

# Show plot
plt.show()

print(f"Plot saved to {os.path.join(output_dir, 'dejmps_fidelity_plot.png')}")