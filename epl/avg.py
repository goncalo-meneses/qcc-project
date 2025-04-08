import numpy as np

FILENAME = "all_density_matrices.npz"

# Load cumulative matrices
data = np.load(FILENAME)
matrices = data["matrices"]  # shape: (N, 4, 4)

# Compute average
avg_matrix = np.mean(matrices, axis=0)

# Print nicely
# np.set_printoptions(precision=4, suppress=True)
print(f"Loaded {matrices.shape[0]} density matrices.")
print("Average density matrix:\n", avg_matrix)
