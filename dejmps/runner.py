import subprocess
import yaml
import numpy as np


network_yaml_path = "network.yaml"
num_runs_per_sim = 100

# Arrays to sweep through
fidelity_values = [0.555, 0.655, 755]
gate_fidelity_values = [1]

# Helper function to update network.yaml
def update_network_yaml(fidelity, gate_fidelity):
    with open(network_yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Update gate_fidelity for both nodes
    for node in config["nodes"]:
        node["gate_fidelity"] = gate_fidelity

    # Update fidelity of the link
    for link in config["links"]:
        link["fidelity"] = fidelity

    with open(network_yaml_path, "w") as f:
        yaml.dump(config, f)

# Sweep all combinations
for fidelity in fidelity_values:
    for gate_fidelity in gate_fidelity_values:
        print(f"\n>>> Running simulations for fidelity={fidelity}, gate_fidelity={gate_fidelity}")

        update_network_yaml(fidelity, gate_fidelity)

        for i in range(num_runs_per_sim):
            # print(f"  - Run {i+1}/{num_runs_per_sim}")
            subprocess.run(["netqasm", "simulate"], check=True)
