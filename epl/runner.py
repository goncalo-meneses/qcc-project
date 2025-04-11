import subprocess
import yaml
import numpy as np


network_yaml_path = "network.yaml"
num_runs_per_sim = 100

# Arrays to sweep through
fidelity_values = [0.7]
gate_fidelity_values = [0.99  , 0.9905, 0.991 , 0.9915, 0.992 , 0.9925, 0.993 , 0.9935,
       0.994 , 0.9945, 0.995 , 0.9955, 0.996 , 0.9965, 0.997 , 0.9975,
       0.998 , 0.9985, 0.999 , 0.9995, 1.    ]

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
            print(f"  - Run {i+1}/{num_runs_per_sim}")
            subprocess.run(["netqasm", "simulate"], check=True)
