import subprocess
import os
import re
import numpy as np
import argparse
import json
from tqdm import tqdm
import time

def run_single_simulation(run_id, temp_dir="temp_runs"):
    """
    Runs a single simulation of the BBPSSW protocol.
    
    Args:
        run_id: ID for this simulation run
        temp_dir: Directory to store temporary run-specific files
        
    Returns:
        success: Boolean indicating if the protocol succeeded
        fidelity: The fidelity of the resulting state (if successful)
    """
    run_dir = os.path.join(temp_dir, f"run_{run_id}") # create a directory for this run if it doesn't exist
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        # execute the netqasm simulate command
        result = subprocess.run(
            ["netqasm", "simulate"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # extract success and fidelity from output
        output = result.stdout
        
        alice_success = "Alice succeeded :-)" in output
        bob_success = "Bob succeeded :-)" in output
        
        # extract the fidelity if succesful
        fidelity = 0.0
        if bob_success:
            print(f"Debug - Full output: {output}")
            fid_match = re.search(r"The fidelity of the final state is: ([0-9.e\-+]+)", output)
            if fid_match:
                fidelity = float(fid_match.group(1))
                print(f"Debug - Extracted fidelity: {fidelity}")
            else:
                print("Debug - No fidelity match found")
        
        return alice_success and bob_success, fidelity
    
    except subprocess.CalledProcessError as e:
        print(f"Error in run {run_id}: {e}")
        print(f"Stderr: {e.stderr}")
        return False, 0.0

def main():
    parser = argparse.ArgumentParser(description='Run multiple simulations of the BBPSSW protocol')
    parser.add_argument('--runs', type=int, default=100, help='Number of protocol runs')
    parser.add_argument('--output', type=str, default='bbpssw_results.json', help='Output file for results')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    
    num_runs = args.runs
    output_file = args.output
    verbose = args.verbose
    
    print(f"Running BBPSSW protocol simulation for {num_runs} iterations...")
    
    temp_dir = "temp_runs"
    os.makedirs(temp_dir, exist_ok=True)
    
    results = {
        "total_runs": num_runs,
        "successful_runs": 0,
        "fidelities": [],
        "success_rate": 0,
        "average_fidelity": 0,
        "std_deviation": 0,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # ========= Run the simulations ==========

    try:
        for i in tqdm(range(num_runs)):
            success, fid = run_single_simulation(i, temp_dir)
            
            if success:
                results["successful_runs"] += 1
                results["fidelities"].append(fid)
                
                if verbose:
                    print(f"Run {i+1}/{num_runs}: Success! Fidelity = {fid:.6f}")
            elif verbose:
                print(f"Run {i+1}/{num_runs}: Failed")
    except KeyboardInterrupt:
        print("\nSimulation interrupted. Calculating statistics for completed runs...")
        results["total_runs"] = i
    
    # calculate statistics
    if results["successful_runs"] > 0:
        results["success_rate"] = results["successful_runs"] / results["total_runs"]
        results["average_fidelity"] = np.mean(results["fidelities"])
        
        if len(results["fidelities"]) > 1:
            results["std_deviation"] = np.std(results["fidelities"])
        
        results["min_fidelity"] = min(results["fidelities"])
        results["max_fidelity"] = max(results["fidelities"])
    
    print("\n====== Results Summary ======")
    print(f"Total runs: {results['total_runs']}")
    print(f"Successful runs: {results['successful_runs']}")
    print(f"Success rate: {results['success_rate']:.2%}")
    
    if results["successful_runs"] > 0:
        print(f"Average fidelity: {results['average_fidelity']:.6f}")
        print(f"Standard deviation: {results['std_deviation']:.6f}")
        print(f"Min fidelity: {results['min_fidelity']:.6f}")
        print(f"Max fidelity: {results['max_fidelity']:.6f}")
    
    with open(output_file, 'w') as f:
        # convert numpy types to Python native types for JSON serialization
        serializable_results = {
            k: v if not isinstance(v, np.ndarray) else v.tolist() 
            for k, v in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # ======== Plot histogram of fidelities =========

    if results["successful_runs"] > 0:
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.hist(results["fidelities"], bins=min(20, len(results["fidelities"])), 
                     alpha=0.7, color='blue', edgecolor='black')
            
            plt.axvline(x=results["average_fidelity"], color='red', linestyle='--', 
                       label=f'Mean: {results["average_fidelity"]:.4f}')
            
            plt.title('Distribution of Fidelities in successful BBPSSW protocol runs')
            plt.xlabel('Fidelity')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_file = os.path.splitext(output_file)[0] + '_histogram.png'
            plt.savefig(plot_file)
            print(f"Fidelity histogram saved to {plot_file}")
            
        except ImportError:
            print("Matplotlib not available. Skipping histogram generation.")
    
    # clean temporary files
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()