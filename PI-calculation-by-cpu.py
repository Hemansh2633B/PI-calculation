import numpy as np
import math
import time
import os
from numba import njit, prange

# The @njit(parallel=True) decorator tells Numba to compile this to machine code
# and automatically distribute 'prange' loops across all CPU cores.
@njit(parallel=True, nogil=True)
def monte_carlo_pi_cpu(n_samples):
    """
    CPU-based Monte Carlo simulation using all available cores.
    """
    hits = 0
    # prange is a parallel range. Numba will split these iterations
    # across your CPU's threads (e.g., 8, 16, or 32 threads).
    for i in prange(n_samples):
        # We use np.random inside the jitted function for speed
        x = np.random.random()
        y = np.random.random()
        
        if x**2 + y**2 <= 1.0:
            hits += 1
            
    return hits

def main():
    # --- Configuration ---
    # 500 million samples is a good balance for a modern CPU. 
    # It uses very little RAM (~0 bytes beyond the overhead) but high CPU.
    total_samples = 500_000_000 
    
    print(f"Starting CPU simulation with {total_samples:,} samples...")
    print(f"Utilizing all available CPU cores via Numba Parallel...")

    # Warm-up (Numba needs a second to compile the code on the first run)
    monte_carlo_pi_cpu(1000)
    
    start_time = time.time()
    
    # Run the actual simulation
    total_inside = monte_carlo_pi_cpu(total_samples)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Calculate Pi
    pi_estimate = 4.0 * total_inside / total_samples
    
    # --- Prepare Output Data ---
    output_text = (
        f"--- CPU Monte Carlo Pi Simulation Results ---\n"
        f"Total Samples Generated: {total_samples:,}\n"
        f"---------------------------------------------\n"
        f"Estimated Pi           : {pi_estimate:.10f}\n"
        f"True Pi                : {math.pi:.10f}\n"
        f"Margin of Error        : {abs(math.pi - pi_estimate):.10f}\n"
        f"Compute Time           : {execution_time:.4f} seconds\n"
        f"Points computed/sec    : {total_samples / execution_time:,.0f}\n"
    )
    
    # Print to console
    print("\n" + output_text)
    
    # --- Save to Local Storage ---
    file_name = "pi_simulation_cpu_results.txt"
    save_path = os.path.join(os.getcwd(), file_name) 
    
    try:
        with open(save_path, "w") as file:
            file.write(output_text)
        print(f"✅ CPU results successfully saved to: {save_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

if __name__ == "__main__":
    main()