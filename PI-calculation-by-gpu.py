import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
import math
import time

# CUDA kernel to calculate PI
@cuda.jit
def monte_carlo_pi_kernel(rng_states, iterations, out):
    """
    GPU kernel to run Monte Carlo simulation.
    """
    # Get absolute thread ID
    thread_id = cuda.grid(1)

    # Check if thread is within bounds
    if thread_id < out.shape[0]:
        # Using integer for exact counting (avoids float precision loss at high iterations)
        inside_circle = 0

        for i in range(iterations):
            x = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)
            y = cuda.random.xoroshiro128p_uniform_float32(rng_states, thread_id)

            # Check if point is inside the unit circle
            if x**2 + y**2 <= 1.0:
                inside_circle += 1

        # Store results of this thread
        out[thread_id] = inside_circle

def print_all_digits(number, file, label=""):
    """
    Print ALL digits of a number without scientific notation
    """
    # Convert to string with maximum precision
    num_str = f"{number:.50f}"  # Start with 50 decimal places

    # Remove trailing zeros that might be from floating point representation
    if '.' in num_str:
        num_str = num_str.rstrip('0').rstrip('.') if '.' in num_str else num_str

    # Write to file
    if label:
        file.write(f"{label}: {num_str}\n")
    else:
        file.write(f"{num_str}\n")

    # Also print to console
    if label:
        print(f"{label}: {num_str}")
    else:
        print(num_str)

def main():
    # Check if CUDA is available
    try:
        cuda.select_device(0)
        # Removed cuda.close() here as it destroys the context needed later
    except:
        print("No CUDA device found. Make sure you're using a GPU runtime in Colab.")
        print("To enable GPU: Runtime -> Change runtime type -> Hardware accelerator -> GPU")
        return

    # --- Configuration ---
    # Using moderate parameters for Colab
    threads_per_block = 1024
    blocks_per_grid = 4096
    total_threads = threads_per_block * blocks_per_grid

    # Higher iterations for better precision
    iterations_per_thread = 50000

    total_samples = total_threads * iterations_per_thread
    print(f"Running {total_threads} threads, {iterations_per_thread} iterations each.")
    print(f"Total samples: {total_samples:,}")

    start_time = time.time()

    # --- Memory Management Optimization ---
    d_results = cuda.device_array(total_threads, dtype=np.uint32)

    # Initialize random number generator states directly on device
    rng_states = create_xoroshiro128p_states(total_threads, seed=42)

    # --- Kernel Execution ---
    monte_carlo_pi_kernel[blocks_per_grid, threads_per_block](
        rng_states, iterations_per_thread, d_results
    )

    # Wait for GPU to finish
    cuda.synchronize()

    # --- Data Retrieval ---
    h_results = d_results.copy_to_host()

    # --- CPU Reduction ---
    total_inside = np.sum(h_results, dtype=np.uint64)
    pi_estimate = 4.0 * float(total_inside) / float(total_samples)

    end_time = time.time()

    # --- Print ALL digits to file and console ---
    filename = "pi_all_digits.txt"

    with open(filename, "w") as file:
        # Header
        file.write("=" * 60 + "\n")
        file.write("PI ESTIMATION USING MONTE CARLO METHOD ON GPU\n")
        file.write("=" * 60 + "\n\n")

        # Configuration
        file.write("CONFIGURATION:\n")
        file.write("-" * 30 + "\n")
        file.write(f"Threads per block: {threads_per_block}\n")
        file.write(f"Blocks per grid: {blocks_per_grid}\n")
        file.write(f"Total threads: {total_threads:,}\n")
        file.write(f"Iterations per thread: {iterations_per_thread:,}\n")
        file.write(f"Total samples: {total_samples:,}\n")
        file.write(f"Time taken: {end_time - start_time:.4f} seconds\n\n")

        # Results
        file.write("RESULTS:\n")
        file.write("-" * 30 + "\n")

        # Print all digits of Pi estimates
        print("\n" + "=" * 60)
        print("ALL DIGITS OF PI ESTIMATES")
        print("=" * 60)

        # Write estimated Pi with ALL digits
        file.write("\nESTIMATED PI (ALL DIGITS):\n")
        print_all_digits(pi_estimate, file, "Estimated Pi")

        # Write true Pi with ALL digits
        file.write("\nTRUE PI (ALL DIGITS):\n")
        print_all_digits(math.pi, file, "True Pi")

        # Write error
        error = abs(math.pi - pi_estimate)
        file.write("\nERROR:\n")
        print_all_digits(error, file, "Error")

        # Additional information
        file.write("\n" + "=" * 60 + "\n")
        file.write("DIGIT COMPARISON:\n")
        file.write("=" * 60 + "\n\n")

        # Convert to strings for comparison
        est_str = f"{pi_estimate:.50f}"
        true_str = f"{math.pi:.50f}"

        # Find where digits start to differ
        file.write("First 50 digits comparison:\n")
        file.write(f"Est: {est_str}\n")
        file.write(f"True: {true_str}\n\n")

        # Count matching digits
        match_count = 0
        for i in range(min(len(est_str), len(true_str))):
            if est_str[i] == true_str[i]:
                match_count += 1
            else:
                break

        file.write(f"First {match_count-2} digits after decimal point are correct\n")
        file.write(f"(excluding '3.' which matches perfectly)\n")

        # Write the full Pi estimate in a clean format
        file.write("\n" + "=" * 60 + "\n")
        file.write("FULL PI ESTIMATE:\n")
        file.write("=" * 60 + "\n")

        # Format Pi in groups of 10 digits for readability
        pi_full = f"{pi_estimate:.50f}"
        file.write("3.")

        # Group digits after decimal
        digits_after_decimal = pi_full.split('.')[1]
        for i in range(0, len(digits_after_decimal), 10):
            group = digits_after_decimal[i:i+10]
            file.write(f"{group} ")
            if (i // 10 + 1) % 5 == 0:  # New line every 5 groups (50 digits)
                file.write("\n ")

        file.write("\n\n")

        # Statistics
        file.write("STATISTICS:\n")
        file.write("-" * 30 + "\n")
        file.write(f"Inside circle count: {total_inside:,}\n")
        file.write(f"Total points: {total_samples:,}\n")
        file.write(f"Ratio (inside/total): {total_inside/total_samples:.10f}\n")
        file.write(f"Pi = 4 * ratio = {pi_estimate}\n")

    print("\n" + "=" * 60)
    print(f"✅ ALL Pi digits have been written to '{filename}'")
    print("=" * 60)

    # Show file size
    import os
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size:,} bytes")

    # Display the content in Colab
    print("\n" + "=" * 60)
    print("FIRST FEW LINES OF THE FILE:")
    print("=" * 60)
    with open(filename, "r") as file:
        lines = file.readlines()[:20]  # Show first 20 lines
        for line in lines:
            print(line.rstrip())

    print("\n" + "=" )
    print("LAST FEW LINES OF THE FILE:")
    print("=" )
    with open(filename, "r") as file:
        lines = file.readlines()[-10:]  # Show last 10 lines
        for line in lines:
            print(line.rstrip())

    # Optional: Download file in Colab
    try:
        from google.colab import files
        print("\n" + "=" )
        print("DOWNLOAD OPTION:")
        print("=" )
        files.download(filename)
    except:
        pass  # Not in Colab environment

if __name__ == "__main__":
    main()