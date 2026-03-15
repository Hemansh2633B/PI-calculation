import numpy as np
import math
import time
import os
import tensorflow as tf

# Try to import TPU specific libraries
try:
    import tensorflow as tf
    from tensorflow.python.profiler import profiler_v2 as profiler
    print("TensorFlow version:", tf.__version__)
except ImportError:
    print("TensorFlow not found. Please install with: pip install tensorflow")
    raise

def initialize_tpu():
    """
    Initialize and connect to TPU
    """
    try:
        # Detect TPU
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("TPU detected:", resolver.cluster_spec().as_dict())
        
        # Connect to TPU
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("TPU initialized successfully!")
        
        # Create TPU strategy
        strategy = tf.distribute.TPUStrategy(resolver)
        print("Number of TPU cores:", strategy.num_replicas_in_sync)
        
        return strategy
    except Exception as e:
        print(f"Error initializing TPU: {e}")
        print("Falling back to CPU/GPU...")
        return None

@tf.function
def monte_carlo_pi_tpu_step(iterations):
    """
    TPU function to run one step of Monte Carlo simulation
    """
    # Generate random points
    random_points = tf.random.uniform((iterations, 2), minval=0, maxval=1, dtype=tf.float32)
    
    # Calculate distance from origin
    distances = tf.reduce_sum(tf.square(random_points), axis=1)
    
    # Count points inside circle
    inside_circle = tf.reduce_sum(tf.cast(distances <= 1.0, tf.int64))
    
    return inside_circle

def monte_carlo_pi_tpu(total_samples, strategy=None):
    """
    TPU implementation of Monte Carlo Pi estimation
    """
    if strategy is None:
        # Fallback to CPU/GPU
        print("No TPU strategy provided, using default device")
        return monte_carlo_pi_cpu_fallback(total_samples)
    
    print(f"\nRunning on TPU with {strategy.num_replicas_in_sync} cores")
    print(f"Total samples: {total_samples:,}")
    
    # Calculate samples per replica
    samples_per_replica = total_samples // strategy.num_replicas_in_sync
    remainder = total_samples % strategy.num_replicas_in_sync
    
    print(f"Samples per TPU core: {samples_per_replica:,}")
    if remainder > 0:
        print(f"Remainder samples: {remainder:,} (will be handled by CPU)")
    
    start_time = time.time()
    
    # Define TPU computation
    @tf.function
    def distributed_computation():
        def per_replica_step():
            return monte_carlo_pi_tpu_step(samples_per_replica)
        
        # Run on all TPU cores
        per_replica_results = strategy.run(per_replica_step)
        
        # Combine results from all cores
        total_inside = strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_results, axis=None
        )
        
        return total_inside
    
    # Run TPU computation
    total_inside_tpu = distributed_computation().numpy()
    
    # Handle remainder on CPU if needed
    if remainder > 0:
        print("Processing remainder samples on CPU...")
        random_points = tf.random.uniform((remainder, 2), minval=0, maxval=1, dtype=tf.float32)
        distances = tf.reduce_sum(tf.square(random_points), axis=1)
        inside_remainder = tf.reduce_sum(tf.cast(distances <= 1.0, tf.int32)).numpy()
        total_inside = total_inside_tpu + inside_remainder
    else:
        total_inside = total_inside_tpu
    
    end_time = time.time()
    
    # Calculate Pi
    pi_estimate = 4.0 * total_inside / total_samples
    
    return pi_estimate, total_inside, total_samples, end_time - start_time

def monte_carlo_pi_cpu_fallback(total_samples):
    """
    Fallback CPU implementation if TPU is not available
    """
    print("Using CPU fallback implementation")
    print(f"Total samples: {total_samples:,}")
    
    start_time = time.time()
    
    # Generate random points in batches for memory efficiency
    batch_size = 1000000000  # 1M samples per batch
    num_batches = total_samples // batch_size
    remainder = total_samples % batch_size
    
    total_inside = 0
    
    for batch in range(num_batches):
        if batch % 10 == 0:
            print(f"Processing batch {batch+1}/{num_batches}")
        
        random_points = tf.random.uniform((batch_size, 2), minval=0, maxval=1, dtype=tf.float32)
        distances = tf.reduce_sum(tf.square(random_points), axis=1)
        inside_batch = tf.reduce_sum(tf.cast(distances <= 1.0, tf.int32)).numpy()
        total_inside += inside_batch
    
    # Handle remainder
    if remainder > 0:
        random_points = tf.random.uniform((remainder, 2), minval=0, maxval=1, dtype=tf.float32)
        distances = tf.reduce_sum(tf.square(random_points), axis=1)
        inside_remainder = tf.reduce_sum(tf.cast(distances <= 1.0, tf.int32)).numpy()
        total_inside += inside_remainder
    
    end_time = time.time()
    
    pi_estimate = 4.0 * total_inside / total_samples
    
    return pi_estimate, total_inside, total_samples, end_time - start_time

def print_all_digits(number, file, label=""):
    """
    Print ALL digits of a number without scientific notation
    """
    # Convert to string with maximum precision
    num_str = f"{number:.50f}"  # Start with 50 decimal places
    
    # Remove trailing zeros
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
    print("=" * 60)
    print("PI ESTIMATION USING MONTE CARLO METHOD ON TPU")
    print("=" * 60)
    
    # Check if running in Google Colab with TPU
    import sys
    in_colab = 'google.colab' in sys.modules
    
    if in_colab:
        print("Running in Google Colab environment")
        
        # TPU setup for Colab
        import os
        if 'COLAB_TPU_ADDR' in os.environ:
            print("TPU is available in this Colab session")
            
            # Initialize TPU
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print(f"TPU initialized with {strategy.num_replicas_in_sync} cores")
        else:
            print("TPU not available. Please enable TPU in Colab:")
            print("Runtime -> Change runtime type -> TPU")
            print("Falling back to CPU/GPU...")
            strategy = None
    else:
        # Try to initialize TPU normally
        strategy = initialize_tpu()
    
    # --- Configuration ---
    # Adjust based on TPU memory and desired precision
    # TPU v2-8 can handle billions of samples efficiently
    
    # For quick test (use smaller number)
    # total_samples = 10_000_000  # 10 million
    
    # For production (use larger number for better precision)
    total_samples = 10000000000000  # 1 billion samples
    
    # For extreme precision (uncomment if you have time)
    # total_samples = 10_000_000_000  # 10 billion samples
    
    print(f"\nConfiguration:")
    print(f"- Device: {'TPU' if strategy else 'CPU/GPU (fallback)'}")
    print(f"- Total samples: {total_samples:,}")
    
    # Run the estimation
    pi_estimate, total_inside, total_samples, elapsed_time = monte_carlo_pi_tpu(
        total_samples, strategy
    )
    
    # --- Print ALL digits to file and console ---
    filename = "pi_tpu_all_digits.txt"
    
    with open(filename, "w") as file:
        # Header
        file.write("=" * 60 + "\n")
        file.write("PI ESTIMATION USING MONTE CARLO METHOD ON TPU\n")
        file.write("=" * 60 + "\n\n")
        
        # Configuration
        file.write("CONFIGURATION:\n")
        file.write("-" * 30 + "\n")
        file.write(f"Device: {'TPU' if strategy else 'CPU/GPU (fallback)'}\n")
        if strategy:
            file.write(f"TPU cores: {strategy.num_replicas_in_sync}\n")
        file.write(f"Total samples: {total_samples:,}\n")
        file.write(f"Time taken: {elapsed_time:.4f} seconds\n\n")
        
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
        
        # Performance metrics
        file.write("\n" + "=" * 60 + "\n")
        file.write("PERFORMANCE METRICS:\n")
        file.write("=" * 60 + "\n")
        samples_per_second = total_samples / elapsed_time
        file.write(f"Samples per second: {samples_per_second:,.0f}\n")
        file.write(f"Time per million samples: {elapsed_time*1e6/total_samples:.4f} seconds\n")
        
        if strategy:
            file.write(f"Samples per second per TPU core: {samples_per_second/strategy.num_replicas_in_sync:,.0f}\n")
    
    print("\n" + "=" * 60)
    print(f"✅ ALL Pi digits have been written to '{filename}'")
    print("=" * 60)
    
    # Show file size
    file_size = os.path.getsize(filename)
    print(f"File size: {file_size:,} bytes")
    
    # Display the content
    print("\n" + "=" * 60)
    print("FIRST FEW LINES OF THE FILE:")
    print("=" * 60)
    with open(filename, "r") as file:
        lines = file.readlines()[:20]
        for line in lines:
            print(line.rstrip())
    
    print("\n" + "=" * 60)
    print("LAST FEW LINES OF THE FILE:")
    print("=" * 60)
    with open(filename, "r") as file:
        lines = file.readlines()[-10:]
        for line in lines:
            print(line.rstrip())
    
    # Performance summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY:")
    print("=" * 60)
    print(f"Total samples: {total_samples:,}")
    print(f"Time taken: {elapsed_time:.4f} seconds")
    print(f"Samples per second: {total_samples/elapsed_time:,.0f}")
    
    # Optional: Download file in Colab
    try:
        from google.colab import files
        print("\n" + "=" * 60)
        print("DOWNLOAD OPTION:")
        print("=" * 60)
        files.download(filename)
    except:
        pass  # Not in Colab environment

if __name__ == "__main__":
    main()