🚀 Monte Carlo Pi Estimation: GPU, CPU, & TPUAn ultra-high-performance suite of scripts designed to estimate the digits of $\pi$ using the Monte Carlo method. This repository demonstrates how to saturate different hardware architectures—from multi-core CPUs to massive NVIDIA GPUs and Google TPUs.🌟 Features⚡ GPU Accelerated: Leverages Numba CUDA to parallelize millions of concurrent threads.💻 CPU Multi-Threading: Uses Python’s multiprocessing and njit to max out every available CPU core.🧠 TPU Power: Utilizes TensorFlow TPUStrategy for massive parallelization on Google Cloud/Colab hardware.🔢 High Precision: Results are written to local .txt files without scientific notation or truncation.📊 Performance Metrics: Real-time tracking of execution time, samples per second, and accuracy.🛠️ Requirements📦 GeneralPython 3.7+NumPyMatplotlib (optional for plotting)🚀 Hardware SpecificsBackendRequirementGPUNVIDIA GPU + CUDA Toolkit + numbaCPUNo special requirements (Standard Library)TPUtensorflow + Google Colab or TPU VM📥 InstallationBash# Clone the repository
git clone https://github.com/yourusername/montecarlo-pi.git
cd montecarlo-pi

# Install dependencies
pip install numpy numba tensorflow
🚀 Usage1. GPU Version (CUDA)Best for systems with high VRAM (e.g., 15GB+).Bashpython pi_gpu.py
Config: Adjust threads_per_block and blocks_per_grid in the script to match your SM count.2. CPU VersionBest for local workstations or servers.Bashpython pi_cpu.py
Config: Toggle use_parallel=True to engage all processor cores.3. TPU VersionBest for Google Colab environments.Bashpython pi_tpu.py
Config: Ensure the "TPU" runtime is selected in Colab settings.📈 Performance ComparisonDeviceTypical SamplesEstimated DigitsRuntime (approx)CPU (1 core)$10^7$3–410–30sCPU (8 cores)$10^8$41–3 minGPU (Mid-range)$10^9$4.510–20sTPU v2-8$10^{10}$51–2 min📄 Output FormatAll versions generate a .txt file (e.g., pi_gpu_results.txt) containing the following breakdown:Plaintext--- Monte Carlo Pi Simulation Results ---
Total Samples Generated: 419,430,400,000
-----------------------------------------
Estimated Pi           : 3.1415926535
True Pi                : 3.1415926535
Margin of Error        : 0.0000000000
Compute Time           : 12.4502 seconds
Points computed/sec    : 33,688,642,110
🛠️ TroubleshootingGPU not detected? Run numba -s to verify your CUDA environment is correctly mapped.TPU Error? Ensure you are connected to a TPU backend in Colab via Runtime -> Change runtime type.Memory Crash? If you have 12GB RAM, avoid setting total_samples too high in the CPU version if using an array-based approach.
