# 🚀 Monte Carlo Pi Estimation: GPU, CPU, & TPU

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy&logoColor=white)
![Numba](https://img.shields.io/badge/Numba-0.55+-00A3E0?logo=numba&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-11.0+-76B900?logo=nvidia&logoColor=white)

An ultra-high-performance suite of scripts designed to estimate the digits of π using the Monte Carlo method. This repository demonstrates how to saturate different hardware architectures—from multi-core CPUs to massive NVIDIA GPUs and Google TPUs.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Performance Comparison](#performance-comparison)
- [Output Format](#output-format)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Features

- ⚡ **GPU Accelerated**: Leverages Numba CUDA to parallelize millions of concurrent threads.
- 💻 **CPU Multi-Threading**: Uses Python's multiprocessing and njit to max out every available CPU core.
- 🧠 **TPU Power**: Utilizes TensorFlow TPUStrategy for massive parallelization on Google Cloud/Colab hardware.
- 🔢 **High Precision**: Results are written to local .txt files without scientific notation or truncation.
- 📊 **Performance Metrics**: Real-time tracking of execution time, samples per second, and accuracy.

## 🛠️ Requirements

### 📦 General
- ![Python](https://img.shields.io/badge/Python-3.7+-3776AB?logo=python&logoColor=white) Python 3.7+
- ![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?logo=numpy&logoColor=white) NumPy
- Matplotlib (optional for plotting)

### 🚀 Hardware Specifics

| Backend | Requirement |
|---------|-------------|
| GPU | NVIDIA GPU + CUDA Toolkit + ![Numba](https://img.shields.io/badge/Numba-0.55+-00A3E0?logo=numba&logoColor=white) Numba |
| CPU | No special requirements (Standard Library) |
| TPU | ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow&logoColor=white) TensorFlow + Google Colab or TPU VM |

## 📥 Installation

```bash
# Clone the repository
git clone https://github.com/Hemansh2633B/PI-calculation.git
cd PI-calculation

# Install dependencies
pip install numpy numba tensorflow
```

## 🚀 Usage

### 1. GPU Version (CUDA)
Best for systems with high VRAM (e.g., 15GB+).

```bash
python PI-calculation-by-gpu.py
```

**Config**: Adjust `threads_per_block` and `blocks_per_grid` in the script to match your SM count.

### 2. CPU Version
Best for local workstations or servers.

```bash
python PI-calculation-by-cpu.py
```

**Config**: Toggle `use_parallel=True` to engage all processor cores.

### 3. TPU Version
Best for Google Colab environments.

```bash
python PI-calculation-by-tpu.py
```

**Config**: Ensure the "TPU" runtime is selected in Colab settings.

## 📈 Performance Comparison

| Device       | Typical Samples | Estimated Digits | Runtime (approx) |
|--------------|-----------------|------------------|------------------|
| CPU (1 core) | $10^7$         | 3–4             | 10–30s          |
| CPU (8 cores)| $10^8$         | 4               | 1–3 min         |
| GPU (Mid-range) | $10^9$      | 4.5             | 10–20s          |
| TPU v2-8     | $10^{10}$      | 5               | 1–2 min         |

## 📄 Output Format

All versions generate a .txt file (e.g., `Results-gpu.txt`) containing the following breakdown:

```
--- Monte Carlo Pi Simulation Results ---
Total Samples Generated: 419,430,400,000
-----------------------------------------
Estimated Pi           : 3.1415926535
True Pi                : 3.1415926535
Margin of Error        : 0.0000000000
Compute Time           : 12.4502 seconds
Points computed/sec    : 33,688,642,110
```

## 🛠️ Troubleshooting

- **GPU not detected?** Run `numba -s` to verify your CUDA environment is correctly mapped.
- **TPU Error?** Ensure you are connected to a TPU backend in Colab via Runtime -> Change runtime type.
- **Memory Crash?** If you have 12GB RAM, avoid setting `total_samples` too high in the CPU version if using an array-based approach.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

**Author**: [Hemansh2633B](https://github.com/Hemansh2633B)
