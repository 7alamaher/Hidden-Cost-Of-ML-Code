# Hidden ML Cost – Milestone 1  
## Configuration and Design Smells in Deep Learning Systems

This milestone investigates how small configuration and coding decisions in machine learning systems can increase computational cost and environmental impact **without changing model accuracy**.

The goal is to detect hidden inefficiencies that are not visible through accuracy metrics, but become measurable when analyzing:

- Runtime  
- Memory usage  
- Energy consumption (kWh)  
- CO₂ emissions  

---

# Model Used in All Experiments

To ensure fair comparison across smells, all experiments use the same deep learning model.

**Dataset:** CIFAR-10  
**Model:** Custom 4-layer Convolutional Neural Network (CNN)  
**Framework:** TensorFlow 2.20.0  
**Python:** 3.12.3 (virtual environment: `tf-gpu`)  
**Execution:** GPU-enabled (WSL2) unless otherwise specified  

## CNN Architecture (Simplified)

- 4 Convolutional layers (ReLU activation)
- MaxPooling layers
- Flatten layer
- Dense hidden layer
- Output layer (Softmax, 10 classes)

All hyperparameters remain constant across baseline and smell runs.

---

# Smell 1 – Improper Model Reuse

## Definition

Improper Model Reuse occurs when a model is constructed multiple times within the same execution instead of being created once and reused properly.

## Core Mechanism

Repeated model construction introduces unnecessary initialization overhead and additional memory allocation.  
The model still trains correctly, but execution becomes less efficient.

---

# Smell 2 – Minibatch Mismatch

## Definition

Minibatch Mismatch occurs when batch sizes are set too large for the available hardware.

## Core Mechanism

Oversized batches increase memory demand for activations and gradients during forward and backward passes.  
Accuracy may remain stable, but memory pressure and computational cost increase.

---

# Smell 3 – Library Path Mismatch

## Definition

Library Path Mismatch occurs when CUDA-related environment variables (such as `LD_LIBRARY_PATH`) are modified in a way that changes how GPU libraries are resolved.

## Core Mechanism

Improper library search configuration increases initialization overhead and runtime inefficiencies without affecting model behavior.

---

# Smell 4 – Unnecessary Parallelism

## Definition

Unnecessary Parallelism occurs when CPU thread counts are manually increased beyond optimal levels in a GPU-based training pipeline.

## Core Mechanism

Excessive threading leads to CPU contention and scheduling overhead, increasing resource usage without improving model accuracy.

---

# Experimental Methodology

For each smell:

1. Execute 10 clean baseline runs  
2. Inject the smell  
3. Execute 10 smell runs  
4. Launch each run as a fresh Python process  
5. Record runtime, memory usage, energy (kWh), and CO₂ emissions  

The following remain constant across experiments:

- Model architecture  
- Dataset  
- Hyperparameters  
- Random seed  
- Hardware configuration  

This ensures that any observed differences are caused only by the injected smell.
