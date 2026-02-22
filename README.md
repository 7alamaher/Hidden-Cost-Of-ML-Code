# Hidden ML Cost – Milestone 1 

This milestone investigates hidden inefficiencies in machine learning systems that increase computational cost and environmental impact without affecting model accuracy.

We analyze four common configuration and programming smells in deep learning workflows and measure their impact on runtime, memory usage, energy consumption, and CO₂ emissions.

All experiments use controlled baseline comparisons, fresh process execution, and consistent hardware configurations.

# Smell 1: Improper Model Reuse
  # Definition
Improper Model Reuse occurs when models are repeatedly constructed within the same execution instead of being built once and reused properly. This can lead to unnecessary object allocation and increased memory usage while the model still trains and evaluates correctly.

  # Core Mechanism
Repeated model construction introduces additional initialization overhead and memory allocation, increasing runtime without improving performance.

# Smell 2: Minibatch Mismatch
   # Definition
Minibatch Mismatch occurs when an excessively large batch size is used during training. This increases memory demand for activations, gradients, and intermediate tensors during forward and backward passes.
Although model accuracy may remain stable, memory pressure and computational overhead increase.

  # Core Mechanism
Over-allocation of memory due to inappropriate batch size selection leads to inefficient resource usage.

# Smell 3: Library Path Mismatch
  # Definition
Library Path Mismatch occurs when CUDA-related environment variables (such as LD_LIBRARY_PATH) are misconfigured, causing inefficient loading and initialization of GPU libraries.
The model continues to train correctly, but runtime overhead and energy consumption increase due to improper library resolution.

  # Core Mechanism
Incorrect runtime library search configuration introduces initialization inefficiencies and increased system overhead.

# Smell 4: Unnecessary Parallelism
  # Definition
Unnecessary Parallelism occurs when CPU thread counts are manually increased beyond optimal levels in a GPU-based training pipeline. This creates thread oversubscription, context switching, and scheduling overhead without improving model accuracy.

# Core Mechanism
Excessive parallel execution leads to CPU contention and inefficient scheduling, increasing resource usage without performance gains.


# Experimental Methodology
For each smell:
A clean baseline is executed (10 runs)
The smell is injected
10 additional runs are executed
Each run is launched as a fresh Python process
Runtime, memory, energy, and CO₂ emissions are recorded

The model architecture, dataset, hyperparameters, and random seeds remain unchanged across baseline and smell experiments.

This ensures that any differences observed are due solely to the injected smell.
