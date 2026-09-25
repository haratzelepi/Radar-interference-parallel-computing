# Radar Interference: Decision Regions and SER

This project studies the computational cost of digital communication simulations in the presence of radar interference and explores how these computations can be accelerated using **OpenMP** and **CUDA**.

## Project goal

The main objective is to **parallelize computationally expensive numerical simulations** and evaluate the performance benefits of different parallel programming approaches.

This project is based on the paper [*Communications System Performance and Design in the Presence of Radar Interference*](https://doi.org/10.1109/TCOMM.2018.2823764) (Nartasilpa et al., 2018), which provides the underlying communication and interference model used in the simulations. An open-access author copy of the PDF can be found [*here*](https://devroye.lab.uic.edu/wp-content/uploads/sites/570/2019/10/Nartasilpa-TCOM-2018.pdf).

The communication receiver uses a Maximum Likelihood (ML) detector under radar interference. Two computationally intensive tasks are considered:

1. **Decision-region computation** on the complex plane.
2. **Monte Carlo Symbol Error Rate (SER) estimation** over a range of Interference-to-Noise Ratio (INR) values.

For both tasks, a serial C implementation is used as a reference and compared against CPU-parallel OpenMP and GPU-parallel CUDA implementations.

Shared constellation and detector code is located in `common/`, while the decision-region and SER implementations are organized in `decision_regions/` and `ser/`, respectively.

## Computational tasks

### Decision regions

A two-dimensional grid of complex received values is generated and every grid point is classified by the ML detector.

The grid resolution can be increased to obtain more detailed decision regions, but this also significantly increases the number of ML evaluations and therefore the computational cost.

### Symbol Error Rate (SER)

SER is estimated through Monte Carlo simulation. For each INR value, a large number of transmitted symbols are generated, radar interference and complex Gaussian noise are added, and the received symbols are passed through the ML detector.

The estimated SER is calculated as:
> `SER = number of incorrectly detected symbols / total transmitted symbols`

Accurate SER estimation requires a large number of Monte Carlo samples, making this part of the simulation computationally expensive as well.

In the OpenMP implementation, different INR points are distributed among CPU threads. In the CUDA implementation, each INR point is assigned to a GPU block, while the threads within the block process Monte Carlo samples in parallel and combine their error counts through a reduction.

## Requirements

GCC with OpenMP support, GNU Scientific Library (`libgsl-dev` on Ubuntu), NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU. 

Build commands should be run from the repository root, where the `Makefile` is located.

## Build and Run

### Decision Regions (Serial, OpenMP, CUDA)
To compile:
```bash
make decision_regions
```
To run:
```bash
make run_decision_regions
```

### Symbol Error Rate - SER (Serial, OpenMP, CUDA)
To compile:
```bash
make SER
```
To run:
```bash
make run_SER
```

### Cleanup
Delete the six executables and object files (leaves CSV result files intact):
```bash
make clean
```

## Thread Settings

To change the number of threads used, edit the corresponding `#define` in the source files and rebuild the task.

**For OpenMP (CPU threads):**
The OpenMP programs call `omp_set_num_threads(NUM_THREADS)`, so change the source definition rather than relying on `OMP_NUM_THREADS`.
```c
// decision_regions/benchmark_decision_regions_openMP.c
// ser/SER_openMP.c
#define NUM_THREADS 16 
```

**For CUDA (GPU threads per block):**
For SER CUDA, use a power-of-two block size with the current reduction loop; it launches one block per INR point.
```c
// decision_regions/decision_regions_cuda.cu
// ser/SER_cuda.cu
#define THREADS_PER_BLOCK 32
```
*Note: The serial versions always use one CPU thread.*

## Results

Each run appends benchmark rows to `decision_regions_timing.csv` or `SER_timings.csv`. 

* The decision-regions CPU benchmarks time the ML computation, while CUDA also times copying the matched indices back to the host. 
* The benchmark programs save timing results only; SER-versus-INR data and decision-region visualization data are not exported by these benchmark runs.