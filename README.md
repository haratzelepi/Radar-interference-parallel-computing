# Radar Interference: Decision Regions and SER

Serial C, OpenMP and CUDA benchmarks for two related tasks: computing ML decision regions on a complex-plane grid and estimating symbol error rate (SER) over an INR sweep. Shared constellation and detector code is located in `common/`, while the decision-region and SER implementations are organized in `decision_regions/` and `ser/`, respectively.

## Requirements

GCC with OpenMP support, GNU Scientific Library (`libgsl-dev` on Ubuntu), NVIDIA CUDA Toolkit (`nvcc`) and a CUDA-capable GPU. Build commands should be run from the repository root, where the `Makefile` is located.

## Build and run

| Task | Compile | Run |
| --- | --- | --- |
| Decision regions (serial, OpenMP, CUDA) | `make decision_regions` | `make run_decision_regions` |
| SER (serial, OpenMP, CUDA) | `make SER` | `make run_SER` |

The run targets automatically build any missing or outdated binaries before execution. `make clean` deletes the six executables and leaves CSV results intact.

## Thread settings

Edit the indicated definition in the source file, then rebuild the corresponding task:

| Implementation | Source file | Setting |
| --- | --- | --- |
| Decision regions, OpenMP | `decision_regions/benchmark_decision_regions_openMP.c` | `#define NUM_THREADS 16` (CPU threads) |
| Decision regions, CUDA | `decision_regions/decision_regions_cuda.cu` | `#define THREADS_PER_BLOCK 32` (GPU threads per block) |
| SER, OpenMP | `ser/SER_openMP.c` | `#define NUM_THREADS 16` (CPU threads) |
| SER, CUDA | `ser/SER_cuda.cu` | `#define THREADS_PER_BLOCK 32` (GPU threads per block) |

The OpenMP programs call `omp_set_num_threads(NUM_THREADS)`, so change the source definition rather than relying on `OMP_NUM_THREADS`. For SER CUDA, use a power-of-two block size with the current reduction loop; it launches one block per INR point. The serial versions use one CPU thread.

## Results

Each run appends benchmark rows to `decision_regions_timing.csv` or `SER_timings.csv`. The decision-regions CPU benchmarks time ML computation, while CUDA also times copying the matched indices to the host. SER CUDA likewise includes the result copy in its timing. The SER benchmarks save timings only: their SER arrays are not exported as SER-versus-INR curves.