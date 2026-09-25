CC = gcc
NVCC = nvcc

CFLAGS = -O3
OMPFLAGS = -fopenmp
INCLUDES = -Icommon -Idecision_regions
LIBS = -lgsl -lgslcblas -lm

COMMON_SRC = common/constellation.c common/detector.c
COMMON_HEADERS = common/constellation.h common/detector.h

REGIONS_SRC = decision_regions/decision_regions.c
REGIONS_HEADER = decision_regions/decision_regions.h
REGIONS_SERIAL_SRC = decision_regions/benchmark_decision_regions_serial.c
REGIONS_OPENMP_SRC = decision_regions/benchmark_decision_regions_openMP.c
REGIONS_CUDA_SRC = decision_regions/decision_regions_cuda.cu

SER_SERIAL_SRC = ser/SER_serial.c
SER_OPENMP_SRC = ser/SER_openMP.c
SER_CUDA_SRC = ser/SER_cuda.cu

REGIONS_BIN = decision_regions_serial decision_regions_openMP decision_regions_cuda
SER_BIN = SER_serial SER_openMP SER_cuda

.PHONY: SER decision_regions run_SER run_decision_regions clean

SER: $(SER_BIN)

decision_regions: $(REGIONS_BIN)

SER_serial: $(SER_SERIAL_SRC) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(SER_SERIAL_SRC) $(COMMON_SRC) -o $@ $(LIBS)

SER_openMP: $(SER_OPENMP_SRC) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCLUDES) $(SER_OPENMP_SRC) $(COMMON_SRC) -o $@ $(LIBS)

SER_cuda: $(SER_CUDA_SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(SER_CUDA_SRC) -o $@

decision_regions_serial: $(REGIONS_SERIAL_SRC) $(REGIONS_SRC) $(REGIONS_HEADER) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(REGIONS_SERIAL_SRC) $(REGIONS_SRC) $(COMMON_SRC) -o $@ $(LIBS)

decision_regions_openMP: $(REGIONS_OPENMP_SRC) $(REGIONS_SRC) $(REGIONS_HEADER) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INCLUDES) $(REGIONS_OPENMP_SRC) $(REGIONS_SRC) $(COMMON_SRC) -o $@ $(LIBS)

decision_regions_cuda: $(REGIONS_CUDA_SRC)
	$(NVCC) $(CFLAGS) $(INCLUDES) $(REGIONS_CUDA_SRC) -o $@

run_SER: SER
	./SER_serial
	./SER_openMP
	./SER_cuda

run_decision_regions: decision_regions
	./decision_regions_serial
	./decision_regions_openMP
	./decision_regions_cuda

clean:
	rm -f $(REGIONS_BIN) $(SER_BIN)