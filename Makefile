CC = gcc
NVCC = nvcc
CFLAGS = -O3
OMPFLAGS = -fopenmp
LIBS = -lgsl -lgslcblas -lm

COMMON_SRC = constellation.c detector.c
COMMON_HEADERS = constellation.h detector.h
REGIONS_SRC = decision_regions.c
REGIONS_HEADER = decision_regions.h

REGIONS_BIN = decision_regions_serial decision_regions_openMP decision_regions_cuda
SER_BIN = SER_serial SER_openMP SER_cuda

.PHONY: decision_regions SER run_decision_regions run_SER clean

decision_regions: $(REGIONS_BIN)

SER: $(SER_BIN)

decision_regions_serial: benchmark_decision_regions_serial.c $(REGIONS_SRC) $(REGIONS_HEADER) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) benchmark_decision_regions_serial.c $(REGIONS_SRC) $(COMMON_SRC) -o $@ $(LIBS)

decision_regions_openMP: benchmark_decision_regions_openMP.c $(REGIONS_SRC) $(REGIONS_HEADER) $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAGS) benchmark_decision_regions_openMP.c $(REGIONS_SRC) $(COMMON_SRC) -o $@ $(LIBS)

decision_regions_cuda: decision_regions_cuda.cu
	$(NVCC) $(CFLAGS) $< -o $@

SER_serial: SER_serial.c $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) SER_serial.c $(COMMON_SRC) -o $@ $(LIBS)

SER_openMP: SER_openMP.c $(COMMON_SRC) $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(OMPFLAGS) SER_openMP.c $(COMMON_SRC) -o $@ $(LIBS)

SER_cuda: SER_cuda.cu
	$(NVCC) $(CFLAGS) $< -o $@

run_decision_regions:
	./decision_regions_serial
	./decision_regions_openMP
	./decision_regions_cuda

run_SER:
	./SER_serial
	./SER_openMP
	./SER_cuda

clean:
	rm -f $(REGIONS_BIN) $(SER_BIN)