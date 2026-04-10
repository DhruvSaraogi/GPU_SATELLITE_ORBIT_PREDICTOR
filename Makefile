# ─── Makefile — GPU Satellite Orbit Simulator ─────────────────────────────────
# Targets:
#   make          → compile with optimizations
#   make debug    → compile with debug symbols
#   make run      → compile + run simulation
#   make plot     → run Python visualizer
#   make all      → compile + run + plot
#   make clean    → remove build artifacts

NVCC     = nvcc
TARGET   = satellite_sim
SRC      = satellite_sim.cu

# Compilation flags:
#   -O3          : maximum optimization
#   -gencode ...  : portable Colab baseline (sm_75) + PTX fallback
#   -Xcompiler   : pass flags to the host (CPU) compiler
#   -lm          : link math library
FLAGS    = -O3 \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_75,code=compute_75 \
	-Xcompiler -Wall -lm

.PHONY: all compile run plot debug clean info

all: compile run plot

compile:
	@echo "Compiling $(SRC) ..."
	$(NVCC) $(FLAGS) $(SRC) -o $(TARGET)
	@echo "Done. Binary: ./$(TARGET)"

debug:
	$(NVCC) -g -G -gencode arch=compute_75,code=sm_75 $(SRC) -o $(TARGET)_debug -lm

run: compile
	@echo "\nRunning simulation ..."
	./$(TARGET)

plot:
	@echo "\nGenerating plots ..."
	python3 visualize.py

clean:
	rm -f $(TARGET) $(TARGET)_debug orbit_results.csv timing.csv
	rm -f orbit_visualization.png

info:
	@nvcc --version
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
