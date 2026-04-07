# GPU-Based Satellite Orbit Feasibility Simulator

This project simulates 2D satellite motion under Earth gravity using CUDA and compares Euler vs RK4 integration. It also generates CSV output and visualizations in Python.

## Files In This Project

- [satellite_sim.cu](satellite_sim.cu) - CUDA simulation and CSV generation
- [visualize.py](visualize.py) - Python plotting script
- [Makefile](Makefile) - build and run shortcuts
- [PCAP MINOR PROJECT SYNOPSIS.docx](PCAP%20MINOR%20PROJECT%20SYNOPSIS.docx) - project synopsis
- [PCAP_SYNOPSIS.pdf](PCAP_SYNOPSIS.pdf) - synopsis PDF

## What You Need In Colab

Upload these files into the Colab working folder:

- [satellite_sim.cu](satellite_sim.cu)
- [visualize.py](visualize.py)
- [Makefile](Makefile)

You do not need to upload the synopsis files to run the simulator. They are only for reference.

## Colab Requirements

1. Turn on GPU runtime:
   - Runtime -> Change runtime type -> Hardware accelerator -> GPU
2. Restart the runtime after changing it.
3. Confirm the GPU is available with:

```bash
!nvidia-smi
```

If `nvidia-smi` does not show a GPU, the simulation cannot run.

## Colab Run Order

Run these cells in order:

```bash
!apt-get update -qq
!apt-get install -y -qq build-essential make
!pip install -q pandas matplotlib numpy
```

Then build and run the simulator:

```bash
!make clean
!make compile
!./satellite_sim
```

Then generate plots:

```bash
!python visualize.py
```

## Output Files

After a successful run, the project generates:

- `orbit_results.csv`
- `timing.csv`
- `trajectory_samples.csv`
- `orbit_visualization.png`

## If You See Errors

### `nvcc: No such file or directory`
- CUDA toolkit is missing in the Colab session.
- Run:

```bash
!apt-get update -qq
!apt-get install -y -qq nvidia-cuda-toolkit
```

### `no CUDA-capable device is detected`
- Colab is not currently using a GPU runtime.
- Recheck Runtime -> Change runtime type -> GPU, then restart the runtime.

### `orbit_results.csv not found`
- The simulator did not finish successfully.
- Fix the CUDA error first, then rerun the simulator.

## Optional Single-Cell Run

If you want one cell that does everything:

```bash
!apt-get update -qq
!apt-get install -y -qq build-essential make
!pip install -q pandas matplotlib numpy
!make clean
!make compile
!./satellite_sim
!python visualize.py
```

## Notes

- The simulation is designed for GPU execution.
- The Python visualizer can run on CPU.
- The Makefile uses portable CUDA flags that work better in Colab than hardcoded GPU-only settings.