"""
visualize.py — Satellite Orbit Simulation Results Visualizer
PCAP Minor Project — Manipal Institute of Technology

Run after satellite_sim produces orbit_results.csv and timing.csv:
    python visualize.py

Requirements:
    pip install pandas matplotlib numpy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── Constants ────────────────────────────────────────────────────────────────
R_EARTH   = 6.371e6      # metres
GM        = 3.986004418e14

CLASS_COLORS = {1: '#2ecc71',   # Stable  → green
                2: '#e74c3c',   # Escape  → red
                3: '#3498db'}   # Crash   → blue
CLASS_LABELS = {1: 'Stable', 2: 'Escape', 3: 'Crash'}
CPU_EULER = 'CPU Euler'
GPU_EULER = 'GPU Euler'
GPU_RK4 = 'GPU RK4'

# ─── Load data ────────────────────────────────────────────────────────────────
if not os.path.exists("orbit_results.csv"):
    print("orbit_results.csv not found. Run satellite_sim first.")
    exit(1)

df = pd.read_csv("orbit_results.csv")
print(f"Loaded {len(df)} satellite records.")

traj_df = None
if os.path.exists("trajectory_samples.csv"):
    traj_df = pd.read_csv("trajectory_samples.csv")
    print(f"Loaded {traj_df['sat_id'].nunique()} sample trajectories.")

# Derive altitude and speed from initial conditions
df['altitude_km'] = (np.sqrt(df['x0_m']**2 + df['y0_m']**2) - R_EARTH) / 1000
df['speed_ms']    = np.sqrt(df['vx0_ms']**2 + df['vy0_ms']**2)
df['speed_kms']   = df['speed_ms'] / 1000

# Compute circular orbit velocity at each altitude (for reference line)
df['v_circ_kms']  = np.sqrt(GM / (R_EARTH + df['altitude_km']*1000)) / 1000
df['v_esc_kms']   = df['v_circ_kms'] * np.sqrt(2)

# ─── Figure setup ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle("GPU-Based Satellite Orbit Feasibility Simulator\n"
             "Manipal Institute of Technology — PCAP Minor Project",
             fontsize=14, fontweight='bold', y=0.98)

# ─── Plot 1: Stability Map — Euler (altitude vs speed) ────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
for cls in [1, 2, 3]:
    sub = df[df['euler_class'] == cls]
    ax1.scatter(sub['speed_kms'], sub['altitude_km'],
                c=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
                s=4, alpha=0.7, rasterized=True)

# Overlay theoretical lines
alts = np.linspace(df['altitude_km'].min(), df['altitude_km'].max(), 300)
v_circ = np.sqrt(GM / (R_EARTH + alts*1000)) / 1000
v_esc  = v_circ * np.sqrt(2)
ax1.plot(v_circ, alts, 'k--', lw=1.2, label='Circular orbit velocity')
ax1.plot(v_esc,  alts, 'k:',  lw=1.2, label='Escape velocity')

ax1.set_xlabel("Initial speed (km/s)")
ax1.set_ylabel("Altitude (km)")
ax1.set_title("Euler: Stability Map")
ax1.legend(fontsize=7, markerscale=3)
ax1.grid(True, alpha=0.3)

# ─── Plot 2: Stability Map — RK4 ──────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
for cls in [1, 2, 3]:
    sub = df[df['rk4_class'] == cls]
    ax2.scatter(sub['speed_kms'], sub['altitude_km'],
                c=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
                s=4, alpha=0.7, rasterized=True)

ax2.plot(v_circ, alts, 'k--', lw=1.2, label='Circular orbit v.')
ax2.plot(v_esc,  alts, 'k:',  lw=1.2, label='Escape velocity')
ax2.set_xlabel("Initial speed (km/s)")
ax2.set_ylabel("Altitude (km)")
ax2.set_title("RK4: Stability Map")
ax2.legend(fontsize=7, markerscale=3)
ax2.grid(True, alpha=0.3)

# ─── Plot 3: Class comparison pie charts side by side ─────────────────────────
ax3 = fig.add_subplot(2, 3, 3)
ax3.axis('off')

# Euler pie
euler_counts = df['euler_class'].value_counts().sort_index()
rk4_counts   = df['rk4_class'].value_counts().sort_index()

ax_pie1 = fig.add_axes([0.67, 0.55, 0.13, 0.25])
ax_pie2 = fig.add_axes([0.80, 0.55, 0.13, 0.25])

labels = [CLASS_LABELS.get(k, str(k)) for k in euler_counts.index]
colors = [CLASS_COLORS.get(k, 'gray') for k in euler_counts.index]
ax_pie1.pie(euler_counts.values, labels=labels, colors=colors,
            autopct='%1.0f%%', textprops={'fontsize': 7}, startangle=90)
ax_pie1.set_title("Euler", fontsize=9)

labels2 = [CLASS_LABELS.get(k, str(k)) for k in rk4_counts.index]
colors2 = [CLASS_COLORS.get(k, 'gray') for k in rk4_counts.index]
ax_pie2.pie(rk4_counts.values, labels=labels2, colors=colors2,
            autopct='%1.0f%%', textprops={'fontsize': 7}, startangle=90)
ax_pie2.set_title("RK4", fontsize=9)

# ─── Plot 4: Orbital Energy Distribution ──────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)

for cls in [1, 2, 3]:
    sub_e = df[df['euler_class'] == cls]['euler_energy']
    sub_r = df[df['rk4_class'] == cls]['rk4_energy']
    # Clip for display
    ax4.hist(sub_e.clip(-1e7, 1e7), bins=60,
             color=CLASS_COLORS[cls], alpha=0.5,
             label=f"Euler {CLASS_LABELS[cls]}", density=True)

ax4.axvline(0, color='black', lw=1.5, linestyle='--', label='ε=0 boundary')
ax4.set_xlabel("Specific orbital energy ε (J/kg)")
ax4.set_ylabel("Density")
ax4.set_title("Energy distribution (Euler)")
ax4.legend(fontsize=7)
ax4.grid(True, alpha=0.3)

# ─── Plot 5: Sample 2D Trajectories (RK4-based) ──────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
earth = plt.Circle((0, 0), R_EARTH / 1000.0, color='#34495e', alpha=0.35, label='Earth')
ax5.add_patch(earth)

if traj_df is not None and not traj_df.empty:
    for sat_id, sub in traj_df.groupby('sat_id'):
        cls = int(sub['class'].iloc[0])
        ax5.plot(sub['x_m'] / 1000.0, sub['y_m'] / 1000.0,
                 color=CLASS_COLORS.get(cls, 'gray'), alpha=0.8, lw=1)
    ax5.set_title("Sample Orbital Trajectories (2D)")
else:
    ax5.text(0.5, 0.5, "Run satellite_sim\nto generate trajectory_samples.csv",
             ha='center', va='center', transform=ax5.transAxes,
             fontsize=11, color='gray')
    ax5.set_title("Sample Orbital Trajectories (2D)")

ax5.set_xlabel("x (km)")
ax5.set_ylabel("y (km)")
ax5.set_aspect('equal', adjustable='box')
ax5.grid(True, alpha=0.3)

# ─── Plot 6: CPU vs GPU Timing (bar chart) ────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)

if os.path.exists("timing.csv"):
    tdf = pd.read_csv("timing.csv")
    methods = tdf['method'].tolist()
    times   = tdf['time_ms'].tolist()
    color_map = {
        CPU_EULER: '#e74c3c',
        GPU_EULER: '#2ecc71',
        GPU_RK4: '#3498db'
    }
    bar_colors = [color_map.get(m, '#95a5a6') for m in methods]

    bars = ax6.bar(methods, times, color=bar_colors[:len(methods)],
                   alpha=0.85, edgecolor='black', lw=0.5)
    for bar, t in zip(bars, times):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{t:.1f} ms", ha='center', fontsize=10, fontweight='bold')

    if CPU_EULER in methods and GPU_EULER in methods:
        cpu_idx = methods.index(CPU_EULER)
        ge_idx = methods.index(GPU_EULER)
        speedup_euler = times[cpu_idx] / times[ge_idx]
        title = f"CPU vs GPU Execution Time\nEuler speedup: {speedup_euler:.1f}×"
        if GPU_RK4 in methods:
            gr_idx = methods.index(GPU_RK4)
            speedup_rk4 = times[cpu_idx] / times[gr_idx]
            title += f" | RK4 speedup: {speedup_rk4:.1f}×"
        ax6.set_title(title)
    else:
        ax6.set_title("Execution Time")

    ax6.set_ylabel("Time (ms)")
    ax6.grid(axis='y', alpha=0.3)
    ax6.set_ylim(0, max(times) * 1.3)
else:
    ax6.text(0.5, 0.5, "Run satellite_sim\nto generate timing.csv",
             ha='center', va='center', transform=ax6.transAxes,
             fontsize=11, color='gray')
    ax6.set_title("CPU vs GPU Timing")

# ─── Finalize and save ────────────────────────────────────────────────────────
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("orbit_visualization.png", dpi=150, bbox_inches='tight')
print("Saved: orbit_visualization.png")
plt.show()
