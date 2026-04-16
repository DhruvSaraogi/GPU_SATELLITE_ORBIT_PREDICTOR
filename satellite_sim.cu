/*
 * GPU-Based Satellite Orbit Feasibility Simulator
 * PCAP Minor Project — Manipal Institute of Technology
 *
 * Authors : Divyanshu Gupta, Kumar Satyam, Dhruv Saraogi
 * Mentor  : Dr G Arul Elango
 *
 * File    : satellite_sim.cu
 * Purpose : Simulates 6000 satellite trajectories in parallel using CUDA.
 *           Implements both Euler and RK4 numerical integration.
 *           Classifies each orbit as: STABLE | ESCAPE | CRASH
 *           Compares CPU vs GPU execution time.
 *           Writes results to CSV for Python visualization.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 1: CONSTANTS & CONFIGURATION
   ══════════════════════════════════════════════════════════════════════════════ */

/* Physical constants */
#define GM          3.986004418e14   /* Earth gravitational parameter (m³/s²)  */
#define R_EARTH     6.371e6          /* Earth mean radius (m)                  */
#define ATMO_HEIGHT 1.0e5            /* Atmosphere top ~100 km (Karman line)   */
#define CRASH_RADIUS (R_EARTH + ATMO_HEIGHT) /* Satellite re-enters below this */

/* Simulation parameters */
#define NUM_SATS    6000             /* Total satellites to simulate            */
#define DT          1.0             /* Time step in seconds                    */
#define NUM_STEPS   5400            /* Steps = 90 min (1 typical GEO period)   */
#define THREADS_PER_BLOCK 256       /* Standard CUDA block size                */
#define TRACE_SATS  12              /* Number of representative trajectories    */

/* Orbit classification codes */
#define ORBIT_STABLE  1             /* Remains bounded, doesn't crash          */
#define ORBIT_ESCAPE  2             /* Escapes Earth's gravity (energy >= 0)   */
#define ORBIT_CRASH   3             /* Falls below atmosphere (r < CRASH_RADIUS)*/

/* Initial condition ranges */
#define ALT_MIN     2.0e5           /* Min altitude above surface = 200 km     */
#define ALT_MAX     2.0e6           /* Max altitude above surface = 2000 km    */
#define VEL_MIN     5000.0          /* Min initial speed (m/s)                 */
#define VEL_MAX     12000.0         /* Max initial speed (m/s)                 */
#define MASS_MIN    80.0            /* Min satellite mass (kg)                 */
#define MASS_MAX    1200.0          /* Max satellite mass (kg)                 */
#define TRAJ_OUTPUT_STRIDE 5        /* Write every Nth step in trajectory CSV  */
/*
 * NOTE: Circular orbit velocity at altitude h is:
 *         v_circ = sqrt(GM / (R_EARTH + h))
 * For h = 400 km (ISS orbit): v_circ ≈ 7670 m/s
 * Escape velocity at surface : v_esc  = sqrt(2*GM / R_EARTH) ≈ 11200 m/s
 */


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 2: CUDA ERROR CHECKING UTILITY
   ══════════════════════════════════════════════════════════════════════════════ */

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while(0)


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 3: DEVICE (GPU) HELPER — GRAVITATIONAL ACCELERATION
   ══════════════════════════════════════════════════════════════════════════════
 *
 * Newton's Law of Gravitation:
 *   F = -GM * m / |r|²  (in direction of -r̂)
 *
 * Since F = m*a:
 *   a_x = -GM * x / |r|³
 *   a_y = -GM * y / |r|³
 *
 * __device__ means this function runs ON THE GPU and can only be
 * called from other GPU functions (kernels or device functions).
 */
__device__ void compute_gravity(double x, double y,
                                 double &ax, double &ay)
{
    double r2 = x*x + y*y;          /* |r|² */
    double r  = sqrt(r2);           /* |r|  */
    double r3 = r2 * r;             /* |r|³ */
    ax = -GM * x / r3;
    ay = -GM * y / r3;
}

void compute_gravity_host(double x, double y, double *ax, double *ay)
{
    double r2 = x*x + y*y;
    double r  = sqrt(r2);
    double r3 = r2 * r;
    *ax = -GM * x / r3;
    *ay = -GM * y / r3;
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 4: EULER INTEGRATION KERNEL
   ══════════════════════════════════════════════════════════════════════════════
 *
 * Euler Method (1st-order):
 *   v(t+dt) = v(t) + a(t) * dt
 *   x(t+dt) = x(t) + v(t+dt) * dt   ← "symplectic" Euler (better energy conservation)
 *
 * Pros : Very fast, simple.
 * Cons : Accumulates error over time. Energy may drift.
 *
 * CUDA THREAD MAPPING:
 *   Each thread gets a unique ID: tid = blockIdx.x * blockDim.x + threadIdx.x
 *   Thread tid handles satellite tid independently — no inter-thread communication.
 *
 * __global__ means this function is a KERNEL — launched by CPU, runs on GPU.
 */
__global__ void euler_kernel(
    const double *x0,         /* initial x positions    [NUM_SATS] */
    const double *y0,         /* initial y positions    [NUM_SATS] */
    const double *vx0,        /* initial x velocities   [NUM_SATS] */
    const double *vy0,        /* initial y velocities   [NUM_SATS] */
    int    *classification,   /* output: STABLE/ESCAPE/CRASH       */
    double *final_x,          /* output: final x position          */
    double *final_y,          /* output: final y position          */
    double *final_energy,     /* output: specific orbital energy    */
    int n                     /* total number of satellites         */
) {
    /* ── Step 1: Get this thread's unique satellite index ── */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;   /* Guard: extra threads do nothing */

    /* ── Step 2: Load initial state from global memory ── */
    double x  = x0[tid];
    double y  = y0[tid];
    double vx = vx0[tid];
    double vy = vy0[tid];
    int    cls = ORBIT_STABLE;  /* Assume stable until proven otherwise */

    /* ── Step 3: Time integration loop ── */
    for (int step = 0; step < NUM_STEPS; step++) {

        /* Compute gravitational acceleration at current position */
        double ax, ay;
        compute_gravity(x, y, ax, ay);

        /* Symplectic Euler update (velocity first, then position) */
        vx += ax * DT;
        vy += ay * DT;
        x  += vx * DT;
        y  += vy * DT;

        /* ── Step 4: Classify current state ── */
        double r  = sqrt(x*x + y*y);
        double v2 = vx*vx + vy*vy;

        /* Specific orbital energy: ε = KE/m + PE/m = v²/2 - GM/r
         * ε < 0  → bound orbit (satellite stays near Earth)
         * ε >= 0 → hyperbolic/parabolic trajectory (escape!)         */
        double energy = 0.5 * v2 - GM / r;

        if (r < CRASH_RADIUS) {
            cls = ORBIT_CRASH;
            break;
        }
        if (energy >= 0.0) {
            cls = ORBIT_ESCAPE;
            break;
        }
    }

    /* ── Step 5: Write outputs ── */
    classification[tid] = cls;
    final_x[tid]        = x;
    final_y[tid]        = y;
    final_energy[tid]   = 0.5*(vx*vx + vy*vy) - GM / sqrt(x*x + y*y);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 5: RK4 INTEGRATION KERNEL
   ══════════════════════════════════════════════════════════════════════════════
 *
 * Runge-Kutta 4th Order Method:
 *
 *   State vector: [x, y, vx, vy]
 *   Derivative  : [vx, vy, ax, ay]
 *
 *   k1 = f(t,        y)
 *   k2 = f(t + dt/2, y + dt/2 * k1)
 *   k3 = f(t + dt/2, y + dt/2 * k2)
 *   k4 = f(t + dt,   y + dt   * k3)
 *   y(t+dt) = y(t) + (dt/6) * (k1 + 2k2 + 2k3 + k4)
 *
 * Pros : 4th-order accuracy — much less energy drift than Euler.
 * Cons : 4x more function evaluations per step (4 calls to compute_gravity).
 */
__global__ void rk4_kernel(
    const double *x0,
    const double *y0,
    const double *vx0,
    const double *vy0,
    int    *classification,
    double *final_x,
    double *final_y,
    double *final_energy,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    double x  = x0[tid];
    double y  = y0[tid];
    double vx = vx0[tid];
    double vy = vy0[tid];
    int    cls = ORBIT_STABLE;

    for (int step = 0; step < NUM_STEPS; step++) {

        double ax, ay;

        /* ── k1: slope at current state ── */
        compute_gravity(x, y, ax, ay);
        double k1_x  = vx,  k1_y  = vy;
        double k1_vx = ax,  k1_vy = ay;

        /* ── k2: slope at midpoint using k1 ── */
        double x2  = x  + 0.5*DT*k1_x;
        double y2  = y  + 0.5*DT*k1_y;
        double vx2 = vx + 0.5*DT*k1_vx;
        double vy2 = vy + 0.5*DT*k1_vy;
        compute_gravity(x2, y2, ax, ay);
        double k2_x  = vx2, k2_y  = vy2;
        double k2_vx = ax,  k2_vy = ay;

        /* ── k3: slope at midpoint using k2 ── */
        double x3  = x  + 0.5*DT*k2_x;
        double y3  = y  + 0.5*DT*k2_y;
        double vx3 = vx + 0.5*DT*k2_vx;
        double vy3 = vy + 0.5*DT*k2_vy;
        compute_gravity(x3, y3, ax, ay);
        double k3_x  = vx3, k3_y  = vy3;
        double k3_vx = ax,  k3_vy = ay;

        /* ── k4: slope at end of step using k3 ── */
        double x4  = x  + DT*k3_x;
        double y4  = y  + DT*k3_y;
        double vx4 = vx + DT*k3_vx;
        double vy4 = vy + DT*k3_vy;
        compute_gravity(x4, y4, ax, ay);
        double k4_x  = vx4, k4_y  = vy4;
        double k4_vx = ax,  k4_vy = ay;

        /* ── Weighted combination (the RK4 magic) ── */
        x  += (DT / 6.0) * (k1_x  + 2.0*k2_x  + 2.0*k3_x  + k4_x);
        y  += (DT / 6.0) * (k1_y  + 2.0*k2_y  + 2.0*k3_y  + k4_y);
        vx += (DT / 6.0) * (k1_vx + 2.0*k2_vx + 2.0*k3_vx + k4_vx);
        vy += (DT / 6.0) * (k1_vy + 2.0*k2_vy + 2.0*k3_vy + k4_vy);

        /* Classify */
        double r      = sqrt(x*x + y*y);
        double v2     = vx*vx + vy*vy;
        double energy = 0.5 * v2 - GM / r;

        if (r < CRASH_RADIUS) { cls = ORBIT_CRASH;  break; }
        if (energy >= 0.0)    { cls = ORBIT_ESCAPE; break; }
    }

    classification[tid] = cls;
    final_x[tid]        = x;
    final_y[tid]        = y;
    final_energy[tid]   = 0.5*(vx*vx + vy*vy) - GM / sqrt(x*x + y*y);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 6: CPU BASELINE (single-threaded, for timing comparison)
   ══════════════════════════════════════════════════════════════════════════════ */
void cpu_euler_simulate(
    const double *x0, const double *y0,
    const double *vx0, const double *vy0,
    int *classification, double *final_energy, int n
) {
    for (int i = 0; i < n; i++) {
        double x  = x0[i],  y  = y0[i];
        double vx = vx0[i], vy = vy0[i];
        int cls = ORBIT_STABLE;

        for (int step = 0; step < NUM_STEPS; step++) {
            double r2 = x*x + y*y;
            double r  = sqrt(r2);
            double r3 = r2 * r;
            double ax = -GM * x / r3;
            double ay = -GM * y / r3;

            vx += ax * DT;
            vy += ay * DT;
            x  += vx * DT;
            y  += vy * DT;

            double ri     = sqrt(x*x + y*y);
            double energy = 0.5*(vx*vx + vy*vy) - GM / ri;

            if (ri < CRASH_RADIUS) { cls = ORBIT_CRASH;  break; }
            if (energy >= 0.0)     { cls = ORBIT_ESCAPE; break; }
        }

        classification[i] = cls;
        final_energy[i]   = 0.5*(vx*vx + vy*vy) - GM / sqrt(x*x + y*y);
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 7: INITIAL CONDITION GENERATOR
   ══════════════════════════════════════════════════════════════════════════════
 *
 * We create a grid of (altitude, velocity) pairs.
 * Each satellite starts at (R_EARTH + altitude, 0) on the x-axis,
 * and moves purely in the y-direction (tangential launch).
 * This gives a clean 2D orbital plane.
 */
void generate_initial_conditions(
    double *x0, double *y0,
    double *vx0, double *vy0,
    double *mass0,
    int n
) {
    /* Grid dimensions */
    int cols = (int)sqrt((double)n) + 1;  /* ~77 × 78 for 6000 sats */

    for (int i = 0; i < n; i++) {
        int row = i / cols;
        int col = i % cols;

        /* Linearly interpolate altitude and velocity */
        double t_alt = (double)col / (cols - 1);  /* 0.0 to 1.0 */
        double t_vel = (double)row / (cols - 1);

        double altitude = ALT_MIN + t_alt * (ALT_MAX - ALT_MIN);
        double speed    = VEL_MIN + t_vel * (VEL_MAX - VEL_MIN);
        double radius   = R_EARTH + altitude;

        /* Start on x-axis, velocity tangential (y-direction) */
        x0[i]  = radius;
        y0[i]  = 0.0;
        vx0[i] = 0.0;
        vy0[i] = speed;
        mass0[i] = MASS_MIN + (i / (double)(n - 1)) * (MASS_MAX - MASS_MIN);
    }
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 8: TIMING UTILITY
   ══════════════════════════════════════════════════════════════════════════════ */
double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 9: CSV WRITER
   ══════════════════════════════════════════════════════════════════════════════ */
void write_results_csv(
    const char *filename,
    const double *x0, const double *y0,
    const double *vx0, const double *vy0,
    const double *mass0,
    const double *euler_final_x,
    const double *euler_final_y,
    const int *euler_cls, const double *euler_energy,
    const double *rk4_final_x,
    const double *rk4_final_y,
    const int *rk4_cls,   const double *rk4_energy,
    int n
) {
    FILE *f = fopen(filename, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return; }

    fprintf(f, "sat_id,mass_kg,x0_m,y0_m,vx0_ms,vy0_ms,"
               "euler_class,euler_energy,euler_final_x_m,euler_final_y_m,"
               "rk4_class,rk4_energy,rk4_final_x_m,rk4_final_y_m\n");

    for (int i = 0; i < n; i++) {
        fprintf(f, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.6e,%.3f,%.3f,%d,%.6e,%.3f,%.3f\n",
                i,
                mass0[i], x0[i], y0[i], vx0[i], vy0[i],
                euler_cls[i], euler_energy[i], euler_final_x[i], euler_final_y[i],
                rk4_cls[i],   rk4_energy[i],   rk4_final_x[i],   rk4_final_y[i]);
    }
    fclose(f);
    printf("Results written to: %s\n", filename);
}

void write_all_trajectories_csv(
    const char *filename,
    const double *x0, const double *y0,
    const double *vx0, const double *vy0,
    const double *mass0,
    const int *rk4_cls,
    int n
) {
    FILE *f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", filename);
        return;
    }

    fprintf(f, "step,sat_id,class,mass_kg,vx_ms,vy_ms,speed_ms,x_m,y_m,r_km\n");

    for (int id = 0; id < n; id++) {
        double x = x0[id], y = y0[id];
        double vx = vx0[id], vy = vy0[id];

        for (int step = 0; step <= NUM_STEPS; step++) {
            if (step % TRAJ_OUTPUT_STRIDE == 0 || step == NUM_STEPS) {
                double speed_ms = sqrt(vx*vx + vy*vy);
                double r_km = sqrt(x*x + y*y) / 1000.0;
                fprintf(f, "%d,%d,%d,%.3f,%.6f,%.6f,%.6f,%.3f,%.3f,%.3f\n",
                        step, id, rk4_cls[id], mass0[id], vx, vy, speed_ms, x, y, r_km);
            }

            if (step == NUM_STEPS) {
                break;
            }

            double ax, ay;

            compute_gravity_host(x, y, &ax, &ay);
            double k1_x = vx,  k1_y = vy;
            double k1_vx = ax, k1_vy = ay;

            double x2 = x + 0.5*DT*k1_x;
            double y2 = y + 0.5*DT*k1_y;
            double vx2 = vx + 0.5*DT*k1_vx;
            double vy2 = vy + 0.5*DT*k1_vy;
            compute_gravity_host(x2, y2, &ax, &ay);
            double k2_x = vx2,  k2_y = vy2;
            double k2_vx = ax,  k2_vy = ay;

            double x3 = x + 0.5*DT*k2_x;
            double y3 = y + 0.5*DT*k2_y;
            double vx3 = vx + 0.5*DT*k2_vx;
            double vy3 = vy + 0.5*DT*k2_vy;
            compute_gravity_host(x3, y3, &ax, &ay);
            double k3_x = vx3,  k3_y = vy3;
            double k3_vx = ax,  k3_vy = ay;

            double x4 = x + DT*k3_x;
            double y4 = y + DT*k3_y;
            double vx4 = vx + DT*k3_vx;
            double vy4 = vy + DT*k3_vy;
            compute_gravity_host(x4, y4, &ax, &ay);
            double k4_x = vx4,  k4_y = vy4;
            double k4_vx = ax,  k4_vy = ay;

            x  += (DT / 6.0) * (k1_x  + 2.0*k2_x  + 2.0*k3_x  + k4_x);
            y  += (DT / 6.0) * (k1_y  + 2.0*k2_y  + 2.0*k3_y  + k4_y);
            vx += (DT / 6.0) * (k1_vx + 2.0*k2_vx + 2.0*k3_vx + k4_vx);
            vy += (DT / 6.0) * (k1_vy + 2.0*k2_vy + 2.0*k3_vy + k4_vy);

            double r = sqrt(x*x + y*y);
            double energy = 0.5 * (vx*vx + vy*vy) - GM / r;
            if (r < CRASH_RADIUS || energy >= 0.0) {
                break;
            }
        }
    }

    fclose(f);
    printf("Trajectory data written to: %s (%d satellites, stride=%d)\n",
           filename, n, TRAJ_OUTPUT_STRIDE);
}


/* ══════════════════════════════════════════════════════════════════════════════
   SECTION 10: MAIN
   ══════════════════════════════════════════════════════════════════════════════ */
int main(void) {

    printf("╔══════════════════════════════════════════════════════╗\n");
    printf("║   GPU-Based Satellite Orbit Feasibility Simulator    ║\n");
    printf("║   Manipal Institute of Technology — PCAP Minor       ║\n");
    printf("╚══════════════════════════════════════════════════════╝\n\n");

    /* ── Print GPU info ── */
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU  : %s\n", prop.name);
    printf("SMs  : %d  |  Max threads/block: %d\n",
           prop.multiProcessorCount, prop.maxThreadsPerBlock);
    printf("Sats : %d  |  Steps: %d  |  dt: %.1f s\n\n",
           NUM_SATS, NUM_STEPS, DT);

    /* ── Sizes ── */
    size_t sz_d = NUM_SATS * sizeof(double);
    size_t sz_i = NUM_SATS * sizeof(int);

    /* ══ STEP A: Allocate HOST memory ══ */
    double *h_x0     = (double*)malloc(sz_d);
    double *h_y0     = (double*)malloc(sz_d);
    double *h_vx0    = (double*)malloc(sz_d);
    double *h_vy0    = (double*)malloc(sz_d);
    double *h_mass0  = (double*)malloc(sz_d);

    /* Euler output */
    int    *h_euler_cls    = (int*)malloc(sz_i);
    double *h_euler_energy = (double*)malloc(sz_d);

    /* RK4 output */
    int    *h_rk4_cls    = (int*)malloc(sz_i);
    double *h_rk4_energy = (double*)malloc(sz_d);

    /* CPU baseline output (for timing comparison) */
    int    *h_cpu_cls    = (int*)malloc(sz_i);
    double *h_cpu_energy = (double*)malloc(sz_d);

    /* Final positions for each integration method */
    double *h_euler_final_x = (double*)malloc(sz_d);
    double *h_euler_final_y = (double*)malloc(sz_d);
    double *h_rk4_final_x   = (double*)malloc(sz_d);
    double *h_rk4_final_y   = (double*)malloc(sz_d);

    /* ══ STEP B: Generate initial conditions ══ */
    generate_initial_conditions(h_x0, h_y0, h_vx0, h_vy0, h_mass0, NUM_SATS);
    printf("Initial conditions generated for %d satellites.\n", NUM_SATS);

    /* ══ STEP C: CPU baseline timing ══ */
    printf("\n── CPU Euler (single-threaded baseline) ──\n");
    double cpu_start = get_time_ms();
    cpu_euler_simulate(h_x0, h_y0, h_vx0, h_vy0,
                       h_cpu_cls, h_cpu_energy, NUM_SATS);
    double cpu_time = get_time_ms() - cpu_start;
    printf("CPU time : %.2f ms\n", cpu_time);

    /* Count CPU classes */
    int cpu_stable=0, cpu_escape=0, cpu_crash=0;
    for (int i=0; i<NUM_SATS; i++) {
        if (h_cpu_cls[i]==ORBIT_STABLE)  cpu_stable++;
        else if (h_cpu_cls[i]==ORBIT_ESCAPE) cpu_escape++;
        else cpu_crash++;
    }
    printf("  Stable: %d  |  Escape: %d  |  Crash: %d\n",
           cpu_stable, cpu_escape, cpu_crash);

    /* ══ STEP D: Allocate DEVICE memory ══ */
    double *d_x0, *d_y0, *d_vx0, *d_vy0;
    int    *d_cls;
    double *d_final_x, *d_final_y, *d_energy;

    CUDA_CHECK(cudaMalloc(&d_x0,      sz_d));
    CUDA_CHECK(cudaMalloc(&d_y0,      sz_d));
    CUDA_CHECK(cudaMalloc(&d_vx0,     sz_d));
    CUDA_CHECK(cudaMalloc(&d_vy0,     sz_d));
    CUDA_CHECK(cudaMalloc(&d_cls,     sz_i));
    CUDA_CHECK(cudaMalloc(&d_final_x, sz_d));
    CUDA_CHECK(cudaMalloc(&d_final_y, sz_d));
    CUDA_CHECK(cudaMalloc(&d_energy,  sz_d));

    /* ══ STEP E: Copy initial conditions Host → Device ══ */
    CUDA_CHECK(cudaMemcpy(d_x0,  h_x0,  sz_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y0,  h_y0,  sz_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx0, h_vx0, sz_d, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy0, h_vy0, sz_d, cudaMemcpyHostToDevice));

    /* ══ STEP F: Launch EULER kernel ══ */
    /*
     * Grid config:
     *   blocks_needed = ceil(NUM_SATS / THREADS_PER_BLOCK)
     *   Total threads = blocks * THREADS_PER_BLOCK  (may slightly exceed NUM_SATS,
     *   which is why we guard with "if (tid >= n) return;" inside the kernel)
     */
    int blocks = (NUM_SATS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("\n── GPU Euler kernel ── (blocks=%d, threads/block=%d)\n",
           blocks, THREADS_PER_BLOCK);

    cudaEvent_t start_evt, stop_evt;
    float gpu_euler_ms;
    float gpu_rk4_ms;
    CUDA_CHECK(cudaEventCreate(&start_evt));
    CUDA_CHECK(cudaEventCreate(&stop_evt));

    CUDA_CHECK(cudaEventRecord(start_evt));
    euler_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_x0, d_y0, d_vx0, d_vy0,
        d_cls, d_final_x, d_final_y, d_energy, NUM_SATS
    );
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaGetLastError());     /* Check for kernel launch errors */
    CUDA_CHECK(cudaEventElapsedTime(&gpu_euler_ms, start_evt, stop_evt));

    printf("GPU Euler time : %.2f ms\n", gpu_euler_ms);
    printf("Speedup        : %.1fx\n", cpu_time / gpu_euler_ms);

    /* Copy results back */
    CUDA_CHECK(cudaMemcpy(h_euler_cls,    d_cls,     sz_i, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_euler_energy, d_energy,  sz_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_euler_final_x, d_final_x, sz_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_euler_final_y, d_final_y, sz_d, cudaMemcpyDeviceToHost));

    int eu_stable=0, eu_escape=0, eu_crash=0;
    for (int i=0; i<NUM_SATS; i++) {
        if (h_euler_cls[i]==ORBIT_STABLE)       eu_stable++;
        else if (h_euler_cls[i]==ORBIT_ESCAPE)  eu_escape++;
        else                                    eu_crash++;
    }
    printf("  Stable: %d  |  Escape: %d  |  Crash: %d\n",
           eu_stable, eu_escape, eu_crash);

    /* ══ STEP G: Launch RK4 kernel ══ */
    printf("\n── GPU RK4 kernel ──\n");

    CUDA_CHECK(cudaEventRecord(start_evt));
    rk4_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_x0, d_y0, d_vx0, d_vy0,
        d_cls, d_final_x, d_final_y, d_energy, NUM_SATS
    );
    CUDA_CHECK(cudaEventRecord(stop_evt));
    CUDA_CHECK(cudaEventSynchronize(stop_evt));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&gpu_rk4_ms, start_evt, stop_evt));

    printf("GPU RK4 time   : %.2f ms\n", gpu_rk4_ms);
    printf("Speedup        : %.1fx\n", cpu_time / gpu_rk4_ms);

    CUDA_CHECK(cudaMemcpy(h_rk4_cls,    d_cls,    sz_i, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rk4_energy, d_energy, sz_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rk4_final_x, d_final_x, sz_d, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rk4_final_y, d_final_y, sz_d, cudaMemcpyDeviceToHost));

    int rk_stable=0, rk_escape=0, rk_crash=0;
    for (int i=0; i<NUM_SATS; i++) {
        if (h_rk4_cls[i]==ORBIT_STABLE)       rk_stable++;
        else if (h_rk4_cls[i]==ORBIT_ESCAPE)  rk_escape++;
        else                                   rk_crash++;
    }
    printf("  Stable: %d  |  Escape: %d  |  Crash: %d\n",
           rk_stable, rk_escape, rk_crash);

    /* ══ STEP H: Write CSV ══ */
    write_results_csv("orbit_results.csv",
                      h_x0, h_y0, h_vx0, h_vy0,
                      h_mass0,
                      h_euler_final_x, h_euler_final_y,
                      h_euler_cls, h_euler_energy,
                      h_rk4_final_x, h_rk4_final_y,
                      h_rk4_cls,   h_rk4_energy,
                      NUM_SATS);

    write_all_trajectories_csv("trajectory_samples.csv",
                                  h_x0, h_y0, h_vx0, h_vy0, h_mass0,
                                  h_rk4_cls, NUM_SATS);

    /* ══ STEP I: Save timing for plot ══ */
    FILE *tf = fopen("timing.csv", "w");
    if (tf) {
        fprintf(tf, "method,time_ms\n");
        fprintf(tf, "CPU Euler,%.2f\n", cpu_time);
        fprintf(tf, "GPU Euler,%.2f\n", gpu_euler_ms);
        fprintf(tf, "GPU RK4,%.2f\n", gpu_rk4_ms);
        fclose(tf);
    }

    /* ══ STEP J: Cleanup ══ */
    cudaFree(d_x0); cudaFree(d_y0); cudaFree(d_vx0); cudaFree(d_vy0);
    cudaFree(d_cls); cudaFree(d_final_x); cudaFree(d_final_y); cudaFree(d_energy);

    free(h_x0); free(h_y0); free(h_vx0); free(h_vy0); free(h_mass0);
    free(h_euler_cls); free(h_euler_energy);
    free(h_rk4_cls);   free(h_rk4_energy);
    free(h_cpu_cls);   free(h_cpu_energy);
    free(h_euler_final_x); free(h_euler_final_y);
    free(h_rk4_final_x);   free(h_rk4_final_y);

    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);

    printf("\nDone.\n");
    return 0;
}
