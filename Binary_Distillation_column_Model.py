# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 04:55:45 2026

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve


# --- Global Constants ---
P_total = 101000.0  # Pa

# DIPPR Constants for Vapor Pressure (Pa) and Temperature (K)
# Col 1: Cyclohexane (A), Col 2: Heptane (B)
DIPPR = np.array([
    [5.1087e1, 8.7829e1],   # C1
    [-5.2264e3, -6.9964e3], # C2
    [-4.2278e0, -9.8802e0], # C3
    [9.7554e-18, 7.2099e-6],# C4
    [6.0000e0, 2.0000e0]    # C5
])

# Wilson Parameters
L12 = 1.618147731
L21 = 0.50253532

def calculate_psat(T):
    """Calculates Saturation Pressure for A and B given Temperature T (K)."""
    # Vectorized for A and B
    Psat = np.exp(
        DIPPR[0, :] + 
        DIPPR[1, :] / T + 
        DIPPR[2, :] * np.log(T) + 
        DIPPR[3, :] * (T ** DIPPR[4, :])
    )
    return Psat # Returns [PsatA, PsatB]


def calculate_gamma(x):
    """Calculates Activity Coefficients (Wilson) given xA."""
    xA = np.clip(x, 1e-9, 1-1e-9)
    xB = 1 - xA
    
    term1_A = -np.log(xA + L12 * xB)
    term2_A = xB * (L12 / (xA + L12 * xB) - L21 / (L21 * xA + xB))
    gammaA = np.exp(term1_A + term2_A)
    
    term1_B = -np.log(xB + L21 * xA)
    term2_B = -xA * (L12 / (xA + L12 * xB) - L21 / (L21 * xA + xB))
    gammaB = np.exp(term1_B + term2_B)
    
    return gammaA, gammaB

def bubble_point_error(T_guess, xA_val):
    """
    Algebraic Constraint: P_total - (yA*P + yB*P) = 0
    P_total - (xA*gammaA*PsatA + xB*gammaB*PsatB) = 0
    """
    gammaA, gammaB = calculate_gamma(xA_val)
    Psats = calculate_psat(T_guess)
    
    P_calc = xA_val * gammaA * Psats[0] + (1 - xA_val) * gammaB * Psats[1]
    return P_calc - P_total

def solve_bubble_point(x_vector, T_guess_vector):
    """
    Newton Solver to find T for all 32 stages simultaneously.
    Since stages are independent for Bubble Point, we can loop or map.
    """
    T_solved = np.zeros_like(x_vector)
    
    # We iterate through each stage to solve for its Temperature
    # Ideally, we use the previous T as the guess for efficiency
    for i in range(len(x_vector)):
        # Use fsolve to find T that makes bubble_point_error = 0
        # We suppress warnings to keep output clean
        T_solved[i] = fsolve(bubble_point_error, T_guess_vector[i], args=(x_vector[i]), xtol=1e-6)[0]
    
    return T_solved

# Global variable to store T for plotting later (since solve_ivp only returns x)
T_history = []
t_history = []

def distill_dae_system(t, x, rr, x_Feed, prev_T, N):
    """
    DAE Model: Solves Algebraic T first, then Differential x.
    """
    # --- 1. Algebraic State Resolution (Calculate T) ---
    # Use the previous temperature as the initial guess for speed
    T_current = solve_bubble_point(x, prev_T)
    
    # Update the guess for the next step (basic stateful hack for ODE solvers)
    # In a class setting, explaining this "State Estimation" is valuable.
    prev_T[:] = T_current 
    
    # Store for visualization
    T_history.append(T_current.copy())
    t_history.append(t)

    # --- 2. Calculate Vapor Composition (y) ---
    # Now that we have the correct T, we calculate Equilibrium y
    gammaA, gammaB = calculate_gamma(x)

    # Psat_stages = np.zeros((N_stages,2))
    # for i in range(N_stages):
    #     Psat_stages[i,:] = calculate_psat(T_current[i])

    Psats_stages = np.array([calculate_psat(T) for T in T_current]) # Shape (32, 2)
    
    # y_i = x_i * gamma_i * Psat_i / P
    y = (x * gammaA * Psats_stages[:, 0]) / P_total

    # --- 3. Ordinary Differential Equations (Mass Balances) ---
    Feed = 24.0 / 60.0
     # Operational calculations based on Reflux Ratio (rr)
    D = x_Feed*Feed           # Distillate Flowrate (fixed split for this model)
    L = rr * D               # Liquid Flow in Rectifying Section
    V = L + D                # Vapor Flow Calculation
    FL = Feed + L            # Liquid Flow in Stripping Section
    W = Feed - D               # Bottom flowrate
    
    # Physics parameters
    vol = 1.6                # Relative Volatility (alpha)
    atray = 0.25             # Molar Holdup on Trays (mol)
    acond = 0.5              # Molar Holdup in Condenser (mol)
    areb = 1.0               # Molar Holdup in Reboiler (mol)

    xdot = np.zeros(N)
    
    # Condenser (0)
    xdot[0] = (1/acond) * (V*y[1] - L*x[0] - D*x[0])

    # Rectifying (1-15)
    for i in range(1, int(N/2)):
        xdot[i] = (1/atray) * (L * (x[i-1] - x[i]) - V * (y[i] - y[i+1]))

    # Feed (16)
    xdot[int(N/2)] = (1/atray) * (Feed * x_Feed + L * x[int(N/2)-1] - FL * x[int(N/2)] - V * (y[int(N/2)] - y[int(N/2)+1]))

    # Stripping (17-30)
    for i in range(int(N/2)+1, N-1):
        xdot[i] = (1/atray) * (FL * (x[i-1] - x[i]) - V * (y[i] - y[i+1]))

    # Reboiler (31)
    xdot[N-1] = (1/areb) * (FL * x[N-2] - W * x[N-1] - V * y[N-1])

    return xdot

# Initial Conditions (Concentration from previous steady state)
N_stages = 32
x_initial = np.zeros(N_stages)

# Initial Temperature Guess (DAE Initialization)
T_guess_init = np.full(N_stages, 350.0) # Start guessing 350K everywhere
# Simulation Params
reflux_ratio = 3
feed_composition = 0.5
final_time = 200

# Clear history
T_history = []
t_history = []

# Wrapper function
# We pass a mutable array 'T_guess_init' that updates in place inside the solver to track state
fun = lambda t, x: distill_dae_system(t, x, reflux_ratio, feed_composition, T_guess_init, N_stages)

print("Solving DAE System (this may take a moment due to internal algebraic loops)...")
sol = solve_ivp(fun, [0, final_time], x_initial, method='BDF', t_eval=np.linspace(0, final_time, 200))
print("Solution Complete.")


# --- Post-Processing Temperature Data ---
# Since T_history was collected during variable timesteps, we need to interpolate it
# to match sol.t (the output time points).

T_array_raw = np.array(T_history)
t_array_raw = np.array(t_history)

T_final = np.zeros((len(sol.t), N_stages))

for stage in range(N_stages):
    # Interpolate raw T history to match the smooth time grid of sol.t
    T_final[:, stage] = np.interp(sol.t, t_array_raw, T_array_raw[:, stage])

# --- Plotting ---
fig, axs = plt.subplots(2, 2, figsize=(16, 12))

# Color Map
all_colors = cm.plasma(np.linspace(0, 1, N_stages))

# --- 1. Composition Plots (Left Column) ---
ax1, ax2 = axs[0, 0], axs[1, 0]

# Top Section
for i in range(0, int(N_stages/2)):
    stage_num = i + 1
    label = f'Stage {stage_num} (Condenser)' if stage_num == 1 else f'Stage {stage_num}'
    style = '--' if stage_num == N_stages else '-'
    width = 2.5 if stage_num == N_stages else 1.0
    ax1.plot(sol.t, sol.y[i], color=all_colors[i], linestyle=style, linewidth=width, label=label)

ax1.set_title(f'Composition: Condenser ({N_stages}) to Stage {int(N_stages/2)+1}', fontsize=16)
ax1.set_ylabel('Mole Fraction ($x_A$)', fontsize=16)
ax1.set_xlabel('Time (min)', fontsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)

# Bottom Section
for i in range(int(N_stages/2), N_stages):
    stage_num = i + 1
    label = f'Stage {stage_num} (Feed)' if stage_num == int(N_stages/2) else (f'Stage {stage_num} (Reboiler)' if stage_num == N_stages else f'Stage {stage_num}')
    style = '--' if stage_num == 1 else '-'
    width = 2.5 if stage_num in [int(N_stages/2), 1] else 1.0
    ax2.plot(sol.t, sol.y[i], color=all_colors[i], linestyle=style, linewidth=width, label=label)

ax2.set_title(f'Composition: Feed ({int(N_stages/2)}) to Reboiler (1)', fontsize=16)
ax2.set_ylabel('Mole Fraction ($x_A$)', fontsize=16)
ax2.set_xlabel('Time (min)', fontsize=16)
ax2.grid(True, linestyle='--', alpha=0.7)

# --- 2. Temperature Plots (Right Column) ---
ax3, ax4 = axs[0, 1], axs[1, 1]

# Top Section Temps
for i in range(0, int(N_stages/2)):
    stage_num = N_stages - i
    label = f'Stage {stage_num} (Condenser)' if stage_num == N_stages else f'Stage {stage_num}'
    style = '--' if stage_num == N_stages else '-'
    width = 2.5 if stage_num == N_stages else 1.0
    ax3.plot(sol.t, T_final[:, i], color=all_colors[i], linestyle=style, linewidth=width, label=label)

ax3.set_title(f'Temperature: Condenser ({N_stages}) to Stage {int(N_stages/2)+1}', fontsize=16)
ax3.set_ylabel('Temperature (K)', fontsize=16)
ax3.set_xlabel('Time (min)', fontsize=16)
ax3.grid(True, linestyle='--', alpha=0.7)
ax3.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=14, ncol=1)


# Bottom Section Temps
for i in range(int(N_stages/2), N_stages):
    stage_num = N_stages - i
    label = f'Stage {stage_num} (Feed)' if stage_num == int(N_stages/2) else (f'Stage {stage_num} (Reboiler)' if stage_num == 1 else f'Stage {stage_num}')
    style = '--' if stage_num == 1 else '-'
    width = 2.5 if stage_num in [int(N_stages/2), 1] else 1.0
    ax4.plot(sol.t, T_final[:, i], color=all_colors[i], linestyle=style, linewidth=width, label = label)

ax4.set_title(f'Temperature: Feed ({int(N_stages/2)}) to Reboiler (1)', fontsize=16)
ax4.set_ylabel('Temperature (K)', fontsize=16)
ax4.set_xlabel('Time (min)', fontsize=16)
ax4.grid(True, linestyle='--', alpha=0.7)
ax4.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=14, ncol=1)

plt.tight_layout()
plt.show()


print(f"Final composition of Cyclohexane top stage = {100*sol.y[0,-1]:.2f}%")
print(f"Final composition of Cyclohexane bottom stage = {100*sol.y[-1,-1]:.2f}%")
print(f"Final temperature of top stage = {T_final[-1,0]:.2f}K")
print(f"Final temperature of bottom stage = {T_final[-1,-1]:.2f}K")
