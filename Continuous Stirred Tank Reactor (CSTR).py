# -*- coding: utf-8 -*-
"""
Created on Tue Dec 30 15:30:56 2025

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- CSTR Model ---
def cstr_model(t, y, params):
    """
    Defines the system of 4 ODEs for the CSTR with variable volume.
    y = [CA, TR, Tj, VR]
    """
    # Unpack the state variables
    CA, TR, Tj = y
    
    # Unpack the parameters
    (ko, E, R, rho, cp, lam, U, AJ, Cao, To, Fj, Tcin, 
     Vj, rhoj, cj, F, VR) = params.values()

    
    # Reaction rate
    k = ko * np.exp(-E / (R * TR))
    rA = k * CA
    
    # Heat transfer rate
    Q = U * AJ * (TR - Tj)
    
    # --- System of 4 Ordinary Differential Equations ---
    dCA_dt = F * (Cao - CA) / VR - rA
    dTR_dt = (F * (To - TR) - (lam * VR * rA + Q) / (rho * cp) ) / VR
    dTj_dt = ((Fj * (Tcin - Tj)) + Q / ( rhoj * cj)) / Vj
    
    return [dCA_dt, dTR_dt, dTj_dt]

# --- Simulation and Plotting ---
# Parameters
params = {
    'ko': 20.75e5,      # s^-1
    'E': 69.71e6,       # J/kmol
    'R': 8314,          # J/(kmol*K)
    'rho': 801,         # kg/m^3
    'cp': 3137,         # J/(kg*K)
    'lam': -69.71e6,    # J/kmol (exothermic)
    'U': 851,           # W/(m^2*K)
    'AJ': 101,          # m^2
    'Cao': 5,           # kmol/m^3
    'To': 350,          # K
    'Fj': 0.015,        # m^3/s (A reasonable steady state value)
    'Tcin': 294,        # K
    'Vj': 10.1,         # m^3
    'rhoj': 1000,       # kg/m^3
    'cj': 4183,         # J/(kg*K)
    'Fo_t': 0.05,       # constant inlet flow
    "VR": 102.0         #Constant reactor volume
}

# Initial conditions
CA0 = params['Cao']     # inlet flowrate
TR0 = 350.0
Tj0 = 350


# Initial condition must be a vector of same size as the derivatives
y0 = [CA0, TR0, Tj0]

# Time span for the simulation
t_hours = 3
t_seconds = t_hours * 3600
t_span = (0, t_seconds)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Solve the ODE system
# Default method is Runge-Kutta 4 ('RK45').
# It can be changed by specifying a different method, e.g., 'Radau', 'BDF', etc.
sol = solve_ivp(
    fun=cstr_model,
    t_span=t_span,
    y0=y0,
    args=(params,),
    t_eval=t_eval,
    method='Radau'
)

# Extract results for plotting
CA = sol.y[0]
TR = sol.y[1]
Tj = sol.y[2]
Conversion = 100 * (params['Cao'] - CA) / params['Cao']

sol_hrs = sol.t / 3600

# --- Plotting the results ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 3, figsize=(25, 10))

# Concentration plot
axs[0].plot(sol_hrs, CA, label=r'$C_A(t)$')
axs[0].plot(sol_hrs, params['Cao'] - CA, label=r'$C_B(t)$')
axs[0].set_title('Reactant Concentration', fontsize=20)
axs[0].set_xlabel('Time (s)', fontsize=20)
axs[0].set_ylabel(r'Concentration $(kmol/m^3)$', fontsize=20)
axs[0].grid(True)
axs[0].legend(fontsize=20)
axs[0].set_xlim(left=0,right=t_hours)

# Reactor Temperature plot
axs[1].plot(sol_hrs, TR, 'r-', label=r'$T_R(t)$')
axs[1].set_title('Reactor Temperature', fontsize=20)
axs[1].set_xlabel('Time (s)', fontsize=20)
axs[1].set_ylabel(r'Temperature $T_R$ (K)', fontsize=20)
axs[1].grid(True)
axs[1].legend(fontsize=20)
axs[1].set_xlim(left=0, right=t_hours)

# Jacket Temperature plot
axs[2].plot(sol_hrs, Tj, 'g-', label=r'$T_j(t)$')
axs[2].set_title('Jacket Temperature', fontsize=20)
axs[2].set_xlabel('Time (s)', fontsize=20)
axs[2].set_ylabel(r'Temperature $T_j$ (K)', fontsize=20)
axs[2].grid(True)
axs[2].legend(fontsize=20)
axs[2].set_xlim(left=0, right=t_hours)


plt.tight_layout()
plt.show()

# Separate Conversion Plot
plt.figure()
plt.plot(sol.t, Conversion, label='Conversion')
plt.xlabel('Time (s)', fontsize=20)
plt.ylabel('Conversion (%)', fontsize=20)
plt.grid(True)
plt.legend(fontsize=20)
plt.xlim(left=0)
plt.title(f"Conversion final value = {Conversion[-1]:.2f}%")
plt.show()
