# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 09:02:25 2026

@author: HP
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# --- Helper function to make the voltage response dynamic ---

def getDynamicVoltage(t):
    return np.piecewise(t, [t < 5, (t>=5) & (t<10), t>=10], [1.0, 0.5, 1.5])

# --- DC motor Model ---
def DC_motor_model(t, y, params):
    """
    Defines the system of 3 ODEs for the DC motor.
    y = [ia, omega]
    """
    # Unpack the state variables
    ia, omega = y
    
    # Unpack the parameters
    (J, b, K, R, L), V = params.values(), getDynamicVoltage(t)


    
    # --- System of 4 Ordinary Differential Equations ---
    dia_dt = (V - K * omega - R * ia) / L
    domega_dt = (K * ia - b * omega)/J    
    return [dia_dt, domega_dt]

# --- Simulation and Plotting ---
# Parameters

params = {
    'J': 0.1,           # s^-1
    'b': 0.1,           # N · m · s/rad
    'K': 0.01,          # N · m/A
    'R': 1,             # Ω
    'L': 0.5,           # H
}

# Initial conditions

ia0 = 0.0
omega0 = 0.0


# Initial condition must be a vector of same size as the derivatives
y0 = [ia0, omega0]

# Time span for the simulation

t_span = (0, 15)
t_eval = np.linspace(t_span[0], t_span[1], 100)

# Solve the ODE system
# Default method is Runge-Kutta 4 ('RK45').
# It can be changed by specifying a different method, e.g., 'Radau', 'BDF', etc.
sol = solve_ivp(
    DC_motor_model,
    t_span=t_span,
    y0=y0,
    args=(params,),
    t_eval=t_eval,
    method='Radau'
)

# Extract results for plotting
ia = sol.y[0]
omega = sol.y[1]


#------ parametric sweep for changing J values ------

J_values = [0.01, 0.05, 0.1]
results = {}

for val in J_values:
    
    params['J'] =  val
    sol = solve_ivp(
        DC_motor_model,
        t_span=t_span,
        y0=y0,
        args=(params,),
        t_eval=t_eval,
        method='Radau'
    )
    results[val] = sol
    
plt.style.use('seaborn-v0_8-whitegrid')
figure, axis = plt.subplots(1, 2, figsize=(14, 6)) # Adjusted height for 1x2 layout

for val, sol in results.items():
    # Plotting Armature Current on the first axis
    # We use an f-string to put the J value in the legend
    axis[0].plot(sol.t, sol.y[0], label=f'J = {val}')
    
    # Plotting Angular Velocity on the second axis
    axis[1].plot(sol.t, sol.y[1], label=f'J = {val}')

# Formatting Axis 0
axis[0].set_title('Armature Current', fontsize=18)
axis[0].set_xlabel('Time (s)', fontsize=14)
axis[0].set_ylabel(r'$i_a(t)$ (A)', fontsize=14)
axis[0].grid(True)
axis[0].legend(title="Inertia (J)", fontsize=10)
axis[0].set_xlim(left=0)

# Formatting Axis 1
axis[1].set_title('Motor Speed', fontsize=18)
axis[1].set_xlabel('Time (s)', fontsize=14)
axis[1].set_ylabel(r'$\dot{\theta}(t)$ (rad/s)', fontsize=14)
axis[1].grid(True)
axis[1].legend(title="Inertia (J)", fontsize=10)
axis[1].set_xlim(left=0)

plt.tight_layout()
plt.show()




# --- Plotting the results ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, axs = plt.subplots(1, 3, figsize=(25, 10))

# Armature Current plot
axs[0].plot(sol.t, ia, label=r'$i_a(t)$')
axs[0].set_title('Armature Current', fontsize=20)
axs[0].set_xlabel('Time (s)', fontsize=20)
axs[0].set_ylabel(r'Armature Current (A)', fontsize=20)
axs[0].grid(True)
axs[0].legend(fontsize=20)
axs[0].set_xlim(left=0)

# Angular velocity plot
axs[1].plot(sol.t, omega, 'r-', label=r'$\dot{\theta}(t)$')
axs[1].set_title('Motor Speed', fontsize=20)
axs[1].set_xlabel('Time (s)', fontsize=20)
axs[1].set_ylabel(r'Angular velocity (rad/s)', fontsize=20)
axs[1].grid(True)
axs[1].legend(fontsize=20)
axs[1].set_xlim(left=0)

# Input voltagw plot
axs[2].plot(sol.t, [getDynamicVoltage(t) for t in sol.t],'g-', label=r'V(t)')
axs[2].set_title('Input Voltage signal', fontsize=20)
axs[2].set_xlabel('Time (s)', fontsize=20)
axs[2].set_ylabel(r'Input Voltage(V)', fontsize=20)
axs[2].grid(True)
axs[2].legend(fontsize=20)
axs[2].set_xlim(left=0)
