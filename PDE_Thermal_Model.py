# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 14:35:43 2026

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  
from scipy.integrate import solve_ivp
# For pretty plots in notebooks
# %matplotlib inline



#%%

L=0.01
Nx = 50
Ny= 50
delta = L/(Nx-1)
k= 130

rho=2330
c_p=700
rho_cp= rho * c_p

alpha=k / rho_cp

delta2_by_alpha = delta**2/alpha



#%%

# 1. Create the coordinate grid
x = np.linspace(0, L, Nx)
y = np.linspace(0, L, Ny)
X, Y = np.meshgrid(x, y)
# 2. Define the core region (Mask)
core1_mask = (X >= 0.2*L) & (X <= 0.4*L) & (Y >= 0.2*L) & (Y <= 0.4*L)
core2_mask = (X >= 0.6*L) & (X <= 0.8*L) & (Y >= 0.6*L) & (Y <= 0.8*L)


def ode_fun(t,x):

    T_matrix = x.reshape(Ny,Nx)
    dT_dt = np.zeros((Ny,Nx))

    for i in range(Ny):
        for j in range(Nx):
            
            if t < 0.05:
                if core1_mask[i,j]:
                    q_source = 5e8
                else:
                    q_source = 0
            else:
                if core2_mask[i,j]:
                    q_source = 5e8
                else:
                    q_source = 0
            if i==0 or i==Ny-1 or j==0 or j==Nx-1:
                dT_dt[i,j] = 0
            else:
                 dT_dt[i,j] = ((T_matrix[i,j+1] - 2 * T_matrix[i,j] + T_matrix[i,j-1]) + (T_matrix[i+1,j] - 2 * T_matrix[i,j] + T_matrix[i-1,j])) / delta2_by_alpha + q_source/rho_cp

    return dT_dt.flatten()
#%%

final_time = 0.1
x_initial = np.full(Nx * Ny, 25)

# Ensures 0.05 and 0.1 are exactly in the results
t_eval = np.sort(np.unique(np.concatenate([np.linspace(0, 0.1, 100), [0.05, 0.1]])))
sol = solve_ivp(ode_fun, [0, final_time], x_initial, method='BDF', t_eval=t_eval)
#%%

def plot_heat_map(t_val, title):
    idx = np.argmin(np.abs(sol.t - t_val))
    T_2d = sol.y[:, idx].reshape((Ny, Nx))
    
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, T_2d, levels=50, cmap='inferno')
    plt.colorbar(label='Temperature (°C)')
    plt.title(title)
    plt.xlabel('L (m)')
    plt.ylabel('L (m)')
    plt.show()

plot_heat_map(0.05, "Temperature at t = 0.05s (Core 1 Active)")
plot_heat_map(0.10, "Temperature at t = 0.10s (Core 2 Active)")



# 1. Setup the Figure
fig, ax = plt.subplots(figsize=(6, 5))

# 2. Determine global min/max temperatures for a constant colorbar
# This is crucial so the colors don't "flicker" or rescale every frame
T_min = np.min(sol.y)
T_max = np.max(sol.y)

# 3. Create the update function for the animation
def update(frame):
    ax.clear()  # Clear the previous frame
    
    # Get the temperature data for the current time step
    # reshaping it back to the 2D grid (Ny, Nx)
    T_current = sol.y[:, frame].reshape((Ny, Nx))
    
    # Create the Contour Plot
    # We enforce vmin/vmax to keep the color scale locked
    cont = ax.contourf(X, Y, T_current, levels=50, cmap='inferno', vmin=T_min, vmax=T_max)
    
    # Add labels and dynamic title
    ax.set_title(f"Time: {sol.t[frame]:.3f} s")
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    
    # Re-draw the core regions (optional visual aid)
    # This draws rectangles so you can see where the cores ARE
    rect1 = plt.Rectangle((0.2*L, 0.2*L), 0.2*L, 0.2*L, linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
    rect2 = plt.Rectangle((0.6*L, 0.6*L), 0.2*L, 0.2*L, linewidth=1, edgecolor='white', facecolor='none', linestyle='--')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    
    return cont,

# 4. Create the Animation
# frames=len(sol.t) ensures we plot every time step you saved
ani = animation.FuncAnimation(fig, update, frames=len(sol.t), interval=100)

# 5. Add a Colorbar (only once)
# We create a dummy map just to make the colorbar appear correctly
# This prevents the colorbar from being redrawn/stacking 50 times
norm = plt.Normalize(vmin=T_min, vmax=T_max)
sm = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Temperature (°C)')

# 6. Save as GIF
print("Generating GIF... this might take a moment.")
ani.save('2D_Heat_Map_Evolution.gif', writer='pillow', fps=10)
plt.show()
