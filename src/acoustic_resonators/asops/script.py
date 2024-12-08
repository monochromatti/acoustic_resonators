from math import floor
import seaborn as sns
import numpy as np

def update_interior(u1, u2, k, idxint):
    u1[i + 1, 1:idxint] = (
        2 * u1[i, 1:idxint] - u1[i-1, 1:idxint]
        + r1**2 * (u1[i, 2:idxint+1] - 2 * u1[i, 1:idxint] + u1[i, 0:idxint-1])
    )
    u2[i + 1, idxint:-1] = (
        2 * u2[i, idxint:-1] - u2[i - 1, idxint:-1]
        + r2**2 * (u2[i, idxint+1:] - 2 * u2[i, idxint:-1] + u2[i, idxint-1:-2])
    )
    u1[i + 1, idxint] += u1[i, idxint] - u2[i, idxint]
    u2[i + 1, idxint] += u2[i, idxint] - u1[i, idxint]

    return u1, u2

def update_boundaries(u1, u2):
    u1[i + 1, 0] = 0  # Rigid boundary. Assumes vibration does not generate sound waves in air.
    u2[i + 1, -1] = -r2 * (u2[i, -1] - u2[i, -2]) + u2[i, -1]  # Radiating boundary. Mimics infinite substrate.
    return u1, u2

# Domain size and physical variables
L_film = 34  # size of domain [nm]
L_sub = 100  # size of substrate [nm]
L = L_film + L_sub # size of simulation domain [nm]
T = 200.0  # total time to simulate [ps]
c1 = 5.4  # speed of sound in thin solid [ps/nm]
c2 = 10.4  # speed of sound in infinite solid [ps/nm]
k = 0.5

# Discretization parameters
dx = 3e-1
dt = 1e-2

r1 = c1 * dt / dx  # Courant number, thin solid
r2 = c2 * dt / dx  # Courant number, infinite solid

assert max(c1, c2) * dt / dx < 1, "Stability condition not satisfied!"

# Define the grid in space
x = np.arange(0, L, dx)
t = np.arange(0, T, dt)
idxint = floor(x.size * L_film / L)  # Index of `x` at interface

u1 = np.zeros((t.size, x.size))  # displacement field in thin solid
u2 = np.zeros((t.size, x.size))  # displacement field in infinite solid

u1[0, :idxint] = 1 # Uniform initial stress

for i in range(1, t.size - 1):
    # Update interior points
    u1, u2 = update_interior(u1, u2, k, idxint)
    u1, u2 = update_boundaries(u1, u2)