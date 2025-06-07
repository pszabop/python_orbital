import numpy as np

mu = 1.32712440018e11  # km^3/s^2
r_earth = 1.496e8      # km
r_mars = 2.279e8       # km

v_earth_circ = np.sqrt(mu / r_earth)  # Earth's orbital speed
delta_v = 1.06                         # km/s, extra tangential velocity at Earth intercept

v_earth_new = v_earth_circ + delta_v

# Calculate C3 at Earth after delta-v
C3_earth = v_earth_new**2 - 2 * mu / r_earth

# Calculate C3 at Mars (circular orbit)
v_mars_circ = np.sqrt(mu / r_mars)
C3_mars = v_mars_circ**2 - 2 * mu / r_mars  # Should be negative, since circular orbit

print(f"Earth circular velocity: {v_earth_circ:.4f} km/s")
print(f"Earth velocity after adding 1.06 km/s: {v_earth_new:.4f} km/s")
print(f"C3 at Earth intercept: {C3_earth:.4f} km^2/s^2")
print(f"C3 at Mars circular orbit: {C3_mars:.4f} km^2/s^2")
