import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
mu_sun = 1.32712440018e11  # km^3/s^2, gravitational parameter of the Sun
r_earth = 1.496e8          # km, average distance from Sun
r_mars = 2.279e8           # km, average distance from Sun
v_earth = np.sqrt(mu_sun / r_earth)
v_mars = np.sqrt(mu_sun / r_mars)
max_periapsis_speed_mars = 7.5  # km/s
total_delta_v = 7  # km/s

def mars_arrival_velocity(delta_v_outbound):
    """
    Computes the arrival velocity at Mars (heliocentric) given the outbound delta-v at Earth.
    Assumes prograde burn from Earth orbit.
    """
    # Compute departure velocity from Earth orbit
    v_depart = v_earth + delta_v_outbound
    
    # Compute C3 (characteristic energy) at departure
    C3 = v_depart**2 - 2 * mu_sun / r_earth  # Specific orbital energy at departure
    
    # Compute velocity at Mars distance using vis-viva equation
    v_arrive = np.sqrt(C3 + 2 * mu_sun / r_mars)  # Specific orbital energy at Mars
    
    return v_arrive

# Function to compute Mars periapsis speed relative to Mars
# Function to compute Mars periapsis speed relative to Mars
def mars_periapsis_speed(delta_v_outbound):
    """
    Computes the Mars periapsis speed relative to Mars, including the effect of Mars gravity
    when dropping from Mars' sphere of influence to periapsis.
    """
    delta_v_braking = total_delta_v - delta_v_outbound
    v_arrive_helio = mars_arrival_velocity(delta_v_outbound)
    #v_rel_to_mars = abs(v_arrive_helio - v_mars)
    v_rel_to_mars = v_arrive_helio - v_mars

    # Mars gravitational parameter (mu_mars)
    mu_mars = 4.282837e4  # km^3/s^2, gravitational parameter of Mars

    r_mars_surface = 3.396e3  # km, Mars radius
    r_periapsis = r_mars_surface  + 70 # periapsis distance from Mars center

    # Compute velocity at periapsis using vis-viva equation
    v_periapsis = np.sqrt(v_rel_to_mars**2 + 2 * mu_mars / r_periapsis)

    # Subtract braking delta-v to get final entry speed
    v_entry = v_periapsis - delta_v_braking

    return v_entry, delta_v_braking, v_rel_to_mars

# Objective: minimize periapsis speed constraint violation
def objective(delta_v_outbound):
    v_entry, delta_v_braking, v_rel_to_mars = mars_periapsis_speed(delta_v_outbound)
    print(f"delta_v_outbound: {delta_v_outbound:.2f}, v_entry: {v_entry:.2f}, delta_v_braking: {delta_v_braking:.2f}, v_rel_to_mars: {v_rel_to_mars:.2f}")
    if v_entry < 0:
        penalty = 1e6  # Large penalty for negative speeds
    else:
        penalty = (v_entry - max_periapsis_speed_mars) ** 2
    return penalty

# Perform optimization to find the optimal outbound delta-v
result = minimize_scalar(objective, bounds=(3, total_delta_v), method='bounded')

# Collect data for plotting
delta_v_range = np.linspace(3, total_delta_v, 100)  # 100 points between 0 and total_delta_v
entry_speeds = []
braking_dvs = []
v_arrives = []
for dv in delta_v_range:
    v_entry, delta_v_braking, v_arrive = mars_periapsis_speed(dv)
    entry_speeds.append(v_entry)
    braking_dvs.append(delta_v_braking)
    v_arrives.append(v_arrive)

# Extract optimization results
result_x = result.x
result_y, result_braking_dv, v_arrive = mars_periapsis_speed(result_x)

# Print the optimization results
print(f"Optimal outbound delta-v: {result_x:.2f} km/s")
print(f"Resulting Mars periapsis speed: {result_y:.2f} km/s")
print(f"Braking delta-v: {result_braking_dv:.2f} km/s")

# Print the range of delta-v and corresponding entry speeds and braking delta-vs
print("Delta-v range, entry speeds, and braking delta-vs:")
for dv, speed, braking_dv, v_arrive in zip(delta_v_range, entry_speeds, braking_dvs, v_arrives):
    print(f"Delta-v: {dv:.2f} km/s, Entry speed: {speed:.2f} km/s, Braking delta-v: {braking_dv:.2f} km/s, arrival {v_arrive:.2f}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(delta_v_range, entry_speeds, label="Mars Periapsis Speed")
plt.plot(delta_v_range, braking_dvs, label="Braking Delta-v")
plt.axvline(result_x, color='r', linestyle='--', label=f"Optimal Delta-v ({result_x:.2f} km/s)")
plt.axhline(max_periapsis_speed_mars, color='g', linestyle='--', label=f"Max Periapsis Speed ({max_periapsis_speed_mars:.2f} km/s)")
plt.xlabel("Outbound Delta-v (km/s)")
plt.ylabel("Speed (km/s)")
plt.title("Mars Periapsis Speed and Braking Delta-v vs Outbound Delta-v")
plt.legend()
plt.grid()
plt.show()