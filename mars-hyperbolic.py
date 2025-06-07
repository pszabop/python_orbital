import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# Constants
MU_SUN = 1.32712440018e11  # km^3/s^2, gravitational parameter of the Sun
MU_MARS = 4.282837e4       # km^3/s^2, gravitational parameter of Mars
R_EARTH = 1.496e8          # km, average distance from Sun
R_MARS = 2.279e8           # km, average distance from Sun
R_MARS_SURFACE = 3.396e3   # km, Mars radius
MAX_PERIAPSIS_SPEED_MARS = 7.5  # km/s
TOTAL_DELTA_V = 7          # km/s

def angular_momentum(r, v):
    """
    Computes angular momentum per unit mass.
    :param r: Distance from the central body (km)
    :param v: Orbital velocity (km/s)
    :return: Angular momentum per unit mass (km^2/s)
    """
    return r * v

def energy(mu, r, v):
    """
    Computes specific orbital energy.
    :param mu: Gravitational parameter (km^3/s^2)
    :param r: Distance from the central body (km)
    :param v: Orbital velocity (km/s)
    :return: Specific orbital energy (km^2/s^2)
    """
    return v**2 / 2 - mu / r

def mars_arrival_velocity(mu_sun, r_earth, r_mars, delta_v_outbound):
    """
    Computes the arrival velocity at Mars using conservation of energy.
    :param mu_sun: Gravitational parameter of the Sun (km^3/s^2)
    :param r_earth: Distance from the Sun to Earth (km)
    :param r_mars: Distance from the Sun to Mars (km)
    :param delta_v_outbound: Outbound delta-v from Earth (km/s)
    :return: Arrival velocity at Mars (km/s)
    """
    v_depart = np.sqrt(mu_sun / r_earth) + delta_v_outbound
    energy_depart = energy(mu_sun, r_earth, v_depart)
    v_arrive = np.sqrt(2 * (energy_depart + mu_sun / r_mars))
    return v_arrive

def mars_periapsis_speed(mu_mars, r_mars_surface, r_periapsis, v_rel_to_mars, delta_v_braking):
    """
    Computes the Mars periapsis speed using conservation of angular momentum and energy.
    :param mu_mars: Gravitational parameter of Mars (km^3/s^2)
    :param r_mars_surface: Radius of Mars (km)
    :param r_periapsis: Periapsis distance from Mars center (km)
    :param v_rel_to_mars: Relative velocity to Mars (km/s)
    :param delta_v_braking: Braking delta-v (km/s)
    :return: Entry speed at Mars periapsis (km/s)
    """
    # Compute angular momentum at Mars arrival
    h = angular_momentum(r_periapsis, v_rel_to_mars)

    # Compute periapsis velocity using angular momentum conservation
    v_periapsis = h / r_periapsis

    # Compute entry speed using conservation of energy
    energy_periapsis = energy(mu_mars, r_periapsis, v_periapsis)
    v_entry = np.sqrt(2 * (energy_periapsis + mu_mars / r_periapsis)) - delta_v_braking

    return v_entry

def objective(delta_v_outbound):
    """
    Objective function for optimization.
    :param delta_v_outbound: Outbound delta-v from Earth (km/s)
    :return: Penalty value for optimization
    """
    v_arrive = mars_arrival_velocity(MU_SUN, R_EARTH, R_MARS, delta_v_outbound)
    delta_v_braking = TOTAL_DELTA_V - delta_v_outbound
    v_rel_to_mars = v_arrive - np.sqrt(MU_SUN / R_MARS)
    r_periapsis = R_MARS_SURFACE + 70  # periapsis distance from Mars center
    v_entry = mars_periapsis_speed(MU_MARS, R_MARS_SURFACE, r_periapsis, v_rel_to_mars, delta_v_braking)
    if v_entry < 0:
        penalty = 1e6  # Large penalty for negative speeds
    else:
        penalty = (v_entry - MAX_PERIAPSIS_SPEED_MARS) ** 2
    return penalty

# Perform optimization to find the optimal outbound delta-v
result = minimize_scalar(objective, bounds=(3, TOTAL_DELTA_V), method='bounded')

# Collect data for plotting
delta_v_range = np.linspace(3, TOTAL_DELTA_V, 100)
entry_speeds = []
for dv in delta_v_range:
    v_arrive = mars_arrival_velocity(MU_SUN, R_EARTH, R_MARS, dv)
    delta_v_braking = TOTAL_DELTA_V - dv
    v_rel_to_mars = v_arrive - np.sqrt(MU_SUN / R_MARS)
    r_periapsis = R_MARS_SURFACE + 70
    v_entry = mars_periapsis_speed(MU_MARS, R_MARS_SURFACE, r_periapsis, v_rel_to_mars, delta_v_braking)
    entry_speeds.append(v_entry)

# Extract optimization results
result_x = result.x
v_arrive = mars_arrival_velocity(MU_SUN, R_EARTH, R_MARS, result_x)
delta_v_braking = TOTAL_DELTA_V - result_x
v_rel_to_mars = v_arrive - np.sqrt(MU_SUN / R_MARS)
r_periapsis = R_MARS_SURFACE + 70
result_y = mars_periapsis_speed(MU_MARS, R_MARS_SURFACE, r_periapsis, v_rel_to_mars, delta_v_braking)

# Print the optimization results
print(f"Optimal outbound delta-v: {result_x:.2f} km/s")
print(f"Resulting Mars periapsis speed: {result_y:.2f} km/s")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(delta_v_range, entry_speeds, label="Mars Periapsis Speed")
plt.axvline(result_x, color='r', linestyle='--', label=f"Optimal Delta-v ({result_x:.2f} km/s)")
plt.axhline(MAX_PERIAPSIS_SPEED_MARS, color='g', linestyle='--', label=f"Max Periapsis Speed ({MAX_PERIAPSIS_SPEED_MARS:.2f} km/s)")
plt.xlabel("Outbound Delta-v (km/s)")
plt.ylabel("Mars Periapsis Speed (km/s)")
plt.title("Mars Periapsis Speed vs Outbound Delta-v")
plt.legend()
plt.grid()
plt.show()