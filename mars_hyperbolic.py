import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from radial_coordinates import VelocityVector
from radial_coordinates import radial_vector_delta_two_orbits


# Constants
MU_SUN = 1.32712440018e11  # km^3/s^2, gravitational parameter of the Sun
MU_MARS = 4.282837e4       # km^3/s^2, gravitational parameter of Mars
MU_EARTH = 3.986004418e5   # km^3/s^2, gravitational parameter of Earth

R_EARTH = 1.496e8          # km, average distance from Sun
R_MARS = 2.279e8           # km, average distance from Sun

EARTH_VELOCITY = VelocityVector(np.sqrt(MU_SUN / R_EARTH), 0)  # Earth's orbital speed around the Sun
MARS_VELOCITY = VelocityVector(np.sqrt(MU_SUN / R_MARS), 0)    # Mars's orbital speed around the Sun

R_MARS_SURFACE = 3.396e3    # km, Mars radius
R_MARS_SOI = 5.6e5          # km, Mars Sphere of Influence (SOI)
R_EARTH_SURFACE = 6.371e3   # km, Earth radius
R_EARTH_SOI = 9.2e5         # km, Earth Sphere of Influence (SOI)

# some hard coded parameters for our simulation
MAX_PERIAPSIS_SPEED_MARS = 7.5  # km/s
TOTAL_DELTA_V = 7.5             # km/s, aka Starship Delta-V budget
EARTH_SOI_DEPARTURE_ANGLE = 0.0


# conic Earth system to SOI
delta_v = TOTAL_DELTA_V / 1.72
braking_delta_v = TOTAL_DELTA_V - delta_v  # remaining delta-v for braking at Mars
v_initial = VelocityVector(np.sqrt(MU_EARTH / R_EARTH_SURFACE) + delta_v, 0.0)  # Earth's orbital velocity + min delta-v all tangential
print(f"Initial velocity at Earth: {v_initial:.2f} km/s")
v_earth_soi = radial_vector_delta_two_orbits(MU_EARTH, R_EARTH_SURFACE, R_EARTH_SOI, v_initial)
print(f"Earth SOI velocity: {v_earth_soi:.2f} km/s")

# conic Heliocentric system to Mars SOI
heliocentric_v_initial = EARTH_VELOCITY + VelocityVector(v_earth_soi.magnitude() * np.cos(EARTH_SOI_DEPARTURE_ANGLE), v_earth_soi.magnitude() * np.sin(EARTH_SOI_DEPARTURE_ANGLE))
print(f"Heliocentric velocity at earth SOI: {heliocentric_v_initial:.2f} km/s")
v_mars_soi = radial_vector_delta_two_orbits(MU_SUN, R_EARTH, R_MARS, heliocentric_v_initial)
print(f"Mars SOI velocity: {v_mars_soi:.2f} km/s")
relative_mars_velocity_heliocentric = v_mars_soi - MARS_VELOCITY
print(f"Heliocentric Relative velocity to Mars at SOI: {relative_mars_velocity_heliocentric:.2f} km/s")

# conic Mars SOI to periapsis
# naive approach just take the magnitude of the heliocentric relative velocity at Mars as radial in the negative direction
mars_soi_relative_velocity = VelocityVector(0, - 1 * relative_mars_velocity_heliocentric.magnitude())
print(f"Relative velocity at Mars SOI: {mars_soi_relative_velocity:.2f} km/s")
mars_periapses_velocity = radial_vector_delta_two_orbits(MU_MARS, R_MARS_SOI, R_MARS_SURFACE, mars_soi_relative_velocity)
print(f"Mars periapsis velocity prior to braking: {mars_periapses_velocity:.2f} km/s")
mars_final_periapses_velocity = mars_periapses_velocity + VelocityVector(0, -braking_delta_v)  # braking at periapsis
print(f"Mars periapsis velocity after braking: {mars_final_periapses_velocity:.2f} km/s")



"""
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
"""