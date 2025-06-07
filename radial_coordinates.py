import numpy as np
from scipy.optimize import minimize_scalar
import unittest

class VelocityVector:
    """
    Represents a velocity vector with tangential and radial components.
    Provides methods to compute magnitude and angle.
    """
    def __init__(self, tangential, radial):
        """
        Initializes the velocity vector.
        :param tangential: Tangential velocity (km/s)
        :param radial: Radial velocity (km/s)
        """
        self.tangential = tangential
        self.radial = radial

    def magnitude(self):
        """
        Computes the magnitude of the velocity vector.
        :return: Magnitude (km/s)
        """
        return np.sqrt(self.tangential**2 + self.radial**2)

    def angle(self):
        """
        Computes the angle of the velocity vector relative to the radial direction.
        :return: Angle in radians
        """
        return np.arctan2(self.tangential, self.radial)

    def __add__(self, other):
        """
        Overloads the + operator to add two velocity vectors.
        :param other: Another VelocityVector
        :return: A new VelocityVector representing the sum
        """
        return VelocityVector(self.tangential + other.tangential, self.radial + other.radial)

    def __sub__(self, other):
        """
        Overloads the - operator to subtract two velocity vectors.
        :param other: Another VelocityVector
        :return: A new VelocityVector representing the difference
        """
        return VelocityVector(self.tangential - other.tangential, self.radial - other.radial)

    def __format__(self, format_spec):
        """
        Formats the velocity vector for printing.
        :param format_spec: Format specification (e.g., '.2f' for 2 decimal places)
        :return: Formatted string representation of the velocity vector
        """
        tangential = format(self.tangential, format_spec)
        radial = format(self.radial, format_spec)
        magnitude = format(self.magnitude(), format_spec)
        angle_degrees = format(np.degrees(self.angle()), format_spec)
        return f"Tangential: {tangential} km/s, Radial: {radial} km/s, Magnitude: {magnitude} km/s, Angle: {angle_degrees}°"

def radial_vector_delta_two_orbits(mu, r_initial, r_final, v_initial):
    """
    Calculates the new velocity vector when traveling between two orbits
    using conservation of energy and angular momentum.
    
    :param mu: Gravitational parameter of the central body (km^3/s^2)
    :param r_initial: Initial orbital radius (km)
    :param r_final: Final orbital radius (km)
    :param v_initial: Initial velocity vector (VelocityVector)
    :return: Final velocity vector (VelocityVector)
    """
    # Compute initial angular momentum per unit mass
    h_initial = r_initial * v_initial.tangential

    # Compute final tangential velocity using conservation of angular momentum
    v_tangential_final = h_initial / r_final

    # Compute initial specific orbital energy
    energy_initial = v_initial.magnitude()**2 / 2 - mu / r_initial

    # Compute final radial velocity using conservation of energy
    v_final_magnitude = np.sqrt(2 * (energy_initial + mu / r_final))
    v_radial_final = np.sqrt(v_final_magnitude**2 - v_tangential_final**2)

    return VelocityVector(v_tangential_final, v_radial_final)


def time_to_traverse_orbits(mu, r_initial, r_final, v_initial, tolerance=3600):
    """
    Calculates the time needed to traverse between two orbits by breaking the radii into smaller increments
    and summing the time for each segment until the error converges within the specified tolerance.
    
    :param mu: Gravitational parameter of the central body (km^3/s^2)
    :param r_initial: Initial orbital radius (km)
    :param r_final: Final orbital radius (km)
    :param v_initial: Initial velocity vector (VelocityVector)
    :param tolerance: Tolerance for the time error in seconds (default: 3600 seconds)
    :return: Total time to traverse the orbits (seconds)
    """
    def segment_time(mu, r1, r2, v1):
        """
        Calculates the time for a single segment between two radii.
        :param mu: Gravitational parameter of the central body (km^3/s^2)
        :param r1: Starting radius of the segment (km)
        :param r2: Ending radius of the segment (km)
        :param v1: Velocity vector at the starting radius (VelocityVector)
        :return: Time for the segment (seconds)
        """
        v2 = radial_vector_delta_two_orbits(mu, r1, r2, v1)  # Compute velocity at the end of the segment
        avg_radial_velocity = (v1.radial + v2.radial) / 2  # Average radial velocity for the segment
        distance = abs(r2 - r1)  # Distance between the radii
        return distance / avg_radial_velocity  # Time = Distance / Average Radial Velocity

    # Initialize variables
    total_time = 0
    current_radius = r_initial
    current_velocity = v_initial
    step_size = abs(r_final - r_initial) / 10  # Start with 10 segments

    while step_size > tolerance:
        next_radius = current_radius + step_size if r_final > r_initial else current_radius - step_size
        segment_time_value = segment_time(mu, current_radius, next_radius, current_velocity)
        total_time += segment_time_value

        # Update for the next iteration
        current_radius = next_radius
        current_velocity = radial_vector_delta_two_orbits(mu, current_radius, next_radius, current_velocity)

        # Reduce step size for finer increments
        step_size /= 2

    return total_time


def find_min_departure_velocity(mu, r_initial, r_final, tolerance=1e-6):
    """
    Finds the minimum departure tangential velocity that results in a final radial velocity
    slightly greater than zero (e.g., 0.001).
    
    :param mu: Gravitational parameter of the central body (km^3/s^2)
    :param r_initial: Initial orbital radius (km)
    :param r_final: Final orbital radius (km)
    :param tolerance: Tolerance for the radial velocity (km/s)
    :return: Minimum tangential departure velocity (km/s)
    """
    def objective(v_tangential_initial):
        #print(f"Trying tangential velocity: {v_tangential_initial:.6f} km/s")
        try:
            # Create the initial velocity vector
            v_initial = VelocityVector(v_tangential_initial, 0.0)
            
            # Compute the final velocity vector
            v_final = radial_vector_delta_two_orbits(mu, r_initial, r_final, v_initial)
            
            # Return the radial velocity (we want it slightly greater than zero)
            #print(f"Final radial velocity: {v_final.radial:.6f} km/s")
            return abs(v_final.radial - tolerance)
        except ValueError:
            #print("Error in computation, returning large value for optimization.")
            return 1

    # Use scipy.optimize to find the minimum tangential velocity
    result = minimize_scalar(objective, bounds=(np.sqrt(mu / r_initial), np.sqrt(mu / r_initial) + 100), method='bounded')

    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed to find a solution.")

def approach_velocity(v1, v2):
    """
    Calculates the approach velocity between two objects based on their velocity vectors.
    
    :param v1: Velocity vector of object 1 (VelocityVector)
    :param v2: Velocity vector of object 2 (VelocityVector)
    :return: Approach velocity vector (VelocityVector)
    """
    # Compute the relative tangential and radial velocities
    return v1 - v2

########################################################################################################################
# Unit tests 
########################################################################################################################
class TestCalculateOrbitalVelocity(unittest.TestCase):
    def test_orbital_velocity_earth_to_mars(self):
        mu_sun = 1.32712440018e11  # Gravitational parameter of the Sun (km^3/s^2)
        r_earth = 1.496e8          # Earth's orbital radius around the Sun (km)
        r_mars = 2.279e8           # Mars's orbital radius around the Sun (km)
        v_initial = VelocityVector(np.sqrt(mu_sun / r_earth) + 2.9435, 0.0)  # Earth's orbital velocity + min delta-v

        v_final = radial_vector_delta_two_orbits(mu_sun, r_earth, r_mars, v_initial)

        # Assert the results
        self.assertAlmostEqual(v_final.radial, 0.037, places=2)  # Radial velocity should remain close to 0
        self.assertAlmostEqual(v_final.tangential, 21.48, places=2)  # Mars's orbital velocity

    def test_orbital_velocity_earth_leo_to_moon(self):
        mu_earth = 3.986004418e5  # Gravitational parameter of Earth (km^3/s^2)
        r_leo = 6.7e3             # Radius of Low Earth Orbit (km)
        r_moon = 3.844e5          # Radius of Moon's orbit around Earth (km)
        v_initial = VelocityVector(np.sqrt(mu_earth / r_leo) + 3.10106, 0.0)  # Orbital velocity in LEO

        v_final = radial_vector_delta_two_orbits(mu_earth, r_leo, r_moon, v_initial)

        # Assert the results
        self.assertAlmostEqual(v_final.radial, 0.032251521309588256, places=1)      # Radial velocity should remain close to 0
        self.assertAlmostEqual(v_final.tangential, 0.18848900207594632 , places=2)  # velocity at Moon's orbit

    def test_min_departure_velocity_earth_to_moon(self):
        mu_earth = 3.986004418e5  # Gravitational parameter of Earth (km^3/s^2)
        r_leo = 6.7e3             # Radius of Low Earth Orbit (km)
        r_moon = 3.844e5          # Radius of Moon's orbit around Earth (km)
        tolerance = 0.01         # Desired final radial velocity (km/s)

        # Find the minimum departure tangential velocity
        min_departure_velocity = find_min_departure_velocity(mu_earth, r_leo, r_moon, tolerance)
        print(f"Minimum departure velocity: {min_departure_velocity:.6f} km/s")

        # Assert the result
        self.assertAlmostEqual(min_departure_velocity, np.sqrt(mu_earth / r_leo) + 3.10106, places=2)

        # calculate given the result
        v_initial = VelocityVector(min_departure_velocity, 0.0)
        v_final = radial_vector_delta_two_orbits(mu_earth, r_leo, r_moon, v_initial)
        print(f"Final velocity vector: {v_final}")
        self.assertAlmostEqual(v_final.radial, 0, places=1)



class TestApproachVelocity(unittest.TestCase):
    def test_approach_velocity(self):
        v1 = VelocityVector(21.48, 0.037)  # Velocity vector of object 1
        v2 = VelocityVector(24.07, 0.0)    # Velocity vector of object 2

        v_relative = approach_velocity(v1, v2)

        # Assert the tangential and radial components
        self.assertAlmostEqual(v_relative.tangential, -2.59, places=2)
        self.assertAlmostEqual(v_relative.radial, 0.037, places=2)

        # Assert the magnitude and angle
        self.assertAlmostEqual(v_relative.magnitude(), 2.59, places=2)
        angle = np.arctan2(-2.59, 0.037)
        #print(f"Angle: {np.degrees(angle)} degrees")
        self.assertAlmostEqual(v_relative.angle(), angle, places=2)


class TestTimeToTraverseOrbits(unittest.TestCase):
    def test_time_to_traverse_orbits_earth_to_moon(self):
        mu_earth = 3.986004418e5  # Gravitational parameter of Earth (km^3/s^2)
        r_leo = 6.7e3             # Radius of Low Earth Orbit (km)
        r_moon = 3.844e5          # Radius of Moon's orbit around Earth (km)
        tolerance = 0.01         # Desired final radial velocity (km/s)

        v_initial_tangential = find_min_departure_velocity(mu_earth, r_leo, r_moon, tolerance) + 0.01  # Add a small delta to ensure we are above the minimum velocity
        v_initial = VelocityVector(v_initial_tangential, 0.0)  # Initial velocity vector

        # Calculate the time to traverse from LEO to Moon's orbit
        time = time_to_traverse_orbits(mu_earth, r_leo, r_moon, v_initial, tolerance=3600)

        # Assert the result (example expected value, adjust based on actual calculation)
        self.assertAlmostEqual(time, 31986.96887181064, places=0)  # Expected time in seconds

# Run the tests if the file is executed directly
if __name__ == "__main__":
    unittest.main()