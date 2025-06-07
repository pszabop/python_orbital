import numpy as np
from scipy.optimize import fsolve

class ConicOrbit:
    def __init__(self, mu, r_vec, v_vec):
        self.mu = mu
        self.r = np.array(r_vec)
        self.v = np.array(v_vec)
        self.h = np.cross(self.r, self.v)
        self.energy = np.linalg.norm(self.v)**2 / 2 - mu / np.linalg.norm(self.r)

    def __repr__(self):
        return (f"ConicOrbit(\n"
                f"  mu={self.mu},\n"
                f"  r={self.r.tolist()},\n"
                f"  v={self.v.tolist()},\n"
                f"  h={self.h.tolist()},\n"
                f"  energy={self.energy:.6f}\n)")

    def __str__(self):
        return (f"ConicOrbit:\n"
                f"  r = [{self.r[0]:.3f}, {self.r[1]:.3f}, {self.r[2]:.3f}]\n"
                f"  v = [{self.v[0]:.3f}, {self.v[1]:.3f}, {self.v[2]:.3f}]\n"
                f"  h = [{self.h[0]:.3f}, {self.h[1]:.3f}, {self.h[2]:.3f}]\n"
                f"  energy = {self.energy:.6f}")

    def __format__(self, format_spec):
        return str(self)

    def propagate_to(self, dt):
        r_mag = np.linalg.norm(self.r)
        v_mag = np.linalg.norm(self.v)
        omega = v_mag / r_mag
        theta = omega * dt
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        r_new = rot_matrix @ self.r
        v_new = rot_matrix @ self.v
        return r_new, v_new

    def rtn_basis(self, r_vec=None, v_vec=None):
        r = np.array(r_vec) if r_vec is not None else self.r
        v = np.array(v_vec) if v_vec is not None else self.v
        R = r / np.linalg.norm(r)
        N = np.cross(r, v)
        N /= np.linalg.norm(N)
        T = np.cross(N, R)
        return R, T, N

    def apply_delta_v_rtn(self, dv_rtn):
        R, T, N = self.rtn_basis()
        dv_vec = dv_rtn[0] * R + dv_rtn[1] * T + dv_rtn[2] * N
        self.v += dv_vec
        self.h = np.cross(self.r, self.v)
        self.energy = np.linalg.norm(self.v)**2 / 2 - self.mu / np.linalg.norm(self.r)


class ConicTransfer:
    def __init__(self, mu_primary):
        self.mu_primary = mu_primary

    def planar_rtn_transfer(self, r_start, v_start, r_end, dt):
        orb = ConicOrbit(self.mu_primary, r_start, v_start)

        # Propagate starting orbit
        r_prop, v_prop = orb.propagate_to(dt)

        # Estimate required final velocity for conic transfer to r_end
        # For now assume coplanar, tangential burn at start
        delta_r = np.linalg.norm(r_end) - np.linalg.norm(r_start)
        semi_major_axis = (np.linalg.norm(r_start) + np.linalg.norm(r_end)) / 2
        v_start_mag = np.sqrt(self.mu_primary * (2/np.linalg.norm(r_start) - 1/semi_major_axis))

        # Required delta-V
        dv_vec = (v_start_mag - np.linalg.norm(v_start)) * (v_start / np.linalg.norm(v_start))

        # Convert delta-V to RTN
        R, T, N = orb.rtn_basis()
        dv_rtn = [np.dot(dv_vec, axis) for axis in (R, T, N)]

        print("dv_rtn:", [f"{x:.3f}" for x in dv_rtn])

        return dv_rtn, r_prop, v_prop


# ==============================
# Unit Tests
# ==============================

def test_simple_plane_change():
    mu_earth = 398600.4418
    r_vec = np.array([6778.0, 0, 0])  # 200 km altitude
    v_vec = np.array([0, 7.73, 0])

    orb = ConicOrbit(mu_earth, r_vec, v_vec)
    R, T, N = orb.rtn_basis()

    # Apply 0.5 km/s out of plane delta-v
    orb.apply_delta_v_rtn([0, 0, 0.5])

    # Check inclination change by comparing angular momentum
    h_before = np.cross(r_vec, v_vec)
    h_after = np.cross(orb.r, orb.v)

    assert not np.allclose(h_before, h_after), "Angular momentum should change due to normal delta-v"
    assert abs(np.dot(h_after, N)) < np.linalg.norm(h_after), "h vector should have tilted"

def test_radial_thrust():
    mu_earth = 398600.4418
    r_vec = np.array([7000.0, 0, 0])
    v_vec = np.array([0.0, 7.5, 0.0])

    orb = ConicOrbit(mu_earth, r_vec, v_vec)
    energy_before = orb.energy
    orb.apply_delta_v_rtn([0.5, 0.0, 0.0])  # Radial boost
    energy_after = orb.energy

    assert energy_after > energy_before, "Radial thrust should increase energy"

def test_propagate():
    mu_earth = 398600.4418
    r_vec = np.array([7000.0, 0.0, 0.0])
    v_vec = np.array([0.0, 7.546, 0.0])  # Circular orbit

    orb = ConicOrbit(mu_earth, r_vec, v_vec)
    r2, v2 = orb.propagate_to(np.pi * 7000 / 7.546)  # Half orbit period

    assert np.allclose(np.linalg.norm(r2), np.linalg.norm(r_vec), atol=1.0), "Radius should stay roughly constant in circular"
    assert r2[0] < 0, "Should be on opposite side of orbit"

def test_planar_rtn_transfer():
    mu_earth = 398600.4418
    r_start = np.array([6678.0, 0, 0])
    v_start = np.array([0, 7.8, 0])
    r_end = np.array([384400.0, 0, 0])  # Approx Moon distance
    dt = 3600 * 18  # 18 hours transfer

    transfer = ConicTransfer(mu_earth)
    dv_rtn, r_final, v_final = transfer.planar_rtn_transfer(r_start, v_start, r_end, dt)

    assert np.linalg.norm(dv_rtn) > 0.01, "Delta-V should be required for transfer"
    assert r_final is not None and v_final is not None, "Output should be valid"

if __name__ == "__main__":
    test_simple_plane_change()
    test_radial_thrust()
    test_propagate()
    test_planar_rtn_transfer()
    print("All tests passed.")

