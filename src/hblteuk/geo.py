# geo.py

import kerrgeopy as kg
import numpy as np

def generate_frequencies(a, p, e, x, constants):
    upR = kg.frequencies.r_frequency(a, p, e, x, constants)
    upTh = kg.frequencies.theta_frequency(a, p, e, x, constants)
    upPh = kg.frequencies.phi_frequency(a, p, e, x, constants, upR, upTh)
    gamma = kg.frequencies.gamma(a, p, e, x, constants, upR, upTh)
    # pybhpt (the mode_frequency/omega validation reference) stores
    # Upsilon_theta UNSIGNED; kerrgeopy signs it by prograde/retrograde.
    # Match pybhpt here -- the signed value is recovered separately in
    # MiniGeo.__init__ for uth(). See dev/PHASE5B.md.
    return np.array((gamma, upR, abs(upTh), upPh))

class MiniGeo:
    def __init__(self, a, p, e, x):
        self.a = a
        self.p = p
        self.e = e
        self.x = x
        self.constants = kg.constants.constants_of_motion(a, p, e, x)
        self.En, self.Lz, self.Qc = self.constants
        self.radial_roots = kg.constants.stable_radial_roots(a, p, e, x, self.constants)
        self.mino_frequencies = generate_frequencies(a, p, e, x, self.constants)
        self.frequencies = self.mino_frequencies[1:]/self.mino_frequencies[0]
        self.radial_solutions = kg.stable.radial_solutions(a, self.constants, self.radial_roots)
        self.polar_roots = kg.stable.stable_polar_roots(a, p, e, x, self.constants)
        self.polar_solutions = kg.stable.polar_solutions(a, self.constants, self.polar_roots)
        # sign(Upsilon_theta): kerrgeopy signs it by prograde/retrograde but
        # generate_frequencies above stores |Upsilon_theta| to match pybhpt;
        # uth() needs the true signed value (cheap formula eval, not a solve).
        self._uth_sign = float(np.sign(
            kg.frequencies.theta_frequency(a, p, e, x, self.constants)))

    @property
    def orbitalenergy(self):
        return self.En
    
    @property
    def orbitalangularmomentum(self):
        return self.Lz
    
    @property
    def blackholespin(self):
        return self.a

    def r(self, qr):
        return self.radial_solutions[0](qr)

    def tr(self, qr):
        return self.radial_solutions[1](qr)

    def phir(self, qr):
        return self.radial_solutions[2](qr)

    def ur(self, qr, r = None):
        if r is None:
            r = self.r(qr)
        Delta = r**2 - 2.*r + self.a**2
        return np.sign(np.sin(qr))*np.sqrt(np.abs((self.En*(r**2 + self.a**2) - self.a*self.Lz)**2 - Delta*(r**2 + (self.Lz - self.a*self.En)**2 + self.Qc)))
    
    def th(self, qth):
        return self.polar_solutions[0](qth)

    def tth(self, qth):
        return self.polar_solutions[1](qth)

    def phith(self, qth):
        return self.polar_solutions[2](qth)

    def uth(self, qth, z = None):
        # z = cos(theta); signed d(theta)/d(lambda). Positive on qth in
        # (0, pi) when Upsilon_theta > 0 (prograde), negative when
        # retrograde -- see dev/PHASE5B.md for the sign convention.
        if z is None:
            z = np.cos(self.th(qth))
        Vth = self.Qc - z**2 * (self.a**2 * (1 - self.En**2)
                                + self.Lz**2 / (1 - z**2))
        return self._uth_sign * np.sign(np.sin(qth)) * np.sqrt(np.abs(Vth))

    def r_velocity(self, qr):
        rr = self.r(qr)
        return self.ur(qr, rr)/rr**2

    def mode_frequency(self, m, k, n):
        return np.dot(self.frequencies, (n, k, m))