# geo.py

import kerrgeopy as kg
import numpy as np

def generate_frequencies(a, p, e, x, constants):
    upR = kg.frequencies.r_frequency(a, p, e, x, constants)
    upTh = kg.frequencies.theta_frequency(a, p, e, x, constants)
    upPh = kg.frequencies.phi_frequency(a, p, e, x, constants, upR, upTh)
    gamma = kg.frequencies.gamma(a, p, e, x, constants, upR, upTh)
    return np.array((gamma, upR, upTh, upPh))

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
    
    def r_velocity(self, qr):
        rr = self.r(qr)
        return self.ur(qr, rr)/rr**2

    def mode_frequency(self, m, k, n):
        return np.dot(self.frequencies, (n, k, m))