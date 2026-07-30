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
        # Recorded handedness of Upsilon_theta (kerrgeopy signs it by
        # prograde/retrograde; generate_frequencies above stores |Upsilon_theta|
        # to match pybhpt). NOT applied to uth() -- see uth() for why.
        self._ups_th_sign = float(np.sign(
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
        r"""Polar velocity in the |Upsilon_theta| convention: positive on
        qth in (0, pi) for BOTH prograde and retrograde orbits.

        This is |Upsilon_theta| * dtheta/dqth, not the signed dtheta/dlambda.
        The distinction matters and was a bug: mino_frequencies[2] stores
        |Upsilon_theta| (pybhpt's mode_frequency convention), so qth is
        parameterized by the unsigned frequency, and the polar velocity has to
        be in that same convention or the source integral mixes handedness.
        Applying sign(Upsilon_theta) here while using |Upsilon_theta| for omega
        left retrograde inclined amplitudes wrong by factors of 0.1-58x versus
        pybhpt; dropping it makes them agree exactly (ratio 1.0000, phase 0).
        Prograde was unaffected because sign(Upsilon_theta) = +1 there, which
        is why this hid until a retrograde orbit was validated.

        The signed handedness is still available as _ups_th_sign for anyone
        who needs true dtheta/dlambda = _ups_th_sign * uth().
        """
        if z is None:
            z = np.cos(self.th(qth))
        Vth = self.Qc - z**2 * (self.a**2 * (1 - self.En**2)
                                + self.Lz**2 / (1 - z**2))
        return np.sign(np.sin(qth)) * np.sqrt(np.abs(Vth))

    def r_velocity(self, qr):
        rr = self.r(qr)
        return self.ur(qr, rr)/rr**2

    def mode_frequency(self, m, k, n):
        return np.dot(self.frequencies, (n, k, m))
    def carter_frequencies(self, nsamples = 2**12):
        # (cR, cTh, cPh), pybhpt conventions (geo.cpp kerr_geo_carter_frequencies
        # with the chi-quadrature reduced to a uniform-qth Mino average; the
        # Upsilon_theta factors cancel -- derivation + verification in
        # dev/PHASE5C_PLAN.md). Uses |Upsilon_theta| like mino_frequencies.
        if not hasattr(self, "_carter_frequencies"):
            a, En, Lz = self.a, self.En, self.Lz
            qth = 2 * np.pi * np.arange(nsamples) / nsamples
            zp = np.cos(self.th(qth))
            freqFactor = a * Lz - a * a * En - np.mean(
                a * Lz - a * a * En * (1 - zp**2))
            mFactor = Lz - a * En - np.mean(Lz / (1 - zp**2) - a * En)
            gamma, upR, upTh, upPh = self.mino_frequencies
            self._carter_frequencies = np.array(
                [freqFactor * upR / gamma,
                 freqFactor * upTh / gamma + upTh,
                 freqFactor * upPh / gamma - mFactor])
        return self._carter_frequencies
