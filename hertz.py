import numpy as np
import pickle
import collocode
from collocode import MultiGridMultiDomainChebyshev, MultiGridChebyshev
import source

from mode import TeukolskyPointParticleModeGridGenerator, TeukolskyPointParticleModeGenerator, PointParticleModeGrid
from teuk import teukolsky_starobinsky_complex_constant
from utils import chop_matrix
from swsh import MultiModeSpinWeightedSpherical

class HertzPointParticleModeGridGenerator(TeukolskyPointParticleModeGridGenerator):
    def solve(self, s, lrange, gauge = 'IRG', reduce = False, **reduce_kwargs):
        if (s != -2):
            raise ValueError(f'Spin-weight {s} is not supported yet. Must be -2.')
        if gauge != 'IRG':
            raise ValueError(f'Gauge {gauge} is not supported yet. Must be IRG.')
        
        Phi = PointParticleModeGrid(s, lrange, [0, 0], [0, 0], self.source)
        Phi.basis = "spheroidal"

        RslmGrid, domains, SslmGrid, amplitudes, eigenvalues = self.solve_teukolsky_mode_data(Phi, reduce = reduce, **reduce_kwargs)
        domainsIn = domains["In"]
        domainsUp = domains["Up"]

        la_chandra = eigenvalues + Phi.s*(Phi.s+1)
        ells = Phi.modes[:, 0]
        emms = Phi.modes[:, 1]
        teuk_starob_constant = teukolsky_starobinsky_complex_constant(ells, emms, Phi.source.a, Phi.frequencies, la_chandra)
        hertz_rescale = (-1.)**(ells + emms)*8./teuk_starob_constant
        amplitudes["Up"] *= hertz_rescale
        amplitudes["In"] *= hertz_rescale

        RslmGrid["In"] = np.ascontiguousarray(amplitudes["In"][:, None, None]*RslmGrid["In"])
        RslmGrid["Up"] = np.ascontiguousarray(amplitudes["Up"][:, None, None]*RslmGrid["Up"])
        if len(RslmGrid["In"].shape) > 2:
            RslmIn = MultiGridMultiDomainChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridMultiDomainChebyshev(RslmGrid["Up"], domainsUp)
        else:
            RslmIn = MultiGridChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridChebyshev(RslmGrid["Up"], domainsUp)

        Phi.Rslm = {"In": RslmIn, "Up": RslmUp}
        Phi.coupling = chop_matrix(SslmGrid, buffer = lrange[-1] - 2, tol=1e-50)
        m_arr = Phi.m_mode_arr
        coeffs_arr = Phi.group_by_m_modes(Phi.coupling)
        gamma_arr = Phi.group_by_m_modes(Phi.source.a*Phi.frequencies)
        Phi.Sslm = MultiModeSpinWeightedSpherical(s, m_arr, gamma_arr, coeffs_arr, tol = None)
        Phi.eigenvalues = eigenvalues
        Phi.amplitudes = amplitudes
        Phi.domains["In"] = domainsIn
        Phi.domains["Up"] = domainsUp

        return Phi
    
    def hertz_from_weyl(self, psi, gauge = 'IRG'):
        if (psi.s != -2):
            raise ValueError(f'Spin-weight {psi.s} is not supported yet. Must be -2.')
        if gauge != 'IRG':
            raise ValueError(f'Gauge {gauge} is not supported yet. Must be IRG.')
        
        Phi = psi.copy()
        la_chandra = Phi.eigenvalues + Phi.s*(Phi.s+1)
        ells = Phi.modes[:, 0]
        emms = Phi.modes[:, 1]
        teuk_starob_constant = teukolsky_starobinsky_complex_constant(ells, emms, Phi.source.a, Phi.frequencies, la_chandra)
        hertz_rescale = (-1.)**(ells + emms)*8./teuk_starob_constant
        Phi.amplitude["Up"] *= hertz_rescale
        Phi.amplitude["In"] *= hertz_rescale
        Phi.Rslm["Up"].coeffs *= hertz_rescale
        Phi.Rslm["In"].coeffs *= hertz_rescale

        return Phi

    def __call__(self, s, lrange, gauge = 'IRG', reduce = False, **reduce_kwargs):
        return self.solve(s, lrange, gauge = gauge, reduce = reduce, **reduce_kwargs)

# class ChebyshevModeGrid:
#     def __init__(self, lmax, solver):
#         self.maxl = lmax
#         self.modeCount = 0
#         self.lmodes = []
#         self.mmodes = []
#         self.modes = {}
#         for l in range(0, self.maxl + 1):
#             for m in range(-l, l + 1):
#                 self.lmodes.append(l)
#                 self.mmodes.append(m)
#                 self.modes[(l,m)] = np.zeros(ptNum, dtype=np.complex128)
        
#         self.ptnum = ptNum
#         self.lmodes = np.array(self.lmodes)
#         self.mmodes = np.array(self.mmodes)

#     def set(self, l, m, val):
#         self.modes[(l, m)] = val

#     def add(self, l, m, val):
#         if (l, m) in self.modes.keys():
#             self.modes[(l, m)] += val
#         else:
#             self.set(l, m, val)
    
#     def get(self, l, m):
#         if (l, m) in self.modes.keys():
#             return self.modes[(l, m)]
#         else:
#             return np.zeros(self.ptnum, dtype=np.complex128)

#     def __call__(self, l, m):
#         return self.get(l, m)
    
class HertzCircularGrid:
    def __init__(self, orbit, lmax, domain):
        self.maxl = lmax
        self.geo = orbit
        self.r0 = self.geo.semilatusrectum
        self.gauge = "IRG"
        self.inner = rpts[rpts <= self.r0]
        self.outer = rpts[rpts >= self.r0]
        self.pts = np.concatenate((self.inner, self.outer))
        self.dnpsi = [
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0]),
            ModeGrid(lmax, self.pts.shape[0])
        ]
        self.maxasymp = 8 # for 0 <= n <= 7 asymptotic coupling coefficients
        self.asymp = ModeGrid(lmax, self.maxasymp)
        self.lmodes = np.arange(2, lmax + 1)
        self.mmodes = np.arange(-lmax, lmax + 1)
        self.filename = ".hertz_grid_a0_" + str(self.geo.blackholespin) + "_r0_" + str(self.r0) + "_gauge_" + str(self.gauge) + "_ptsNum_" + str(self.pts.shape[0]) + "_modeNum_" + str(self.lmodes.shape[0])

    def sanity_check(self, RInner, ROuter, hertz):
        r0UpIter = 0
        r0InIter = self.inner.shape[0] - 1
        Rup = ROuter.solution('Up', r0UpIter)
        Rin = RInner.solution('In', r0InIter)
        assert (np.abs(1. - Rin/hertz.homogeneousradialsolution('In', 0)) < 1.e-8), "Horizon-side Hertz field values of {} and {} do not match for mode {}".format(Rin, hertz.homogeneousradialsolution('In', 0), (RInner.spinweight, RInner.spheroidalmode, RInner.azimuthalmode, RInner.frequency))
        assert (np.abs(1. - Rup/hertz.homogeneousradialsolution('Up', 0)) < 1.e-8), "Infinity-side Hertz field values of {} and {} do not match for mode {}".format(Rup, hertz.homogeneousradialsolution('Up', 0),  (ROuter.spinweight, ROuter.spheroidalmode, ROuter.azimuthalmode, ROuter.frequency))

    def solve(self):
        if self.gauge in ["IRG", "ASAAB0", "SAAB0"]:
            s = -2
        else:
            s = 2
        for m in self.mmodes:
            lmin = np.max([2, abs(m)])
            lmax = self.maxl + 4 + int(20*self.geo.blackholespin)
            jmodes = np.arange(lmin, lmax + 1)
            for j in jmodes:
                teuk = TeukolskyMode(s, j, m, 0, 0, self.geo)
                teuk.solve(self.geo)
                hertz = HertzMode(teuk, self.gauge)
                hertz.solve()
                PsiUpJ = hertz.amplitude('Up')
                PsiInJ = hertz.amplitude('In')

                if self.outer.shape[0] > 0:
                    asymp = self.outer[self.outer > 1e5]
                    if asymp.shape[0] > 0:
                        nonasymp = self.outer[~(self.outer > 1e5)]
                        ROuter0 = RadialTeukolsky(s, j, m, teuk.blackholespin, teuk.frequency, nonasymp)
                        ROuter1 = RadialTeukolsky(s, j, m, teuk.blackholespin, teuk.frequency, asymp)
                        ROuter0.solve(bc='Up')
                        ROuter1.solve(method='ASYM', bc='Up')
                        d0Rup = np.concatenate((ROuter0.radialsolutions('Up'), ROuter1.radialsolutions('Up')))
                        d1Rup = np.concatenate((ROuter0.radialderivatives('Up'), ROuter1.radialderivatives('Up')))
                        d2Rup = np.concatenate((ROuter0.radialderivatives2('Up'), ROuter1.radialderivatives2('Up')))
                        d3Rup = np.concatenate((third_derivative(ROuter0, 'Up'), third_derivative(ROuter1, 'Up')))
                        d4Rup = np.concatenate((fourth_derivative(ROuter0, 'Up'), fourth_derivative(ROuter1, 'Up')))
                        d5Rup = np.concatenate((fifth_derivative(ROuter0, 'Up'), fifth_derivative(ROuter1, 'Up')))
                        d6Rup = np.concatenate((sixth_derivative(ROuter0, 'Up'), sixth_derivative(ROuter1, 'Up')))
                    else:
                        ROuter = RadialTeukolsky(s, j, m, teuk.blackholespin, teuk.frequency, self.outer)
                        ROuter.solve(bc='Up')
                        d0Rup = ROuter.radialsolutions('Up')
                        d1Rup = ROuter.radialderivatives('Up')
                        d2Rup = ROuter.radialderivatives2('Up')
                        d3Rup = third_derivative(ROuter, 'Up')
                        d4Rup = fourth_derivative(ROuter, 'Up')
                        d5Rup = fifth_derivative(ROuter, 'Up')
                        d6Rup = sixth_derivative(ROuter, 'Up')
                else:
                    d0Rup = np.array([])
                    d1Rup = np.array([])
                    d2Rup = np.array([])
                    d3Rup = np.array([])
                    d4Rup = np.array([])
                    d5Rup = np.array([])
                    d6Rup = np.array([])
                    
                if self.inner.shape[0] > 0:  
                    RInner = RadialTeukolsky(s, j, m, teuk.blackholespin, teuk.frequency, self.inner)
                    RInner.solve(bc='In')
                    d0Rin = RInner.radialsolutions('In')
                    d1Rin = RInner.radialderivatives('In')
                    d2Rin = RInner.radialderivatives2('In')
                    d3Rin = third_derivative(RInner, 'In')
                    d4Rin = fourth_derivative(RInner, 'In')
                    d5Rin = fifth_derivative(RInner, 'In')
                    d6Rin = sixth_derivative(RInner, 'In')
                else:
                    d0Rin = np.array([])
                    d1Rin = np.array([])
                    d2Rin = np.array([])
                    d3Rin = np.array([])
                    d4Rin = np.array([])
                    d5Rin = np.array([])
                    d6Rin = np.array([])

                for l in range(abs(m), self.maxl + 1):
                    PsiIn = hertz.couplingcoefficient(l)*PsiInJ
                    PsiUp = hertz.couplingcoefficient(l)*PsiUpJ
                    d0Psi = np.concatenate((PsiIn*d0Rin, PsiUp*d0Rup)) 
                    d1Psi = np.concatenate((PsiIn*d1Rin, PsiUp*d1Rup)) 
                    d2Psi = np.concatenate((PsiIn*d2Rin, PsiUp*d2Rup))
                    d3Psi = np.concatenate((PsiIn*d3Rin, PsiUp*d3Rup))
                    d4Psi = np.concatenate((PsiIn*d4Rin, PsiUp*d4Rup))
                    d5Psi = np.concatenate((PsiIn*d5Rin, PsiUp*d5Rup))
                    d6Psi = np.concatenate((PsiIn*d6Rin, PsiUp*d6Rup))  
                    self.dnpsi[0].add(l, m, d0Psi)
                    self.dnpsi[1].add(l, m, d1Psi)
                    self.dnpsi[2].add(l, m, d2Psi)
                    self.dnpsi[3].add(l, m, d3Psi)
                    self.dnpsi[4].add(l, m, d4Psi)
                    self.dnpsi[5].add(l, m, d5Psi)
                    self.dnpsi[6].add(l, m, d6Psi)
                    asymp_modes = np.array([PsiUp*(hertz.eigenvalue**n) for n in range(self.maxasymp)])
                    self.asymp.add(l, m, asymp_modes)

    def set(self, l, m, val0, val1, val2, val3, val4, val5, val6):
        self.dnpsi[0].set(l, m, val0)
        self.dnpsi[1].set(l, m, val1)
        self.dnpsi[2].set(l, m, val2)
        self.dnpsi[3].set(l, m, val3)
        self.dnpsi[4].set(l, m, val4)
        self.dnpsi[5].set(l, m, val5)
        self.dnpsi[6].set(l, m, val6)

    def save(self, fn = None):
        if fn is None:
            fn = self.filename + ".pkl"
        with open(fn, 'wb') as outp:
            pickle.dump(self.dnpsi, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, fn = None):
        if fn is None:
            fn = self.filename + ".pkl"
        with open(fn, 'rb') as inp:
            tmp_array = pickle.load(inp)
            self.dnpsi = tmp_array
        
    def __call__(self, l, m, deriv=0):
        if deriv <= 6 and deriv >= 0:
            return self.dnpsi[deriv](l, m)
        else:
            return self.dnpsi[0](l, m)
