# mode.py
import numpy as np
from geo import MiniGeo
from cheby import MultiGridChebyshev, MultiGridMultiDomainChebyshev
from collocode import (CollocationODEFixedStepSolver,
                       CollocationODESolver,
                       CollocationODEFixedMultiDomainFixedStepSolver,
                       CollocationODEMultiDomainFixedStepSolver)
from source import integrate_source
from spheroidal import SpinWeightedSpheroidalHarmonic
from teuk import (sigma_r, 
                  sigma_r_deriv,
                  sigma_r_deriv_sigma, 
                  sigma_r_deriv_sigma_deriv,
                  HyperboloidalTeukolsky,
                  HyperboloidalTeukolskySolver,
                  teuk_sys
                )

class PointParticleMode:
    def __init__(self, s, l, m, k, n, source, Rslm = {"In": None, "Up": None}, Sslm = None, amplitudes = {"In": None, "Up": None}, precisions = {"In": None, "Up": None}, domains = None):
        self.source = source
        self.a = source.a
        self.s = s
        self.l = l
        self.m = m
        self.k = k
        self.n = n
        self.frequency = source.mode_frequency(m, k, n)
        self.kappa = np.sqrt(1 - self.a**2)

        self.Sslm = Sslm
        self.Rslm = Rslm
        self.amplitudes = amplitudes
        self.precisions = precisions

        if domains is None:
            self.domains = {"In": None, "Up": None}
            self.domain = {"In": None, "Up": None}
        else:
            self.domains = domains
            self.domain = dict.fromkeys(domains)
            for key in self.domain.keys():
                self.domain[key] = np.array([self.domains[key][0], self.domains[key][-1]])
    
    @property
    def blackholespin(self):
        return self.a
    
    @property
    def spinweight(self):
        return self.s
    
    @property
    def eigenvalue(self):
        return self.Sslm.eigenvalue
    
    def amplitude(self, bc):
        return self.amplitudes[bc]
    
    def precision(self, bc):
        return self.precisions[bc]

    def radialsolution_tslicing(self, bc, sigma):
        return self.amplitude[bc]*self.homogeneousradialsolution_tslicing(bc, sigma)
    
    def homogeneousradialsolution_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative2_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 2) + 2.*self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv2(sigma)*self.Rslm.psi[bc](sigma)
    
    def radialsolution(self, bc, sigma):
        return self.amplitude[bc]*self.radialsolution(bc, sigma)
    
    def homogeneousradialsolution(self, bc, sigma):
        return self.Rslm.psi[bc](sigma)
    
    def polarsolution(self, z):
        return self.Sslm(z)
    
class PointParticleModeGrid:
    def __init__(self, s, lrange, krange, nrange, source, RslmGrid = {"In": None, "Up": None}, SslmGrid = {"In": None, "Up": None}, domains = None, basis = 'spheroidal'):
        self.source = source
        self.a = source.a
        self.kappa = np.sqrt(1 - self.a**2)
        self.s = s

        if isinstance(lrange, int):
            self.lmin = np.abs(s)
            self.lmax = lrange
        else:
            self.lmin, self.lmax = lrange

        self.kmin = np.min(krange)
        self.kmax = np.max(krange)
        self.nmin = np.min(nrange)
        self.nmax = np.max(nrange)

        self.larray = np.arange(self.lmin, self.lmax+1, dtype = np.int64)
        self.karray = np.arange(self.kmin, self.kmax+1, dtype = np.int64)
        self.narray = np.arange(self.nmin, self.nmax+1, dtype = np.int64)

        self.klength = self.kmax - self.kmin + 1
        self.nlength = self.nmax - self.nmin + 1

        self.modeNum = (1 + self.lmax - self.lmin)*(1 + self.lmax + self.lmin)*self.klength*self.nlength

        self.basis = basis
        self.Sslm = SslmGrid
        self.Rslm = RslmGrid

        if domains is None:
            self.domains = {"In": None, "Up": None}
            self.domain = {"In": None, "Up": None}
        else:
            self.domains = domains
            self.domain = dict.fromkeys(domains)
            for key in self.domain.keys():
                self.domain[key] = np.array([self.domains[key][0], self.domains[key][-1]])

    def mode_index(self, l, m, k, n):
        return ((l - self.lmin)*(l + self.lmin) + (l + m))*((k - self.kmin + 1)*self.nlength) + ((k - self.kmin)*self.nlength + (n - self.nmin))
    
    @property
    def blackholespin(self):
        return self.a
    
    @property
    def spinweight(self):
        return self.s

    def radialsolution(self, bc, sigma):
        return self.amplitude[bc]*self.homogeneousradialsolution(bc, sigma)
    
    def homogeneousradialsolution(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative2(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 2) + 2.*self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv2(sigma)*self.Rslm.psi[bc](sigma)
    
    def hyperboloidalsolution(self, bc, sigma):
        return self.amplitude[bc]*self.Rslm.psi[bc](sigma)
    
    def homogeneoushyperboloidalsolution(self, bc, sigma):
        return self.Rslm.psi[bc](sigma)
    
    def polarsolution(self, z):
        return self.Sslm(z)
    
class TeukolskyPointParticleMode(PointParticleMode):
    pass

# class TeukolskyPointParticleModeGrid(PointParticleModeGrid):
#     def __init__(self, s, lrange, krange, nrange, source, Rslm = None, Sslm = None, amplitude = None, precision = None, domains = None):
#         if not (s == -2 or s == 0):
#             raise ValueError(f'Spin-weight {s} is not supported yet. Must be 0 or -2.')
        
#         super().__init__(s, lrange, krange, nrange, source)

#         self.Sslm = Sslm
#         self.Rslm = Rslm

#         if amplitude is None:
#             self.amplitude = {"In": None, "Up": None}
#         else:
#             self.amplitude = amplitude

#         if precision is None:
#             self.precision = {"In": None, "Up": None}
#         else:
#             self.precision = precision

#         if domains is None:
#             self.domains = {"In": None, "Up": None}
#             self.domain = {"In": None, "Up": None}
#         else:
#             self.domains = domains
#             self.domain = dict.fromkeys(domains)
#             for key in self.domain.keys():
#                 self.domain[key] = np.array([self.domains[key][0], self.domains[key][-1]])
    
class TeukolskyPointParticleModeGenerator:
    def __init__(self, source, solver = None, solver_kwargs = {}, integrate_kwargs = {}):
        rmin = source.radial_roots[1]
        rmax = source.radial_roots[0]
        self.a = source.a
        self.kappa = np.sqrt(1-self.a**2)
        self.smin = sigma_r(rmax, self.kappa)
        self.smax = sigma_r(rmin, self.kappa)

        self.source = source
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.teuk_solver = HyperboloidalTeukolskySolver(domains = [[0, self.smax], [self.smin, 1]], solver = solver, solver_kwargs = solver_kwargs)
        self.integrate_kwargs = integrate_kwargs

    def reduce(self, mode, **reduce_kwargs):
        out = mode.copy()
        out.Rslm = self.teuk_solver.reduce(mode.Rslm, **reduce_kwargs)
        return out

    def generate_mode(self, s, l, m, k, n, teuk_solver = None, integrate_kwargs = {}, reduce = False, **reduce_kwargs):
        if k != 0:
            raise ValueError('Non-zero k-modes are not supported yet')
        
        if not (s == -2 or s == 0):
            raise ValueError(f'Spin-weight {s} is not supported yet. Must be 0 or -2.')
        
        teuk = TeukolskyPointParticleMode(s, l, m, k, n, self.source)

        teuk.Sslm = SpinWeightedSpheroidalHarmonic(s, l, m, teuk.a*teuk.frequency)
        teuk.Rslm = teuk_solver(teuk.a, s, l, m, teuk.frequency, teuk.eigenvalue, rescale = True, reduce = reduce, **reduce_kwargs)

        amplitude_data = integrate_source(s, l, m, k, n, self.source, Sslm = teuk.Sslm, Rslm = teuk.Rslm, **integrate_kwargs)
        teuk.amplitudes["In"], teuk.amplitudes["Up"] = amplitude_data[0]
        teuk.precisions["In"], teuk.precisions["Up"] = amplitude_data[1]

        teuk.domain = {"In": teuk.Rslm.psi["In"].domain, "Up": teuk.Rslm.psi["Up"].domain}
        teuk.domains = {"In": teuk.Rslm.psi["In"].domains, "Up": teuk.Rslm.psi["Up"].domains}

        return teuk
    
    def __call__(self, s, l, m, k, n, reduce = False, **reduce_kwargs):
        return self.generate_mode(s, l, m, k, n, self.teuk_solver, self.integrate_kwargs, reduce = reduce, **reduce_kwargs)
    
class TeukolskyPointParticleModeGridGenerator(TeukolskyPointParticleModeGenerator):
    def __init__(self, source, solver = None, solver_kwargs = {}, integrate_kwargs = {}):
        if solver is None:
            self.solver = CollocationODEFixedMultiDomainFixedStepSolver(self.solver.n)
            self.solver_kwargs = {}
        else:
            self.solver = solver
            self.solver_kwargs = solver_kwargs

        if self.solver.type == 'dynamic':
            print(f"Warning: Provided solver has dynamic domain structure. Solver must have a fixed domain structure. Changing to fixed solver with {self.solver.n} nodes.")
            self.solver = CollocationODEFixedMultiDomainFixedStepSolver(self.solver.n)

        rmin = source.radial_roots[1]
        rmax = source.radial_roots[0]
        self.smin = sigma_r(rmax, self.kappa)
        self.smax = sigma_r(rmin, self.kappa)

        self.source = source
        self.integrate_kwargs = integrate_kwargs

    def optimize(self, s, lmax, solver = None, **solver_kwargs):
        # optimize the subdomain structure for a mode grid ranging up to lmax
        if solver.n != self.solver.n:
            print(f"Warning: The provided solver uses {solver.n} nodes, while the class solver uses {self.solver.n}. Resulting domains may not be optimized for class solver.")

        test = self.generate_mode(s, lmax, 1, 0, 0, solver = solver, solver_kwargs = solver_kwargs, integrate_kwargs = self.integrate_kwargs)
        self.domains = test.domains
        self.solver_kwargs["subdomains"] = test.domains

    def __call__(self, s, lrange, reduce = False, **reduce_kwargs):
        out = TeukolskyPointParticleModeGrid(s, lrange, [0, 0], [0, 0], self.source)

        l = lrange[0]
        m = 0
        teuk = self.generate_mode(s, l, m, 0, 0, self.solver, self.solver_kwargs, self.integrate_kwargs, reduce = reduce, **reduce_kwargs)
        coeffs = teuk.Rslm.psi["Up"].coeffs
        domainsIn = teuk.domains["In"]
        domainsUp = teuk.domains["Up"]
        cshape = coeffs.shape

        RslmGrid = {"In": np.empty((out.modeNum,) + cshape, dtype = np.complex128),
                      "Up": np.empty((out.modeNum,) + cshape, dtype = np.complex128)}
        amplitude = {"In": np.empty((out.modeNum,), dtype = np.complex128),
                      "Up": np.empty((out.modeNum,), dtype = np.complex128)}
        SslmGrid = np.empty((out.modeNum, out.lmax + 100))
        larray = np.empty(out.modeNum)
        marray = np.empty(out.modeNum)
        eigenvalues = np.empty(out.modeNum)
        frequencies = np.empty(out.modeNum)

        i = out.mode_index(l, m, 0, 0)
        RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
        RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
        amplitude["In"][i] = teuk.amplitude["In"]
        amplitude["Up"][i] = teuk.amplitude["Up"]
        coupling = teuk.Sslm.coefficients
        SslmGrid[i][:len(coupling)] = coupling
        eigenvalues[i] = teuk.eigenvalue
        frequencies[i] = self.source.mode_frequency(m, 0, 0)
        larray[i] = l
        marray[i] = m

        for l in lrange[1:]:
            teuk = self.generate_mode(s, l, m, 0, 0, self.solver, self.solver_kwargs, self.integrate_kwargs, reduce = reduce, reduce_kwargs = reduce_kwargs)
            i = out.mode_index(l, m, 0, 0)
            RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
            RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
            amplitude["In"][i] = teuk.amplitude["In"]
            amplitude["Up"][i] = teuk.amplitude["Up"]
            coupling = teuk.Sslm.coefficients
            SslmGrid[i][:len(coupling)] = coupling
            eigenvalues[i] = teuk.eigenvalue
            frequencies[i] = self.source.mode_frequency(m, 0, 0)
            larray[i] = l
            marray[i] = m

        for l in lrange:
            for m in range(1, l + 1):
                teuk = self.generate_mode(s, l, m, 0, 0, self.solver, self.solver_kwargs, self.integrate_kwargs, reduce = reduce, reduce_kwargs = reduce_kwargs)
                i = out.mode_index(l, m, 0, 0)
                RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
                RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
                amplitude["In"][i] = teuk.amplitude["In"]
                amplitude["Up"][i] = teuk.amplitude["Up"]
                coupling = teuk.Sslm.coefficients
                SslmGrid[i][:len(coupling)] = coupling
                eigenvalues[i] = teuk.eigenvalue
                frequencies[i] = self.source.mode_frequency(m, 0, 0)
                larray[i] = l
                marray[i] = m

        RslmGrid["In"] = np.ascontiguousarray(np.moveaxis(RslmGrid["In"], 0, -1))
        RslmGrid["Up"] = np.ascontiguousarray(np.moveaxis(RslmGrid["Up"], 0, -1))
        if len(RslmGrid["In"].shape) > 2:
            RslmIn = MultiGridMultiDomainChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridMultiDomainChebyshev(RslmGrid["Up"], domainsUp)
        else:
            RslmIn = MultiGridChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridChebyshev(RslmGrid["Up"], domainsUp)

        Rslm = HyperboloidalTeukolsky(out.a, s, larray, marray, frequencies, eigenvalues, psi = {"In": RslmIn, "Up": RslmUp})
    
class TeukolskyPointParticleMode(PointParticleMode):
    pass

class TeukolskyPointParticleModeGrid(PointParticleModeGrid):
    def __init__(self, s, lrange, krange, nrange, source, Rslm = None, Sslm = None, amplitude = None, precision = None, domains = None):
        if not (s == -2 or s == 0):
            raise ValueError(f'Spin-weight {s} is not supported yet. Must be 0 or -2.')
        
        super().__init__(s, lrange, krange, nrange, source)

        self.Sslm = Sslm
        self.Rslm = Rslm

        if amplitude is None:
            self.amplitude = {"In": None, "Up": None}
        else:
            self.amplitude = amplitude

        if precision is None:
            self.precision = {"In": None, "Up": None}
        else:
            self.precision = precision

        if domains is None:
            self.domains = {"In": None, "Up": None}
            self.domain = {"In": None, "Up": None}
        else:
            self.domains = domains
            self.domain = dict.fromkeys(domains)
            for key in self.domain.keys():
                self.domain[key] = np.array([self.domains[key][0], self.domains[key][-1]])
    
class LMModeMultiDomain(MultiGridMultiDomainChebyshev):
    def __init__(self, lrange, coeffList, domainList):
        super().__init__(coeffList, domainList)
        self.lmin, self.lmax = lrange
        self.modeNum = (1 + self.lmax - self.lmin)*(1 + self.lmax + self.lmin)
        assert self.modeNum == self.coeffs.shape[0]

    def mode_index(self, l, m):
        # locate index of (l, m)-mode
        return (l + self.lmin)*(l - self.lmin) + l + m