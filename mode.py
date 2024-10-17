# mode.py
import numpy as np
from utils import chop_matrix
from geo import MiniGeo
from cheby import MultiGridChebyshev, MultiGridMultiDomainChebyshev
from collocode import (CollocationODEFixedStepSolver,
                       CollocationODESolver,
                       CollocationODEFixedMultiDomainFixedStepSolver,
                       CollocationODEMultiDomainFixedStepSolver)
from source import integrate_source
from swsh import SpinWeightedSpheroidalHarmonic
from swsh import MultiModeSpinWeightedSpherical
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
        return self.amplitudes[bc]*self.homogeneousradialsolution_tslicing(bc, sigma)
    
    def homogeneousradialsolution_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma)
    
    def homogeneousradialderivative2_tslicing(self, bc, sigma):
        return self.Rslm.Z_sigma(sigma)*self.Rslm.psi[bc](sigma, deriv = 2) + 2.*self.Rslm.Z_sigma_deriv(sigma)*self.Rslm.psi[bc](sigma, deriv = 1) + self.Rslm.Z_sigma_deriv2(sigma)*self.Rslm.psi[bc](sigma)
    
    def radialsolution(self, bc, sigma):
        return self.amplitudes[bc]*self.homogeneousradialsolution(bc, sigma)
    
    def homogeneousradialsolution(self, bc, sigma):
        return self.Rslm.psi[bc](sigma)
    
    def polarsolution(self, z):
        return self.Sslm(z)
    
def lmkn_mode_count(lmin, lmax, knum, nnum):
    return (1 + lmax - lmin)*(1 + lmax + lmin)*knum*nnum
    
class PointParticleModeGrid(PointParticleMode):
    def __init__(self, s, lrange, krange, nrange, source, Rslm = {"In": None, "Up": None}, Sslm = None, amplitudes = {"In": None, "Up": None}, precisions = {"In": None, "Up": None}, eigenvalues = None, domains = None, coupling = None, basis = 'spheroidal'):
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

        self.mode_num = lmkn_mode_count(self.lmin, self.lmax, self.klength, self.nlength)
        self.m_zero_final_index = self.mode_index_positive_m(self.lmax, 0, self.kmax, self.nmax)

        self.basis = basis
        self.Sslm = Sslm
        self.Rslm = Rslm

        self.precisions = precisions
        self.amplitudes = amplitudes
        self.eigenvalues = eigenvalues
        self.coupling = coupling

        if domains is None:
            self.domains = {"In": None, "Up": None}
            self.domain = {"In": None, "Up": None}
        else:
            self.domains = domains
            self.domain = dict.fromkeys(domains)
            for key in self.domain.keys():
                self.domain[key] = np.array([self.domains[key][0], self.domains[key][-1]])

        self.frequencies = np.empty(self.mode_num)
        self.modes = np.empty((self.mode_num, 4), dtype=np.int64)
        mkn_positive_mode_arr = []
        mkn_negative_mode_arr = []
        for l in self.larray:
            for m in range(-l, l + 1):
                for k in self.karray:
                    for n in self.narray:
                        self.frequencies[self.mode_index(l, m, k, n)] = source.mode_frequency(m, k, n)
                        self.modes[self.mode_index(l, m, k, n)] = [l, m, k, n]
                        if l == self.lmax:
                            if m >= 0:
                                mkn_positive_mode_arr.append([m,k,n])
                            else:
                                mkn_negative_mode_arr.append([m,k,n])
        mkn_positive_mode_arr = np.array(mkn_positive_mode_arr, dtype = np.int64)
        mkn_negative_mode_arr = np.array(mkn_negative_mode_arr, dtype = np.int64)
        self.mkn_mode_arr = np.concatenate((mkn_positive_mode_arr, mkn_negative_mode_arr))
        self.m_mode_arr = np.concatenate((np.arange(0, self.lmax + 1), np.arange(-self.lmax, 0)))
        
    
    def mode_index_positive_m(self, l, m, k, n):
        if m < 2:
            return (((self.lmax - 1)*m + (l - 2))*self.klength + k - self.kmin)*self.nlength + n - self.nmin
        return (((2*self.lmax - m + 3)*m // 2 - 3 + (l - m))*self.klength + k - self.kmin)*self.nlength + n - self.nmin
    
    def mode_index(self, l, m, k = 0, n = 0):
        if m < 0:
            return self.mode_num + self.m_zero_final_index - self.mode_index_positive_m(l, -m, k, n)
        else:
            return self.mode_index_positive_m(l, m, k, n)
        
    def m_mode_indices(self, m, k=0, n=0):
        lmin = np.max([self.lmin, np.abs(m)])
        return [self.mode_index(l, m, k, n) for l in range(lmin, self.lmax + 1)]
        
    def mode_count_for_m(self, m):
        if np.abs(m) < 2:
            return self.nlength*self.klength*(self.lmax - 1)
        return self.nlength*self.klength*(self.lmax - np.abs(m) + 1)
    
    def l_mode_count_for_m(self, m):
        if np.abs(m) < 2:
            return (self.lmax - 1)
        return (self.lmax - np.abs(m) + 1)
    
    def group_by_m_modes(self, arr):
        assert arr.shape[0] == self.mode_num
        mode_count_m_arr = np.cumsum([self.mode_count_for_m(m) for m in self.m_mode_arr])
        return np.split(arr, mode_count_m_arr[:-1])
    
    def group_by_mkn_modes(self, arr):
        assert arr.shape[0] == self.mode_num
        mode_count_mkn_arr = np.cumsum([self.l_mode_count_for_m(m) for m, k, n in self.mkn_mode_arr])
        return np.split(arr, mode_count_mkn_arr[:-1])
    
class TeukolskyPointParticleMode(PointParticleMode):
    pass
    
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
        self.solver_kwargs = solver_kwargs.copy()
        self.teuk_solver = HyperboloidalTeukolskySolver(domains = [[0, self.smax], [self.smin, 1]], solver = self.solver, solver_kwargs = self.solver_kwargs)
        self.integrate_kwargs = integrate_kwargs.copy()

    def reduce(self, mode, **reduce_kwargs):
        psi4 = mode.copy()
        psi4.Rslm = self.teuk_solver.reduce(mode.Rslm, **reduce_kwargs)
        return psi4

    def generate_mode(self, s, l, m, k, n, teuk_solver = None, integrate_kwargs = {}, rescale = True, reduce = False, **reduce_kwargs):
        if k != 0:
            raise ValueError('Non-zero k-modes are not supported yet')
        
        if not (s == -2 or s == 0):
            raise ValueError(f'Spin-weight {s} is not supported yet. Must be 0 or -2.')
        
        if teuk_solver is None:
            teuk_solver = self.teuk_solver
        
        teuk = TeukolskyPointParticleMode(s, l, m, k, n, self.source)

        teuk.Sslm = SpinWeightedSpheroidalHarmonic(s, l, m, teuk.a*teuk.frequency)
        teuk.Rslm = teuk_solver(teuk.a, s, l, m, teuk.frequency, teuk.eigenvalue, rescale = rescale, reduce = reduce, **reduce_kwargs)
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
            solver = CollocationODEFixedMultiDomainFixedStepSolver()
            solver_kwargs = {}

        if solver.type == 'dynamic':
            # print(f"Warning: Provided solver has dynamic domain structure. Solver must have a fixed domain structure. Changing to fixed solver with {self.solver.n} nodes.")
            solver = CollocationODEFixedMultiDomainFixedStepSolver(solver.n)

        super().__init__(source, solver = solver, solver_kwargs=solver_kwargs, integrate_kwargs=integrate_kwargs)

    def optimize(self, s, lmax, solver = None, solver_kwargs = {}, integrate_kwargs = None):
        # optimize the subdomain structure for a mode grid ranging up to lmax
        if solver is not None:
            if solver.n != self.solver.n:
                print(f"Warning: The provided solver uses {solver.n} nodes, while the class solver uses {self.solver.n}. Resulting domains may not be optimized for class solver.")

        if solver is None:
            teuk_solver = self.teuk_solver
        else:
            teuk_solver = HyperboloidalTeukolskySolver(domains = [[0, self.smax], [self.smin, 1]], solver = solver, solver_kwargs = solver_kwargs)

        if integrate_kwargs is None:
            integrate_kwargs = self.integrate_kwargs

        test = self.generate_mode(s, lmax, 1, 0, 0, teuk_solver = teuk_solver, integrate_kwargs = integrate_kwargs)
        self.domains = test.domains
        self.solver_kwargs["subdomains"] = test.domains
        self.teuk_solver = HyperboloidalTeukolskySolver(domains = [[0, self.smax], [self.smin, 1]], solver = self.solver, solver_kwargs = self.solver_kwargs)

    def solve_teukolsky_mode_data(self, psi4, reduce = False, **reduce_kwargs):
        lrange = psi4.larray
        s = psi4.s

        l = lrange[0]
        m = 0
        teuk = self.generate_mode(s, l, m, 0, 0, self.teuk_solver, self.integrate_kwargs, reduce = reduce, **reduce_kwargs)
        coeffsIn = teuk.Rslm.psi["In"].coeffs
        coeffsUp = teuk.Rslm.psi["Up"].coeffs
        cshapeIn = coeffsIn.shape
        cshapeUp = coeffsUp.shape

        RslmGrid = {"In": np.empty((psi4.mode_num,) + cshapeIn, dtype = np.complex128),
                      "Up": np.empty((psi4.mode_num,) + cshapeUp, dtype = np.complex128)}
        domains = {"In": teuk.domains["In"], "Up": teuk.domains["Up"]}
        amplitudes = {"In": np.empty((psi4.mode_num,), dtype = np.complex128),
                      "Up": np.empty((psi4.mode_num,), dtype = np.complex128)}
        SslmGrid = np.zeros((psi4.mode_num, psi4.lmax + 100))
        eigenvalues = np.empty(psi4.mode_num)

        i = psi4.mode_index(l, m, 0, 0)
        RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
        RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
        amplitudes["In"][i] = teuk.amplitudes["In"]
        amplitudes["Up"][i] = teuk.amplitudes["Up"]
        coupling = teuk.Sslm.coefficients
        SslmGrid[i][:len(coupling)] = coupling
        eigenvalues[i] = teuk.eigenvalue

        for l in psi4.larray[1:]:
            m = 0
            teuk = self.generate_mode(s, l, m, 0, 0, self.teuk_solver, self.integrate_kwargs, reduce = reduce, reduce_kwargs = reduce_kwargs)
            i = psi4.mode_index(l, m, 0, 0)
            RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
            RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
            amplitudes["In"][i] = teuk.amplitudes["In"]
            amplitudes["Up"][i] = teuk.amplitudes["Up"]
            coupling = teuk.Sslm.coefficients
            SslmGrid[i][:len(coupling)] = coupling
            eigenvalues[i] = teuk.eigenvalue

        for l in psi4.larray:
            for m in range(1, l + 1):
                teuk = self.generate_mode(s, l, m, 0, 0, self.teuk_solver, self.integrate_kwargs, reduce = reduce, reduce_kwargs = reduce_kwargs)
                i = psi4.mode_index(l, m, 0, 0)
                RslmGrid["In"][i] = teuk.Rslm.psi["In"].coeffs
                RslmGrid["Up"][i] = teuk.Rslm.psi["Up"].coeffs
                amplitudes["In"][i] = teuk.amplitudes["In"]
                amplitudes["Up"][i] = teuk.amplitudes["Up"]
                coupling = teuk.Sslm.coefficients
                SslmGrid[i][:len(coupling)] = coupling
                eigenvalues[i] = teuk.eigenvalue

                # teuk = self.generate_mode(s, l, -m, 0, 0, self.teuk_solver, self.integrate_kwargs, reduce = reduce, reduce_kwargs = reduce_kwargs)
                i = psi4.mode_index(l, -m, 0, 0)
                lmin = np.max([2, abs(m)])
                RslmGrid["In"][i] = np.conj(teuk.Rslm.psi["In"].coeffs)
                RslmGrid["Up"][i] = np.conj(teuk.Rslm.psi["Up"].coeffs)
                amplitudes["In"][i] = (-1)**l*np.conj(teuk.amplitudes["In"])
                amplitudes["Up"][i] = (-1)**l*np.conj(teuk.amplitudes["Up"])
                coupling = teuk.Sslm.coefficients.copy() # TODO: should i reverse the ordering because of the negative indexing of m<0 modes?
                coupling[::2] *= -1
                if np.abs(l + lmin) % 2 == 0:
                    coupling *= -1
                SslmGrid[i][:len(coupling)] = coupling
                eigenvalues[i] = teuk.eigenvalue

        return RslmGrid, domains, SslmGrid, amplitudes, eigenvalues

    def solve(self, s, lrange, reduce = False, **reduce_kwargs):
        psi4 = PointParticleModeGrid(s, lrange, [0, 0], [0, 0], self.source)
        psi4.basis = "spheroidal"

        RslmGrid, domains, SslmGrid, amplitudes, eigenvalues = self.solve_teukolsky_mode_data(psi4, reduce = reduce, **reduce_kwargs)
        domainsIn = domains["In"]
        domainsUp = domains["Up"]

        RslmGrid["In"] = np.ascontiguousarray(amplitudes["In"][:, None, None]*RslmGrid["In"])
        RslmGrid["Up"] = np.ascontiguousarray(amplitudes["Up"][:, None, None]*RslmGrid["Up"])
        if len(RslmGrid["In"].shape) > 2:
            RslmIn = MultiGridMultiDomainChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridMultiDomainChebyshev(RslmGrid["Up"], domainsUp)
        else:
            RslmIn = MultiGridChebyshev(RslmGrid["In"], domainsIn)
            RslmUp = MultiGridChebyshev(RslmGrid["Up"], domainsUp)

        psi4.Rslm = {"In": RslmIn, "Up": RslmUp}
        psi4.coupling = chop_matrix(SslmGrid, buffer = lrange[-1] - 2, tol=1e-50)
        m_arr = psi4.m_mode_arr
        coeffs_arr = psi4.group_by_m_modes(psi4.coupling)
        gamma_arr = psi4.group_by_m_modes(psi4.source.a*psi4.frequencies)
        psi4.Sslm = MultiModeSpinWeightedSpherical(s, m_arr, gamma_arr, coeffs_arr, tol = None)
        psi4.eigenvalues = eigenvalues
        psi4.amplitudes = amplitudes
        psi4.domains["In"] = domainsIn
        psi4.domains["Up"] = domainsUp

        return psi4
    
    def cast_to_spherical(self, psi4, lmax):
        lrange = [psi4.larray[0], lmax]
        psi4_Ylm = PointParticleModeGrid(psi4.s, lrange, psi4.karray, psi4.narray, self.source, amplitudes=psi4.amplitudes, precisions=psi4.precisions, eigenvalues=psi4.eigenvalues, domains=psi4.domains, coupling=psi4.coupling)
        psi4_Ylm.basis = "spherical"
        jmax = psi4.lmax

        mmax = np.min([lmax, jmax])
        domain_num_up = len(psi4.Rslm["Up"].coeffs)
        coeff_num_up = len(psi4.Rslm["Up"].coeffs[0])
        new_mode_num = psi4_Ylm.mode_num
        psi4_up = np.empty((domain_num_up, new_mode_num, coeff_num_up), dtype = np.complex128)
        domain_num_in = len(psi4.Rslm["In"].coeffs)
        coeff_num_in = len(psi4.Rslm["In"].coeffs[0])
        psi4_in = np.empty((domain_num_in, new_mode_num, coeff_num_in), dtype = np.complex128)

        psi4_grouped = psi4.group_by_mkn_modes(psi4.coupling)

        for j in range(domain_num_up):
            coeffsUp = psi4.Rslm["Up"].coeffs[j]
            coeff4Up = psi4.group_by_mkn_modes(coeffsUp.T)
            for m in range(-mmax, mmax + 1):
                m_mode_indices = psi4_Ylm.m_mode_indices(m)
                psi4_up[j, m_mode_indices] = (psi4_grouped[m].T[:len(m_mode_indices)] @ coeff4Up[m])

        for j in range(domain_num_in):
            coeffsIn = psi4.Rslm["In"].coeffs[j]
            coeff4In = psi4.group_by_mkn_modes(coeffsIn.T)
            for m in range(-mmax, mmax + 1):
                m_mode_indices = psi4_Ylm.m_mode_indices(m)
                psi4_in[j, m_mode_indices] = (psi4_grouped[m].T[:len(m_mode_indices)] @ coeff4In[m])

        RslmIn = MultiGridMultiDomainChebyshev(np.moveaxis(psi4_in, 1, 0), psi4_Ylm.domains["In"])
        RslmUp = MultiGridMultiDomainChebyshev(np.moveaxis(psi4_up, 1, 0), psi4_Ylm.domains["Up"])
        psi4_Ylm.Rslm = {"In": RslmIn, "Up": RslmUp}

        return psi4_Ylm
    
    def __call__(self, s, lrange, reduce = False, **reduce_kwargs):
        return self.solve(s, lrange, reduce = reduce, **reduce_kwargs)

class TeukolskyPointParticleModeGrid(PointParticleModeGrid):
    pass
    
class LMModeMultiDomain(MultiGridMultiDomainChebyshev):
    def __init__(self, lrange, coeffList, domainList):
        super().__init__(coeffList, domainList)
        self.lmin, self.lmax = lrange
        self.mode_num = (1 + self.lmax - self.lmin)*(1 + self.lmax + self.lmin)
        assert self.mode_num == self.coeffs.shape[0]

    def mode_index(self, l, m):
        # locate index of (l, m)-mode
        return (l + self.lmin)*(l - self.lmin) + l + m