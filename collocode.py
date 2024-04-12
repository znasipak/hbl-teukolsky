# collocode.py
"""
Python classes for solving second-order homogeneous ODEs using a Chebyshev collocation method
Author: Zach Nasipak
"""

import numpy as np
import numpy as xp
import numpy.polynomial.chebyshev as ch
from cheby import (cheby_D_jit,
                   _CHEBY_DMAX,
                   _CHEBY_dsize,
                   Chebyshev,
                   MultiDomainChebyshev,
                   MultiGridChebyshev,
                   MultiGridMultiDomainChebyshev)
from numba import njit
from numba.experimental import jitclass
import numba as nb

# I precompute everything with N = 4 and N = 64 points, since these are the most common sampling numbers I use
_precomputed_grid_numbers = np.arange(4, 64)

_CHEBY_TYPES = [1,2]
_CHEBY_nodes = dict.fromkeys(_CHEBY_TYPES)
_CHEBY_Tmat0 = dict.fromkeys(_CHEBY_TYPES)
_CHEBY_Tmat1 = dict.fromkeys(_CHEBY_TYPES)
_CHEBY_Tmat2 = dict.fromkeys(_CHEBY_TYPES)
_CHEBY_Tmat0Inv = dict.fromkeys(_CHEBY_TYPES)

for chtype in _CHEBY_TYPES:
    _CHEBY_nodes[chtype] = dict.fromkeys(_precomputed_grid_numbers)
    _CHEBY_Tmat0[chtype] = dict.fromkeys(_precomputed_grid_numbers)
    _CHEBY_Tmat1[chtype] = dict.fromkeys(_precomputed_grid_numbers)
    _CHEBY_Tmat2[chtype] = dict.fromkeys(_precomputed_grid_numbers)
    _CHEBY_Tmat0Inv[chtype] = dict.fromkeys(_precomputed_grid_numbers)

_BC_dsize = 4
_BC_TYPES = [1, 2]
_BC_NUMS = [i for i in range(_BC_dsize)]

_CHEBY_bcs = dict.fromkeys(_BC_TYPES)
_CHEBY_bcs_end = dict.fromkeys(_BC_TYPES)
for bctype in _BC_TYPES:
    _CHEBY_bcs[bctype] = dict.fromkeys(_precomputed_grid_numbers)
    _CHEBY_bcs_end[bctype] = dict.fromkeys(_precomputed_grid_numbers)
    for num in _precomputed_grid_numbers:
        _CHEBY_bcs[bctype][num] = dict.fromkeys(_BC_NUMS)
        _CHEBY_bcs_end[bctype][num] = dict.fromkeys(_BC_NUMS)

# _CHEBY_bc1_2 = dict.fromkeys(_BC_TYPES, dict.fromkeys(_precomputed_grid_numbers))
# _CHEBY_bc2_2 = dict.fromkeys(_BC_TYPES, dict.fromkeys(_precomputed_grid_numbers))

for num in _precomputed_grid_numbers:
    _CHEBY_nodes[1][num] = ch.chebpts1(num)
    _CHEBY_Tmat0[1][num] = ch.chebvander(_CHEBY_nodes[1][num], num-1)
    _CHEBY_Tmat1[1][num] = xp.matmul(_CHEBY_Tmat0[1][num], _CHEBY_DMAX[:num,:num])
    _CHEBY_Tmat2[1][num] = xp.matmul(_CHEBY_Tmat1[1][num], _CHEBY_DMAX[:num,:num])
    _CHEBY_Tmat0Inv[1][num] = xp.linalg.inv(_CHEBY_Tmat0[1][num])

    _CHEBY_nodes[2][num] = ch.chebpts2(num)
    _CHEBY_Tmat0[2][num] = ch.chebvander(_CHEBY_nodes[2][num], num-1)
    _CHEBY_Tmat1[2][num] = xp.matmul(_CHEBY_Tmat0[2][num], _CHEBY_DMAX[:num,:num])
    _CHEBY_Tmat2[2][num] = xp.matmul(_CHEBY_Tmat1[2][num], _CHEBY_DMAX[:num,:num])
    _CHEBY_Tmat0Inv[2][num] = xp.linalg.inv(_CHEBY_Tmat0[2][num])

    _CHEBY_bcs[1][num][0] = xp.array(ch.chebvander(-1, num-1))
    _CHEBY_bcs_end[1][num][0] = xp.array(ch.chebvander(1, num-1))
    for bcnum in _BC_NUMS:
        if bcnum > 0:
            _CHEBY_bcs[1][num][bcnum] = xp.matmul(_CHEBY_bcs[1][num][bcnum - 1], _CHEBY_DMAX[:num,:num])
            _CHEBY_bcs_end[1][num][bcnum] = xp.matmul(_CHEBY_bcs_end[1][num][bcnum - 1], _CHEBY_DMAX[:num,:num])
        _CHEBY_bcs[2][num][bcnum] = xp.array(ch.chebvander(_CHEBY_nodes[2][num][bcnum], num-1))
        _CHEBY_bcs_end[2][num][bcnum] = xp.array(ch.chebvander(_CHEBY_nodes[2][num][-1-bcnum], num-1))

def x_of_z(z, zmin, zmax):
    return (2*z - (zmax + zmin))/(zmax - zmin)

def z_of_x(x, zmin, zmax):
    return ((zmax - zmin)*x + zmax + zmin)/2

def dx_dz(zmin, zmax):
    return 2/(zmax - zmin)

def dz_dx(zmin, zmax):
    return (zmax - zmin)/2

# @jitclass([
#     ('coeffs', nb.complex128[:]),
#     ('error', nb.float64)
# ])
# class CoefficientGenerator:
#     def __init__(self, Pmat, Qmat, Umat, Tmat2, Tmat1, Tmat0, y0, bcs):
#         Mmat = xp.multiply(Pmat, Tmat2) + xp.multiply(Qmat, Tmat1) + xp.multiply(Umat, Tmat0)
#         source = xp.zeros(Pmat.shape, dtype = Mmat.dtype)
#         MmatS = xp.ascontiguousarray(Mmat.copy())

#         # dealing with large numbers
#         norm = 1
#         if xp.abs(y0[0]) > 1:
#             norm = y0[0]
#         for i in range(len(y0)):
#             MmatS[i] = bcs[i]
#             source[i,0] = y0[i]/norm

#         coeffs_temp = xp.linalg.solve(MmatS, source)
#         self.error = xp.linalg.norm(Mmat @ coeffs_temp)
#         self.coeffs = coeffs_temp.flatten()

class CoefficientGenerator:
    def __init__(self, Pmat, Qmat, Umat, Tmat2, Tmat1, Tmat0, y0, bcs):
        Mmat = Pmat*Tmat2 + Qmat*Tmat1 + Umat*Tmat0
        source = xp.zeros(Pmat.shape, dtype = Mmat.dtype)
        MmatS = xp.ascontiguousarray(Mmat.copy())

        # dealing with large numbers
        norm = 1
        if xp.abs(y0[0]) > 1:
            norm = y0[0]
        for i in range(len(y0)):
            MmatS[i] = bcs[i]
            source[i,0] = y0[i]/norm

        coeffs_temp = xp.linalg.solve(MmatS, source)
        self.error = xp.linalg.norm(Mmat @ coeffs_temp)
        self.coeffs = norm*coeffs_temp.flatten()

class CollocationODEFixedStepSolver:
    def __init__(self, n = 16, chtype = 1, bctype = 1):
        self.n = n
        self.chtype = chtype
        self.bctype = bctype

        assert chtype in _CHEBY_TYPES
        assert bctype in _BC_TYPES

        if n in _precomputed_grid_numbers:
            self.nodes = _CHEBY_nodes[chtype][n]
            self.Tmat0 = _CHEBY_Tmat0[chtype][n]
            self.Tmat1 = _CHEBY_Tmat1[chtype][n]
            self.Tmat2 = _CHEBY_Tmat2[chtype][n]
            self.bcs = _CHEBY_bcs[bctype][n]
        elif n < _CHEBY_dsize:
            D = _CHEBY_DMAX[:n,:n]
            self.bcs = dict.fromkeys(_BC_NUMS)

            if chtype == 1:
                self.nodes = ch.chebpts1(n)
            else:
                self.nodes = ch.chebpts2(n)
            self.Tmat0 = xp.array(ch.chebvander(self.nodes, n-1))
            self.Tmat1 = xp.matmul(self.Tmat0, D)
            self.Tmat2 = xp.matmul(self.Tmat1, D)

            if bctype == 1:
                self.bcs[0] = xp.array(ch.chebvander(-1, n - 1))
                for i in range(1, _BC_dsize):
                    self.bcs[i] = xp.matmul(self.bcs[i-1], D)                    
            else:
                temp = ch.chebpts2(n)
                for num in self.bcs.keys():
                    self.bcs[num] = xp.array(ch.chebvander(temp[num], n - 1))
        else:
            raise ValueError(f"Too many collocation points requested. Maximum number of collocation points is set to {_CHEBY_dsize}.")
        
        self.bcs = xp.asarray(list(self.bcs.values()))

    @property
    def type(self):
        return "fixed"

    def solve_ode_coeffs(self, ode_sys, y0, args = (), domain = [-1, 1]):
        [zmin, zmax] = domain
    
        z_nodes = z_of_x(self.nodes, zmin, zmax)
        dzdx = dz_dx(zmin, zmax)
        # z_nodes_T = z_nodes.reshape(self.n, 1)

        Pmat_T, Qmat_T, Umat_T = ode_sys(z_nodes, *args)
        Pmat = Pmat_T.reshape(self.n, 1)
        Qmat = dzdx*(Qmat_T.reshape(self.n, 1))
        Umat = dzdx**2*(Umat_T.reshape(self.n, 1))
        # Pmat, Qmat, Umat = ode_sys(z_nodes_T, *args)
        result = CoefficientGenerator(Pmat, Qmat, Umat, self.Tmat2, self.Tmat1, self.Tmat0, xp.asarray(y0), self.bcs)

        return result.coeffs, result.error
    
    def solve_ode(self, ode_sys, y0, args = (), domain = [-1, 1]):
        assert (len(y0) >= 2)
        yi = y0.copy()
        if self.bctype == 1:
            dzdx = dz_dx(domain[0], domain[1])
            for i in range(len(yi)):
                yi[i] *= (dzdx)**(i)

        coeffs, error = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = domain)
        cheby = Chebyshev(coeffs, domain = domain)
        cheby.error = error
        return cheby
    
    def __call__(self, ode_sys, y0, args = (), domain = [-1, 1], **kwargs):
        return self.solve_ode(ode_sys, y0, args = args, domain = domain)
    
class CollocationODESolver:
    def __init__(self, n = 16, chtype = 1, bctype = 1, adapative = True, nmax = 64, n_buffer_size = 10):
        self.adaptive = adapative
        self.nmax = nmax
        if self.nmax < n or self.adaptive is False:
            self.nmax = n

        delta_n = self.nmax - n
        n_step_size = int(xp.ceil(delta_n / n_buffer_size))
   
        i = 0 
        nstep = n + i*n_step_size
        n_keys = [nstep]
        while nstep < self.nmax:
            i += 1
            nstep = n + i*n_step_size
            n_keys.append(nstep)

        n_keys[-1] = self.nmax

        self.solver = dict.fromkeys(n_keys)

        for nn in self.solver.keys():
            self.solver[nn] = CollocationODEFixedStepSolver(nn, chtype, bctype)
    
    @property
    def type(self):
        return "dynamic"

    def solve_ode(self, ode_sys, y0, args = (), domain = [-1, 1], tol = 1e-8):
        n_keys = list(self.solver.keys())

        i = 0
        n = n_keys[0]
        sol = self.solver[n](ode_sys, y0, args = args, domain = domain)
        error = sol.error
        while error > tol and n < self.nmax:
            i += 1
            n = n_keys[i]
            sol = self.solver[n](ode_sys, y0, args = args, domain = domain)
            error = sol.error

        return sol

    def __call__(self, ode_sys, y0, args = (), domain = [-1, 1], tol = 1e-8, **kwargs):
        return self.solve_ode(ode_sys, y0, args = args, domain = domain, tol = tol)
    
class CollocationTransformation:
    def __init__(self, n = 16, chtype = 1):
        self.n = n
        self.chtype = chtype

        assert chtype in _CHEBY_TYPES

        if n in _precomputed_grid_numbers:
            self.nodes = _CHEBY_nodes[chtype][n]
            self.Tmat0 = _CHEBY_Tmat0[chtype][n]
            self.Tmat1 = _CHEBY_Tmat1[chtype][n]
            self.Tmat0Inv = _CHEBY_Tmat0Inv[chtype][n]
        elif n < _CHEBY_dsize:
            D = _CHEBY_DMAX[:n,:n]
            if chtype == 1:
                self.nodes = ch.chebpts1(n)
            else:
                self.nodes = ch.chebpts2(n)

            self.Tmat0 = xp.array(ch.chebvander(self.nodes, n-1))
            self.Tmat1 = xp.matmul(self.Tmat0, D)
            self.Tmat0Inv = xp.linalg.inv(self.Tmat0)
        else:
            raise ValueError(f"Too many collocation points requested. Maximum number of collocation points is set to {_CHEBY_dsize}.")

    def transform_coeffs(self, transform_sys, coeffs, args = (), domain = [-1, 1]):
        [zmin, zmax] = domain
    
        z_nodes = z_of_x(self.nodes, zmin, zmax)
        dzdx = dz_dx(zmin, zmax)
        z_nodes_T = z_nodes.reshape(self.n, 1)

        Fmat, Gmat = transform_sys(z_nodes_T, *args)
        Gmat = dzdx*Gmat

        Mmat = xp.multiply(Gmat, self.Tmat1) + xp.multiply(Fmat, self.Tmat0)

        new_coeffs = xp.matmul(self.Tmat0Inv, xp.matmul(Mmat, coeffs))
        return new_coeffs
    
    def transform(self, transform_sys, coeffs, args = (), domain = [-1, 1]):
        coeffs = self.solve_transform_coeffs(transform_sys, coeffs, args = args, domain = domain)
        return Chebyshev(coeffs, domain = domain)
    
    def __call__(self, transform_sys, coeffs, args = (), domain = [-1, 1], **kwargs):
        return self.transform(transform_sys, coeffs, args = args, domain = domain)
    
class CollocationAlgebra:
    def __init__(self, n = 16, chtype = 1):
        self.n = n
        self.chtype = chtype

        assert chtype in _CHEBY_TYPES
        
        if n in _precomputed_grid_numbers:
            self.nodes = _CHEBY_nodes[chtype][n]
            self.Tmat0 = _CHEBY_Tmat0[chtype][n]
            self.Tmat0Inv = _CHEBY_Tmat0Inv[chtype][n]
        elif n < _CHEBY_dsize:
            D = _CHEBY_DMAX[:n,:n]
            if chtype == 1:
                self.nodes = ch.chebpts1(n)
            else:
                self.nodes = ch.chebpts2(n)

            self.Tmat0 = xp.array(ch.chebvander(self.nodes, n-1))
            self.Tmat0Inv = xp.linalg.inv(self.Tmat0)
        else:
            raise ValueError(f"Too many collocation points requested. Maximum number of collocation points is set to {_CHEBY_dsize}.")
    
    def solve_sys_coeffs(self, algebraic_sys, args = (), domain = [-1, 1], include_Tmat = False):
        [zmin, zmax] = domain
    
        z_nodes = z_of_x(self.nodes, zmin, zmax)
        z_nodes_T = z_nodes.reshape(self.n, 1)

        if include_Tmat:
            Fmat = algebraic_sys(z_nodes_T, self.Tmat0, *args)
        else:
            Fmat = algebraic_sys(z_nodes_T, *args)

        new_coeffs = xp.matmul(self.Tmat0Inv, Fmat).flatten()
        return new_coeffs
    
    def solve_sys(self, algebraic_sys, args = (), domain = [-1, 1], include_Tmat = False):
        coeffs = self.solve_sys_coeffs(algebraic_sys, args = args, domain = domain, include_Tmat = include_Tmat)
        return Chebyshev(coeffs, domain = domain)
    
    def interpolate(self, algebraic_sys, args = (), domain = [-1, 1]):
        def temp_sys(x, *args2):
            z = z_of_x(x, domain[0], domain[1])
            return algebraic_sys(z, *args2)

        coeffs = ch.chebinterpolate(temp_sys, self.n - 1, args = args)

        return Chebyshev(coeffs, domain = domain)
    
    def __call__(self, algebraic_sys, args = (), domain = [-1, 1], include_Tmat = False, **kwargs):
        return self.solve_sys(algebraic_sys, args = args, domain = domain, include_Tmat = include_Tmat)
    
class CollocationAlgebraMultiDomain(CollocationAlgebra):
    def solve_sys(self, algebraic_sys, args = (), subdomains = [-1, 1], include_Tmat = False):
        ndomains = len(subdomains) - 1
        test = algebraic_sys(0.5, *args)
        coeffs = np.empty((ndomains, self.n), dtype = type(test))
        for i in range(ndomains):
            coeffs[i] = self.solve_sys_coeffs(algebraic_sys, args = args, domain = [subdomains[i], subdomains[i+1]], include_Tmat = include_Tmat)
        return MultiDomainChebyshev(coeffs, subdomains)
    
    def __call__(self, algebraic_sys, args = (), domain = [-1, 1], include_Tmat = False, **kwargs):
        return self.solve_sys(algebraic_sys, args = args, subdomains = domain, include_Tmat = include_Tmat)
    
class CollocationODEFixedMultiDomainFixedStepSolver(CollocationODEFixedStepSolver):
    def __init__(self, n = 16, chtype = 1, bctype = 1):
        super().__init__(n, chtype, bctype)

        if n in _precomputed_grid_numbers:
            self.bcs_end = _CHEBY_bcs_end[bctype][n]
        elif n < _CHEBY_dsize:
            D = _CHEBY_DMAX[:n,:n]
            self.bcs_end = dict.fromkeys(_BC_NUMS)

            if bctype == 1:
                self.bcs_end[0] = xp.array(ch.chebvander(1, n - 1))
                for i in range(1, _BC_dsize):
                    self.bcs_end[i] = xp.matmul(self.bcs_end[i-1], D)                    
            else:
                temp = ch.chebpts2(n)
                for num in self.bcs_end.keys():
                    self.bcs_end[num] = xp.array(ch.chebvander(temp[-1-num], n - 1))
    
    @property
    def type(self):
        return "fixed"

    def solve_ode_multi_coeffs(self, ode_sys, y0, subdomains, args=()):
        domain_num = len(subdomains) - 1

        coeff_array = []
        yi = y0.copy()
        subdomain = [subdomains[0], subdomains[1]]
        coeff_temp = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
        coeff_array.append(coeff_temp[0])
        for j in range(len(yi)):
            yi[j] = xp.dot(self.bcs_end[j], coeff_array[0])

        for i in range(1, domain_num):
            if self.bctype == 1:
                dzdx0 = dz_dx(subdomains[i-1], subdomains[i])
                dzdx1 = dz_dx(subdomains[i], subdomains[i+1])
                for jj in range(len(yi)):
                    yi[jj] *= (dzdx1/dzdx0)**(jj)
            subdomain = [subdomains[i], subdomains[i+1]]
            coeff_temp = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
            coeff_array.append(coeff_temp[0])
            for j in range(len(yi)):
                yi[j] = xp.dot(self.bcs_end[j], coeff_array[i])
        coeff_array = np.array(coeff_array)

        return coeff_array
    
    def solve_ode_multi_domain(self, ode_sys, y0, args=(), domain=[-1, 1], subdomains = 16, spacing = 'linear'):
        if isinstance(subdomains, int):
            subdomains = generate_subdomain_list(domain, subdomains, spacing)
        assert (isinstance(subdomains, list) or isinstance(subdomains, tuple) or isinstance(subdomains, xp.ndarray))
        
        yi = y0.copy()
        if self.bctype == 1:
            dzdx = dz_dx(subdomains[0], subdomains[1])
            for i in range(len(yi)):
                yi[i] *= (dzdx)**(i)

        coeff_array = self.solve_ode_multi_coeffs(ode_sys, yi, subdomains, args = args)

        return MultiDomainChebyshev(coeff_array, subdomains)
    
    def __call__(self, ode_sys, y0, args=(), domain=[-1, 1], subdomains = 16, spacing = 'linear', **kwargs):
        return self.solve_ode_multi_domain(ode_sys, y0, args = args, domain = domain, subdomains = subdomains, spacing = spacing)

def subdivide_subdomains(subdomains, i):
    subdomains_temp = xp.empty(len(subdomains) + 1)
    subdomains_temp[:i] = subdomains[:i]
    subdomains_temp[i] = subdomains[i]
    subdomains_temp[i+1] = 0.5*(subdomains[i] + subdomains[i+1])
    subdomains_temp[i+2:] = subdomains[i+1:]
    return subdomains_temp

class CollocationODEMultiDomainFixedStepSolver(CollocationODEFixedStepSolver):
    def __init__(self, n = 16, chtype = 1, bctype = 1):
        super().__init__(n, chtype, bctype)

        if n in _precomputed_grid_numbers:
            self.bcs_end = _CHEBY_bcs_end[bctype][n]
        elif n < _CHEBY_dsize:
            D = _CHEBY_DMAX[:n,:n]
            self.bcs_end = dict.fromkeys(_BC_NUMS)

            if bctype == 1:
                self.bcs_end[0] = xp.array(ch.chebvander(1, n - 1))
                for i in range(1, _BC_dsize):
                    self.bcs_end[i] = xp.matmul(self.bcs_end[i-1], D)                    
            else:
                temp = ch.chebpts2(n)
                for num in self.bcs_end.keys():
                    self.bcs_end[num] = xp.array(ch.chebvander(temp[-1-num], n - 1))

    @property
    def type(self):
        return "dynamic"

    def solve_ode_multi_coeffs(self, ode_sys, y0, subdomains, args=(), tol = 1e-12):
        domain_num = len(subdomains) - 1
        itermax = 100
        atol = 5e-16/tol

        coeff_array = []
        yi = y0.copy()
        subdomain = [subdomains[0], subdomains[1]]
        coeff_temp, error0 = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
        diff0 = xp.abs(1 - (atol + xp.dot(self.bcs_end[0], coeff_temp))/(atol + xp.dot(self.bcs_end[0][0,:-1], coeff_temp[:-1])))
        diff1 = xp.abs(1 - (atol + xp.dot(self.bcs_end[1], coeff_temp))/(atol + xp.dot(self.bcs_end[1][0,:-1], coeff_temp[:-1])))
        error = xp.max([error0, diff0[0], diff1[0]])
        # print("Before")
        # print(error, coeff_temp)
        # print(subdomains)
        # print(0, domain_num)

        ii = 0
        while error > tol and ii < itermax:
            subdomains_temp = subdivide_subdomains(subdomains, 0)
            if self.bctype == 1:
                dzdx0 = dz_dx(subdomains[0], subdomains[1])
                dzdx1 = dz_dx(subdomains_temp[0], subdomains_temp[1])
                for jj in range(len(yi)):
                    yi[jj] *= (dzdx1/dzdx0)**(jj)
            subdomains = subdomains_temp.copy()
            subdomain = [subdomains[0], subdomains[1]]
            coeff_temp, error0 = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
            diff0 = xp.abs(1 - (atol + xp.dot(self.bcs_end[0], coeff_temp))/(atol + xp.dot(self.bcs_end[0][0,:-1], coeff_temp[:-1])))
            diff1 = xp.abs(1 - (atol + xp.dot(self.bcs_end[1], coeff_temp))/(atol + xp.dot(self.bcs_end[1][0,:-1], coeff_temp[:-1])))
            error = xp.max([error0, diff0[0], diff1[0]])
            ii += 1
        domain_num += ii
        # print("After")
        # print(error, coeff_temp)
        # print(subdomains)
        # print(0, domain_num)

        coeff_array.append(coeff_temp)
        for j in range(len(yi)):
            yi[j] = xp.dot(self.bcs_end[j], coeff_array[0])[0]

        i = 1
        while i < domain_num: 
            if self.bctype == 1:
                dzdx0 = dz_dx(subdomains[i-1], subdomains[i])
                dzdx1 = dz_dx(subdomains[i], subdomains[i+1])
                for jj in range(len(yi)):
                    yi[jj] *= (dzdx1/dzdx0)**(jj)
            subdomain = [subdomains[i], subdomains[i+1]]
            coeff_temp, error0 = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
            diff0 = xp.abs(1 - (atol + xp.dot(self.bcs_end[0], coeff_temp))/(atol + xp.dot(self.bcs_end[0][0,:-1], coeff_temp[:-1])))
            diff1 = xp.abs(1 - (atol + xp.dot(self.bcs_end[1], coeff_temp))/(atol + xp.dot(self.bcs_end[1][0,:-1], coeff_temp[:-1])))
            error = xp.max([error0, diff0[0], diff1[0]])
            # print("Before")
            # print(error, coeff_temp)
            # print(subdomains)
            # print(i, domain_num)

            ii = 0
            while error > tol and ii < itermax:
                subdomains_temp = subdivide_subdomains(subdomains, i)
                if np.abs(subdomains_temp[i+1] - subdomains_temp[i]) > 5e-6:
                    if self.bctype == 1:
                        dzdx0 = dz_dx(subdomains[i], subdomains[i+1])
                        dzdx1 = dz_dx(subdomains_temp[i], subdomains_temp[i+1])
                        for jj in range(len(yi)):
                            yi[jj] *= (dzdx1/dzdx0)**(jj)
                    subdomains = subdomains_temp.copy()
                    subdomain = [subdomains[i], subdomains[i+1]]
                    coeff_temp, error0 = self.solve_ode_coeffs(ode_sys, yi, args = args, domain = subdomain)
                    diff0 = xp.abs(1 - (atol + xp.dot(self.bcs_end[0], coeff_temp))/(atol + xp.dot(self.bcs_end[0][0,:-1], coeff_temp[:-1])))
                    diff1 = xp.abs(1 - (atol + xp.dot(self.bcs_end[1], coeff_temp))/(atol + xp.dot(self.bcs_end[1][0,:-1], coeff_temp[:-1])))
                    error = xp.max([error0, diff0[0], diff1[0]])
                    ii += 1
                else:
                    print("ERROR!!!!!!!!!!!!!!!!!!!!!!")
                    error = 0

            domain_num += ii
            # print("After")
            # print(error, coeff_temp)
            # print(subdomains)
            # print(i, domain_num)
            coeff_array.append(coeff_temp)
            for j in range(len(yi)):
                yi[j] = xp.dot(self.bcs_end[j], coeff_array[i])[0]
            
            i += 1
        
        coeff_array = np.array(coeff_array)

        return coeff_array, subdomains
    
    def solve_ode_multi_domain(self, ode_sys, y0, args=(), domain=[-1, 1], subdomains = 16, tol = 1e-12, spacing = 'linear'):
        if isinstance(subdomains, int):
            subdomains = generate_subdomain_list(domain, subdomains, spacing)
        assert (isinstance(subdomains, list) or isinstance(subdomains, tuple) or isinstance(subdomains, xp.ndarray))
        
        yi = y0.copy()
        if self.bctype == 1:
            dzdx = dz_dx(subdomains[0], subdomains[1])
            for i in range(len(yi)):
                yi[i] *= (dzdx)**(i)

        coeff_array, subdomains_out = self.solve_ode_multi_coeffs(ode_sys, yi, subdomains, args = args, tol = tol)

        return MultiDomainChebyshev(coeff_array, subdomains_out)
    
    def __call__(self, ode_sys, y0, args=(), domain=[-1, 1], subdomains = 16, tol = 1e-12, spacing = 'linear', **kwargs):
        return self.solve_ode_multi_domain(ode_sys, y0, args = args, domain = domain, subdomains = subdomains, tol = tol, spacing = spacing)
    

def generate_subdomain_list(domain, ndomains, spacing = 'linear'):
        if spacing == 'log':
            subdomains = xp.logspace(np.log10(domain[0]), np.log10(domain[1]), ndomains + 1)
        elif spacing == 'sqrt':
            subdomains = xp.sqrt(xp.linspace(domain[0]**2, domain[1]**2, ndomains + 1))
        elif spacing == 'third':
            subdomains = (xp.linspace(domain[0]**3, domain[1]**3, ndomains + 1))**(1./3.)
        elif spacing == 'linear':
            subdomains = xp.linspace(domain[0], domain[1], ndomains + 1)
        elif spacing == 'arcsinh':
            subdomains = 0.5*(xp.arcsinh(xp.linspace(xp.sinh(-3), xp.sinh(3), ndomains + 1))*(domain[1] - domain[0])/3 + (domain[1] + domain[0]))
        elif spacing == 'arcsinh5':
            subdomains = 0.5*(xp.arcsinh(xp.linspace(xp.sinh(-5), xp.sinh(5), ndomains + 1))*(domain[1] - domain[0])/5 + (domain[1] + domain[0]))
        elif spacing == 'arcsinh7':
            subdomains = 0.5*(xp.arcsinh(xp.linspace(xp.sinh(-7), xp.sinh(7), ndomains + 1))*(domain[1] - domain[0])/7 + (domain[1] + domain[0]))
        else:
            subdomains = xp.linspace(domain[0], domain[1], ndomains + 1)

        return subdomains