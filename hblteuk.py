# hblteuk.py
"""
Teukolsky solvers using hyperboloidal slicing, compactification, and numerical collocation
Author: Zachary Nasipak
"""

# Load necessary modules
# use_gpu = False # No GPU capability at the moment, but maybe in the future?
# if use_gpu:
#     import cupy as xp
# else:
#     import numpy as xp
    
import numpy as np
import numpy as xp # This is leaving open the introduction of GPUs in the future
import numpy.polynomial.chebyshev as ch
#import scipy
import time

"""
GLOBAL VARIABLES
I precompute some matrices and arrays corresponding to the Chebyshev polynomials
and the collocation points at the Chebyshev nodes
"""

# construct the differential matrix '_HBLTEUK_D100' that gives the derivative of a Chebyshev series
_HBLTEUK_dsize = 500
_HBLTEUK_D100 = xp.zeros([_HBLTEUK_dsize,_HBLTEUK_dsize], np.float32)
_HBLTEUK_j = 1
while _HBLTEUK_j < _HBLTEUK_dsize:
    _HBLTEUK_D100[0][_HBLTEUK_j] = _HBLTEUK_j
    _HBLTEUK_j += 2
for _HBLTEUK_i in range(1,_HBLTEUK_dsize):
    _HBLTEUK_j = _HBLTEUK_i + 1
    while _HBLTEUK_j < _HBLTEUK_dsize:
        _HBLTEUK_D100[_HBLTEUK_i][_HBLTEUK_j] = 2*_HBLTEUK_j
        _HBLTEUK_j += 2

# I precompute everything with N = 16 and N = 32 points, since these are the most common sampling numbers I use
_HBLTEUK_nodes_16=ch.chebpts1(16)
_HBLTEUK_Tmat0_16=ch.chebvander(_HBLTEUK_nodes_16, 15)
_HBLTEUK_Tmat1_16 = xp.matmul(_HBLTEUK_Tmat0_16, _HBLTEUK_D100[:16,:16])
_HBLTEUK_Tmat2_16 = xp.matmul(_HBLTEUK_Tmat1_16, _HBLTEUK_D100[:16,:16])
_HBLTEUK_Tmat0Inv_16 = xp.linalg.inv(_HBLTEUK_Tmat0_16)
_HBLTEUK_bc1_16 = xp.array(ch.chebvander(-1, 15))
_HBLTEUK_bc2_16 = xp.matmul(_HBLTEUK_bc1_16, _HBLTEUK_D100[:16,:16])

_HBLTEUK_nodes_32=ch.chebpts1(32)
_HBLTEUK_Tmat0_32=ch.chebvander(_HBLTEUK_nodes_32, 31)
_HBLTEUK_Tmat1_32 = xp.matmul(_HBLTEUK_Tmat0_32, _HBLTEUK_D100[:32,:32])
_HBLTEUK_Tmat2_32 = xp.matmul(_HBLTEUK_Tmat1_32, _HBLTEUK_D100[:32,:32])
_HBLTEUK_Tmat0Inv_32 = xp.linalg.inv(_HBLTEUK_Tmat0_32)
_HBLTEUK_bc1_32 = xp.array(ch.chebvander(-1, 31))
_HBLTEUK_bc2_32 = xp.matmul(_HBLTEUK_bc1_32, _HBLTEUK_D100[:32,:32])

"""
class MultiChebyshev
This class joins together M Chebyshev series f_i of length N that are joined on adjacent domains [x_i, x_{i+1}] 
for x_i \in (x0, x1, ..., x_M). 

For example, consider M = 3 Chebyshev series of length 3: 
    f_0(x) = \sum_{n=0}^2 c_{0,n} T_n(X_0(x)), x \in [x_0, x_1]
    f_1(x) = \sum_{n=0}^2 c_{1,n} T_n(X_1(x)), x \in [x_1, x_2]
    f_2(x) = \sum_{n=0}^2 c_{2,n} T_n(X_2(x)), x \in [x_2, x_3]
    X_i(x) = (2*x - x_i - x_{i+1})/(x_{i+1} - x_{i})
Then MultiChebyshev joins together these solutions to create f(x), where x is no valid for [x_0, x_3]

Methods
-------

    __init__(coeffList, domainList)
        Input
        -----
        - coeffList: (array) (M x N)-dimensional numpy array of the Chebyshev coefficients c_{i, n}. In the above 
            example the coeffList would have the form
                [[c_{0,0}, c_{0,1}, c_{0,2}], 
                 [c_{1,0}, c_{1,1}, c_{1,2}],
                 [c_{2,0}, c_{2,1}, c_{2,2}]]
        - domainList: (array) M-dimensional numpy array of the subdomain boundaries x_i. In the above example
            the domainList would have the form
                [x_0, x_1, x_2, x_3]
    
    chebeval(x, deriv=0)
        Input
        -----
        - x: (float or array) point(s) at which we evaluate f(x)
        - deriv: (int) Setting deriv = i > 0 takes the ith-derivative of f
        
        Output
        ------
          (float or array) f(x) or its ith-derivative
        
    deriv(i)
        Input
        -----
        - i: (int) the ith-derivative of f(x)
        
        Output
        ------
          (MultiChebyshev) \partial_x^i f(x)
          
    __call__(sigma, deriv=0)
        This is equivalent to calling chebeval
"""

class MultiChebyshev:
    def __init__(self, coeffList, domainList):
        self.coeffs = coeffList
        self.domains = domainList
        self.n_cs, self.n_sample = self.coeffs.shape
        self.chebylist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            self.chebylist[i] = ch.Chebyshev(self.coeffs[i], domain=[self.domains[i], self.domains[i+1]])
        if self.domains[0] > self.domains[1]:
            self.sorted_chebylist = self.chebylist[::-1] # reverse
            self.sorted_domains = self.domains[::-1]
        else:
            self.sorted_chebylist = self.chebylist
            self.sorted_domains = self.domains
    
    def __deriv_list(self, m):
        derivlist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            derivlist[i] = self.sorted_chebylist[i].deriv(m)
            
        return derivlist
            
    def chebeval(self, sigma, deriv=0):
        if isinstance(sigma, np.ndarray):
            if deriv == 0:
                return multi_chebyshev_vec_no_deriv(sigma, self.sorted_chebylist, self.sorted_domains)
            else:
                return multi_chebyshev_vec_no_deriv(sigma, self.__deriv_list(deriv), self.sorted_domains)
        else:
            if deriv == 0:
                return multi_chebyshev_no_deriv(sigma, self.sorted_chebylist, self.sorted_domains)
            else:
                return multi_chebyshev_no_deriv(sigma, self.__deriv_list(deriv), self.sorted_domains)
    
    def deriv(self, m):
        D = _HBLTEUK_D100[:self.n_sample-1, :self.n_sample-1] # differential matrix
        derivcoefflist = np.empty((self.n_cs, self.n_sample - 1), dtype=xp.cdouble)
        for i in range(self.n_cs):
            derivcoefflist[i] = self.coeffs[i][:-1] # throw away last coefficient because you lose one-order in the series for each derivative
            dx = 2/(self.domains[i+1]-self.domains[i]) # jacobian for variable transformation
            for _ in range(m): # apply derivative operator m-times
                derivcoefflist[i] = dx*xp.matmul(D, derivcoefflist[i])
            
        return MultiChebyshev(derivcoefflist, self.domains)
        
    def __call__(self, sigma, deriv=0):
        return self.chebeval(sigma, deriv)

def multi_chebyshev_no_deriv(sigma, funcs, domains):
    # search to see if sigma is within a certain subdomain
    # If it is in the ith subdomain, then evaluate the chebyshev series
    # that represents the solution in that subdomain
    for i in range(funcs.size):
        if sigma <= domains[i+1] and sigma >= domains[i]:
            return funcs[i](sigma)
    if sigma > domains[-1]:
        return funcs[-1](sigma)
    if sigma < domains[0]:
        return funcs[0](sigma)
    return 0

def multi_chebyshev_vec_no_deriv(sigmaList, funcs, domains):
    # vectorized version of multi_chebyshev_no_deriv for sigma
    chebevalList = xp.empty(sigmaList.size, dtype=complex)
    for i,sigma in enumerate(sigmaList):
        chebevalList[i] = multi_chebyshev_no_deriv(sigma, funcs, domains)
    
    return chebevalList

"""
class HyperboloidalTeukolskySolution
This class acts as a container for the hyperboloidal-sliced, compactified, and rescaled Teukolsky solutions
$\Psi(\sigma) = R(\sigma)/Z(\sigma)$

Methods
-------

    __init__(s, l, omega, bc, mch)
        Input
        -----
        - s: (int) spin-weight of the solution
        - l: (int) spheroidal multipole mode number of the solution
        - omega: (float) frequency of the solution
        - bc: (str) boundary conditions that generated the solution. 
            Common options include 'In' or 'Up'.
        - mch: (MultiChebyshev) An instance of MultiChebyshev that holds the underlying function that 
            represents the Teukolsky solutions
            
    sol(sigma)
        Input
        -----
        - sigma: (float or array) compactified radial coordinate
        
        Output
        ------
          (float or array) Solution evaluated at sigma \Psi(\sigma)
    
    deriv(sigma)
        Input
        -----
        - sigma: (float or array) compactified radial coordinate
        
        Output
        ------
          (float or array) Derivative of the solution \partial_\sigma \Psi(\sigma)
          
    __call__(sigma, deriv=0)
        Input
        -----
        - sigma: (float or array) compactified radial coordinate
        - deriv: (int) Setting deriv = i > 0 takes the ith-derivative of the solution
        
        Output
        ------
          (float or array) \partial_\sigma^{deriv} \Psi(\sigma)
"""
    
class HyperboloidalTeukolskySolution:    
    def __init__(self, s, l, omega, bc, mch):
        self.s = s
        self.l = l
        self.frequency = omega
        self.bc = bc
        
        self.spin = 0
        self.shifted_eigenvalue = self.l*(self.l + 1)
        self.eigenvalue = self.shifted_eigenvalue - self.s*(self.s + 1)
        
        self.mch = mch
        self.domains = mch.domains
        self.coeffs = mch.coeffs
        self.mch_deriv = mch.deriv(1)

        if self.domains[0] < self.domains[-1]:
            self.domain = [self.domains[0], self.domains[-1]]
        else:
            self.domain = [self.domains[-1], self.domains[0]]
    
    def linspace_var(self, n=100):
        return xp.linspace(self.domain[0], self.domain[1], n)
    
    def linspace(self, n=100):
        sigma = self.linspace_var(n)
        return [sigma, self.sol(sigma)]
        
    def sol(self, sigma):
        return self.mch.chebeval(sigma)
    
    def deriv(self, sigma):
        return self.mch_deriv.chebeval(sigma)
    
    def _repr_latex_(self):
        bc_str = str(self.bc)
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$\psi^\mathrm{'+bc_str+'}_{sl\omega}(\sigma; '+sub_str+')$, $\sigma \in '+domain_str+' $}'
    
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.sol(sigma)
        elif deriv == 1:
            return self.deriv(sigma)
        else:
            self.mch(sigma, deriv)
            
"""
class RadialTeukolskySolution
This class acts as a container for the Teukolsky solutions in traditional Boyer-Lindquist coordinates
$R(r) = Z(\sigma(r)) \Psi(\sigma(r))$

Methods
-------

    __init__(s, l, omega, bc, mch)
        Input
        -----
        - s: (int) spin-weight of the solution
        - l: (int) spheroidal multipole mode number of the solution
        - omega: (float) frequency of the solution
        - bc: (str) boundary conditions that generated the solution. 
            Common options include 'In' or 'Up'.
        - mch: (MultiChebyshev) An instance of MultiChebyshev that holds the underlying function that 
            represents the Teukolsky solutions
            
    sol(r)
        Input
        -----
        - r: (float or array) Boyer-Lindquist radial coordinate
        
        Output
        ------
          (float or array) R(r)
    
    deriv(r)
        Input
        -----
        - r: (float or array) Boyer-Lindquist radial coordinate
        
        Output
        ------
          (float or array) \partial_r R(r)
          
    __call__(r, deriv=0)
        Input
        -----
        - r: (float or array) Boyer-Lindquist radial coordinate
        - deriv: (int) Setting deriv = i > 0 takes the ith-derivative of the solution
        
        Output
        ------
          (float or array) R(r) or \partial_r R(r)
"""
            
class RadialTeukolskySolution:    
    def __init__(self, s, l, omega, bc, mch):
        self.s = s
        self.l = l
        self.frequency = omega
        self.bc = bc
        
        self.spin = 0
        self.shifted_eigenvalue = self.l*(self.l + 1)
        self.eigenvalue = self.shifted_eigenvalue - self.s*(self.s + 1)
        
        self.mch = mch
        self.domains = mch.domains
        self.coeffs = mch.coeffs
        self.mch_deriv = mch.deriv(1)
        
        if self.domains[0] == 0.:
            domain_max = xp.inf
        else:
            domain_max = self.r_of_sigma(self.domains[0])
        if self.domains[-1] == 0.:
            domain_min = xp.inf
        else:
            domain_min = self.r_of_sigma(self.domains[-1])
        if domain_min < domain_max:
            self.domain = [domain_min, domain_max]
        else:
            self.domain = [domain_max, domain_min]
        
    @staticmethod
    def sigma_of_r(r):
        return 2./r
    
    @staticmethod
    def dsigma_dr(r):
        return -2./r**2
    
    @staticmethod
    def r_of_sigma(sigma):
        return 2./sigma
    
    @staticmethod
    def dr_dsigma(r):
        return -r**2/2.
    
    def f_teukolsky_transformation(self, r):
        return r**(-2*self.s - 1)*np.exp(1j*self.frequency*r)*(1 - 2./r)**(-self.s)*(2./r*(1 - 2./r))**(-2j*self.frequency)
    
    def f_teukolsky_transformation_deriv(self, r):
        return r**(-2*self.s - 1)*np.exp(1j*self.frequency*r)*(1 - 2./r)**(-self.s)*(2./r*(1 - 2./r))**(-2j*self.frequency)*((2 - r + 1j*self.frequency*(-8 + r**2) + 2*self.s - 2*r*self.s)/((-2 + r)*r))
    
    def g_teukolsky_transformation_deriv(self, r):
        return (-2/r**2)*r**(-2*self.s - 1)*np.exp(1j*self.frequency*r)*(1 - 2./r)**(-self.s)*(2./r*(1 - 2./r))**(-2j*self.frequency)
    
    def linspace_var(self, cutoff, n=100):
        rmin, rmax = self.domain
        if rmax == np.inf:
            rmax = cutoff
        if rmin == 2.:
            rmin = cutoff
        return xp.linspace(rmin, rmax, n)
    
    def linspace(self, cutoff, n=100):
        r = self.linspace_var(cutoff, n)
        return [r, self.sol(r)]
        
    def sol(self, r):
        sigma = self.sigma_of_r(r)
        alpha = self.f_teukolsky_transformation(r)
        return alpha*self.mch.chebeval(sigma)
    
    def deriv(self, r):
        sigma = self.sigma_of_r(r)
        alpha = self.f_teukolsky_transformation_deriv(r)
        beta = self.g_teukolsky_transformation_deriv(r)
        return alpha*self.mch.chebeval(sigma) + beta*self.mch_deriv.chebeval(sigma)
    
    def _repr_latex_(self):
        bc_str = str(self.bc)
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$R^\mathrm{'+bc_str+'}_{sl\omega}(r; '+sub_str+')$, $r \in '+domain_str+' $}'
    
    def __call__(self, r, deriv=0):
        if deriv == 0:
            return self.sol(r)
        elif deriv == 1:
            return self.deriv(r)
        else:
            print('Error')

"""
class TeukolskySolver
This class generates Teukolsky solutions using hyperboloidal slicing, radial compactification, and 
Chebyshev collocation methods across a series of adjacent subdomains. Homogeneous solutions can
be constructed to either satisfy the traditional 'In' or 'Up' boundary conditions defined in the
literature (e.g., see Sasaki and Tagoshi, Living Reviews in Relativity). This serves as a base
class for RadialTeukolsky and HyperboloidalTeukolsky, and should not be directly called/accessed
by the user.

Methods
-------

    __init__(s, l, omega)
        Input
        -----
        - s: (int) spin-weight of the solution
        - l: (int) spheroidal multipole mode number of the solution
        - omega: (float) frequency of the solution
"""

class TeukolskySolver:
    def __init__(self, s, l, omega):
        self.s = s
        self.l = l
        self.frequency = omega
        self.blackholespin = 0.
        self.shifted_eigenvalue = self.l*(self.l + 1)
        self.eigenvalue = self.shifted_eigenvalue - self.s*(self.s + 1)

        self.bcs = ['In', 'Up']
        self.domains = dict.fromkeys(self.bcs)
        self.mch = dict.fromkeys(self.bcs)
        self.solution = dict.fromkeys(self.bcs)
        if abs(self.s) > 0:
            self.spinkeys = [self.s, -self.s]
        else:
            self.spinkeys = [self.s]
            
        self.coeffs = dict.fromkeys(self.spinkeys)
        self.coeffs[self.s] = dict.fromkeys(self.bcs)
        self.coeffs[-self.s] = dict.fromkeys(self.bcs)
        
        self.domain = dict.fromkeys(self.bcs)
        self.spinsign = {'In': -1, 'Up': 1}
    
    @staticmethod
    def p_hbl(sigma):
        return sigma**2*(1 - sigma)
    
    @staticmethod
    def q_hbl(sigma, s, omega):
        return 2*sigma*(1+s) - sigma**2*(3 + s - 8j*omega) - 4j*omega 

    @staticmethod
    def u_hbl(sigma, s, lam, omega):
        return -4j*omega*(4j*omega + s) - (1 - 4j*omega)*(1 + s - 4j*omega)*sigma - lam
    
    @staticmethod
    def xOfSigma(sigma, smin, smax):
        return (2*sigma - (smax + smin))/(smax - smin)

    @staticmethod
    def sigmaOfX(x, smin, smax):
        return ((smax - smin)*x + smax + smin)/2

    @staticmethod
    def dxOfSigma(smin, smax):
        return 2/(smax - smin)
    
    @staticmethod
    def fTS_plus_to_minus_2(sigma, lam, omega):
        return (2*omega)**(-4)*1/16.*((256j)*omega**3*sigma - lam*(2 + lam)*(-1 + sigma)*sigma**4 + 256*omega**4*(1 + sigma) + (4j)*omega*sigma**3*(lam*(-6 + 5*sigma) + sigma*(-5 + 6*sigma)) + 16*omega**2*sigma**2*(lam*(-3 + 2*sigma**2) + sigma*(-5 + sigma + 3*sigma**2)))/(1 - sigma)**3
    
    @staticmethod
    def gTS_plus_to_minus_2(sigma, lam, omega):
        return (2*omega)**(-4)*((-0.25j)*omega*sigma**2*(16*omega**2 + sigma**2*(2*lam*(-1 + sigma) + sigma*(-2 + 3*sigma))))/(-1 + sigma)**3
    
    @staticmethod
    def fTS_minus_to_plus_2(sigma, lam, omega):
        return (16*(128j*omega**3 + 256*omega**4 + lam*(2 + lam)*(-1 + sigma)**2*sigma**2 + 4j*omega*sigma*(sigma*(-3 + 5*sigma) + lam*(-2 + sigma + sigma**2)) + 16*omega**2*(lam*(-1 + sigma)*(1 + 2*sigma**2) + sigma*(-3 + sigma*(3 + sigma*(-2 + 3*sigma))))))/sigma**6/(128j*omega*(-1 + 2j*omega)*(-1 + 4j*omega)*(1 + 4j*omega))
    
    @staticmethod
    def gTS_minus_to_plus_2(sigma, lam, omega):
        return (64j*omega*(-1 + sigma)*(16*omega**2 + 2*lam*(-1 + sigma)*sigma**2 + sigma**3*(-2 + 3*sigma)))/sigma**6/(128j*omega*(-1 + 2j*omega)*(-1 + 4j*omega)*(1 + 4j*omega))
    
    @staticmethod # NOT YET IMPLEMENTED!!!!
    def fTS_plus_to_minus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod # NOT YET IMPLEMENTED!!!!
    def gTS_plus_to_minus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod # NOT YET IMPLEMENTED!!!!
    def fTS_minus_to_plus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod # NOT YET IMPLEMENTED!!!!
    def gTS_minus_to_plus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod
    def sigmaOfR(r):
        return 2./r
    
    @staticmethod
    def rOfSigma(sigma):
        return 2./sigma
        
    
    def __flip_spin_coeffs(self, clist, dlist):
        s = -abs(self.s)
        la = self.shifted_eigenvalue - s*(s+1)
        domain_num, n = clist.shape
    
        if n == 16:
            nodes = _HBLTEUK_nodes_16
            Tmat0 = _HBLTEUK_Tmat0_16
            Tmat1 = _HBLTEUK_Tmat1_16
            Tmat0Inv = _HBLTEUK_Tmat0Inv_16
        elif n == 32:
            nodes = _HBLTEUK_nodes_32
            Tmat0 = _HBLTEUK_Tmat0_32
            Tmat1 = _HBLTEUK_Tmat1_32
            Tmat0Inv = _HBLTEUK_Tmat0Inv_32
        else:
            D = _HBLTEUK_D100[:n,:n]
            nodes = ch.chebpts1(n)
            Tmat0 = xp.array(ch.chebvander(nodes, n-1))
            Tmat1 = xp.matmul(Tmat0, D)
            Tmat0Inv = xp.linalg.inv(Tmat0)
    
        sigmaNodesList = np.empty((domain_num, n))
        dxdsigma = np.empty(domain_num)
        for i in range(domain_num):
            sigmaNodesList[i] = self.sigmaOfX(nodes, dlist[i], dlist[i+1])
            dxdsigma[i] = self.dxOfSigma(dlist[i], dlist[i+1])
        
        sigmaNodesTList = sigmaNodesList.reshape(domain_num, n, 1)
        Mmat = np.empty((domain_num, n, n), dtype=np.cdouble)
        if self.s == 2:
            for i, sigmaNodes in enumerate(sigmaNodesTList):
                Fmat = self.fTS_minus_to_plus_2(sigmaNodes, la, self.frequency)
                Gmat = dxdsigma[i]*self.gTS_minus_to_plus_2(sigmaNodes, la, self.frequency)
                Mmat[i] = xp.multiply(Fmat, Tmat0) + xp.multiply(Gmat, Tmat1)
        elif self.s == -2:
            for i, sigmaNodes in enumerate(sigmaNodesTList):
                Fmat = self.fTS_plus_to_minus_2(sigmaNodes, la, self.frequency)
                Gmat = dxdsigma[i]*self.gTS_plus_to_minus_2(sigmaNodes, la, self.frequency)
                Mmat[i] = xp.multiply(Fmat, Tmat0) + xp.multiply(Gmat, Tmat1)
        elif self.s == 1: # NOT YET IMPLEMENTED!!!!
            for i, sigmaNodes in enumerate(sigmaNodesTList):
                Fmat = self.fTS_minus_to_plus_1(sigmaNodes, la, self.frequency)
                Gmat = dxdsigma[i]*self.gTS_minus_to_plus_1(sigmaNodes, la, self.frequency)
                Mmat[i] = xp.multiply(Fmat, Tmat0) + xp.multiply(Gmat, Tmat1)
        elif self.s == -1: # NOT YET IMPLEMENTED!!!!
            for i, sigmaNodes in enumerate(sigmaNodesTList):
                Fmat = self.fTS_plus_to_minus_1(sigmaNodes, la, self.frequency)
                Gmat = dxdsigma[i]*self.gTS_plus_to_minus_1(sigmaNodes, la, self.frequency)
                Mmat[i] = xp.multiply(Fmat, Tmat0) + xp.multiply(Gmat, Tmat1)
  
        vvec = xp.array(clist.reshape((domain_num, n, 1)))
        Tmat0InvTile = xp.tile(Tmat0Inv, (domain_num, 1)).reshape((domain_num, n, n))

        clist_new = xp.matmul(Tmat0InvTile, xp.matmul(Mmat, vvec))
        return clist_new.reshape((domain_num, n))
    
    def flip_spin(self, bc=['In','Up']):
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.flip_spin(condition)
            return None
        if self.coeffs[self.s][bc] is None:
            if self.coeffs[-self.s][bc] is None:
                print("ERROR: No coefficients to flip")
                return None
            self.coeffs[self.s][bc] = self.__flip_spin_coeffs(self.coeffs[-self.s][bc], self.domains[bc])
        elif self.coeffs[-self.s][bc] is None:
            self.coeffs[-self.s][bc] = self.__flip_spin_coeffs(self.coeffs[self.s][bc], self.domains[bc])
        return None
    
    def solve_hyperboloidal_coeffs_domain(self, s, psi0, dpsi0, domain, bc, n=16):
        [smin, smax] = domain
    
        if n == 16:
            nodes = _HBLTEUK_nodes_16
            Tmat0 = _HBLTEUK_Tmat0_16
            Tmat1 = _HBLTEUK_Tmat1_16
            Tmat2 = _HBLTEUK_Tmat2_16
            bc1 = _HBLTEUK_bc1_16
            bc2 = _HBLTEUK_bc2_16
        elif n == 32:
            nodes = _HBLTEUK_nodes_32
            Tmat0 = _HBLTEUK_Tmat0_32
            Tmat1 = _HBLTEUK_Tmat1_32
            Tmat2 = _HBLTEUK_Tmat2_32
            bc1 = _HBLTEUK_bc1_32
            bc2 = _HBLTEUK_bc2_32
        else:
            D = _HBLTEUK_D100[:n,:n]
            nodes = ch.chebpts1(n)
            Tmat0 = xp.array(ch.chebvander(nodes, n-1))
            Tmat1 = xp.matmul(Tmat0, D)
            Tmat2 = xp.matmul(Tmat1, D)
            bc1 = xp.array(ch.chebvander(-1, n - 1))
            bc2 = xp.matmul(bc1, D)
    
        la = self.shifted_eigenvalue - s*(1+s)
        sigmaNodes = self.sigmaOfX(nodes, smin, smax)
        dsigmadx = 1./self.dxOfSigma(smin, smax)
        sigmaNodesT = sigmaNodes.reshape(n, 1)
        Pmat = self.p_hbl(sigmaNodesT)
        Qmat = dsigmadx*self.q_hbl(sigmaNodesT, s, self.frequency)
        Umat = dsigmadx**2*self.u_hbl(sigmaNodesT, s, la, self.frequency)

        Mmat = xp.multiply(Pmat, Tmat2) + xp.multiply(Qmat, Tmat1) + xp.multiply(Umat, Tmat0)
        Mmat[0] = bc1
        Mmat[1] = bc2

        source = xp.zeros((n,1), dtype=complex)
        source[0,0] = psi0
        source[1,0] = dpsi0

        coeffs = xp.linalg.solve(Mmat, source).flatten()
        return coeffs
        
    def __solve_hyperboloidal_teukolsky_coeffs(self, s, bc, cutoff, subdomains, nsample=32):
        
        [smin, smax] = self.domain[bc]
        la = self.shifted_eigenvalue - s*(s+1)
        
        if subdomains == 0:
            boundaryNum = np.amax([16, 4*self.l])
        else:
            boundaryNum = subdomains
    
        mincut, maxcut = cutoff
            
        if bc == 'Up':
            if maxcut > 0. and maxcut < 1.:
                smax = maxcut
            a1 = a1sigma0(s, la, self.frequency)
            a2 = a2sigma0(s, la, self.frequency)
            smin = 0.5*xp.min(np.abs([1/a1, a1/a2]))
            if smin > smax:
                smin = smax
                boundaryNum = 1
        elif bc == 'In':
            if mincut > 0 and mincut < 1:
                smin = mincut
            b1 = b1sigma1(s, la, self.frequency)
            smax = 1 - 1/np.abs(b1)
            if smax < smin:
                smax = smin
                boundaryNum = 1
        else:
            print("Error")
        
        self.domains[bc] = np.zeros(boundaryNum + 1)
        self.domains[bc][1:] = smin*(smax/smin)**np.linspace(0, 1, num=boundaryNum)
        if bc == 'In':
            self.domains[bc][0] = 1
            self.domains[bc][1:] = self.domains[bc][1:][::-1] # reverse list

        smin = self.domains[bc][0]
        smax = self.domains[bc][1]
        dsigmadx = 1./dxOfSigma(smin, smax)
    
        if bc == 'In':
            psi0 = 2*np.exp(-4j*self.frequency)
            dpsi0 = -dsigmadx*b1sigma1(s, la, self.frequency)*psi0
        else:
            psi0 = 1.
            dpsi0 = dsigmadx*a1sigma0(s, la, self.frequency)

        self.coeffs[s][bc] = xp.empty((boundaryNum, nsample), dtype=xp.cdouble)
        
        i=0
        self.coeffs[s][bc][i] = self.solve_hyperboloidal_coeffs_domain(s, psi0, dpsi0, [smin, smax], bc, n=nsample)
        
        for boundary in self.domains[bc][2:]:
            psi0 = xp.sum(self.coeffs[s][bc][i])
            dpsi0 = xp.sum(xp.matmul(_HBLTEUK_D100[:nsample,:nsample],self.coeffs[s][bc][i]))/dsigmadx
            smin = smax
            smax = boundary
            dsigmadx = 1./dxOfSigma(smin, smax)
            dpsi0 *= dsigmadx
            i += 1
            
            self.coeffs[s][bc][i] = self.solve_hyperboloidal_coeffs_domain(s, psi0, dpsi0, [smin, smax], bc, n=nsample)
            
def default_smax():
    return 0.5

def default_smin(omega):
    return 0.03*omega**(2/3)

"""
class RadialTeukolsky

Methods
-------
"""
        
class RadialTeukolsky(TeukolskySolver):
    
    def solve(self, bc=['In','Up'], use_ts_transform=True, cutoff=[2,np.inf], subdomains=0, chebyshev_samples=16):
        # set domain for solver
        self.domain = {'In': [default_smin(self.frequency), 1.], 'Up': [0., default_smax()]}
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff, subdomains=subdomains, chebyshev_samples=chebyshev_samples)
            return None
            
        start = time.time()
        if bc not in self.bcs:
            print('ERROR: Invalid key type {}'.format(bc))
            return None
                
        if use_ts_transform:
            s = self.spinsign[bc]*abs(self.s)
        else:
            s = self.s
            
        if isinstance(cutoff, list) or isinstance(cutoff, np.ndarray):
            rmin, rmax = cutoff
            if rmax is np.inf:
                mincut = 0.
                maxcut = self.sigmaOfR(rmin)
            else:
                maxcut, mincut = self.sigmaOfR(np.array(cutoff))
        else:
            mincut = self.sigmaOfR(cutoff)
            maxcut = mincut
            
        self.domain = {'In': [default_smin(self.frequency), 1.], 'Up': [0., default_smax()]}
        self._TeukolskySolver__solve_hyperboloidal_teukolsky_coeffs(s, bc, [mincut, maxcut], subdomains, nsample=chebyshev_samples)
                
        if use_ts_transform and s*self.s < 0:
            self.flip_spin(bc)
        
        self.mch[bc] = MultiChebyshev(self.coeffs[self.s][bc], self.domains[bc])
        self.domains[bc] = self.mch[bc].domains
        smin = self.domains[bc][0]
        smax = self.domains[bc][-1]
        if smin > smax:
            if smax == 0.:
                rmax = np.inf
            else:
                rmax = self.rOfSigma(smax)
            rmin = self.rOfSigma(smin)
        else:
            if smin == 0.:
                rmax = np.inf
            else:
                rmax = self.rOfSigma(smin)
            rmin = self.rOfSigma(smax)
        
        self.domain[bc][1] = rmax
        self.domain[bc][0] = rmin

        self.solution[bc] = RadialTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
    
    def get_hyperboloidal(self, bc=None):
        if bc is None:
            hbl = dict.fromkeys(self.bcs)
            for bc in self.bcs:
                hbl[bc] = HyperboloidalTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
            return hbl
        elif bc in self.bcs:
            return HyperboloidalTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
        else:
            print('ERROR: Invalid key type {}'.format(bc))
            
    def get_teukolsky(self, bc=None):
        if bc is None:
            return self.solution
        elif bc in self.bcs:
            return self.solution[bc]
        else:
            print('ERROR: Invalid key type {}'.format(bc))
    
    def __call__(self, bc=None):
        return self.get_teukolsky(bc=bc)
    
    def _repr_latex_(self):
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        return r'$R^\mathrm{In/Up}_{sl\omega}(r; '+sub_str+')$'
    
"""
class HyperboloidalTeukolsky

Methods
-------
"""
    
class HyperboloidalTeukolsky(TeukolskySolver):
    def solve(self, bc=['In','Up'], use_ts_transform=True, cutoff=[0,1], subdomains=0, chebyshev_samples=16):
        # set domain for solver
        self.domain = {'In': [default_smin(self.frequency), 1.], 'Up': [0., default_smax()]}
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff, subdomains=subdomains, chebyshev_samples=chebyshev_samples)
            return None
            
        start = time.time()
        if bc not in self.bcs:
            print('ERROR: Invalid key type {}'.format(bc))
            return None
                
        if use_ts_transform:
            s = self.spinsign[bc]*abs(self.s)
        else:
            s = self.s
            
        self._TeukolskySolver__solve_hyperboloidal_teukolsky_coeffs(s, bc, cutoff, subdomains, nsample=chebyshev_samples)
                
        if use_ts_transform and s*self.s < 0:
            self.flip_spin(bc)
        
        self.mch[bc] = MultiChebyshev(self.coeffs[self.s][bc], self.domains[bc])
        self.domains[bc] = self.mch[bc].domains
        if self.domains[bc][0] < self.domains[bc][-1]:
            smin = self.domains[bc][0]
            smax = self.domains[bc][-1]
        else:
            smax = self.domains[bc][0]
            smin = self.domains[bc][-1]
        
        self.domain[bc][1] = smax
        self.domain[bc][0] = smin

        self.solution[bc] = HyperboloidalTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
        
    def test_accuracy(self, bc):
        if self.mch[bc] is None:
            print('ERROR: No solution available for {}'.format(bc))
        else:
            sigma = np.linspace(self.domain[bc][0], self.domain[bc][1])
            psi = self.mch[bc](sigma)
            dpsi = self.mch[bc].deriv(1)(sigma)
            dpsi2 = self.mch[bc].deriv(2)(sigma)
            return [sigma, self.p_hbl(sigma)*dpsi2/psi + self.q_hbl(sigma, self.s, self.frequency)*dpsi/psi + self.u_hbl(sigma, self.s, self.eigenvalue, self.frequency)]
    
    def get_teukolsky(self, bc=None):
        if bc is None:
            teuk = dict.fromkeys(self.bcs)
            for bc in self.bcs:
                teuk[bc] = RadialTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
            return teuk
        elif bc in self.bcs:
            return RadialTeukolskySolution(self.s, self.l, self.frequency, bc, self.mch[bc])
        else:
            print('ERROR: Invalid key type {}'.format(bc))
            
    def get_hyperboloidal(self, bc=None):
        if bc is None:
            return self.solution
        elif bc in self.bcs:
            return self.solution[bc]
        else:
            print('ERROR: Invalid key type {}'.format(bc))
    
    def __call__(self, bc=None):
        return self.get_hyperboloidal(bc=bc)
    
    def _repr_latex_(self):
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        return r'$\Psi^\mathrm{In/Up}_{sl\omega}(\sigma; '+sub_str+')$'

def a1sigma0(s, lam, omega):
    return -(lam + 4j*omega*(s + 4j*omega))/(4j*omega)

def a2sigma0(s, lam, omega):
    return (lam**2 + 2*lam*(-1 + 4j*omega)*(1 + 4j*omega + s) + 4j*omega*(-1 - 16*omega**2*(-1 + 4j*omega) + (1 + 4j*omega)*(-3 + 8j*omega)*s + (-2 + 4j*omega)*s**2))/(-32*omega**2)

def b1sigma1(s, lam, omega):
    return -(lam + (1 - 4j*omega)*(1 - 4j*omega + s) + 4j*omega*(4j*omega + s))/(-3 + 4j*omega - s + 2*(1 + s))

def xOfSigma(sigma, smin, smax):
    return (2*sigma - (smax + smin))/(smax - smin)

def sigmaOfX(x, smin, smax):
    return ((smax - smin)*x + smax + smin)/2

def dxOfSigma(smin, smax):
    return 2/(smax - smin)

def hfunc(sigma):
    return 2/sigma - 2*np.log(sigma) - 2*np.log(1 - sigma)

def rescaleteuksigma(s, omega, sigma):
    return 0.5*sigma**(1+2*s)*(4*(1 - sigma))**(-s)*np.exp(1j*omega*hfunc(sigma))

def rescaleteuk(s, omega, r):
    return rescaleteuksigma(s, omega, 2/r)