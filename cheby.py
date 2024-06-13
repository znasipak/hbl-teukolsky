# cheby.py

import numpy as np
import numpy as xp # This is leaving open the introduction of GPUs in the future
import numpy.polynomial.chebyshev as ch
from numpy.polynomial.chebyshev import Chebyshev as ChebyshevNP
from numba import njit
import numba as nb
from numba import config

def isvec(var):
    return isinstance(var, xp.ndarray) or isinstance(var, list)

@njit(cache=True)
def cheby_D_jit(n, dtype = np.float64):
    Dmat = xp.zeros((n, n), dtype=dtype)
    j = 1
    while j < n:
        Dmat[0,j] = j
        j += 2
    for i in range(1,n):
        j = i + 1
        while j < n:
            Dmat[i,j] = 2*j
            j += 2
    return Dmat

_CHEBY_dsize = 2000
_CHEBY_DMAX = cheby_D_jit(_CHEBY_dsize)

# @njit([nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:](nb.complex128[:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:](nb.float64[:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:](nb.complex128[:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_evaluation(coeffs, domain, x):
    # x = 0.5*((domain[1] - domain[0])*x_shift + domain[1] + domain[0])
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp1 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp0 = coeffs[N - 1]*(np.zeros(x.shape) + 1)
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:,:](nb.float64[:,:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_evaluation_2d(coeffs, domain, x):
    # x = 0.5*((domain[1] - domain[0])*x_shift + domain[1] + domain[0])
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    
    N = coeffs.shape[0]
    outshape = coeffs.shape + x.shape

    bkp2 = xp.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp1 = xp.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*xp.ones(outshape[1:])
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:](nb.complex128[:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:](nb.float64[:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:](nb.complex128[:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_deriv_evaluation(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp1 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp0 = coeffs[N - 1]*(np.zeros(x.shape) + 1)
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:,:](nb.float64[:,:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_deriv_evaluation_2d(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    
    N = coeffs.shape[0]
    outshape = coeffs.shape + x.shape

    bkp2 = xp.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp1 = xp.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*xp.ones(outshape[1:])
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64(nb.float64[:], nb.float64[:], nb.float64),
#        nb.complex128(nb.complex128[:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_evaluation_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = 0.*coeffs[N - 1]
    bkp1 = 0.*coeffs[N - 1]
    bkp0 = coeffs[N - 1]
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64),
#        nb.complex128[:](nb.complex128[:,:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_evaluation_2d_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])

    N = coeffs.shape[0]

    bkp2 = xp.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp1 = xp.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*xp.ones(coeffs.shape[1])
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64(nb.float64[:], nb.float64[:], nb.float64),
#        nb.complex128(nb.complex128[:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_deriv_evaluation_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = 0.*coeffs[N - 1]
    bkp1 = 0.*coeffs[N - 1]
    bkp0 = coeffs[N - 1]
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64),
#        nb.complex128[:](nb.complex128[:,:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_deriv_evaluation_2d_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])

    N = coeffs.shape[0]

    bkp2 = xp.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp1 = xp.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*xp.ones(coeffs.shape[1])
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit(nb.int64(nb.float64, nb.float64[:], nb.int64), cache=True)
def find_domain(sigma, domains, i0):
    for i in range(i0, len(domains) - 1):
        if xp.real(sigma) <= domains[i+1] and xp.real(sigma) >= domains[i]:
            return i
        
    for i in range(i0, -1, -1):
        if xp.real(sigma) <= domains[i+1] and xp.real(sigma) >= domains[i]:
            return i
        
    if xp.real(sigma) > domains[-1]:
        return len(domains) - 2
    
    if xp.real(sigma) < domains[0]:
        return 0
    
    return 0

def multi_chebyshev(sigma, coef_array, domains, ordering = 1, i0 = 0):
    # search to see if sigma is within a certain subdomain
    # If it is in the ith subdomain, then evaluate the chebyshev series
    # that represents the solution in that subdomain

    i0 = find_domain(sigma, domains, i0)
    return i0, clenshaw_evaluation_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigma)

def multi_chebyshev_deriv(sigma, coef_array, domains, ordering = 1, i0 = 0):
    # search to see if sigma is within a certain subdomain
    # If it is in the ith subdomain, then evaluate the chebyshev series
    # that represents the solution in that subdomain

    i0 = find_domain(sigma, domains, i0)
    return i0, clenshaw_deriv_evaluation_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigma)

# @njit(nb.int64[:](nb.float64[:], nb.float64[:], nb.int64), cache=True)
def find_domain_vec(sigmas, domains, i0):
    domain_ii = xp.empty(len(sigmas), dtype = xp.int64)
    for ii, sigma in enumerate(sigmas):
        i0 = find_domain(sigma, domains, i0)
        domain_ii[ii] = i0
    
    return domain_ii

def multi_chebyshev_vec(sigmas, coef_array, domains, ordering):
    sigmas_shape = sigmas.shape
    sigmas_1d = sigmas.flatten()
    return multi_chebyshev_jit_vec(sigmas_1d, coef_array, domains, ordering).reshape(sigmas_shape)

def multi_chebyshev_deriv_vec(sigmas, coef_array, domains, ordering):
    sigmas_shape = sigmas.shape
    sigmas_1d = sigmas.flatten()
    return multi_chebyshev_deriv_jit_vec(sigmas_1d, coef_array, domains, ordering).reshape(sigmas_shape)

# @njit([nb.float64[:](nb.float64[:], nb.float64[:,:], nb.float64[:], nb.int64),
#        nb.complex128[:](nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.int64)], cache=True)
def multi_chebyshev_jit_vec(sigmas, coef_array, domains, ordering):
    coefdtype = coef_array[0].dtype
    chebevalList = xp.empty(len(sigmas), dtype=coefdtype)
    domain_ii = find_domain_vec(sigmas, domains, 0)
    for i in range(len(sigmas)):
        i0 = domain_ii[i]
        chebevalList[i] = clenshaw_evaluation_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigmas[i])

    return chebevalList

# @njit([nb.float64[:,:](nb.float64[:], nb.float64[:,:,:], nb.float64[:], nb.int64),
#        nb.complex128[:,:](nb.float64[:], nb.complex128[:,:,:], nb.float64[:], nb.int64)], cache=True)
def multi_chebyshev_jit_vec_2d(sigmas, coef_array, domains, ordering):
    coefdtype = coef_array[0].dtype
    chebevalList = xp.empty((len(sigmas), coef_array.shape[-1]), dtype=coefdtype)
    domain_ii = find_domain_vec(sigmas, domains, 0)
    for i in range(len(sigmas)):
        i0 = domain_ii[i]
        chebevalList[i] = clenshaw_evaluation_2d_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigmas[i])

    return chebevalList

# @njit([nb.float64[:](nb.float64[:], nb.float64[:,:], nb.float64[:], nb.int64),
#        nb.complex128[:](nb.float64[:], nb.complex128[:,:], nb.float64[:], nb.int64)], cache=True)
def multi_chebyshev_deriv_jit_vec(sigmas, coef_array, domains, ordering):
    coefdtype = coef_array[0].dtype
    chebevalList = xp.empty(len(sigmas), dtype=coefdtype)
    domain_ii = find_domain_vec(sigmas, domains, 0)
    for i in range(len(sigmas)):
        i0 = domain_ii[i]
        chebevalList[i] = clenshaw_deriv_evaluation_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigmas[i])

    return chebevalList

# @njit([nb.float64[:,:](nb.float64[:], nb.float64[:,:,:], nb.float64[:], nb.int64),
#        nb.complex128[:,:](nb.float64[:], nb.complex128[:,:,:], nb.float64[:], nb.int64)], cache=True)
def multi_chebyshev_deriv_jit_vec_2d(sigmas, coef_array, domains, ordering):
    coefdtype = coef_array[0].dtype
    chebevalList = xp.empty((len(sigmas), coef_array.shape[-1]), dtype=coefdtype)
    domain_ii = find_domain_vec(sigmas, domains, 0)
    for i in range(len(sigmas)):
        i0 = domain_ii[i]
        chebevalList[i] = clenshaw_deriv_evaluation_2d_scalar(coef_array[i0], domains[i0:i0+2][::ordering], sigmas[i])

    return chebevalList

def multi_chebyshev_no_deriv(sigma, funcs, domains):
    # search to see if sigma is within a certain subdomain
    # If it is in the ith subdomain, then evaluate the chebyshev series
    # that represents the solution in that subdomain
    for i in range(funcs.size):
        if xp.real(sigma) <= domains[i+1] and xp.real(sigma) >= domains[i]:
            return funcs[i](sigma)
    if xp.real(sigma) > domains[-1]:
        return funcs[-1](sigma)
    if xp.real(sigma) < domains[0]:
        return funcs[0](sigma)
    return 0

def multi_chebyshev_vec_no_deriv(sigmaList, funcs, domains):
    # vectorized version of multi_chebyshev_no_deriv for sigma
    coefdtype = funcs[0].coef[0].dtype
    chebevalList = xp.empty(sigmaList.size, dtype=coefdtype)
    for i,sigma in enumerate(sigmaList):
        chebevalList[i] = multi_chebyshev_no_deriv(sigma, funcs, domains)
    
    return chebevalList


class Chebyshev(ChebyshevNP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atol = 1e-10
        diff0 = xp.abs(1 - (atol + xp.dot(np.ones(len(self.coef)), self.coef))/(atol + xp.dot(np.ones(len(self.coef) - 1), self.coef[:-1])))
        self.error = diff0

    def custom_eval(self, arg):
        if isinstance(arg, np.ndarray):
            return clenshaw_evaluation(self.coef, self.domain, np.atleast_1d(arg))
        else:
            return clenshaw_evaluation_scalar(self.coef, self.domain, arg)
    
    def custom_deriv(self, arg):
        if isinstance(arg, np.ndarray):
            return clenshaw_deriv_evaluation(self.coef, self.domain, np.atleast_1d(arg))
        else:
            return clenshaw_deriv_evaluation_scalar(self.coef, self.domain, arg)
    
    def __call__(self, arg, deriv = 0):
        if deriv == 0:
            return self.custom_eval(arg)
        elif deriv == 1:
            return self.custom_deriv(arg)
        else:
            return super().deriv(deriv)(arg)
        
    @property
    def coeffs(self):
        return self.coef

"""
class MultiChebyshev
This class joins together M Chebyshev series f_i of length N that are joined on adjacent domains [x_i, x_{i+1}] 
for x_i \in (x0, x1, ..., x_M). 

For example, consider M = 3 Chebyshev series of length 3: 
    f_0(x) = \sum_{n=0}^2 c_{0,n} T_n(X_0(x)), x \in [x_0, x_1]
    f_1(x) = \sum_{n=0}^2 c_{1,n} T_n(X_1(x)), x \in [x_1, x_2]
    f_2(x) = \sum_{n=0}^2 c_{2,n} T_n(X_2(x)), x \in [x_2, x_3]
    X_i(x) = (2*x - x_i - x_{i+1})/(x_{i+1} - x_{i})
Then MultiChebyshev joins together these solutions to represent f(x) in the domain [x_0, x_3]

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

def chebder_jit(c, m, scl):
    return ch.chebder(c, m, scl)

def chebmul_jit(c1, c2):
    return ch.chebmul(c1, c2)

def chebdiv_jit(c1, c2):
    return ch.chebdiv(c1, c2)

def chebpow_jit(c1, pow):
    return ch.chebpow(c1, pow)

# @njit([nb.float64[:,:](nb.float64[:,::1], nb.int64, nb.float64[:]),
#        nb.complex128[:,:](nb.complex128[:,::1], nb.int64, nb.float64[:])], cache=True)
def chebdiff_jit(c, m, scl):
    # function for differentiating chebyshev series
    ndomain, nsample = c.shape
    assert ndomain == scl.shape[0]
    D = cheby_D_jit(nsample, dtype = c.dtype) # differential matrix
    derivcoefflist = np.empty((ndomain, nsample - m), dtype = c.dtype)
    for i in range(ndomain):
        derivcoeff = c[i] # throw away last coefficient because you lose one-order in the series for each derivative
        dx = scl[i] # jacobian for variable transformation
        for _ in range(m): # apply derivative operator m-times
            # derivcoefflist[i] = dx*xp.matmul(D, derivcoefflist[i])
            derivcoeff = dx*(D @ derivcoeff)
        derivcoefflist[i] = derivcoeff[:-m]
        
    return derivcoefflist

# @njit(cache=True)
def chebdiff_2d_jit(c, m, scl):
    # function for differentiating chebyshev series
    ndomain, nsample, ngrid = c.shape
    assert ndomain == scl.shape[0]
    D = cheby_D_jit(nsample, dtype = c.dtype) # differential matrix
    derivcoefflist = np.empty((ndomain, nsample - m, ngrid), dtype = c.dtype)
    for j in range(ngrid):
        for i in range(ndomain):
            derivcoeff = c[i,:,j]
            dx = scl[i] # jacobian for variable transformation
            for _ in range(m): # apply derivative operator m-times
                derivcoeff = dx*(D @ derivcoeff)
            derivcoefflist[i,:,j] = derivcoeff[:-m]

    return derivcoefflist


class MultiDomainChebyshev:
    def __init__(self, coeffList, domainList):
        self.coeffs = xp.asarray(coeffList)
        self.domains = xp.asarray(domainList)
        self.ndomain, self.nsample = self.coeffs.shape

        self.domain_shift = -(self.domains[:-1] + self.domains[1:])/(self.domains[1:] - self.domains[:-1])
        self.domain_scale = 2./(self.domains[1:] - self.domains[:-1])

        if self.domains[0] > self.domains[1]:
            # reverse
            self.sorted_coeffs = self.coeffs[::-1]
            self.sorted_domains = self.domains[::-1]
            self.sorted_domain_shift = self.domain_shift[::-1]
            self.sorted_domain_scale = self.domain_scale[::-1]
            self.ordering = -1
        else:
            self.sorted_coeffs = self.coeffs
            self.sorted_domains = self.domains
            self.sorted_domain_shift = self.domain_shift
            self.sorted_domain_scale = self.domain_scale
            self.ordering = 1

        self.domain = [self.domains[0], self.domains[-1]]

        self.i0 = 0
            
    def eval(self, sigma):
        if isvec(sigma):
            return multi_chebyshev_vec(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering)
        else:
            self.i0, val = multi_chebyshev(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering, self.i0)
            return val
        
    def deriv(self, sigma):
        if isvec(sigma):
            return multi_chebyshev_deriv_vec(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering)
        else:
            self.i0, val = multi_chebyshev_deriv(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering, self.i0)
            return val
        
    def derivn(self, sigma, n = 2):
        if n == 0:
            return self.eval(self, sigma)
        elif n == 1:
            return self.eval_deriv(self, sigma)
        
        derivlist = chebdiff_jit(self.sorted_coeffs, n, self.domain_scale)
        if isvec(sigma):
            return multi_chebyshev_vec(sigma, derivlist, self.sorted_domains, self.ordering)
        else:
            self.i0, val = multi_chebyshev(sigma, derivlist, self.sorted_domains, self.ordering, self.i0)
            return val
    
    def diff(self, m):
        return MultiDomainChebyshev(chebdiff_jit(self.coeffs, m, self.domain_scale), self.domains)
    
    def __mul__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            coefflist = []
            assert other.ndomain == self.ndomain
            assert np.all(other.domains == self.domains)
            for i in range(self.ndomain):
                coefflist.append(chebmul_jit(self.coeffs[i], other.coeffs[i]))

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs*other

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __rmul__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            coefflist = []
            assert other.ndomain == self.ndomain
            assert np.all(other.domains == self.domains)
            for i in range(self.ndomain):
                coefflist.append(chebmul_jit(self.coeffs[i], other.coeffs[i]))

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs*other

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __add__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs + other.coeffs

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs + other

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __radd__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs + other.coeffs

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs + other

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __sub__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs - other.coeffs

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs - other

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __rsub__(self, other):
        if isinstance(other, MultiDomainChebyshev):
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = other.coeffs - self.coeffs

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = other - self.coeffs

            return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __pow__(self, other):
        coefflist = []
        for i in range(self.ndomain):
            coefflist.append(chebpow_jit(self.coeffs[i], other))

        return MultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.eval(sigma)
        elif deriv == 1:
            return self.deriv(sigma)
        else:
            return self.derivn(sigma, deriv)
        
class MultiGridMultiDomainChebyshev:
    # Note that this class assumes that you are working on a fixed multi-domain for each slice of the grid
    def __init__(self, coeffList, domainList):
        self.coeffs = xp.asarray(coeffList)
        self.ngrid, self.ndomain, self.nsample = self.coeffs.shape
        self.coeffs = xp.ascontiguousarray(xp.moveaxis(self.coeffs, 0, -1)) # this is the better way to store the data
        self.domains = xp.asarray(domainList)

        self.domain_shift = -(self.domains[:-1] + self.domains[1:])/(self.domains[1:] - self.domains[:-1])
        self.domain_scale = 2./(self.domains[1:] - self.domains[:-1])

        if self.domains[0] > self.domains[1]:
            # reverse
            self.sorted_coeffs = self.coeffs[::-1]
            self.sorted_domains = self.domains[::-1]
            self.sorted_domain_shift = self.domain_shift[::-1]
            self.sorted_domain_scale = self.domain_scale[::-1]
            self.ordering = -1
        else:
            self.sorted_coeffs = self.coeffs
            self.sorted_domains = self.domains
            self.sorted_domain_shift = self.domain_shift
            self.sorted_domain_scale = self.domain_scale
            self.ordering = 1
        
        self.dtype = self.coeffs.dtype

        self.domain = [self.domains[0], self.domains[-1]]

        self.i0 = 0
            
    def eval(self, sigma):
        if isvec(sigma):
            # out = np.zeros((self.ngrid, len(sigma)), dtype = self.dtype)
            # for j in range(self.ngrid):
            #     out[j] = multi_chebyshev_vec(sigma, self.sorted_coeffs[j], self.sorted_domains, self.ordering)
            out = multi_chebyshev_jit_vec_2d(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering)
        else:
            out = np.zeros(self.ngrid, dtype = self.dtype)
            for j in range(self.ngrid): 
                self.i0, val = multi_chebyshev(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering, self.i0)
                out[j] = val
        
        return out
        
    def deriv(self, sigma):
        if isvec(sigma):
            out = multi_chebyshev_deriv_jit_vec_2d(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering)
        else:
            out = np.zeros((self.ngrid), dtype = self.dtype)
            for j in range(self.ngrid): 
                self.i0, val = multi_chebyshev_deriv(sigma, self.sorted_coeffs, self.sorted_domains, self.ordering, self.i0)
                out[j] = val
        
        return out
        
    def derivn(self, sigma, n = 2):
        if n == 0:
            return self.eval(sigma)
        
        derivlist = chebdiff_2d_jit(self.sorted_coeffs, n, self.sorted_domain_scale)
        if isvec(sigma):
            out = multi_chebyshev_jit_vec_2d(sigma, derivlist, self.sorted_domains, self.ordering)
        else:
            out = np.zeros(self.ngrid, dtype = self.dtype)
            for j in range(self.ngrid): 
                self.i0, val = multi_chebyshev(sigma, derivlist, self.sorted_domains, self.ordering, self.i0)
                out[j] = val
        
        return out
    
    def diff(self, m):
        return MultiGridMultiDomainChebyshev(chebdiff_2d_jit(self.coeffs, m, self.domain_scale), self.domains)
    
    def __mul__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            coefflist = []
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert np.all(other.domains == self.domains)
            for j in range(self.ngrid):
                coeffsublist = []
                for i in range(self.ndomain):
                    coeffsublist.append(chebmul_jit(self.coeffs[j,i], other.coeffs[j,i]))
                coefflist.append(coeffsublist)

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs*other

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __rmul__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            coefflist = []
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert np.all(other.domains == self.domains)
            for j in range(self.ngrid):
                coeffsublist = []
                for i in range(self.ndomain):
                    coeffsublist.append(chebmul_jit(self.coeffs[j,i], other.coeffs[j,i]))
                coefflist.append(coeffsublist)
        else:
            coefflist = self.coeffs*other

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __add__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs + other.coeffs

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs + other

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __radd__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs + other.coeffs

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs + other

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __sub__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = self.coeffs - other.coeffs

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = self.coeffs - other

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __rsub__(self, other):
        if isinstance(other, MultiGridMultiDomainChebyshev):
            assert other.ngrid == self.ngrid
            assert other.ndomain == self.ndomain
            assert other.nsample == self.nsample
            assert np.all(other.domains == self.domains)
            coefflist = other.coeffs - self.coeffs

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        else:
            coefflist = other - self.coeffs

            return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __pow__(self, other):
        coefflist = []
        for j in range(self.ngrid):
            coeffsublist = []
            for i in range(self.ndomain):
                coeffsublist.append(chebpow_jit(self.coeffs[j, i], other))
            coefflist.append(coeffsublist)

        return MultiGridMultiDomainChebyshev(np.array(coefflist), self.domains)
        
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.eval(sigma)
        elif deriv == 1:
            return self.deriv(sigma)
        else:
            return self.derivn(sigma, deriv)
        
class MultiGridChebyshev:
    # Note that this class assumes that you are working on a fixed single-domain for each slice of the grid
    def __init__(self, coeffList, domain):
        self.coeffs = xp.asarray(coeffList)
        self.coeffs_internal = xp.ascontiguousarray(xp.moveaxis(self.coeffs, 0, -1)) # this is the better way to store the data
        self.ngrid, self.nsample = self.coeffs.shape
        self.domain = domain

        self.domain_shift = -(self.domain[1] + self.domain[0])/(self.domain[1] - self.domain[0]) + np.zeros(self.ngrid)
        self.domain_scale = 2./(self.domain[1] - self.domain[0]) + np.zeros(self.ngrid)
        
        self.dtype = self.coeffs.dtype
            
    def eval(self, sigma):
        if isvec(sigma):
            # out = np.zeros((self.ngrid, len(sigma)), dtype = self.dtype)
            # for j in range(self.ngrid):
            #     out[j] = clenshaw_evaluation(self.coeffs[j], self.domain, sigma)
            out = clenshaw_evaluation_2d(self.coeffs_internal, self.domain, sigma)
        else:
            out = np.zeros((self.ngrid), dtype = self.dtype)
            for j in range(self.ngrid): 
                out[j] = clenshaw_evaluation_scalar(self.coeffs[j], self.domain, sigma)
        
        return out
        
    def deriv(self, sigma):
        if isvec(sigma):
            # out = np.zeros((self.ngrid, len(sigma)), dtype = self.dtype)
            # for j in range(self.ngrid):
            #     out[j] = clenshaw_deriv_evaluation(self.coeffs[j], self.domain, sigma)
            out = clenshaw_deriv_evaluation_2d(self.coeffs_internal, self.domain, sigma)
        else:
            out = np.zeros((self.ngrid), dtype = self.dtype)
            for j in range(self.ngrid): 
                out[j] = clenshaw_deriv_evaluation_scalar(self.coeffs[j], self.domain, sigma)
        
        return out
        
    def derivn(self, sigma, n = 2):
        if n == 0:
            return self.eval(sigma)
        elif n == 1:
            return self.eval_deriv(sigma)
        
        derivlist = chebdiff_jit(self.coeffs, n, self.domain_scale)
        derivlist = xp.ascontiguousarray(xp.moveaxis(derivlist, 0, -1))
        
        if isvec(sigma):
            # out = np.zeros((self.ngrid, len(sigma)), dtype = self.dtype)
            # for j in range(self.ngrid):
            #     out[j] = clenshaw_evaluation(derivlist[j], self.domain, sigma)
            out = clenshaw_evaluation_2d(derivlist, self.domain, sigma)
        else:
            out = np.zeros((self.ngrid), dtype = self.dtype)
            for j in range(self.ngrid): 
                out[j] = clenshaw_evaluation_scalar(derivlist[j], self.domain, sigma)
        
        return out
    
    def diff(self, m):
        return MultiGridChebyshev(chebdiff_jit(self.coeffs, m, self.domain_scale), self.domain)
    
    def __mul__(self, other):
        if isinstance(other, MultiGridChebyshev):
            coefflist = []
            assert other.ngrid == self.ngrid
            assert np.all(other.domain == self.domain)
            for j in range(self.ngrid):
                coefflist.append(chebmul_jit(self.coeffs[j], other.coeffs[j]))
        else:
            coefflist = self.coeffs*other

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __rmul__(self, other):
        if isinstance(other, MultiGridChebyshev):
            coefflist = []
            assert other.ngrid == self.ngrid
            assert np.all(other.domain == self.domain)
            for j in range(self.ngrid):
                coefflist.append(chebmul_jit(self.coeffs[j], other.coeffs[j]))
        else:
            coefflist = self.coeffs*other

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __add__(self, other):
        if isinstance(other, MultiGridChebyshev):
            assert other.ngrid == self.ngrid
            assert other.nsample == self.nsample
            assert np.all(other.domain == self.domain)
            coefflist = self.coeffs + other.coeffs
        else:
            coefflist = self.coeffs + other

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __radd__(self, other):
        if isinstance(other, MultiGridChebyshev):
            assert other.ngrid == self.ngrid
            assert other.nsample == self.nsample
            assert np.all(other.domain == self.domain)
            coefflist = self.coeffs + other.coeffs
        else:
            coefflist = self.coeffs + other

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __sub__(self, other):
        if isinstance(other, MultiGridChebyshev):
            assert other.ngrid == self.ngrid
            assert other.nsample == self.nsample
            assert np.all(other.domain == self.domain)
            coefflist = self.coeffs - other.coeffs
        else:
            coefflist = self.coeffs - other

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __rsub__(self, other):
        if isinstance(other, MultiGridChebyshev):
            assert other.ngrid == self.ngrid
            assert other.nsample == self.nsample
            assert np.all(other.domain == self.domain)
            coefflist = other.coeffs - self.coeffs
        else:
            coefflist = other - self.coeffs

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __pow__(self, other):
        coefflist = []
        for j in range(self.ngrid):
            coefflist.append(chebpow_jit(self.coeffs[j], other))

        return MultiGridChebyshev(np.array(coefflist), self.domain)
        
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.eval(sigma)
        elif deriv == 1:
            return self.deriv(sigma)
        else:
            return self.derivn(sigma, deriv)
    