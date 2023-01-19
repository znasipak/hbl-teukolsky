use_gpu = False
import numpy as np
if use_gpu:
    import cupy as xp
else:
    import numpy as xp
import numpy.polynomial.chebyshev as ch
from numba import jit, njit
import numba as nb
import scipy
import time

dsize = 500
D100 = xp.zeros([dsize,dsize], np.float32)
j = 1
while j < dsize:
    D100[0][j] = j
    j += 2
for i in range(1,dsize):
    j = i + 1
    while j < dsize:
        D100[i][j] = 2*j
        j += 2

nodes_16=ch.chebpts1(16)
Tmat0_16=ch.chebvander(nodes_16, 15)
Tmat1_16 = xp.matmul(Tmat0_16, D100[:16,:16])
Tmat2_16 = xp.matmul(Tmat1_16, D100[:16,:16])
Tmat0Inv_16 = xp.linalg.inv(Tmat0_16)
bc1_16 = xp.array(ch.chebvander(-1, 15))
bc2_16 = xp.matmul(bc1_16, D100[:16,:16])

nodes_32=ch.chebpts1(32)
Tmat0_32=ch.chebvander(nodes_32, 31)
Tmat1_32 = xp.matmul(Tmat0_32, D100[:32,:32])
Tmat2_32 = xp.matmul(Tmat1_32, D100[:32,:32])
Tmat0Inv_32 = xp.linalg.inv(Tmat0_32)
bc1_32 = xp.array(ch.chebvander(-1, 31))
bc2_32 = xp.matmul(bc1_32, D100[:32,:32])

#@njit
def default_smax():
    return 0.5

#@njit
def default_smin(omega):
    return 0.03*omega**(2/3)

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
            
    def chebeval(self, sigma, deriv=0):
        if isinstance(sigma, np.ndarray):
            if deriv == 0:
                return multi_chebyshev_vec_no_deriv(sigma, self.sorted_chebylist, self.sorted_domains)
            else:
                return multi_chebyshev_vec_no_deriv(sigma, self.deriv_list(deriv), self.sorted_domains)
        else:
            if deriv == 0:
                return multi_chebyshev_no_deriv(sigma, self.sorted_chebylist, self.sorted_domains)
            else:
                return multi_chebyshev_no_deriv(sigma, self.deriv_list(deriv), self.sorted_domains)
    
    def deriv_list(self, m):
        derivlist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            derivlist[i] = self.sorted_chebylist[i].deriv(m)
            
        return derivlist
    
    def deriv(self, m):
        D = D100[:self.n_sample-1, :self.n_sample-1] # differential matrix
        derivcoefflist = np.empty((self.n_cs, self.n_sample - 1), dtype=xp.cdouble)
        for i in range(self.n_cs):
            derivcoefflist[i] = self.coeffs[i][:-1] # throw away last coefficient because you lose one-order in the series for each derivative
            dx = 2/(self.domains[i+1]-self.domains[i]) # jacobian for variable transformation
            for _ in range(m): # apply derivative operator m-times
                derivcoefflist[i] = dx*xp.matmul(D, derivcoefflist[i])
            
        return MultiChebyshev(derivcoefflist, self.domains)
        
    def __call__(self, sigma, deriv=0):
        return self.chebeval(sigma, deriv)
    
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
        
        self.domain = [self.domains[0], self.domains[-1]]
        
    def sol_func(self, sigma):
        return self.mch.chebeval(sigma)
    
    def deriv_func(self, sigma):
        return self.mch_deriv.chebeval(sigma)
    
    def _repr_latex_(self):
        bc_str = str(self.bc)
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$\Psi^\mathrm{'+bc_str+'}_{sl\omega}(\sigma; '+sub_str+')$, $\sigma \in '+domain_str+' $'
    
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.sol_func(sigma)
        elif deriv == 1:
            return self.deriv_func(sigma)
        else:
            self.mch(sigma, deriv)
            
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
        
    def sol_func(self, r):
        sigma = self.sigma_of_r(r)
        alpha = self.f_teukolsky_transformation(r)
        return alpha*self.mch.chebeval(sigma)
    
    def deriv_func(self, r):
        sigma = self.sigma_of_r(r)
        alpha = self.f_teukolsky_transformation_deriv(r)
        beta = self.g_teukolsky_transformation_deriv(r)
        return alpha*self.mch.chebeval(sigma) + beta*self.mch_deriv.chebeval(sigma)
    
    def _repr_latex_(self):
        bc_str = str(self.bc)
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$R^\mathrm{'+bc_str+'}_{sl\omega}(r; '+sub_str+')$, $r \in '+domain_str+' $'
    
    def __call__(self, r, deriv=0):
        if deriv == 0:
            return self.sol_func(r)
        elif deriv == 1:
            return self.deriv_func(r)
        else:
            print('Error')
            

#@njit
def pSchwFull(sigma):
    return sigma**2*(1 - sigma)

#@njit
def qSchwFull(sigma, s, omega):
    return 2*sigma*(1+s) - sigma**2*(3 + s - 8j*omega) - 4j*omega 

#@njit
def uSchwFull(sigma, s, lam, omega):
    return -4j*omega*(4j*omega + s) - (1 - 4j*omega)*(1 + s - 4j*omega)*sigma - lam

def fTSp2SchwFull(sigma, lam, omega):
    return (2*omega)**(-4)*1/16.*((256j)*omega**3*sigma - lam*(2 + lam)*(-1 + sigma)*sigma**4 + 256*omega**4*(1 + sigma) + (4j)*omega*sigma**3*(lam*(-6 + 5*sigma) + sigma*(-5 + 6*sigma)) + 16*omega**2*sigma**2*(lam*(-3 + 2*sigma**2) + sigma*(-5 + sigma + 3*sigma**2)))/(1 - sigma)**3

#@njit
def gTSp2SchwFull(sigma, lam, omega):
    return (2*omega)**(-4)*((-1j/4)*omega*sigma**2*(16*omega**2 + sigma**2*(2*lam*(-1 + sigma) + sigma*(-2 + 3*sigma))))/(-1 + sigma)**3

#@njit
def fTSm2SchwFull(sigma, lam, omega):
    Omega = 1j*omega
    return (16*(-128*Omega**3 + 256*Omega**4 + lam*(2 + lam)*(-1 + sigma)**2*sigma**2 + 4*Omega*sigma*(sigma*(-3 + 5*sigma) + lam*(-2 + sigma + sigma**2)) - 16*Omega**2*(lam*(-1 + sigma)*(1 + 2*sigma**2) + sigma*(-3 + sigma*(3 + sigma*(-2 + 3*sigma))))))/sigma**6/(128*Omega*(-1 + 2*Omega)*(-1 + 4*Omega)*(1 + 4*Omega))

#@njit
def gTSm2SchwFull(sigma, lam, omega):
    Omega = 1j*omega
    return (64*Omega*(-1 + sigma)*(-16*Omega**2 + 2*lam*(-1 + sigma)*sigma**2 + sigma**3*(-2 + 3*sigma)))/sigma**6/(128*Omega*(-1 + 2*Omega)*(-1 + 4*Omega)*(1 + 4*Omega))

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
        
        self.domain = {'In': [default_smin(self.frequency), 1.], 'Up': [0., default_smax()]}
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
    
    @staticmethod
    def fTS_plus_to_minus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod
    def gTS_plus_to_minus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod
    def fTS_minus_to_plus_1(sigma, lam, omega):
        return 1.
    
    @staticmethod
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
            nodes = nodes_16
            Tmat0 = Tmat0_16
            Tmat1 = Tmat1_16
            Tmat0Inv = Tmat0Inv_16
        elif n == 32:
            nodes = nodes_32
            Tmat0 = Tmat0_32
            Tmat1 = Tmat1_32
            Tmat0Inv = Tmat0Inv_32
        else:
            D = D100[:n,:n]
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
        elif self.s == 1:
            for i, sigmaNodes in enumerate(sigmaNodesTList):
                Fmat = self.fTS_minus_to_plus_1(sigmaNodes, la, self.frequency)
                Gmat = dxdsigma[i]*self.gTS_minus_to_plus_1(sigmaNodes, la, self.frequency)
                Mmat[i] = xp.multiply(Fmat, Tmat0) + xp.multiply(Gmat, Tmat1)
        elif self.s == -1:
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
            nodes = nodes_16
            Tmat0 = Tmat0_16
            Tmat1 = Tmat1_16
            Tmat2 = Tmat2_16
            bc1 = bc1_16
            bc2 = bc2_16
        elif n == 32:
            nodes = nodes_32
            Tmat0 = Tmat0_32
            Tmat1 = Tmat1_32
            Tmat2 = Tmat2_32
            bc1 = bc1_32
            bc2 = bc2_32
        else:
            D = D100[:n,:n]
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
            dpsi0 = xp.sum(xp.matmul(D100[:nsample,:nsample],self.coeffs[s][bc][i]))/dsigmadx
            smin = smax
            smax = boundary
            dsigmadx = 1./dxOfSigma(smin, smax)
            dpsi0 *= dsigmadx
            i += 1
            
            self.coeffs[s][bc][i] = self.solve_hyperboloidal_coeffs_domain(s, psi0, dpsi0, [smin, smax], bc, n=nsample)
        
class RadialTeukolsky(TeukolskySolver):
    
    def solve(self, bc=['In','Up'], use_ts_transform=True, cutoff=[2,np.inf], subdomains=0):
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff, subdomains=subdomains)
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
            
        self._TeukolskySolver__solve_hyperboloidal_teukolsky_coeffs(s, bc, [mincut, maxcut], subdomains, nsample=16)
                
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
        
    def test_accuracy(self, bc):
        if self.mch[bc] is None:
            print('ERROR: No solution available for {}'.format(bc))
        else:
            return 1   
    
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
    
class HyperboloidalTeukolsky(TeukolskySolver):
    def solve(self, bc=['In','Up'], use_ts_transform=True, cutoff=[0,1], subdomains=0):
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff, subdomains=subdomains)
            return None
            
        start = time.time()
        if bc not in self.bcs:
            print('ERROR: Invalid key type {}'.format(bc))
            return None
                
        if use_ts_transform:
            s = self.spinsign[bc]*abs(self.s)
        else:
            s = self.s
            
        self._TeukolskySolver__solve_hyperboloidal_teukolsky_coeffs(s, bc, cutoff, subdomains, nsample=16)
                
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
            return 1   
    
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

def sigma_of_r(r, *args):
    return 2/r

def dr_dsigma(r, *args):
    return -r**2/2

def alpha_teuk(r, s, l, omega):
    return r**(-2*s - 1)*np.exp(1j*omega*r)*(1 - 2./r)**(-s)*(2./r*(1 - 2./r))**(-2j*omega)

def beta_teuk(r, *args):
    return 0

def alpha_teuk_deriv(r, s, l, omega):
    return r**(-2*s - 1)*np.exp(1j*omega*r)*(1 - 2./r)**(-s)*(2./r*(1 - 2./r))**(-2j*omega)*((2 - r + 1j*omega*(-8 + r**2) + 2*s - 2*r*s)/((-2 + r)*r))
    
def beta_teuk_deriv(r, s, l, omega):
    return r**(-2*s - 1)*np.exp(1j*omega*r)*(1 - 2./r)**(-s)*(2./r*(1 - 2./r))**(-2j*omega)

def multi_chebyshev(sigma, funcs, domains, deriv=0):
    if deriv == 0:
        return multi_chebyshev(sigma, funcs, domains)
    else:
        return multi_chebyshev_deriv(sigma, funcs, domains)

def multi_chebyshev_no_deriv(sigma, funcs, domains):
    for i in range(funcs.size):
        if sigma <= domains[i+1] and sigma >= domains[i]:
            return funcs[i](sigma)
    return 0

def multi_chebyshev_deriv(sigma, funcs, domains, deriv=1):
    for i in range(funcs.size):
        if sigma <= domains[i+1] and sigma >= domains[i]:
            return funcs[i].deriv(deriv)(sigma)
    return 0

def multi_chebyshev_vec(sigmaList, funcs, domains, deriv=0):
    chebevalList = xp.empty(sigmaList.size, dtype=complex)
    
    if deriv == 0:
        for i,sigma in enumerate(sigmaList):
            chebevalList[i] = multi_chebyshev_no_deriv(sigma, funcs, domains)
    else:
        for i,sigma in enumerate(sigmaList):
            chebevalList[i] = multi_chebyshev_deriv(sigma, funcs, domains, deriv)
            
    return chebevalList

def multi_chebyshev_vec_no_deriv(sigmaList, funcs, domains):
    chebevalList = xp.empty(sigmaList.size, dtype=complex)
    for i,sigma in enumerate(sigmaList):
        chebevalList[i] = multi_chebyshev_no_deriv(sigma, funcs, domains)
    
    return chebevalList

def a1sigma0(s, lam, omega):
    return -(lam + 4j*omega*(s + 4j*omega))/(4j*omega)

def a2sigma0(s, lam, omega):
    return (lam**2 + 2*lam*(-1 + 4j*omega)*(1 + 4j*omega + s) + 4j*omega*(-1 - 16*omega**2*(-1 + 4j*omega) + (1 + 4j*omega)*(-3 + 8j*omega)*s + (-2 + 4j*omega)*s**2))/(-32*omega**2)

def b1sigma1(s, lam, omega):
    return -(lam + (1 - 4j*omega)*(1 - 4j*omega + s) + 4j*omega*(4j*omega + s))/(-3 + 4j*omega - s + 2*(1 + s))

def chebode(p, q, u, x0, psi0, dpsi0, n=16):
    if n == 16:
        nodes = nodes_16
        Tmat0 = Tmat0_16
        Tmat1 = Tmat1_16
        Tmat2 = Tmat2_16
        bc1 = bc1_16
        bc2 = bc2_16
    elif n == 32:
        nodes = nodes_32
        Tmat0 = Tmat0_32
        Tmat1 = Tmat1_32
        Tmat2 = Tmat2_32
        bc1 = bc1_32
        bc2 = bc2_32
    else:
        D = D100[:n,:n]
        nodes = ch.chebpts1(n)
        Tmat0 = xp.array(ch.chebvander(nodes, n-1))
        Tmat1 = xp.matmul(Tmat0, D)
        Tmat2 = xp.matmul(Tmat1, D)
        bc1 = xp.array(ch.chebvander(-1, n - 1))
        bc2 = xp.matmul(bc1, D)
    
    nodesT = nodes.reshape(n, 1)
    Pmat = p(nodesT)
    Qmat = q(nodesT)
    Umat = u(nodesT)

    Mmat = xp.multiply(Pmat, Tmat2) + xp.multiply(Qmat, Tmat1) + xp.multiply(Umat, Tmat0)
    Mmat[0] = bc1
    Mmat[1] = bc2

    source = xp.zeros((n,1), dtype=np.cdouble)
    source[0,0] = psi0
    source[1,0] = dpsi0

    coeffs = xp.linalg.solve(Mmat, source)
    return coeffs

def checkode(p, q, u, f):
    nodes, psi = f.linspace()
    nodes, dpsi = f.deriv().linspace()
    nodes, d2psi = f.deriv(2).linspace()

    return p(nodes)*d2psi + q(nodes)*dpsi + u(nodes)*psi

def checkode2(p, q, u, f):
    nodes, psi = f.linspace()
    nodes, dpsi = f.deriv().linspace()
    nodes, d2psi = f.deriv(2).linspace()

    return p(nodes)*d2psi/psi + q(nodes)*dpsi/psi + u(nodes)

def chebtransform(f, g, v):
    n = v.shape[0]
    D = D100[:n,:n]
    
    if n == 16:
        nodes = nodes_16
        Tmat0 = Tmat0_16
        Tmat1 = Tmat1_16
        Tmat0Inv = Tmat0Inv_16
    elif n == 32:
        nodes = nodes_32
        Tmat0 = Tmat0_32
        Tmat1 = Tmat1_32
        Tmat0Inv = Tmat0Inv_32
    else:
        nodes = ch.chebpts1(n)
        Tmat0 = xp.array(ch.chebvander(nodes, n-1))
        Tmat1 = xp.matmul(Tmat0, D)
        Tmat0Inv = xp.linalg.inv(Tmat0)
    
    Fmat = xp.diag(f(nodes))
    Gmat = xp.diag(g(nodes))
  
    Mmat = xp.matmul(Fmat, Tmat0) + xp.matmul(Gmat, Tmat1)
    vvec = xp.array(v.reshape([n,1]))

    coeffs = xp.matmul(xp.matmul(Tmat0Inv, Mmat), vvec)
    return coeffs

def xOfSigma(sigma, smin, smax):
    return (2*sigma - (smax + smin))/(smax - smin)

def sigmaOfX(x, smin, smax):
    return ((smax - smin)*x + smax + smin)/2

def dxOfSigma(smin, smax):
    return 2/(smax - smin)

def flip_plus_2_teuk(l, omega, coeffs, domain):
    s = -2
    la = l*(l+1) - s*(s+1)
    smin = domain[0]
    smax = domain[1]
    dxdsigma = dxOfSigma(smin, smax)
    def ftrans(x):
        return fTSp2SchwFull(sigmaOfX(x, smin, smax), la, omega)
    def gtrans(x):
        return dxdsigma*gTSp2SchwFull(sigmaOfX(x, smin, smax), la, omega)

    newcoeffs = chebtransform(ftrans, gtrans, coeffs).flatten()
    return newcoeffs

def flip_minus_2_teuk(l, omega, coeffs, domain):
    s = -2
    la = l*(l+1) - s*(s+1)
    smin = domain[0]
    smax = domain[1]
    dxdsigma = dxOfSigma(smin, smax)
    def ftrans(x):
        return fTSm2SchwFull(sigmaOfX(x, smin, smax), la, omega)
    def gtrans(x):
        return dxdsigma*gTSm2SchwFull(sigmaOfX(x, smin, smax), la, omega)

    newcoeffs = chebtransform(ftrans, gtrans, coeffs).flatten()
    return newcoeffs

def chebteuk(s, l, omega):
    la = l*(l+1) - s*(s+1)
    a1 = a1sigma0(s, la, omega)
    a2 = a2sigma0(s, la, omega)
    smax = 0.5*xp.min(np.abs([1/a1, a1/a2]))
    if smax > default_smax():
        smax = default_smax()
    smin = 0
    dsigmadx = 1./dxOfSigma(smin, smax)
    def pode(x):
        return pSchwFull(sigmaOfX(x, smin, smax))
    def qode(x):
        return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, omega)
    def uode(x):
        return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, omega)
    psi0 = 1
    dpsi0 = -uode(-1)*psi0/qode(-1)

    coeffs = chebode(pode, qode, uode, -1, psi0, dpsi0, 32).flatten()

    return coeffs, [smin, smax]

def multichebteuk(stemp, l, omega, bc='Up', cutoff=0, subdomains=0):
    spinsign = 1
    if bc=='Up':
        spinsign = -1
    if spinsign*stemp == 2:
        s = -stemp
    else:
        s = stemp
    la = l*(l+1) - s*(s+1)
    
    smin = 0
    smax = 1
    if subdomains == 0:
        boundaryNum = np.amax([16, 4*l])
    else:
        boundaryNum = subdomains
    
    boundaryNode = -1
    if bc == 'Up':
        if cutoff <= 0 or cutoff >= 1:
            smax = default_smax()
        else:
            smax = cutoff
        a1 = a1sigma0(s, la, omega)
        a2 = a2sigma0(s, la, omega)
        smin = 0.5*xp.min(np.abs([1/a1, a1/a2]))
        if smin > smax:
            smin = smax
            boundaryNum = 1
    elif bc == 'In':
        if cutoff <= 0 or cutoff >= 1:
            smin = default_smin(omega)
        else:
            smin = cutoff
        b1 = b1sigma1(s, la, omega)
        smax = 1 - 1/np.abs(b1)
        if smax < smin:
            smax = smin
            boundaryNum = 1
    else:
        print("Error")
        
    boundaryList = np.zeros(boundaryNum + 1)
    boundaryList[1:] = smin*(smax/smin)**np.linspace(0, 1, num=boundaryNum)
    if bc == 'In':
        boundaryList[0] = 1
        boundaryList[1:] = boundaryList[1:][::-1]
    #print(boundaryList)

    smin = boundaryList[0]
    smax = boundaryList[1]
    dsigmadx = 1./dxOfSigma(smin, smax)
    #print(dsigmadx)
    #print(sigmaOfX(-1, smin, smax))
    
    def pode(x):
        return pSchwFull(sigmaOfX(x, smin, smax))
    def qode(x):
        return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, omega)
    def uode(x):
        return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, omega)
    
    psi0 = 1.
    dpsi0 = -uode(boundaryNode)*psi0/qode(boundaryNode)
    coeffs = chebode(pode, qode, uode, boundaryNode, psi0, dpsi0, 32).flatten()
    coeffsList = np.empty(boundaryNum, dtype=object)
    #print(coeffs)
    
    i=0
    if spinsign*stemp == 2:
        if stemp == -2:
            coeffsList[i] = flip_plus_2_teuk(l, omega, coeffs, [smin, smax])
        else:
            coeffsList[i] = flip_minus_2_teuk(l, omega, coeffs, [smin, smax])
    else:
        coeffsList[i] = coeffs
  
    for boundary in boundaryList[2:]:
        i += 1
        if use_gpu:
            coeffs = coeffs.get()
        f = ch.Chebyshev(coeffs, domain=[smin, smax])
        psi0 = f(smax)
        dpsi0 = f.deriv()(smax)
        smin = smax
        smax = boundary
        dsigmadx = 1./dxOfSigma(smin, smax)
        #print(smin, smax, dsigmadx)
        dpsi0 *= dsigmadx
        def pode(x):
            return pSchwFull(sigmaOfX(x, smin, smax))
        def qode(x):
            return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, omega)
        def uode(x):
            return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, omega)

        coeffs = chebode(pode, qode, uode, boundaryNode, psi0, dpsi0, 16).flatten()
        if spinsign*stemp == 2:
            if stemp == -2:
                coeffsList[i] = flip_plus_2_teuk(l, omega, coeffs, [smin, smax])
            else:
                coeffsList[i] = flip_minus_2_teuk(l, omega, coeffs, [smin, smax])
        else:
            coeffsList[i] = coeffs

    return coeffsList, boundaryList

def hfunc(sigma):
    return 2/sigma - 2*np.log(sigma) - 2*np.log(1 - sigma)

def rescaleteuksigma(s, omega, sigma):
    return 0.5*sigma**(1+2*s)*(4*(1 - sigma))**(-s)*np.exp(1j*omega*hfunc(sigma))

def rescaleteuk(s, omega, r):
    return rescaleteuksigma(s, omega, 2/r)

def checkteuk(s, l, omega, f):
    la = l*(l+1) - s*(s+1)
    def pode(x):
        return pSchwFull(x)
    def qode(x):
        return qSchwFull(x, s, omega)
    def uode(x):
        return uSchwFull(x, s, la, omega)

    return checkode2(pode, qode, uode, f)