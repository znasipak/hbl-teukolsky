use_gpu = False
import numpy as np
if use_gpu:
    import cupy as xp
else:
    import numpy as xp
import numpy.polynomial.chebyshev as ch

dsize=500
D100 = xp.zeros([dsize,dsize])
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
Tmat0_16=ch.chebvander(nodes_16,15)
Tmat1_16 = xp.matmul(Tmat0_16, D100[:16,:16])
Tmat2_16 = xp.matmul(Tmat1_16, D100[:16,:16])
Tmat0Inv_16 = xp.linalg.inv(Tmat0_16)

nodes_32=ch.chebpts1(32)
Tmat0_32=ch.chebvander(nodes_32,31)
Tmat1_32 = xp.matmul(Tmat0_32, D100[:32,:32])
Tmat2_32 = xp.matmul(Tmat1_32, D100[:32,:32])
Tmat0Inv_32 = xp.linalg.inv(Tmat0_32)

def default_smax():
    return 0.5

def default_smin(omega):
    return 0.03*omega**(2/3)

class MultiChebyshev:
    def __init__(self, coeffList, domainList):
        self.coeffs = coeffList
        self.domains = domainList
        self.n_cs = self.coeffs.size
        self.chebylist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            self.chebylist[i] = ch.Chebyshev(self.coeffs[i], domain=[self.domains[i], self.domains[i+1]])
        if self.domains[0] > self.domains[1]:
            self.chebylist = self.chebylist[::-1] # reverse
            self.domains = self.domains[::-1]
            
    def chebeval(self, sigma, deriv=0):
        if isinstance(sigma, np.ndarray):
            if deriv == 0:
                return multi_chebyshev_vec_no_deriv(sigma, self.chebylist, self.domains)
            else:
                return multi_chebyshev_vec_no_deriv(sigma, self.deriv_list(deriv), self.domains)
        else:
            if deriv == 0:
                return multi_chebyshev_no_deriv(sigma, self.chebylist, self.domains)
            else:
                return multi_chebyshev_no_deriv(sigma, self.deriv_list(deriv), self.domains)
    
    def deriv_list(self, m):
        derivlist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            derivlist[i] = self.chebylist[i].deriv(m)
            
        return derivlist
    
    def deriv(self, m):
        derivcoefflist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            derivcoefflist[i] = self.chebylist[i].deriv(m).coef
            
        return MultiChebyshev(derivcoefflist, self.domains)
        
    def __call__(self, sigma, deriv=0):
        return self.chebeval(sigma, deriv)
        
class RadialSolution:
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
        
    def transform_solution(self, transformation):
        if isinstance(transformation, list) or isinstance(transformation, np.ndarray):
            return TransformedRadialSolution(self.s, self.l, self.frequency, self.bc, self.mch, transformation)
        else:
            print('ERROR: Transformation must be a list of functions')
        
    def sol_func(self, sigma):
        return self.mch.chebeval(sigma)
    
    def deriv_func(self, sigma):
        return self.mch_deriv.chebeval(sigma)
    
    def __repr__(self):
        return 'RadialSolution[{}, domain=[{:.3f}, {:.3f}]]'.format(self.bc, self.domain[0], self.domain[1])
    
    def __call__(self, sigma, deriv=0):
        if deriv == 0:
            return self.sol_func(sigma)
        elif deriv == 1:
            return self.deriv_func(sigma)
        else:
            print('ERROR: Only first derivative implemented')
            
class TransformedRadialSolution(RadialSolution):
    def __init__(self, s, l, omega, bc, mch, transformation):
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
        
        if isinstance(transformation, list):
            if callable(transformation[0]):
                self.transformation = transformation
            else:
                print('ERROR: Must specify callable function for transforming solution')
        else:
            print('ERROR: Must specify list of functions for transforming solution')
            
    def sol_func(self, sigma):
        alpha = self.transformation[0](sigma, self.s, self.l, self.frequency)
        beta = self.transformation[1](sigma, self.s, self.l, self.frequency)
        return alpha*self.mch.chebeval(sigma) + beta*self.mch_deriv.chebeval(sigma)
    
    def deriv_func(self, sigma):
        alpha = self.transformation[2](sigma, self.s, self.l, self.frequency)
        beta = self.transformation[3](sigma, self.s, self.l, self.frequency)
        return alpha*self.mch.chebeval(sigma) + beta*self.mch_deriv.chebeval(sigma)
    
class VariableTransformedRadialSolution(TransformedRadialSolution):            
    def sol_func(self, sigma):
        x_of_sigma = self.transformation[0](sigma, self.s, self.l, self.frequency)
        return self.mch.chebeval(x_of_sigma)
    
    def deriv_func(self, sigma):
        x_of_sigma = self.transformation[0](sigma, self.s, self.l, self.frequency)
        dsigma_dx = self.transformation[1](sigma, self.s, self.l, self.frequency)
        return dsigma_dx*self.mch_deriv.chebeval(x_of_sigma)
    
class FullyTransformedRadialSolution(TransformedRadialSolution):            
    def sol_func(self, sigma):
        x_of_sigma = self.transformation[0](sigma, self.s, self.l, self.frequency)
        dsigma_dx = self.transformation[1](sigma, self.s, self.l, self.frequency)
        alpha = self.transformation[2](sigma, self.s, self.l, self.frequency)
        beta = dsigma_dx*self.transformation[3](sigma, self.s, self.l, self.frequency)
        return alpha*self.mch.chebeval(x_of_sigma) + beta*self.mch_deriv.chebeval(x_of_sigma)
    
    def deriv_func(self, sigma):
        x_of_sigma = self.transformation[0](sigma, self.s, self.l, self.frequency)
        dsigma_dx = self.transformation[1](sigma, self.s, self.l, self.frequency)
        alpha = self.transformation[4](sigma, self.s, self.l, self.frequency)
        beta = dsigma_dx*self.transformation[5](sigma, self.s, self.l, self.frequency)
        return alpha*self.mch.chebeval(x_of_sigma) + beta*self.mch_deriv.chebeval(x_of_sigma)
        
class HyperboloidalTeukolsky:
    def __init__(self, s, l, omega):
        self.s = s
        self.l = l
        self.frequency = omega
        self.blackholespin = 0.
        self.shifted_eigenvalue = self.l*(self.l + 1)
        self.eigenvalue = self.shifted_eigenvalue - self.s*(self.s + 1)

        self.bcs = ['In', 'Up']
        self.coeffs = dict.fromkeys(self.bcs)
        self.domains = dict.fromkeys(self.bcs)
        self.mch = dict.fromkeys(self.bcs)
        self.hbl = dict.fromkeys(self.bcs)
        self.teukolsky = dict.fromkeys(self.bcs)
        
        self.domain = {'In': [default_smin(self.frequency), 1], 'Up': [0, default_smax()]}
        self.spinsign = {'In': -1, 'Up': 1}
        
    def solve_teukolsky(self, bc=['In','Up'], use_ts_transform=True, save_negative_s=False, cutoff=0, subdomains=0):
        if isinstance(bc, list) or isinstance(bc, np.ndarray):
            for condition in bc:
                self.solve_teukolsky(condition, use_ts_transform=use_ts_transform, save_negative_s=save_negative_s, cutoff=cutoff, subdomains=subdomains)
            return None
            
        if bc not in self.bcs:
            print('ERROR: Invalid key type {}'.format(bc))
            return None
        
        self.coeffs[bc], self.domains[bc] = multichebteuk(self.s, self.l, self.frequency, bc=bc)
        
        if use_ts_transform:
            s = self.spinsign[bc]*abs(self.s)
        else:
            s = self.s
        la = self.l*(self.l+1) - s*(s+1)
    
        [smin, smax] = self.domain[bc]
        if subdomains == 0:
            boundaryNum = np.amax([16, 4*self.l])
        else:
            boundaryNum = subdomains
    
        boundaryNode = -1
        if bc == 'Up':
            if cutoff > 0 and cutoff < 1:
                smax = cutoff
            a1 = a1sigma0(s, la, self.frequency)
            a2 = a2sigma0(s, la, self.frequency)
            smin = 0.5*xp.min(np.abs([1/a1, a1/a2]))
            if smin > smax:
                smin = smax
                boundaryNum = 1
        elif bc == 'In':
            if cutoff > 0 and cutoff < 1:
                smin = cutoff
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
    
        def pode(x):
            return pSchwFull(sigmaOfX(x, smin, smax))
        def qode(x):
            return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, self.frequency)
        def uode(x):
            return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, self.frequency)
    
        if bc == 'In':
            psi0 = 2*np.exp(-4j*self.frequency)
        else:
            psi0 = 1.
        dpsi0 = -uode(boundaryNode)*psi0/qode(boundaryNode)
        coeffs = chebode(pode, qode, uode, boundaryNode, psi0, dpsi0, 32).flatten()
        self.coeffs[bc] = np.empty(boundaryNum, dtype=object)
    
        i=0
        if self.spinsign[bc]*self.s == -2:
            if self.s == -2:
                self.coeffs[bc][i] = flip_plus_2_teuk(self.l, self.frequency, coeffs, [smin, smax])
            else:
                self.coeffs[bc][i] = flip_minus_2_teuk(self.l, self.frequency, coeffs, [smin, smax])
        else:
            self.coeffs[bc][i] = coeffs
  
        for boundary in self.domains[bc][2:]:
            i += 1
            if use_gpu:
                coeffs = coeffs.get()
            f = ch.Chebyshev(coeffs, domain=[smin, smax])
            psi0 = f(smax)
            dpsi0 = f.deriv()(smax)
            smin = smax
            smax = boundary
            dsigmadx = 1./dxOfSigma(smin, smax)
            dpsi0 *= dsigmadx
            
            def pode(x):
                return pSchwFull(sigmaOfX(x, smin, smax))
            def qode(x):
                return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, self.frequency)
            def uode(x):
                return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, self.frequency)

            coeffs = chebode(pode, qode, uode, boundaryNode, psi0, dpsi0, 16).flatten()
            if self.spinsign[bc]*self.s == -2:
                if self.s == -2:
                    self.coeffs[bc][i] = flip_plus_2_teuk(self.l, self.frequency, coeffs, [smin, smax])
                else:
                    self.coeffs[bc][i] = flip_minus_2_teuk(self.l, self.frequency, coeffs, [smin, smax])
            else:
                self.coeffs[bc][i] = coeffs
        
        self.mch[bc] = MultiChebyshev(self.coeffs[bc], self.domains[bc])
        self.domains[bc] = self.mch[bc].domains
        self.coeffs[bc] = self.mch[bc].coeffs
        self.domain[bc][0] = self.domains[bc][0]
        self.domain[bc][1] = self.domains[bc][-1]
        
        self.hbl[bc] = RadialSolution(self.s, self.l, self.frequency, bc, self.mch[bc])
        transformations = [sigma_of_r, dr_dsigma, alpha_teuk, beta_teuk, alpha_teuk_deriv, beta_teuk_deriv]
        self.teukolsky[bc] = FullyTransformedRadialSolution(self.s, self.l, self.frequency, bc, self.mch[bc], transformations)
        
    def get_hyperboloidal(self, bc=None):
        if bc == None:
            return self.hbl
        elif bc in self.bcs:
            return self.hbl[bc]
        else:
            print('ERROR: Invalid key type {}'.format(bc))
            
    def get_teukolsky(self, bc=None):
        if bc == None:
            return self.teukolsky
        elif bc in self.bcs:
            return self.teukolsky[bc]
        else:
            print('ERROR: Invalid key type {}'.format(bc))
    
    def __call__(self, bc=None):
        return self.get_teukolsky(bc=bc)

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

#multi_chebyshev_vec = np.vectorize(multi_chebyshev, excluded=[1, 2])
        
def pSchwFull(sigma):
    return sigma**2*(1 - sigma)

def qSchwFull(sigma, s, omega):
    return 2*sigma*(1+s) - sigma**2*(3 + s - 8j*omega) - 4j*omega 

def uSchwFull(sigma, s, lam, omega):
    return -4j*omega*(4j*omega + s) - (1 - 4j*omega)*(1 + s - 4j*omega)*sigma - lam

def fTSp2SchwFull(sigma, lam, omega):
    return (2*omega)**(-4)*1/16.*((256j)*omega**3*sigma - lam*(2 + lam)*(-1 + sigma)*sigma**4 + 256*omega**4*(1 + sigma) + (4j)*omega*sigma**3*(lam*(-6 + 5*sigma) + sigma*(-5 + 6*sigma)) + 16*omega**2*sigma**2*(lam*(-3 + 2*sigma**2) + sigma*(-5 + sigma + 3*sigma**2)))/(1 - sigma)**3

def gTSp2SchwFull(sigma, lam, omega):
    return (2*omega)**(-4)*((-1j/4)*omega*sigma**2*(16*omega**2 + sigma**2*(2*lam*(-1 + sigma) + sigma*(-2 + 3*sigma))))/(-1 + sigma)**3

def fTSm2SchwFull(sigma, lam, omega):
    Omega = 1j*omega
    return (16*(-128*Omega**3 + 256*Omega**4 + lam*(2 + lam)*(-1 + sigma)**2*sigma**2 + 4*Omega*sigma*(sigma*(-3 + 5*sigma) + lam*(-2 + sigma + sigma**2)) - 16*Omega**2*(lam*(-1 + sigma)*(1 + 2*sigma**2) + sigma*(-3 + sigma*(3 + sigma*(-2 + 3*sigma))))))/sigma**6/(128*Omega*(-1 + 2*Omega)*(-1 + 4*Omega)*(1 + 4*Omega))

def gTSm2SchwFull(sigma, lam, omega):
    Omega = 1j*omega
    return (64*Omega*(-1 + sigma)*(-16*Omega**2 + 2*lam*(-1 + sigma)*sigma**2 + sigma**3*(-2 + 3*sigma)))/sigma**6/(128*Omega*(-1 + 2*Omega)*(-1 + 4*Omega)*(1 + 4*Omega))

def a1sigma0(s, lam, omega):
    Omega = 1j*omega
    return -(lam + 4*Omega*(s + 4.*Omega))/(4.*Omega)

def a2sigma0(s, lam, omega):
    Omega = 1j*omega
    return (lam**2 + 2*lam*(-1 + 4*Omega)*(1 + 4*Omega + s) + 4*Omega*(-1 + 16*Omega**2*(-1 + 4*Omega) + (1 + 4*Omega)*(-3 + 8*Omega)*s + (-2 + 4*Omega)*s**2))/(32*Omega**2)

def b1sigma1(s, lam, omega):
    Omega = 1j*omega
    return (lam + (1 - 4*Omega)*(1 - 4*Omega + s) + 4*Omega*(4*Omega + s))/(-3 + 4*Omega - s + 2*(1 + s))

def chebode(p, q, u, x0, psi0, dpsi0, n):
    D = D100[:n,:n]
    
    if n == 16:
        nodes = nodes_16
        Tmat0 = Tmat0_16
        Tmat1 = Tmat1_16
        Tmat2 = Tmat2_16
    elif n == 32:
        nodes = nodes_32
        Tmat0 = Tmat0_32
        Tmat1 = Tmat1_32
        Tmat2 = Tmat2_32
    else:
        nodes = ch.chebpts1(n)
        Tmat0 = xp.array(ch.chebvander(nodes, n-1))
        Tmat1 = xp.matmul(Tmat0, D)
        Tmat2 = xp.matmul(Tmat1, D)
    
    Pmat = xp.diag(p(nodes))
    Qmat = xp.diag(q(nodes))
    Umat = xp.diag(u(nodes))

    Mmat = xp.matmul(Pmat, Tmat2) + xp.matmul(Qmat, Tmat1) + xp.matmul(Umat, Tmat0)
    Mmat[0] = xp.array(ch.chebvander(x0, n - 1))
    Mmat[1] = xp.matmul(Mmat[0], D)

    source = xp.zeros([n,1], dtype=complex)
    source[0,0] = psi0
    source[1,0] = dpsi0

    coeffs = xp.matmul(xp.linalg.inv(Mmat), source)
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