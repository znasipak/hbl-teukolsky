use_gpu = False
import numpy as np
if use_gpu:
    import cupy as xp
else:
    import numpy as xp
import numpy.polynomial.chebyshev as ch

zeros16=xp.zeros([16,16], dtype=complex)
nodes16=ch.chebpts1(16)
Tmat16=ch.chebvander(nodes16,15)

zeros32=xp.zeros([32,32], dtype=complex)
nodes32=ch.chebpts1(32)
Tmat32=ch.chebvander(nodes32,31)

smaxGlobal = 0.5
sminGlobal = 0.

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

class MultiChebyshev:
    def __init__(self, coeffList, domainList):
        self.coeffs = coeffList
        self.domains = domainList
        self.n_cs = self.coeffs.size
        self.chebylist = np.empty(self.n_cs, dtype=object)
        for i in range(self.n_cs):
            self.chebylist[i] = ch.Chebyshev(self.coeffs[i], domain=[self.domains[i], self.domains[i+1]])
            
    def chebeval(self, sigma):
        if isinstance(sigma, np.ndarray):
            return multi_chebyshev_vec(sigma, self.chebylist, self.domains)
        else:
            return multi_chebyshev(sigma, self.chebylist, self.domains)
        
class HyperboloidalTeukolsky:
    domain = [sminGlobal, smaxGlobal]
    
    def __init__(self, s, l, omega):
        self.s = s
        self.l = l
        self.frequency = omega
        self.spin = 0.
        self.eigenvalue = self.l*(self.l + 1) - self.s*(self.s + 1)
        
    def solve_teukolsky(self):
        self.coeffs, self.domains = multichebteuk(self.s, self.l, self.frequency)
        self.mch = MultiChebyshev(self.coeffs, self.domains)
        
    def hteval(self, sigma):
        return self.mch.chebeval(sigma)
    
    def rteval(self, r):
        return rescaleteuk(self.s, self.frequency, r)*self.hteval(2/r)
    
def multi_chebyshev(sigma, funcs, domains):
    for i in range(funcs.size):
        if sigma <= domains[i+1]:
            return funcs[i](sigma)
    return 0

multi_chebyshev_vec = np.vectorize(multi_chebyshev, excluded=[1, 2])
        
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

def a1sigma0(s, lam, omega):
    Omega = 1j*omega
    return -(lam + 4*Omega*(s + 4.*Omega))/(4.*Omega)

def a2sigma0(s, lam, omega):
    Omega = 1j*omega
    return (lam**2 + 2*lam*(-1 + 4*Omega)*(1 + 4*Omega + s) + 4*Omega*(-1 + 16*Omega**2*(-1 + 4*Omega) + (1 + 4*Omega)*(-3 + 8*Omega)*s + (-2 + 4*Omega)*s**2))/(32*Omega**2)

def chebode(p, q, u, x0, psi0, dpsi0, n):
    nodes = ch.chebpts1(n)
    D = D100[:n,:n]

    Tmat0 = xp.array(ch.chebvander(nodes, n-1))
    Tmat1 = xp.matmul(Tmat0, D)
    Tmat2 = xp.matmul(Tmat1, D)
    Pmat = xp.zeros([n, n], dtype=complex)
    Qmat = xp.zeros([n, n], dtype=complex)
    Umat = xp.zeros([n, n], dtype=complex)
    for i in range(n):
        Pmat[i][i] = p(nodes[i])
        Qmat[i][i] = q(nodes[i])
        Umat[i][i] = u(nodes[i])

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
    nodes = ch.chebpts1(n)
    D = D100[:n,:n]

    Tmat0 = xp.array(ch.chebvander(nodes, n-1))
    Tmat1 = xp.matmul(Tmat0, D)
    Tmat0Inv = xp.linalg.inv(Tmat0)
    Fmat = xp.zeros([n, n], dtype=complex)
    Gmat = xp.zeros([n, n], dtype=complex)
    for i in range(n):
        Fmat[i][i] = f(nodes[i])
        Gmat[i][i] = g(nodes[i])
  
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

def chebteuk(s, l, omega):
    la = l*(l+1) - s*(s+1)
    a1 = a1sigma0(s, la, omega)
    a2 = a2sigma0(s, la, omega)
    smax = 0.5*xp.min(np.abs([1/a1, a1/a2]))
    if smax > smaxGlobal:
        smax = smaxGlobal
    smin = sminGlobal
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

def multichebteuk(stemp, l, omega):
    if stemp == -2:
        s = 2
    else:
        s = stemp
    la = l*(l+1) - s*(s+1)
    a1 = a1sigma0(s, la, omega)
    a2 = a2sigma0(s, la, omega)
    s1 = 0.5*xp.min(np.abs([1/a1, a1/a2]))
    boundaryNum = np.amax([16, 4*l])
    if s1 > smaxGlobal:
        s1 = smaxGlobal
        boundaryNum = 1
    boundaryList = np.zeros(boundaryNum + 1)
    boundaryList[1:] = s1*(smaxGlobal/s1)**np.linspace(0, 1, num=boundaryNum)

    smin = sminGlobal
    smax = boundaryList[1]
    dsigmadx = 1./dxOfSigma(smin, s1)
    def pode(x):
        return pSchwFull(sigmaOfX(x, smin, smax))
    def qode(x):
        return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, omega)
    def uode(x):
        return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, omega)
    psi0 = 1
    dpsi0 = -uode(-1)*psi0/qode(-1)
    coeffs = chebode(pode, qode, uode, -1, psi0, dpsi0, 32).flatten()
    coeffsList = np.empty(boundaryNum, dtype=object)
    
    i=0
    if stemp == -2:
        coeffsList[i] = flip_plus_2_teuk(l, omega, coeffs, [smin, smax])
    else:
        coeffsList[i] = coeffs
  
    for boundary in boundaryList[2:]:
        i+=1
        if use_gpu:
            coeffs = coeffs.get()
        f=ch.Chebyshev(coeffs, domain=[smin, smax])
        psi0 = f(smax)
        dpsi0 = f.deriv()(smax)
        smin = smax
        smax = boundary
        dsigmadx = 1./dxOfSigma(smin, smax)
        dpsi0 *= dsigmadx
        def pode(x):
            return pSchwFull(sigmaOfX(x, smin, smax))
        def qode(x):
            return dsigmadx*qSchwFull(sigmaOfX(x, smin, smax), s, omega)
        def uode(x):
            return dsigmadx**2*uSchwFull(sigmaOfX(x, smin, smax), s, la, omega)

        coeffs = chebode(pode, qode, uode, -1, psi0, dpsi0, 16).flatten()
        if stemp == -2:
            coeffsList[i] = flip_plus_2_teuk(l, omega, coeffs, [smin, smax])
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