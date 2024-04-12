from spherical import Wigner3j
from numpy import np
from numba import njit
import pickle

@njit
def muCoupling(l, n):
    if (l + n < 0) or (l - n + 1 < 0):
        return 0
    return np.sqrt((l - n + 1)*(l + n))

@njit
def C3Product(l1, m1, s1, l2, m2, s2, l3, m3, s3):
    return (-1.)**(m1 + s1)*np.sqrt((2.*l1 + 1)*(2.*l2 + 1)*(2.*l3 + 1)/(4.*np.pi))*Wigner3j(l1, l2, l3, s1, -s2, -s3)*Wigner3j(l1, l2, l3, -m1, m2, m3)

class MetricReconstructor:
    def __init__(self, hertz):
        self.blackholespin = hertz.a
        self.kappa = np.sqrt(1 - hertz.a**2)
        self.hertz_class = hertz
        
        self.maxl = hertz.maxl - 4
        self.maxl_hertz = hertz.maxl
        self.geo = hertz.geo
        self.r0 = self.geo.p
        self.inner = hertz.inner
        self.outer = hertz.outer
        self.pts = hertz.pts
        self.h22 = [ModeGrid(self.maxl, self.pts.shape[0]) for i in range(5)] # four derivatives plus solutions
        self.h23 = [ModeGrid(self.maxl, self.pts.shape[0]) for i in range(5)]
        self.h24 = [ModeGrid(self.maxl, self.pts.shape[0]) for i in range(5)]
        self.h33 = [ModeGrid(self.maxl, self.pts.shape[0]) for i in range(5)]
        self.h44 = [ModeGrid(self.maxl, self.pts.shape[0]) for i in range(5)]
        self.lmodes = np.unique(self.h22[0].lmodes)
        self.mmodes = np.unique(self.h22[0].mmodes)

        self.frequency = self.geo.frequencies[-1]

    def hertz_mode(self, l, m, deriv=0):
        return self.hertz_class(l, m, deriv=deriv)
    
    def hertz_dagger_mode(self, l, m, deriv=0):
        return (-1)**(l+m)*self.hertz_class(l, m, deriv=deriv)
    
    def h22_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*np.sqrt(np.pi)*(-28*kappa*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)-28*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)+28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)+28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*m**(2)*Omega**(2)+28*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)-28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)+2*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*m**(2)*Omega**(2)*sigma**(2)-6*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)*sigma**(2)+14*kappa*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)*sigma**(2)-4*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-4*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-2*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-14*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+8*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+2*m*Omega*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*q**(3)*sigma**(2)+m*Omega*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(-28j)+kappa*m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(-14j)+kappa*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*sigma**(2)*(-2j)+2*m*Omega*C3Product(l, m, 0, 2, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*(-1+kappa**(2))*(2*m*Omega*kappa**(2)*(7-7*sigma+2*sigma**(2))+sigma**(2)*(3*m*Omega+(-7j))-7*kappa*sigma*(-2+sigma)*(m*Omega+(-1j)))+C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*sigma**(2)*(2j)+m*Omega*sigma*C3Product(l, m, 0, 3, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(210)*q**(3)*(2j)*(kappa*m*Omega*(-2+sigma)+sigma*(-1*m*Omega+(1j)))+kappa*sigma*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*(4j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(14j)+kappa*m*Omega*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*(28j)-140*kappa*m*Omega*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*muCoupling(l1, 2)+140*kappa*m*Omega*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*muCoupling(l1, 2)-140*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)+140*m*Omega*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)+140*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)-140*m*Omega*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)-28*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)+70*kappa*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)+28*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)-70*kappa*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*muCoupling(l1, 2)-7*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*muCoupling(l1, 2)-42*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+42*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+7*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+4*m*Omega*C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(21)*q**(3)*sigma**(2)*muCoupling(l1, 2)-4*m*Omega*C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(21)*q**(3)*sigma**(2)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*(-70j)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(-35j)*muCoupling(l1, 2)+q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(-35j)*muCoupling(l1, 2)+kappa*m*Omega*sigma*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*(-28j)*muCoupling(l1, 2)+kappa*m*Omega*sigma*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*(-28j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(-14j)*muCoupling(l1, 2)+kappa*m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)*muCoupling(l1, 2)+kappa*m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*sigma*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(28j)*muCoupling(l1, 2)+m*Omega*sigma*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(28j)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(35j)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(35j)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*(70j)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(5)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(5)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(5)*kappa**(2)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(5)*kappa**(2)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*(-70j)*muCoupling(l1, 1)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(-35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(-35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*(70j)*muCoupling(l1, 1)*muCoupling(l1, 2)))/(420)
        if l == l1:
            hab += ((-35*self.hertz_dagger_mode(l, m, 0)*muCoupling(l, 1)*muCoupling(l, 2)*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))-35*self.hertz_mode(l, m, 0)*muCoupling(l, 1)*muCoupling(l, 2)*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))))/(420)
        return hab

    def h23_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*np.sqrt(np.pi)*(14*m*Omega*q*C3Product(l, m, 1, 1, 0, -1, l1, m, 2)*np.sqrt(3)*(kappa*sigma*self.hertz_dagger_mode(l1, m, 1)*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+3*sigma**(2)+2*kappa**(2)*(5-5*sigma+sigma**(2)))+self.hertz_dagger_mode(l1, m, 0)*(-1j)*(3*m*sigma**(3)*(2*Omega-1*q)+kappa*sigma**(2)*(-4*m*Omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+2*kappa**(3)*(5*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-10*m*Omega+(3j))+(5j))+sigma*kappa**(2)*(-2*m*(q*(5-5*sigma+sigma**(2))+Omega*(-10+3*sigma**(2)))+(5j)*(4-3*sigma+sigma**(2)))))+sigma**(2)*(kappa*sigma*self.hertz_dagger_mode(l1, m, 1)*(-1+sigma)+self.hertz_dagger_mode(l1, m, 0)*(kappa*(3-1*sigma+m*Omega*sigma*(-2j))+m*sigma*(-1j)*(2*Omega-1*q)))*(14*C3Product(l, m, 1, 2, 0, -1, l1, m, 2)*np.sqrt(15)*(-1+kappa**(2))+2*m*Omega*C3Product(l, m, 1, 3, 0, -1, l1, m, 2)*np.sqrt(42)*q**(3)-7*C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*np.sqrt(10)*muCoupling(l1, 2)*(-1+kappa**(2)))))/(840*kappa**(2))
        if l == l1:
            hab += ((35*kappa*sigma*self.hertz_dagger_mode(l, m, 1)*np.sqrt(2)*muCoupling(l, 2)*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+2*sigma**(2)+kappa**(2)*(6-6*sigma+sigma**(2)))+self.hertz_dagger_mode(l, m, 0)*np.sqrt(2)*(-35j)*muCoupling(l, 2)*(2*m*sigma**(3)*(2*Omega-1*q)+kappa*sigma**(2)*(-2*m*Omega*(-6+sigma)+3*m*q*(-2+sigma)+(-2j)*(-3+sigma))+sigma*kappa**(2)*(-1*m*(4*Omega*(-3+sigma**(2))+q*(6-6*sigma+sigma**(2)))+(3j)*(4-3*sigma+sigma**(2)))+kappa**(3)*(6*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-12*m*Omega+(3j))+(6j)))))/(840*kappa**(2))
        return hab

    def h24_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*np.sqrt(np.pi)*(14*m*Omega*q*C3Product(l, m, -1, l1, m, -2, 1, 0, 1)*np.sqrt(3)*(kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+3*sigma**(2)+2*kappa**(2)*(5-5*sigma+sigma**(2)))+self.hertz_mode(l1, m, 0)*(-1j)*(3*m*sigma**(3)*(2*Omega-1*q)+kappa*sigma**(2)*(-4*m*Omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+2*kappa**(3)*(5*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-10*m*Omega+(3j))+(5j))+sigma*kappa**(2)*(-2*m*(q*(5-5*sigma+sigma**(2))+Omega*(-10+3*sigma**(2)))+(5j)*(4-3*sigma+sigma**(2)))))+sigma**(2)*(-1*kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)+self.hertz_mode(l1, m, 0)*(m*sigma*(1j)*(2*Omega-1*q)+kappa*(-3+sigma+m*Omega*sigma*(2j))))*(14*C3Product(l, m, -1, l1, m, -2, 2, 0, 1)*np.sqrt(15)*(-1+kappa**(2))-2*m*Omega*C3Product(l, m, -1, l1, m, -2, 3, 0, 1)*np.sqrt(42)*q**(3)-7*C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*np.sqrt(10)*muCoupling(l1, 2)*(-1+kappa**(2)))))/(840*kappa**(2))
        if l == l1:
            hab += ((-35*kappa*sigma*self.hertz_mode(l, m, 1)*np.sqrt(2)*muCoupling(l, 2)*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+2*sigma**(2)+kappa**(2)*(6-6*sigma+sigma**(2)))+self.hertz_mode(l, m, 0)*np.sqrt(2)*(35j)*muCoupling(l, 2)*(2*m*sigma**(3)*(2*Omega-1*q)+kappa*sigma**(2)*(-2*m*Omega*(-6+sigma)+3*m*q*(-2+sigma)+(-2j)*(-3+sigma))+sigma*kappa**(2)*(-1*m*(4*Omega*(-3+sigma**(2))+q*(6-6*sigma+sigma**(2)))+(3j)*(4-3*sigma+sigma**(2)))+kappa**(3)*(6*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-12*m*Omega+(3j))+(6j)))))/(840*kappa**(2))
        return hab

    def h33_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*C3Product(l, m, 2, 1, 0, 0, l1, m, 2)*np.sqrt(3*np.pi)*(kappa*sigma*(-1+sigma)*(kappa*q*sigma*self.hertz_dagger_mode(l1, m, 2)*(1j)*(-1+sigma)+2*self.hertz_dagger_mode(l1, m, 1)*(m*sigma*(-1+2*Omega*q)+m*sigma*kappa**(2)+2*kappa*q*(m*Omega*sigma+(1j))))+self.hertz_dagger_mode(l1, m, 0)*(-1j)*(m**(2)*sigma**(2)*(-4*Omega+q+4*q*Omega**(2))+kappa*m*sigma*(-1+2*Omega*q)*(sigma*(4*m*Omega+(-1j))+(4j))+m*sigma*kappa**(3)*(sigma*(4*m*Omega+(-1j))+(4j))+kappa**(2)*(4*Omega*m**(2)*sigma**(2)+q*(-6+m*sigma**(2)*(-1*m+4*m*Omega**(2)+Omega*(-2j))+sigma*(4+m*Omega*(8j)))))))/(48*kappa**(4))
        if l == l1:
            hab += ((-3*sigma*self.hertz_dagger_mode(l, m, 2)*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-1*sigma)+3*self.hertz_dagger_mode(l, m, 0)*(m**(2)*sigma**(2)*(-1+4*Omega*q-4*Omega**(2))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*Omega**(2))+m*Omega*(-2j)*(4-2*sigma+sigma**(2)))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*Omega*q*(-2+sigma)+sigma+4*Omega**(2)*(-4+sigma))+m*(-1j)*(4*Omega*(2+sigma)-1*q*(4-2*sigma+sigma**(2))))+kappa*m*sigma*(m*(-2+8*Omega*q+sigma-4*Omega**(2)*(2+sigma))+(1j)*(-4+sigma)*(2*Omega-1*q)))+kappa*self.hertz_dagger_mode(l, m, 1)*(6j)*(-1+sigma)*(m*sigma**(2)*(-2*Omega+q)+2*kappa**(2)*(m*Omega*sigma*(-2+sigma)+(-1j))-1*kappa*sigma*(4*m*Omega+m*q*(-2+sigma)+(2j)))))/(48*kappa**(4))
        return hab

    def h44_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (q*C3Product(l, m, -2, l1, m, -2, 1, 0, 0)*np.sqrt(3*np.pi)*(2j)*(-1*kappa*sigma*(-1+sigma)*(kappa*sigma*self.hertz_mode(l1, m, 2)*(-1+sigma)+self.hertz_mode(l1, m, 1)*(-2j)*(m*sigma*(2*Omega-1*q)+2*kappa*(m*Omega*sigma+(1j))))+self.hertz_mode(l1, m, 0)*(m**(2)*sigma**(2)*(1-4*Omega*q+4*Omega**(2))+kappa*m*sigma*(2*Omega-1*q)*(sigma*(4*m*Omega+(-1j))+(4j))+kappa**(2)*(-6+m*sigma**(2)*(-1*m+4*m*Omega**(2)+Omega*(-2j))+sigma*(4+m*Omega*(8j))))))/(48*kappa**(4))
        if l == l1:
            hab += ((-3*sigma*self.hertz_mode(l, m, 2)*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-1*sigma)+3*self.hertz_mode(l, m, 0)*(m**(2)*sigma**(2)*(-1+4*Omega*q-4*Omega**(2))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*Omega**(2))+m*Omega*(-2j)*(4-2*sigma+sigma**(2)))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*Omega*q*(-2+sigma)+sigma+4*Omega**(2)*(-4+sigma))+m*(-1j)*(4*Omega*(2+sigma)-1*q*(4-2*sigma+sigma**(2))))+kappa*m*sigma*(m*(-2+8*Omega*q+sigma-4*Omega**(2)*(2+sigma))+(1j)*(-4+sigma)*(2*Omega-1*q)))+kappa*self.hertz_mode(l, m, 1)*(6j)*(-1+sigma)*(m*sigma**(2)*(-2*Omega+q)+2*kappa**(2)*(m*Omega*sigma*(-2+sigma)+(-1j))-1*kappa*sigma*(4*m*Omega+m*q*(-2+sigma)+(2j)))))/(48*kappa**(4))
        return hab