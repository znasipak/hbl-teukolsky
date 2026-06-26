from spherical import Wigner3j
from .teuk import sigma_r, r_sigma
import numpy as np
#from numba import njit
from .collocode import multi_domain_chebyshev_nodes
from .cheby import MultiDomainChebyshev, MultiGridMultiDomainChebyshev
import numpy.polynomial.chebyshev as ch
import functools

#@njit
def muCoupling(l, n):
    if (l + n < 0) or (l - n + 1 < 0):
        return 0
    return np.sqrt((l - n + 1)*(l + n))

#@njit
@functools.lru_cache(maxsize=None)
def C3Product(l1, m1, s1, l2, m2, s2, l3, m3, s3):
    return (-1.)**(m1 + s1)*np.sqrt((2.*l1 + 1)*(2.*l2 + 1)*(2.*l3 + 1)/(4.*np.pi))*Wigner3j(l1, l2, l3, s1, -s2, -s3)*Wigner3j(l1, l2, l3, -m1, m2, m3)

class MetricReconstructor:
    def __init__(self, hertz, nsamples = 32):
        self.blackholespin = hertz.a
        self.kappa = hertz.kappa
        self.PhiRslm = hertz.Rslm
        self.hertz = hertz
        
        self.lmax = hertz.lmax - 4
        self.lmax_hertz = hertz.lmax
        self.source = hertz.source
        self.r0 = self.source.p
        self.domains = hertz.domains
        self.pts_inup = {"In": multi_domain_chebyshev_nodes(nsamples, hertz.domains["In"]).flatten(),
                    "Up": multi_domain_chebyshev_nodes(nsamples, hertz.domains["Up"]).flatten()}
        self.PhiRslm_inup = {"In": {0: hertz.Rslm["In"](self.pts_inup["In"], deriv = 0), 
                               1: hertz.Rslm["In"](self.pts_inup["In"], deriv = 1),
                               2: hertz.Rslm["In"](self.pts_inup["In"], deriv = 2)},
                        "Up": {0: hertz.Rslm["Up"](self.pts_inup["Up"], deriv = 0), 
                               1: hertz.Rslm["Up"](self.pts_inup["Up"], deriv = 1),
                               2: hertz.Rslm["Up"](self.pts_inup["Up"], deriv = 2)}}
        self.pts = np.concatenate((self.pts_inup["Up"], self.pts_inup["In"][::-1]))
        self.PhiRslm = {}
        for i in range(3):
            self.PhiRslm[i] = np.concatenate((self.PhiRslm_inup["Up"][i], self.PhiRslm_inup["In"][i][::-1]))

        self.nodes = ch.chebpts1(nsamples)
        self.nsamples = nsamples
        self.Tmat0 = np.array(ch.chebvander(self.nodes, nsamples - 1))
        self.Tmat0Inv = np.linalg.inv(self.Tmat0)

        # self.hab = {"In": None, "Up": None} # four derivatives plus solutions
        # self.h23 = {"In": None, "Up": None}
        # self.h24 = {"In": None, "Up": None}
        # self.h33 = {"In": None, "Up": None}
        # self.h44 = {"In": None, "Up": None}

        self.frequency = self.source.frequencies[-1]

    def mode_index(self, l, m):
        return self.hertz.mode_index(l, m)

    def hertz_mode(self, l, m, deriv=0):
        if l < 2 or l < abs(m):
            return 0.*self.PhiRslm[0][:, 0]  
        else:
            return self.PhiRslm[deriv][:, self.mode_index(l, m)]
    
    def hertz_dagger_mode(self, l, m, deriv=0):
        if l < 2 or l < abs(m):
            return 0.*self.PhiRslm[0][:, 0] 
        else:
            return (-1.)**(l+m)*self.PhiRslm[deriv][:, self.mode_index(l, m)]
    
    def extract_chebyshev_coefficients(self, field):
        domain_in_num = len(self.domains["In"]) - 1
        domain_up_num = len(self.domains["Up"]) - 1
        field_split_domains = field.reshape(domain_in_num + domain_up_num, self.nsamples)
        field_up_domains = field_split_domains[:domain_up_num]
        field_in_domains = field_split_domains[domain_up_num:]
        new_coeffs_in = np.empty((domain_in_num, self.nsamples), dtype = np.complex128)
        new_coeffs_up = np.empty((domain_up_num, self.nsamples), dtype = np.complex128)

        for i, subdomain in enumerate(field_in_domains):
            new_coeffs_in[i] = (self.Tmat0Inv @ subdomain).flatten()

        for i, subdomain in enumerate(field_up_domains):
            new_coeffs_up[i] = (self.Tmat0Inv @ subdomain).flatten()

        return {"In": MultiDomainChebyshev(new_coeffs_in, self.domains["In"][::-1]), 
                "Up": MultiDomainChebyshev(new_coeffs_up, self.domains["Up"])}

    def h22_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*np.sqrt(np.pi)*(-28*kappa*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)-28*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)+28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)+28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*m**(2)*Omega**(2)+28*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)-28*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)+2*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*m**(2)*Omega**(2)*sigma**(2)-6*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)*sigma**(2)+14*kappa*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*m**(2)*Omega**(2)*sigma**(2)-4*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-4*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-2*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*m**(2)*Omega**(2)*sigma**(2)-14*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(10)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+2*C3Product(l, m, 0, l1, m, -2, 4, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(10)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+8*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(4)*m**(2)*Omega**(2)*sigma**(2)+2*m*Omega*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*q**(3)*sigma**(2)+m*Omega*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(-28j)+kappa*m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(-14j)+kappa*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*sigma**(2)*(-2j)+2*m*Omega*C3Product(l, m, 0, 2, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*(-1+kappa**(2))*(2*m*Omega*kappa**(2)*(7-7*sigma+2*sigma**(2))+sigma**(2)*(3*m*Omega+(-7j))-7*kappa*sigma*(-2+sigma)*(m*Omega+(-1j)))+C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*sigma**(2)*(2j)+m*Omega*sigma*C3Product(l, m, 0, 3, 0, -2, l1, m, 2)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(210)*q**(3)*(2j)*(kappa*m*Omega*(-2+sigma)+sigma*(-1*m*Omega+(1j)))+kappa*sigma*C3Product(l, m, 0, l1, m, -2, 3, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(210)*m**(2)*Omega**(2)*q**(3)*(4j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)+m*Omega*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(14j)+kappa*m*Omega*sigma*C3Product(l, m, 0, l1, m, -2, 2, 0, 2)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*(28j)-140*kappa*m*Omega*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*muCoupling(l1, 2)+140*kappa*m*Omega*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*muCoupling(l1, 2)-140*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)+140*m*Omega*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)+140*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)-140*m*Omega*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*muCoupling(l1, 2)-28*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)+70*kappa*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)+28*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)-70*kappa*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*muCoupling(l1, 2)+4*m*Omega*q*C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(21)*sigma**(2)*muCoupling(l1, 2)-4*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(21)*sigma**(2)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*muCoupling(l1, 2)-7*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*muCoupling(l1, 2)-42*m*Omega*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+42*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)-4*m*Omega*q*C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(21)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+4*m*Omega*q*C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(21)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+7*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*(-70j)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(-35j)*muCoupling(l1, 2)+q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(-35j)*muCoupling(l1, 2)+kappa*m*Omega*sigma*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*(-28j)*muCoupling(l1, 2)+kappa*m*Omega*sigma*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*(-28j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(-14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*sigma**(2)*(-14j)*muCoupling(l1, 2)+kappa*m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)*muCoupling(l1, 2)+kappa*m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(2)*sigma**(2)*(14j)*muCoupling(l1, 2)+m*Omega*sigma*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(28j)*muCoupling(l1, 2)+m*Omega*sigma*C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(30)*kappa**(3)*(28j)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(35j)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*self.hertz_mode(l1, m, 0)*np.sqrt(6)*sigma**(2)*(35j)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(6)*(70j)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(5)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)+7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(5)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(5)*kappa**(2)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)-7*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(5)*kappa**(2)*sigma**(2)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*(-70j)*muCoupling(l1, 1)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(-35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(-35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_dagger_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+q*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*sigma**(2)*(35j)*muCoupling(l1, 1)*muCoupling(l1, 2)+kappa*q*sigma*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*self.hertz_mode(l1, m, 0)*np.sqrt(3)*(70j)*muCoupling(l1, 1)*muCoupling(l1, 2)))/(420)
        if l == l1:
            hab += ((-35*self.hertz_dagger_mode(l, m, 0)*muCoupling(l, 1)*muCoupling(l, 2)*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))-35*self.hertz_mode(l, m, 0)*muCoupling(l, 1)*muCoupling(l, 2)*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))))/(420)
        return hab

    def h23_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (np.sqrt(np.pi)*(7*m*Omega*q*C3Product(l, m, 1, 1, 0, -1, l1, m, 2)*np.sqrt(3)*(4*kappa*self.hertz_dagger_mode(l1, m, 1)*sigma**(2)*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+3*sigma**(2)+2*kappa**(2)*(5-5*sigma+sigma**(2)))+self.hertz_dagger_mode(l1, m, 0)*(1j)*(16*m*Omega*kappa**(4)*(-5+10*sigma-6*sigma**(2)+sigma**(3))+m*sigma**(4)*(-12*Omega+11*q+q**(3))+kappa**(2)*sigma**(2)*(-4*m*Omega*(36-36*sigma+7*sigma**(2))+m*q*(40-40*sigma+9*sigma**(2))+(-20j)*(4-3*sigma+sigma**(2)))+8*sigma*kappa**(3)*(m*Omega*(-20+30*sigma-12*sigma**(2)+sigma**(3))+(1j)*(-5+5*sigma-3*sigma**(2)+sigma**(3)))+4*kappa*sigma**(3)*(m*(-2+sigma)*(8*Omega-5*q)+(3j)*(-3+sigma))))+2*sigma**(2)*(2*m*Omega*q*C3Product(l, m, 1, 3, 0, -1, l1, m, 2)*np.sqrt(42)*(-1*kappa*self.hertz_dagger_mode(l1, m, 1)*sigma**(2)*(-1+sigma)*(-1+kappa**(2))+self.hertz_dagger_mode(l1, m, 0)*q**(2)*(1j)*(2*m*Omega*kappa**(2)*(-1+sigma)+m*sigma**(2)*(-1*Omega+q)+kappa*sigma*(m*Omega*(-2+sigma)+(1j)*(-3+sigma))))+C3Product(l, m, 1, 2, 0, -1, l1, m, 2)*np.sqrt(15)*(14j)*(-1+kappa**(2))*(kappa*self.hertz_dagger_mode(l1, m, 1)*sigma**(2)*(-1j)*(-1+sigma)+self.hertz_dagger_mode(l1, m, 0)*(2*m*Omega*kappa**(2)*(-1+sigma)+m*sigma**(2)*(-1*Omega+q)+kappa*sigma*(m*Omega*(-2+sigma)+(1j)*(-3+sigma))))+7*C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*np.sqrt(10)*muCoupling(l1, 2)*(-1+kappa**(2))*(-1*kappa*self.hertz_dagger_mode(l1, m, 1)*sigma**(2)*(-1+sigma)+self.hertz_dagger_mode(l1, m, 0)*(-1j)*(2*m*Omega*kappa**(2)*(-1+sigma)+m*sigma**(2)*(-1*Omega+q)+kappa*sigma*(m*Omega*(-2+sigma)+(1j)*(-3+sigma)))))))/(840*sigma*kappa**(2))
        if l == l1:
            hab += ((35*kappa*self.hertz_dagger_mode(l, m, 1)*np.sqrt(2)*sigma**(2)*muCoupling(l, 2)*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+2*sigma**(2)+kappa**(2)*(6-6*sigma+sigma**(2)))+self.hertz_dagger_mode(l, m, 0)*np.sqrt(2)*(35j)*muCoupling(l, 2)*(2*m*Omega*kappa**(4)*(-6+12*sigma-7*sigma**(2)+sigma**(3))+2*m*sigma**(4)*(-1*Omega+q)+kappa**(2)*sigma**(2)*(m*Omega*(-22+22*sigma-4*sigma**(2))+m*q*(6-6*sigma+sigma**(2))+(-3j)*(4-3*sigma+sigma**(2)))+sigma*kappa**(3)*(m*Omega*(-24+36*sigma-14*sigma**(2)+sigma**(3))+(1j)*(-6+6*sigma-3*sigma**(2)+sigma**(3)))+kappa*sigma**(3)*(m*(-2+sigma)*(5*Omega-3*q)+(2j)*(-3+sigma)))))/(840*sigma*kappa**(2))
        return hab

    def h24_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (4*np.sqrt(np.pi)*(7*m*Omega*q*C3Product(l, m, -1, l1, m, -2, 1, 0, 1)*np.sqrt(3)*(4*kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+3*sigma**(2)+2*kappa**(2)*(5-5*sigma+sigma**(2)))+self.hertz_mode(l1, m, 0)*(-1j)*(m*sigma**(3)*(24*Omega-1*q*(11+q**(2)))-4*kappa*sigma**(2)*(4*m*Omega*(-5+sigma)-5*m*q*(-2+sigma)+(3j)*(-3+sigma))+8*kappa**(3)*(5*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-10*m*Omega+(3j))+(5j))+sigma*kappa**(2)*(m*Omega*(80-24*sigma**(2))+m*q*(-40+40*sigma-9*sigma**(2))+(20j)*(4-3*sigma+sigma**(2)))))-2*sigma**(2)*(14*C3Product(l, m, -1, l1, m, -2, 2, 0, 1)*np.sqrt(15)*(-1+kappa**(2))*(kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)+self.hertz_mode(l1, m, 0)*(kappa*(3-1*sigma+m*Omega*sigma*(-2j))+m*sigma*(-1j)*(2*Omega-1*q)))+2*m*Omega*q*C3Product(l, m, -1, l1, m, -2, 3, 0, 1)*np.sqrt(42)*(kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)*(-1+kappa**(2))+self.hertz_mode(l1, m, 0)*q**(2)*(m*sigma*(1j)*(2*Omega-1*q)+kappa*(-3+sigma+m*Omega*sigma*(2j))))+7*C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*np.sqrt(10)*muCoupling(l1, 2)*(-1+kappa**(2))*(-1*kappa*sigma*self.hertz_mode(l1, m, 1)*(-1+sigma)+self.hertz_mode(l1, m, 0)*(m*sigma*(1j)*(2*Omega-1*q)+kappa*(-3+sigma+m*Omega*sigma*(2j)))))))/(3360*kappa**(2))
        if l == l1:
            hab += ((-140*kappa*sigma*self.hertz_mode(l, m, 1)*np.sqrt(2)*muCoupling(l, 2)*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+2*sigma**(2)+kappa**(2)*(6-6*sigma+sigma**(2)))+self.hertz_mode(l, m, 0)*np.sqrt(2)*(35j)*muCoupling(l, 2)*(m*sigma**(3)*(16*Omega-1*q*(7+q**(2)))-4*kappa*sigma**(2)*(2*m*Omega*(-6+sigma)-3*m*q*(-2+sigma)+(2j)*(-3+sigma))+4*kappa**(3)*(6*sigma*(2*m*Omega+(-1j))+sigma**(3)*(2*m*Omega+(-1j))+sigma**(2)*(-12*m*Omega+(3j))+(6j))+sigma*kappa**(2)*(m*q*(-24+24*sigma-5*sigma**(2))-16*m*Omega*(-3+sigma**(2))+(12j)*(4-3*sigma+sigma**(2))))))/(3360*kappa**(2))
        return hab

    def h33_coupling(self, l, m, l1):
        Omega = self.frequency
        q = self.blackholespin
        kappa = self.kappa
        sigma = self.pts
        hab = (2*q*C3Product(l, m, 2, 1, 0, 0, l1, m, 2)*np.sqrt(3*np.pi)*(kappa*sigma*(-1+sigma)*(kappa*sigma*self.hertz_dagger_mode(l1, m, 2)*(1j)*(-1+sigma)+self.hertz_dagger_mode(l1, m, 1)*(4*m*Omega*sigma+4*kappa*m*Omega*sigma-2*m*q*sigma+kappa*(4j)))+self.hertz_dagger_mode(l1, m, 0)*(-1j)*(m**(2)*sigma**(2)*(1-4*Omega*q+4*Omega**(2))+kappa*m*sigma*(2*Omega-1*q)*(sigma*(4*m*Omega+(-1j))+(4j))+kappa**(2)*(-6+m*sigma**(2)*(-1*m+4*m*Omega**(2)+Omega*(-2j))+sigma*(4+m*Omega*(8j))))))/(48*kappa**(4))
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
    

def lmin_of_sm(s, m):
    return np.max([np.abs(s), np.abs(m)])

def lmin_of_m(m):
    return np.abs(m)

_SPINWEIGHT_OF_ORG_METRIC_COMPONENT = {
    '11': 0,
    '13': 1,
    '14': -1,
    '33': 2,
    '44': -2
}

_SPINWEIGHT_OF_IRG_METRIC_COMPONENT = {
    '22': 0,
    '23': 1,
    '24': -1,
    '33': 2,
    '44': -2
}

from scipy.special import factorial, ellipk
def spin_zero_to_s_coupling(s, j, l, m):
    if s > 0:
        return (-1.)**(m + s)*np.sqrt(4**s*factorial(s)**2*(2*l + 1)*(2*j + 1)/factorial(2*s))*Wigner3j(s, l, j, 0, m, -m)*Wigner3j(s, l, j, s, -s, 0)
    else:
        return (-1.)**(m)*np.sqrt(4**(-s)*factorial(-s)**2*(2*l + 1)*(2*j + 1)/factorial(-2*s))*Wigner3j(-s, l, j, 0, m, -m)*Wigner3j(-s, l, j, s, -s, 0)

def metric_scalings_IRG(kappa, sigma):
    pref = 8.*kappa**3
    sigma = np.asarray(sigma)
    ones = np.zeros(len(sigma)) + 1
    return pref*np.array([[ones, ones, ones, ones],
                          [ones, (1-sigma)**2/sigma**5, (1-sigma)/sigma**4, (1-sigma)/sigma**4],
                          [ones, (1-sigma)/sigma**4, 1/sigma, ones],
                          [ones, (1-sigma)/sigma**4, ones, 1/sigma]])

def metric_rho_factors_IRG(kappa, sigma, z):
    sigma = np.asarray(sigma)
    r = r_sigma(sigma, kappa)
    a = np.sqrt(1. - kappa**2)
    rho = -1/(r - 1j*a*z)
    rhob = -1/(r + 1j*a*z)
    ones = np.zeros(len(sigma)) + 1
    return np.array([[ones, ones,           ones,           ones],
                     [ones, rho**2*rhob**2, rho**2*rhob**1, rho**1*rhob**2],
                     [ones, rho**2*rhob**1, rhob,           ones],
                     [ones, rho**1*rhob**2, ones,       rho]])

def hab_rescaled_IRG(kappa, sigma, z, h22, h24, h44):
    scalings = metric_scalings_IRG(kappa, sigma)
    rho_factors = metric_rho_factors_IRG(kappa, sigma, z)
    rescaling = rho_factors*scalings
    return rescaling[1,1]*h22, rescaling[1,3]*h24, rescaling[3,3]*h44

def metric_IRG_22_nodagger_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = h[0]*(1/12)*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))
	terms[1] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(kappa*(-2+sigma)-1*sigma)
	terms[2] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/6)
	terms[3] = omega*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j))
	terms[4] = omega*sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(sigma+omega*sigma*(-1j)*(-1+kappa)+kappa*omega*(2j))
	terms[5] = h[0]*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*sigma**(2)*(1/21)
	terms[6] = q*h[0]*np.sqrt(np.pi*(1/6))*(1/5)*(2*omega*(10-5*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(-10+sigma*(10-3*sigma)))+sigma*(5j)*(kappa*(-2+sigma)-1*sigma))
	terms[7] = sigma*h[0]*np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))
	terms[8] = omega*h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(-2/5)

	return terms

def metric_IRG_22_nodagger_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(6*(-1+kappa)*(-1+sigma)+q**(2)*(-6+4*sigma))+h[1]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(2*h[0]*(kappa*(-1+sigma)-1*sigma)+sigma*h[1]*(kappa*(-2+sigma)-1*sigma))
	terms[2] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*h[1])
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[1]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(sigma*h[1]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))+2*h[0]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(1j))))
	terms[5] = sigma*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*h[1])
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(h[1]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+2*h[0]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(sigma*h[1]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+h[0]*(-4*kappa*omega+sigma*(4*omega*(-1+kappa)+(2j))))
	terms[8] = omega*sigma*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(2*h[0]+sigma*h[1])

	return terms

def metric_IRG_22_nodagger_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(-6+6*kappa+4*q**(2))+4*h[1]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[2]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(2*h[0]*(-1+kappa)+4*h[1]*(-1*kappa+sigma*(-1+kappa))+sigma*h[2]*(-2*kappa+sigma*(-1+kappa)))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+4*h[1]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[2]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(h[0]*(2+omega*(-2j)*(-1+kappa))+sigma*h[2]*(sigma+omega*(-1j)*(-2*kappa+sigma*(-1+kappa)))+4*h[1]*(sigma+omega*(-1j)*(-1*kappa+sigma*(-1+kappa))))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(2*h[0]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[2]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+4*h[1]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(sigma*h[2]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+h[0]*(4*omega*(-1+kappa)+(2j))+h[1]*(-8*kappa*omega+8*omega*sigma*(-1+kappa)+sigma*(4j)))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))

	return terms

def metric_IRG_22_nodagger_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(6*h[1]*(-3+3*kappa+2*q**(2))+6*h[2]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[3]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(6*h[1]*(-1+kappa)+6*h[2]*(-1*kappa+sigma*(-1+kappa))+sigma*h[3]*(-2*kappa+sigma*(-1+kappa)))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(6*h[1]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+6*h[2]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[3]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(h[1]*(6+omega*(-6j)*(-1+kappa))+sigma*h[3]*(sigma+omega*(-1j)*(-2*kappa+sigma*(-1+kappa)))+6*h[2]*(sigma+omega*(-1j)*(-1*kappa+sigma*(-1+kappa))))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(6*h[1]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[3]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+6*h[2]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(6*h[1]*(2*omega*(-1+kappa)+(1j))+sigma*h[3]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+6*h[2]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j))))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))

	return terms

def metric_IRG_22_nodagger_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (h[2]*(-3+3*kappa+2*q**(2))+(1/12)*(8*h[3]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[4]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(12*h[2]*(-1+kappa)+8*h[3]*(-1*kappa+sigma*(-1+kappa))+sigma*h[4]*(-2*kappa+sigma*(-1+kappa)))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(12*h[2]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+8*h[3]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[4]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(12*h[2]*(1+omega*(-1j)*(-1+kappa))+sigma*h[4]*(sigma+omega*(-1j)*(-2*kappa+sigma*(-1+kappa)))+8*h[3]*(sigma+omega*(-1j)*(-1*kappa+sigma*(-1+kappa))))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(12*h[2]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[4]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+8*h[3]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(12*h[2]*(2*omega*(-1+kappa)+(1j))+sigma*h[4]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+8*h[3]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j))))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))

	return terms

def metric_IRG_22_dagger_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = h[0]*(1/12)*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))
	terms[1] = q*h[0]*np.sqrt(np.pi*(1/6))*(1/5)*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))
	terms[2] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(1j)/(2)*(kappa*(-2+sigma)-1*sigma)
	terms[3] = omega*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j))
	terms[4] = sigma*h[0]*np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))
	terms[5] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/6)
	terms[6] = omega*sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))
	terms[7] = omega*h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(2/5)
	terms[8] = h[0]*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*sigma**(2)*(1/21)

	return terms

def metric_IRG_22_dagger_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(6*(-1+kappa)*(-1+sigma)+q**(2)*(-6+4*sigma))+h[1]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(h[1]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+2*h[0]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(2*h[0]*(kappa*(-1+sigma)-1*sigma)+sigma*h[1]*(kappa*(-2+sigma)-1*sigma))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[1]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(sigma*h[1]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+h[0]*(-4*kappa*omega+sigma*(4*omega*(-1+kappa)+(2j))))
	terms[5] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*h[1])
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(sigma*h[1]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))+2*h[0]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(1j))))
	terms[7] = omega*sigma*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(2*h[0]+sigma*h[1])
	terms[8] = sigma*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*h[1])

	return terms

def metric_IRG_22_dagger_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(-6+6*kappa+4*q**(2))+4*h[1]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[2]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(2*h[0]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[2]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+4*h[1]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(2*h[0]*(-1+kappa)+4*h[1]*(-1*kappa+sigma*(-1+kappa))+sigma*h[2]*(-2*kappa+sigma*(-1+kappa)))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+4*h[1]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[2]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(sigma*h[2]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+h[0]*(4*omega*(-1+kappa)+(2j))+h[1]*(-8*kappa*omega+8*omega*sigma*(-1+kappa)+sigma*(4j)))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(2*h[0]*(omega*(-1+kappa)+(1j))+sigma*h[2]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))+4*h[1]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))

	return terms

def metric_IRG_22_dagger_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(6*h[1]*(-3+3*kappa+2*q**(2))+6*h[2]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[3]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(6*h[1]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[3]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+6*h[2]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(6*h[1]*(-1+kappa)+6*h[2]*(-1*kappa+sigma*(-1+kappa))+sigma*h[3]*(-2*kappa+sigma*(-1+kappa)))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(6*h[1]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+6*h[2]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[3]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(6*h[1]*(2*omega*(-1+kappa)+(1j))+sigma*h[3]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+6*h[2]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j))))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(6*h[1]*(omega*(-1+kappa)+(1j))+sigma*h[3]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))+6*h[2]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))

	return terms

def metric_IRG_22_dagger_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (h[2]*(-3+3*kappa+2*q**(2))+(1/12)*(8*h[3]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[4]*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(12*h[2]*(2*omega*(-5+5*kappa+3*q**(2))+(-5j)*(-1+kappa))+h[4]*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))+8*h[3]*(2*omega*(5*(-1+kappa)*(-1+sigma)+q**(2)*(-5+3*sigma))+(5j)*(kappa+sigma-1*kappa*sigma)))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(12*h[2]*(-1+kappa)+8*h[3]*(-1*kappa+sigma*(-1+kappa))+sigma*h[4]*(-2*kappa+sigma*(-1+kappa)))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(12*h[2]*(omega*(-7+4*q**(2))+7*kappa*(omega+(-1j))+(7j))+8*h[3]*(7*omega*(-1+kappa)*(-1+sigma)+omega*q**(2)*(-7+4*sigma)+(7j)*(kappa+sigma-1*kappa*sigma))+h[4]*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+7*kappa*sigma*(-2+sigma)*(omega+(-1j))+sigma**(2)*(7j)))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(1j)*(12*h[2]*(2*omega*(-1+kappa)+(1j))+sigma*h[4]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j)))+8*h[3]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(1j))))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(12*h[2]*(omega*(-1+kappa)+(1j))+sigma*h[4]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(1j)))+8*h[3]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))

	return terms

def metric_IRG_23_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(sigma*h[1]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[0]*(sigma*(3*(-1+q**(2))*(4+sigma*(-3+sigma))+m*q*(1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+omega*(4j)*(3+q**(2)*(-3+sigma**(2))))+kappa*(-6+q**(2)*(6+sigma*(-6+3*sigma-1*sigma**(2)+omega*(-2j)*(6+sigma*(-6+sigma))))+m*q*sigma**(2)*(3j)*(-2+sigma)+3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(sigma*h[1]*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+h[0]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*sigma**(2)*(-1+q**(2))**(-1)*(1/2)*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(3*kappa-1*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j)))
	terms[3] = np.sqrt(np.pi*(1/10))*q**(2)*sigma**(2)*(-1+q**(2))**(-1)*(1/6)*(-1*kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(-3+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[4] = omega*np.sqrt(np.pi*(1/42))*q**(3)*sigma**(2)*(-1+q**(2))**(-1)*(1/5)*(-1*kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(-3+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))

	return terms

def metric_IRG_23_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(3*h[0]*(-4+2*kappa+6*sigma-6*kappa*sigma-3*sigma**(2)+3*kappa*sigma**(2)+q**(2)*(4-2*kappa-6*sigma+2*kappa*sigma+3*sigma**(2)-1*kappa*sigma**(2)+omega*(-2j)*(2+2*kappa-4*kappa*sigma+sigma**(2)*(-2+kappa)))+m*q*(1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+omega*(4j)*(1+kappa))+sigma*(h[2]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[1]*(1j)*(12*omega+12*kappa*omega+m*q*(-6+3*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(6+sigma*(-6+sigma)))+sigma*(9j)*(kappa*(-2+sigma)-1*sigma)+q**(2)*(4*omega*(-3+sigma**(2))-1*kappa*(6+sigma*(-6+sigma))*(2*omega+(3j))+sigma*(9j)*(-2+sigma))+(18j)*(kappa+sigma))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(h[0]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+sigma*(h[2]*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+h[1]*(1j)*(20*omega*(1+kappa)+5*m*q*(-2+sigma*(-2+sigma)*(-1+kappa))+2*m*q**(3)*(5+sigma*(-5+sigma))+2*omega*q**(2)*(-10-2*kappa*(5+sigma*(-5+sigma))+3*sigma**(2))+kappa*q**(2)*(-6j)*(5+sigma*(-5+sigma))+kappa*(15j)*(2+sigma*(-2+sigma))+sigma*(15j)*(-2+sigma)*(-1+q**(2)))))
	terms[2] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(sigma**(2)*(kappa*h[2]*(-1+sigma)+h[1]*(3*kappa+omega*(-2j)*(1+kappa)+m*q*(1j)))+h[0]*(6*kappa-3*kappa*sigma+omega*sigma*(-6j)*(1+kappa)+m*q*sigma*(3j)))
	terms[3] = sigma*np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(sigma**(2)*(-1*kappa*h[2]*(-1+sigma)+h[1]*(-1j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j)))+3*h[0]*(kappa*(-2+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[4] = omega*sigma*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(sigma**(2)*(-1*kappa*h[2]*(-1+sigma)+h[1]*(-1j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j)))+3*h[0]*(kappa*(-2+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))

	return terms

def metric_IRG_23_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(6*kappa*h[2]+12*sigma*h[2]-42*kappa*sigma*h[2]-6*kappa*h[2]*q**(2)-12*sigma*h[2]*q**(2)+42*kappa*sigma*h[2]*q**(2)+sigma*h[3]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))-45*h[2]*sigma**(2)+45*kappa*h[2]*sigma**(2)+45*h[2]*q**(2)*sigma**(2)-39*kappa*h[2]*q**(2)*sigma**(2)+21*h[2]*sigma**(3)-21*kappa*h[2]*sigma**(3)-21*h[2]*q**(2)*sigma**(3)+7*kappa*h[2]*q**(2)*sigma**(3)+omega*sigma*h[2]*q**(2)*(-12j)+kappa*omega*sigma*h[2]*q**(2)*(-12j)+m*q*sigma*h[2]*(-6j)+kappa*m*q*h[2]*sigma**(2)*(-6j)+m*h[2]*q**(3)*sigma**(2)*(-6j)+m*q*h[2]*sigma**(3)*(-3j)+kappa*omega*h[2]*q**(2)*sigma**(3)*(-2j)+m*h[2]*q**(3)*sigma**(3)*(1j)+6*h[0]*(-1*(-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+omega*q**(2)*(-2j)*(kappa*(-2+sigma)-2*sigma)+m*q*(1j)*((-2+3*sigma)*(-1+kappa)+q**(2)*(-2+sigma)))+kappa*m*q*h[2]*sigma**(3)*(3j)+omega*h[2]*q**(2)*sigma**(3)*(4j)+m*sigma*h[2]*q**(3)*(6j)+m*q*h[2]*sigma**(2)*(6j)+h[1]*(6j)*(4*omega*(1+kappa)+m*q*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))-2*omega*q**(2)*(2+kappa*(2+sigma*(-4+sigma))-2*sigma**(2))+(-1j)*(-2-2*kappa-3*sigma+3*sigma*(kappa+sigma-1*kappa*sigma)+q**(2)*(2-3*sigma*(-1+sigma)+kappa*(2+sigma*(-5+sigma)))))+omega*sigma*h[2]*(12j)+kappa*omega*sigma*h[2]*(12j)+kappa*omega*h[2]*q**(2)*sigma**(2)*(12j))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(2*h[1]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*(-1+2*sigma)*(h[2]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*h[1]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+h[0]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+2*h[1]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[2]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(6*h[0]*(kappa-1*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j))+sigma*(6*h[1]*(kappa+kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j))+sigma*(kappa*sigma*h[3]*(-1+sigma)+h[2]*(-3*kappa+7*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j)))))
	terms[3] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(6*h[0]*(kappa*(-1+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(sigma*(-1*kappa*sigma*h[3]*(-1+sigma)+h[2]*(3*kappa-7*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(1+sigma))))
	terms[4] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(6*h[0]*(kappa*(-1+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(sigma*(-1*kappa*sigma*h[3]*(-1+sigma)+h[2]*(3*kappa-7*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(1+sigma))))

	return terms

def metric_IRG_23_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(6*(2*h[1]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[2]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+3*(-1+2*sigma)*(2*h[1]*(3-3*kappa+q**(2)*(-3+kappa))+4*h[2]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[3]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+sigma*(-1+sigma)*(6*h[2]*(3-3*kappa+q**(2)*(-3+kappa))+6*h[3]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[4]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+6*h[0]*(3*(-1+kappa)+q**(2)*(3-1*kappa+omega*(-2j)*(-2+kappa))+m*q*(1j)*(-3+3*kappa+q**(2)))+18*h[1]*(-1*(-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+omega*q**(2)*(-2j)*(kappa*(-2+sigma)-2*sigma)+m*q*(1j)*((-2+3*sigma)*(-1+kappa)+q**(2)*(-2+sigma)))+9*h[2]*(-4+6*sigma-3*sigma**(2)+m*q*(1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))+q**(2)*(4-6*sigma+3*sigma**(2)+kappa*(-2+2*sigma-1*sigma**(2)+omega*(-2j)*(2+sigma*(-4+sigma)))+omega*(4j)*(-1+sigma**(2))))+h[3]*(sigma*(3*(-1+q**(2))*(4+sigma*(-3+sigma))+m*q*(1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+omega*(4j)*(3+q**(2)*(-3+sigma**(2))))+kappa*(-6+q**(2)*(6+sigma*(-6+3*sigma-1*sigma**(2)+omega*(-2j)*(6+sigma*(-6+sigma))))+m*q*sigma**(2)*(3j)*(-2+sigma)+3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(6*(h[2]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*h[1]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+3*(-1+2*sigma)*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(6*h[2]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[4]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+6*h[3]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+3*h[1]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+3*h[2]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[3]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa)))+6*h[0]*(5*(-1+kappa)+m*q*(1j)*(-5+5*kappa+2*q**(2))+q**(2)*(5-2*kappa+kappa*omega*(-4j)+omega*(6j))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1j)/(2)*(6*h[0]*(-2*omega*(1+kappa)+m*q+kappa*(1j))-6*h[1]*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma))+sigma**(2)*(9*h[2]*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+kappa*sigma*h[4]*(-1j)*(-1+sigma)+h[3]*(-2*omega*sigma*(1+kappa)+m*q*sigma+kappa*(1j)*(6-11*sigma))))
	terms[3] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(6*h[0]*(kappa+m*q*(-1j)+omega*(2j)*(1+kappa))+sigma**(2)*(-1*kappa*sigma*h[4]*(-1+sigma)+h[2]*(-9j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+h[3]*(6*kappa-11*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma)))
	terms[4] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(6*h[0]*(kappa+m*q*(-1j)+omega*(2j)*(1+kappa))+sigma**(2)*(-1*kappa*sigma*h[4]*(-1+sigma)+h[2]*(-9j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+h[3]*(6*kappa-11*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma)))

	return terms

def metric_IRG_23_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(12*(2*h[1]*(3-3*kappa+q**(2)*(-3+kappa))+4*h[2]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[3]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+4*(-1+2*sigma)*(6*h[2]*(3-3*kappa+q**(2)*(-3+kappa))+6*h[3]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[4]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+sigma*(-1+sigma)*(12*h[3]*(3-3*kappa+q**(2)*(-3+kappa))+8*h[4]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[5]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+24*h[1]*(3*(-1+kappa)+q**(2)*(3-1*kappa+omega*(-2j)*(-2+kappa))+m*q*(1j)*(-3+3*kappa+q**(2)))+36*h[2]*(-1*(-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+omega*q**(2)*(-2j)*(kappa*(-2+sigma)-2*sigma)+m*q*(1j)*((-2+3*sigma)*(-1+kappa)+q**(2)*(-2+sigma)))+12*h[3]*(-4+6*sigma-3*sigma**(2)+m*q*(1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))+q**(2)*(4-6*sigma+3*sigma**(2)+kappa*(-2+2*sigma-1*sigma**(2)+omega*(-2j)*(2+sigma*(-4+sigma)))+omega*(4j)*(-1+sigma**(2))))+h[4]*(sigma*(3*(-1+q**(2))*(4+sigma*(-3+sigma))+m*q*(1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+omega*(4j)*(3+q**(2)*(-3+sigma**(2))))+kappa*(-6+q**(2)*(6+sigma*(-6+3*sigma-1*sigma**(2)+omega*(-2j)*(6+sigma*(-6+sigma))))+m*q*sigma**(2)*(3j)*(-2+sigma)+3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(12*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+4*(-1+2*sigma)*(6*h[2]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[4]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+6*h[3]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(12*h[3]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[5]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+8*h[4]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+6*h[2]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+4*h[3]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[4]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa)))+24*h[1]*(5*(-1+kappa)+m*q*(1j)*(-5+5*kappa+2*q**(2))+q**(2)*(5-2*kappa+kappa*omega*(-4j)+omega*(6j))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1j)/(2)*(24*h[1]*(-2*omega*(1+kappa)+m*q)-12*h[2]*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma))+sigma*(-12*h[3]*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(-1+5*sigma))+sigma*(kappa*sigma*h[5]*(-1j)*(-1+sigma)+h[4]*(-2*omega*sigma*(1+kappa)+m*q*sigma+kappa*(3j)*(3-5*sigma)))))
	terms[3] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(h[1]*(-24j)*(-2*omega*(1+kappa)+m*q)+sigma*(12*h[3]*(kappa-5*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(-1*kappa*sigma*h[5]*(-1+sigma)+h[4]*(9*kappa-15*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))))+h[2]*(12j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma)))
	terms[4] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(h[1]*(-24j)*(-2*omega*(1+kappa)+m*q)+sigma*(12*h[3]*(kappa-5*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(-1*kappa*sigma*h[5]*(-1+sigma)+h[4]*(9*kappa-15*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))))+h[2]*(12j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma)))

	return terms

def metric_IRG_24_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(-1*sigma*h[1]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[0]*(sigma*(-3*(-1+q**(2))*(4+sigma*(-3+sigma))+omega*(-4j)*(3+q**(2)*(-3+sigma**(2)))+m*q*(-1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma))))+kappa*(6+m*q*sigma**(2)*(-3j)*(-2+sigma)+q**(2)*(-6+sigma*(6-3*sigma+sigma**(2)+omega*(2j)*(6+sigma*(-6+sigma))))-3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(sigma*h[1]*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+h[0]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*sigma**(2)*(-1+q**(2))**(-1)*(1/2)*(-1*kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(-3+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[3] = omega*np.sqrt(np.pi*(1/42))*q**(3)*sigma**(2)*(-1+q**(2))**(-1)*(1/5)*(-1*kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(-3+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[4] = np.sqrt(np.pi*(1/10))*q**(2)*sigma**(2)*(-1+q**(2))**(-1)*(1/6)*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(3*kappa-1*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j)))

	return terms

def metric_IRG_24_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(3*h[0]*(4-2*kappa-6*sigma+6*kappa*sigma+3*sigma**(2)-3*kappa*sigma**(2)+omega*(-4j)*(1+kappa)+m*q*(-1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+q**(2)*(-4+sigma*(6-3*sigma+omega*sigma*(-4j))+kappa*(2-2*sigma+sigma**(2)+omega*(2j)*(2+sigma*(-4+sigma)))+omega*(4j)))-1*sigma*(h[2]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[1]*(1j)*(12*omega+12*kappa*omega+m*q*(-6+3*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(6+sigma*(-6+sigma)))+sigma*(9j)*(kappa*(-2+sigma)-1*sigma)+q**(2)*(4*omega*(-3+sigma**(2))-1*kappa*(6+sigma*(-6+sigma))*(2*omega+(3j))+sigma*(9j)*(-2+sigma))+(18j)*(kappa+sigma))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(h[0]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+sigma*(h[2]*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+h[1]*(1j)*(20*omega*(1+kappa)+5*m*q*(-2+sigma*(-2+sigma)*(-1+kappa))+2*m*q**(3)*(5+sigma*(-5+sigma))+2*omega*q**(2)*(-10-2*kappa*(5+sigma*(-5+sigma))+3*sigma**(2))+kappa*q**(2)*(-6j)*(5+sigma*(-5+sigma))+kappa*(15j)*(2+sigma*(-2+sigma))+sigma*(15j)*(-2+sigma)*(-1+q**(2)))))
	terms[2] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(sigma**(2)*(-1*kappa*h[2]*(-1+sigma)+h[1]*(-1j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j)))+3*h[0]*(kappa*(-2+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[3] = omega*sigma*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(sigma**(2)*(-1*kappa*h[2]*(-1+sigma)+h[1]*(-1j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j)))+3*h[0]*(kappa*(-2+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))
	terms[4] = sigma*np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(sigma**(2)*(kappa*h[2]*(-1+sigma)+h[1]*(3*kappa+omega*(-2j)*(1+kappa)+m*q*(1j)))+h[0]*(6*kappa-3*kappa*sigma+omega*sigma*(-6j)*(1+kappa)+m*q*sigma*(3j)))

	return terms

def metric_IRG_24_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(-6*kappa*h[2]-12*sigma*h[2]+42*kappa*sigma*h[2]+6*kappa*h[2]*q**(2)+12*sigma*h[2]*q**(2)-42*kappa*sigma*h[2]*q**(2)-1*sigma*h[3]*(-1+sigma)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+45*h[2]*sigma**(2)-45*kappa*h[2]*sigma**(2)-45*h[2]*q**(2)*sigma**(2)+39*kappa*h[2]*q**(2)*sigma**(2)-21*h[2]*sigma**(3)+21*kappa*h[2]*sigma**(3)+21*h[2]*q**(2)*sigma**(3)-7*kappa*h[2]*q**(2)*sigma**(3)+omega*sigma*h[2]*(-12j)+kappa*omega*sigma*h[2]*(-12j)+kappa*omega*h[2]*q**(2)*sigma**(2)*(-12j)+m*sigma*h[2]*q**(3)*(-6j)+m*q*h[2]*sigma**(2)*(-6j)+omega*h[2]*q**(2)*sigma**(3)*(-4j)+kappa*m*q*h[2]*sigma**(3)*(-3j)+m*h[2]*q**(3)*sigma**(3)*(-1j)+h[1]*(-6j)*(4*omega*(1+kappa)+m*q*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))-2*omega*q**(2)*(2+kappa*(2+sigma*(-4+sigma))-2*sigma**(2))+(-1j)*(-2-2*kappa-3*sigma+3*sigma*(kappa+sigma-1*kappa*sigma)+q**(2)*(2-3*sigma*(-1+sigma)+kappa*(2+sigma*(-5+sigma)))))+kappa*omega*h[2]*q**(2)*sigma**(3)*(2j)+6*h[0]*((-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+m*q*(1j)*(2*(-1+kappa+q**(2))-1*sigma*(-3+3*kappa+q**(2)))+omega*q**(2)*(2j)*(kappa*(-2+sigma)-2*sigma))+m*q*h[2]*sigma**(3)*(3j)+m*q*sigma*h[2]*(6j)+kappa*m*q*h[2]*sigma**(2)*(6j)+m*h[2]*q**(3)*sigma**(2)*(6j)+omega*sigma*h[2]*q**(2)*(12j)+kappa*omega*sigma*h[2]*q**(2)*(12j))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(2*h[1]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*(-1+2*sigma)*(h[2]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*h[1]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+h[0]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+2*h[1]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[2]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(6*h[0]*(kappa*(-1+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(sigma*(-1*kappa*sigma*h[3]*(-1+sigma)+h[2]*(3*kappa-7*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(1+sigma))))
	terms[3] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(6*h[0]*(kappa*(-1+sigma)+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(sigma*(-1*kappa*sigma*h[3]*(-1+sigma)+h[2]*(3*kappa-7*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(1+sigma))))
	terms[4] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1/6)*(6*h[0]*(kappa-1*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j))+sigma*(6*h[1]*(kappa+kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j))+sigma*(kappa*sigma*h[3]*(-1+sigma)+h[2]*(-3*kappa+7*kappa*sigma+omega*sigma*(-2j)*(1+kappa)+m*q*sigma*(1j)))))

	return terms

def metric_IRG_24_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(-6*(2*h[1]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[2]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+3*(1-2*sigma)*(2*h[1]*(3-3*kappa+q**(2)*(-3+kappa))+4*h[2]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[3]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))-1*sigma*(-1+sigma)*(6*h[2]*(3-3*kappa+q**(2)*(-3+kappa))+6*h[3]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[4]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+18*h[1]*((-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+m*q*(1j)*(2*(-1+kappa+q**(2))-1*sigma*(-3+3*kappa+q**(2)))+omega*q**(2)*(2j)*(kappa*(-2+sigma)-2*sigma))+6*h[0]*(3-3*kappa+m*q*(-1j)*(-3+3*kappa+q**(2))+q**(2)*(-3+kappa+omega*(2j)*(-2+kappa)))+9*h[2]*(4-2*kappa-6*sigma+6*kappa*sigma+3*sigma**(2)-3*kappa*sigma**(2)+omega*(-4j)*(1+kappa)+m*q*(-1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+q**(2)*(-4+sigma*(6-3*sigma+omega*sigma*(-4j))+kappa*(2-2*sigma+sigma**(2)+omega*(2j)*(2+sigma*(-4+sigma)))+omega*(4j)))+h[3]*(sigma*(-3*(-1+q**(2))*(4+sigma*(-3+sigma))+omega*(-4j)*(3+q**(2)*(-3+sigma**(2)))+m*q*(-1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma))))+kappa*(6+m*q*sigma**(2)*(-3j)*(-2+sigma)+q**(2)*(-6+sigma*(6-3*sigma+sigma**(2)+omega*(2j)*(6+sigma*(-6+sigma))))-3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(6*(h[2]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+2*h[1]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+3*(-1+2*sigma)*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(6*h[2]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[4]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+6*h[3]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+3*h[1]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+3*h[2]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[3]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa)))+6*h[0]*(5*(-1+kappa)+m*q*(1j)*(-5+5*kappa+2*q**(2))+q**(2)*(5-2*kappa+kappa*omega*(-4j)+omega*(6j))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(6*h[0]*(kappa+m*q*(-1j)+omega*(2j)*(1+kappa))+sigma**(2)*(-1*kappa*sigma*h[4]*(-1+sigma)+h[2]*(-9j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+h[3]*(6*kappa-11*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma)))
	terms[3] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(6*h[0]*(kappa+m*q*(-1j)+omega*(2j)*(1+kappa))+sigma**(2)*(-1*kappa*sigma*h[4]*(-1+sigma)+h[2]*(-9j)*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+h[3]*(6*kappa-11*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+h[1]*(6j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma)))
	terms[4] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1j)/(6)*(6*h[0]*(-2*omega*(1+kappa)+m*q+kappa*(1j))-6*h[1]*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(2+sigma))+sigma**(2)*(9*h[2]*(-2*omega*(1+kappa)+m*q+kappa*(-3j))+kappa*sigma*h[4]*(-1j)*(-1+sigma)+h[3]*(-2*omega*sigma*(1+kappa)+m*q*sigma+kappa*(1j)*(6-11*sigma))))

	return terms

def metric_IRG_24_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(-1+q**(2))**(-1)*(1/12)*(-12*(2*h[1]*(3-3*kappa+q**(2)*(-3+kappa))+4*h[2]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[3]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+4*(1-2*sigma)*(6*h[2]*(3-3*kappa+q**(2)*(-3+kappa))+6*h[3]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[4]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))-1*sigma*(-1+sigma)*(12*h[3]*(3-3*kappa+q**(2)*(-3+kappa))+8*h[4]*(3*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(3+kappa*(-3+sigma)-3*sigma))+h[5]*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma))))+36*h[2]*((-1+sigma)*(3-3*kappa+q**(2)*(-3+kappa))+m*q*(1j)*(2*(-1+kappa+q**(2))-1*sigma*(-3+3*kappa+q**(2)))+omega*q**(2)*(2j)*(kappa*(-2+sigma)-2*sigma))+24*h[1]*(3-3*kappa+m*q*(-1j)*(-3+3*kappa+q**(2))+q**(2)*(-3+kappa+omega*(2j)*(-2+kappa)))+12*h[3]*(4-2*kappa-6*sigma+6*kappa*sigma+3*sigma**(2)-3*kappa*sigma**(2)+omega*(-4j)*(1+kappa)+m*q*(-1j)*(-2+sigma*(-4+3*sigma)*(-1+kappa)+q**(2)*(2+sigma*(-4+sigma)))+q**(2)*(-4+sigma*(6-3*sigma+omega*sigma*(-4j))+kappa*(2-2*sigma+sigma**(2)+omega*(2j)*(2+sigma*(-4+sigma)))+omega*(4j)))+h[4]*(sigma*(-3*(-1+q**(2))*(4+sigma*(-3+sigma))+omega*(-4j)*(3+q**(2)*(-3+sigma**(2)))+m*q*(-1j)*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma))))+kappa*(6+m*q*sigma**(2)*(-3j)*(-2+sigma)+q**(2)*(-6+sigma*(6-3*sigma+sigma**(2)+omega*(2j)*(6+sigma*(-6+sigma))))-3*sigma*(2+sigma*(-3+sigma)+omega*(4j)))))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-1)*(1/10)*(12*(2*h[1]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[3]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+4*h[2]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+4*(-1+2*sigma)*(6*h[2]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[4]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+6*h[3]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+sigma*(-1+sigma)*(12*h[3]*(5-5*kappa+q**(2)*(-5+2*kappa))+h[5]*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(5+sigma*(-5+sigma)))+8*h[4]*(5*(-1+kappa+sigma-1*kappa*sigma)+q**(2)*(5-5*sigma+kappa*(-5+2*sigma))))+6*h[2]*(-6*(-1+sigma)*(5-5*kappa+q**(2)*(-5+2*kappa))+omega*q**(2)*(-4j)*(-9*sigma+2*kappa*(-5+3*sigma))+m*q*(-2j)*(-5*(-2+3*sigma)*(-1+kappa)+q**(2)*(10-6*sigma)))+4*h[3]*(m*(1j)*(5*q*(-2+sigma*(-4+3*sigma)*(-1+kappa))+2*q**(3)*(5+sigma*(-10+3*sigma)))+q**(2)*(20+15*sigma*(-2+sigma)+2*kappa*(-5+6*sigma-3*sigma**(2)+omega*(-2j)*(5+sigma*(-10+3*sigma)))+omega*(2j)*(-10+9*sigma**(2)))+5*(-4-3*sigma*(-2+sigma)+omega*(4j)+kappa*(2+3*sigma*(-2+sigma)+omega*(4j))))+h[4]*(m*sigma*q**(3)*(2j)*(5+sigma*(-5+sigma))+q**(2)*(5*sigma*(4+sigma*(-3+sigma))+2*kappa*(5+sigma*(-5+3*sigma-1*sigma**(2)+omega*(-2j)*(5+sigma*(-5+sigma))))+omega*sigma*(2j)*(-10+3*sigma**(2)))+5*(-2*kappa-3*sigma**(2)*(-1+kappa)+sigma**(3)*(-1+kappa)+2*sigma*(-2+kappa+omega*(2j)*(1+kappa)))+m*q*sigma*(5j)*(-2+sigma*(-2+sigma)*(-1+kappa)))+24*h[1]*(5*(-1+kappa)+m*q*(1j)*(-5+5*kappa+2*q**(2))+q**(2)*(5-2*kappa+kappa*omega*(-4j)+omega*(6j))))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-1+q**(2))**(-1)*(1/2)*(h[1]*(-24j)*(-2*omega*(1+kappa)+m*q)+sigma*(12*h[3]*(kappa-5*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(-1*kappa*sigma*h[5]*(-1+sigma)+h[4]*(9*kappa-15*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))))+h[2]*(12j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma)))
	terms[3] = omega*np.sqrt(np.pi*(1/42))*q**(3)*(-1+q**(2))**(-1)*(1/5)*(h[1]*(-24j)*(-2*omega*(1+kappa)+m*q)+sigma*(12*h[3]*(kappa-5*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))+sigma*(-1*kappa*sigma*h[5]*(-1+sigma)+h[4]*(9*kappa-15*kappa*sigma+m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa))))+h[2]*(12j)*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma)))
	terms[4] = np.sqrt(np.pi*(1/10))*q**(2)*(-1+q**(2))**(-1)*(1j)/(6)*(24*h[1]*(-2*omega*(1+kappa)+m*q)-12*h[2]*(6*omega*sigma*(1+kappa)-3*m*q*sigma+kappa*(1j)*(1+5*sigma))+sigma*(-12*h[3]*(2*omega*sigma*(1+kappa)-1*m*q*sigma+kappa*(1j)*(-1+5*sigma))+sigma*(kappa*sigma*h[5]*(-1j)*(-1+sigma)+h[4]*(-2*omega*sigma*(1+kappa)+m*q*sigma+kappa*(3j)*(3-5*sigma)))))

	return terms

def metric_IRG_33_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(sigma*h[2]*(-1+sigma)**(2)*(-1+q**(2))*(kappa*(-2+sigma)-1*sigma)+h[1]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[0]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1/8)*(sigma*(-1+sigma)*(sigma*h[2]*(-1j)*(-1+sigma)*(-1+q**(2))+h[1]*(4*omega*sigma*(1+kappa)-2*kappa*m*q*sigma-4*q**(2)*(omega*sigma+(1j))+(4j)))+h[0]*(2*sigma*(omega*(1+kappa)*(4+sigma*(-1+omega*(-4j)))+(-2j))+q**(2)*((-6j)+sigma*(sigma*m**(2)*(-1j)+2*(1+omega*(2j))*(omega*sigma+(2j))))+m*q*sigma*(kappa*(-4+sigma)+omega*sigma*(4j)*(1+kappa))+(6j)))

	return terms

def metric_IRG_33_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(2*h[0]*(-2+m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)-2*omega*(1+kappa)*(4*omega+(1j))+2*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma)))+m*q*(4*omega*(1+kappa+q**(2)*(-1+sigma))+(1j)*(-1+2*kappa+sigma+q**(2)-1*sigma*(kappa+q**(2)))))+(-1+sigma)*(sigma*h[3]*(-1+sigma)*(-1+q**(2))*(kappa*(-2+sigma)-1*sigma)+2*h[2]*(sigma*(1j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(1j)+sigma*(2j))+kappa*(1+q**(2)*(-1+2*sigma*(-2+sigma)*(1+omega*(-1j)))+m*q*sigma**(2)*(1j)-2*sigma*(-2+sigma+omega*(2j)))))+h[1]*(2+6*kappa+4*sigma-4*omega*sigma*(1+kappa)*(4*omega+(5j))+m*sigma*q**(3)*(-2+sigma)*(4*omega+(5j))+m*q*sigma*(8*omega*(1+kappa)+(5j)*(2+sigma*(-1+kappa)))+q**(2)*(-2+sigma*(-4-1*sigma*m**(2)+4*omega*(4*omega-1*omega*sigma+(5j)))+kappa*(-6+sigma*(-2+sigma)*(m**(2)-2*omega*(2*omega+(5j)))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(-1j)/(8)*(sigma*(-1+sigma)*(sigma*h[3]*(-1+sigma)*(-1+q**(2))+2*h[2]*(-1-2*sigma+omega*sigma*(-2j)*(-1-1*kappa+q**(2))+q*(q+2*q*sigma+kappa*m*sigma*(-1j))))+2*h[0]*(2+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(-2+sigma))+2*omega*(1+kappa)*(4*omega*sigma+sigma*(-1j)+(2j))+q**(2)*(-2+sigma*m**(2)-4*sigma*omega**(2)+omega*(2j)*(-2+sigma)))+h[1]*(-2-1*m*q*sigma**(2)*(4*omega*(1+kappa)+kappa*(5j))+2*sigma*(-2+omega*sigma*(1+kappa)*(4*omega+(5j)))+q**(2)*(2+sigma*(4+sigma*(m**(2)-2*omega*(2*omega+(5j)))))))

	return terms

def metric_IRG_33_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(2*sigma*h[2]*(-1+q**(2))*(2-3*sigma+kappa*(-4+3*sigma))+2*(-1+sigma)*(-1+q**(2))*(h[2]+sigma*h[3])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(2*h[3]+sigma*h[4])*(kappa*(-2+sigma)-1*sigma)+q*h[0]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+2*h[1]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+2*h[2]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[3]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[1]*(4j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j)))+h[2]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(-1j)/(8)*(4*h[1]*(sigma*m**(2)*q**(2)-4*sigma*omega**(2)*(-2-2*kappa+q**(2))+omega*(-2j)*(1+2*sigma)*(-1-1*kappa+q**(2))+m*q*(-4*omega*sigma*(1+kappa)+kappa*(-1j)*(1+2*sigma)))+2*h[0]*(-4*m*omega*q*(1+kappa)+m**(2)*q**(2)+2*omega*(1+kappa)*(4*omega+(-1j))+kappa*m*q*(1j)+omega*q**(2)*(-4*omega+(2j)))+sigma*(h[2]*(12*sigma*(-1+q**(2))+sigma*m**(2)*q**(2)-4*sigma*omega**(2)*(-2*(1+kappa)+q**(2))+omega*(-2j)*(-4+11*sigma)*(-1-1*kappa+q**(2))+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(4-11*sigma)))+sigma*(-1+sigma)*(h[4]*(-1+sigma)*(-1+q**(2))+h[3]*(-2j)*(kappa*m*q+2*q**(2)*(omega+(2j))-2*(omega+kappa*omega+(2j))))))

	return terms

def metric_IRG_33_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(6*sigma*h[2]*(-1+kappa)*(-1+q**(2))+6*(-1+q**(2))*(h[2]+sigma*h[3])*(2-3*sigma+kappa*(-4+3*sigma))+3*(-1+sigma)*(-1+q**(2))*(2*h[3]+sigma*h[4])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(3*h[4]+sigma*h[5])*(kappa*(-2+sigma)-1*sigma)+3*q*h[1]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+3*h[2]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+3*h[3]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[4]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[3]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j)))+q*h[1]*(12j)*(-2*kappa*omega*q+m*(-1+kappa+q**(2)))+h[2]*(12j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(-1j)/(8)*(6*h[1]*(-4*m*omega*q*(1+kappa)+m**(2)*q**(2)+kappa*m*q*(-1j)-2*omega*q**(2)*(2*omega+(1j))+2*omega*(1+kappa)*(4*omega+(1j)))+sigma*((-1+sigma)*(sigma*h[5]*(-1+sigma)*(-1+q**(2))+2*h[4]*(1-6*sigma+q**(2)*(-1+6*sigma+omega*sigma*(-2j))+kappa*m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+6*h[2]*(-4+q**(2)*(4+m**(2)-4*omega**(2)+omega*(-10j))+2*omega*(1+kappa)*(4*omega+(5j))-1*m*q*(4*omega*(1+kappa)+kappa*(5j)))+h[3]*(16-36*sigma+8*sigma*omega**(2)*(1+kappa)+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(8-17*sigma))+omega*(2j)*(-8+17*sigma)*(1+kappa)+q**(2)*((16j)*(omega+(1j))+sigma*(36+m**(2)-2*omega*(2*omega+(17j)))))))

	return terms

def metric_IRG_33_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(24*(-1+kappa)*(-1+q**(2))*(h[2]+sigma*h[3])+12*(-1+q**(2))*(2*h[3]+sigma*h[4])*(2-3*sigma+kappa*(-4+3*sigma))+4*(-1+sigma)*(-1+q**(2))*(3*h[4]+sigma*h[5])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(4*h[5]+sigma*h[6])*(kappa*(-2+sigma)-1*sigma)+6*q*h[2]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+4*h[3]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+4*h[4]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[5]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[4]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j)))+h[3]*(24j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j)))+q*h[2]*(48j)*(-2*kappa*omega*q+m*(-1+kappa+q**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1/8)*(4*(-1+2*sigma)*(sigma*h[5]*(-1j)*(-1+sigma)*(-1+q**(2))+h[4]*(4*omega*sigma*(1+kappa-1*q**(2))+(-1j)*(-1-6*sigma+q*(q+6*q*sigma+kappa*m*sigma*(-2j))))-6*h[3]*(-2*omega*(1+kappa)+kappa*m*q+(-1j)+q**(2)*(2*omega+(1j))))+12*(sigma*h[4]*(-1j)*(-1+sigma)*(-1+q**(2))-2*h[3]*(kappa*m*q*sigma+(-1j)-2*sigma*(omega+kappa*omega+(1j))+q**(2)*((1j)+2*sigma*(omega+(1j))))+h[2]*(8*omega*(1+kappa)-4*kappa*m*q-2*q**(2)*(4*omega+(1j))+(2j)))+sigma*(-1+sigma)*(sigma*h[6]*(-1j)*(-1+sigma)*(-1+q**(2))-2*sigma*h[5]*(kappa*m*q+2*q**(2)*(omega+(2j))-2*(omega+kappa*omega+(2j)))-4*h[4]*(-4*omega*(1+kappa)+2*kappa*m*q+(-3j)+q**(2)*(4*omega+(3j))))+6*h[2]*(m**(2)*q**(2)*(-2j)+4*omega*(-1-1*kappa+q**(2)+omega*(2j)*(-2*(1+kappa)+q**(2)))+2*m*q*(kappa+omega*(4j)*(1+kappa)))+8*h[3]*(2*omega*(1+kappa)*(2+sigma*(-1+omega*(-4j)))+(-2j)+q**(2)*(sigma*m**(2)*(-1j)+2*(1+omega*(2j))*(omega*sigma+(1j)))+m*q*(kappa*(-2+sigma)+omega*sigma*(4j)*(1+kappa)))+h[4]*(2*sigma*(omega*(1+kappa)*(4+sigma*(-1+omega*(-4j)))+(-2j))+q**(2)*((-6j)+sigma*(sigma*m**(2)*(-1j)+2*(1+omega*(2j))*(omega*sigma+(2j))))+m*q*sigma*(kappa*(-4+sigma)+omega*sigma*(4j)*(1+kappa))+(6j)))

	return terms

def metric_IRG_44_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(sigma*h[2]*(-1+sigma)**(2)*(-1+q**(2))*(kappa*(-2+sigma)-1*sigma)+h[1]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[0]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1/8)*(sigma*(-1+sigma)*(sigma*h[2]*(1j)*(-1+sigma)*(-1+q**(2))+2*h[1]*(-2*omega*sigma*(1+kappa)+kappa*m*q*sigma+(-2j)+2*q**(2)*(omega*sigma+(1j))))+h[0]*((-6j)+m*q*sigma*(-1*kappa*(-4+sigma)+omega*sigma*(-4j)*(1+kappa))+q**(2)*(1j)*(6+sigma*(sigma*m**(2)-2*(2*omega+(-1j))*(omega*sigma+(2j))))+2*sigma*((2j)+omega*(1+kappa)*(-4+sigma+omega*sigma*(4j)))))

	return terms

def metric_IRG_44_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(2*h[0]*(-2+m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)-2*omega*(1+kappa)*(4*omega+(1j))+2*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma)))+m*q*(4*omega*(1+kappa+q**(2)*(-1+sigma))+(1j)*(-1+2*kappa+sigma+q**(2)-1*sigma*(kappa+q**(2)))))+(-1+sigma)*(sigma*h[3]*(-1+sigma)*(-1+q**(2))*(kappa*(-2+sigma)-1*sigma)+2*h[2]*(sigma*(1j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(1j)+sigma*(2j))+kappa*(1+q**(2)*(-1+2*sigma*(-2+sigma)*(1+omega*(-1j)))+m*q*sigma**(2)*(1j)-2*sigma*(-2+sigma+omega*(2j)))))+h[1]*(2+6*kappa+4*sigma-4*omega*sigma*(1+kappa)*(4*omega+(5j))+m*sigma*q**(3)*(-2+sigma)*(4*omega+(5j))+m*q*sigma*(8*omega*(1+kappa)+(5j)*(2+sigma*(-1+kappa)))+q**(2)*(-2+sigma*(-4-1*sigma*m**(2)+4*omega*(4*omega-1*omega*sigma+(5j)))+kappa*(-6+sigma*(-2+sigma)*(m**(2)-2*omega*(2*omega+(5j)))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1j)/(8)*(sigma*(-1+sigma)*(sigma*h[3]*(-1+sigma)*(-1+q**(2))+2*h[2]*(-1-2*sigma+omega*sigma*(-2j)*(-1-1*kappa+q**(2))+q*(q+2*q*sigma+kappa*m*sigma*(-1j))))+2*h[0]*(2+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(-2+sigma))+2*omega*(1+kappa)*(4*omega*sigma+sigma*(-1j)+(2j))+q**(2)*(-2+sigma*m**(2)-4*sigma*omega**(2)+omega*(2j)*(-2+sigma)))+h[1]*(-2-1*m*q*sigma**(2)*(4*omega*(1+kappa)+kappa*(5j))+2*sigma*(-2+omega*sigma*(1+kappa)*(4*omega+(5j)))+q**(2)*(2+sigma*(4+sigma*(m**(2)-2*omega*(2*omega+(5j)))))))

	return terms

def metric_IRG_44_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(2*sigma*h[2]*(-1+q**(2))*(2-3*sigma+kappa*(-4+3*sigma))+2*(-1+sigma)*(-1+q**(2))*(h[2]+sigma*h[3])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(2*h[3]+sigma*h[4])*(kappa*(-2+sigma)-1*sigma)+q*h[0]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+2*h[1]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+2*h[2]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[3]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[1]*(4j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j)))+h[2]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1j)/(8)*(4*h[1]*(sigma*m**(2)*q**(2)-4*sigma*omega**(2)*(-2-2*kappa+q**(2))+omega*(-2j)*(1+2*sigma)*(-1-1*kappa+q**(2))+m*q*(-4*omega*sigma*(1+kappa)+kappa*(-1j)*(1+2*sigma)))+2*h[0]*(-4*m*omega*q*(1+kappa)+m**(2)*q**(2)+2*omega*(1+kappa)*(4*omega+(-1j))+kappa*m*q*(1j)+omega*q**(2)*(-4*omega+(2j)))+sigma*(h[2]*(12*sigma*(-1+q**(2))+sigma*m**(2)*q**(2)-4*sigma*omega**(2)*(-2*(1+kappa)+q**(2))+omega*(-2j)*(-4+11*sigma)*(-1-1*kappa+q**(2))+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(4-11*sigma)))+sigma*(-1+sigma)*(h[4]*(-1+sigma)*(-1+q**(2))+h[3]*(-2j)*(kappa*m*q+2*q**(2)*(omega+(2j))-2*(omega+kappa*omega+(2j))))))

	return terms

def metric_IRG_44_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(6*sigma*h[2]*(-1+kappa)*(-1+q**(2))+6*(-1+q**(2))*(h[2]+sigma*h[3])*(2-3*sigma+kappa*(-4+3*sigma))+3*(-1+sigma)*(-1+q**(2))*(2*h[3]+sigma*h[4])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(3*h[4]+sigma*h[5])*(kappa*(-2+sigma)-1*sigma)+3*q*h[1]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+3*h[2]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+3*h[3]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[4]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[3]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j)))+q*h[1]*(12j)*(-2*kappa*omega*q+m*(-1+kappa+q**(2)))+h[2]*(12j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1j)/(8)*(6*h[1]*(-4*m*omega*q*(1+kappa)+m**(2)*q**(2)+kappa*m*q*(-1j)-2*omega*q**(2)*(2*omega+(1j))+2*omega*(1+kappa)*(4*omega+(1j)))+sigma*((-1+sigma)*(sigma*h[5]*(-1+sigma)*(-1+q**(2))+2*h[4]*(1-6*sigma+q**(2)*(-1+6*sigma+omega*sigma*(-2j))+kappa*m*q*sigma*(-1j)+omega*sigma*(2j)*(1+kappa)))+6*h[2]*(-4+q**(2)*(4+m**(2)-4*omega**(2)+omega*(-10j))+2*omega*(1+kappa)*(4*omega+(5j))-1*m*q*(4*omega*(1+kappa)+kappa*(5j)))+h[3]*(16-36*sigma+8*sigma*omega**(2)*(1+kappa)+m*q*(-4*omega*sigma*(1+kappa)+kappa*(1j)*(8-17*sigma))+omega*(2j)*(-8+17*sigma)*(1+kappa)+q**(2)*((16j)*(omega+(1j))+sigma*(36+m**(2)-2*omega*(2*omega+(17j)))))))

	return terms

def metric_IRG_44_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 2
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (-1+q**(2))**(-2)*(1/16)*(24*(-1+kappa)*(-1+q**(2))*(h[2]+sigma*h[3])+12*(-1+q**(2))*(2*h[3]+sigma*h[4])*(2-3*sigma+kappa*(-4+3*sigma))+4*(-1+sigma)*(-1+q**(2))*(3*h[4]+sigma*h[5])*(1-5*kappa+3*sigma*(-1+kappa))+(-1+sigma)**(2)*(-1+q**(2))*(4*h[5]+sigma*h[6])*(kappa*(-2+sigma)-1*sigma)+6*q*h[2]*(2*q*m**(2)*(-1+kappa)+8*m*omega*q**(2)+m*(-2j)*(-1+kappa+q**(2))+4*omega*q*(-2*omega*(1+kappa)+kappa*(1j)))+4*h[3]*(-4+2*m**(2)*q**(2)*(kappa*(-1+sigma)-1*sigma)+8*m*omega*q*(1+kappa+q**(2)*(-1+sigma))+m*q*(-2j)*(1-2*kappa+sigma*(-1+kappa)+q**(2)*(-1+sigma))-4*omega*(1+kappa)*(4*omega+(1j))+4*q**(2)*(1-2*omega**(2)*(-2+kappa*(-1+sigma)+sigma)+omega*(1j)*(1+kappa*(-1+sigma))))+4*h[4]*(-4*(-1+q**(2))*(-1+kappa+2*sigma)+omega*(-4j)*(2*(-1+2*sigma)*(1+kappa)+q**(2)*(2-4*sigma+kappa*(2+3*sigma*(-2+sigma))))+m*q*(2j)*(-2+sigma*(6-2*kappa-3*sigma+3*kappa*sigma)+q**(2)*(2+3*sigma*(-2+sigma))))+h[5]*(-1+sigma)*(sigma*(2j)*(-1+q**(2))*(4*omega+m*q*(-2+sigma)+(2j))+kappa*(4-4*q**(2)+omega*sigma*(-4j)*(2+q**(2)*(-2+sigma))+m*q*sigma**(2)*(2j)))+h[4]*(-2*(-1+q**(2))*(3+kappa-2*sigma)+sigma*m**(2)*q**(2)*(kappa*(-2+sigma)-1*sigma)-4*sigma*omega**(2)*(4*(1+kappa)+q**(2)*(-4+kappa*(-2+sigma)+sigma))+omega*(2j)*(-2*(1+kappa)*(2+sigma)+q**(2)*(2*(2+sigma)+kappa*(4+sigma*(-2+sigma))))+m*q*(8*omega*sigma*(1+kappa)+sigma*(-1j)*(2+kappa*(-4+sigma)-1*sigma)+q**(2)*((-4j)+sigma*(-2+sigma)*(4*omega+(-1j)))+(4j)))+h[3]*(24j)*(-4*omega*(1+kappa)+m*q*(3-1*kappa+3*sigma*(-1+kappa)+3*q**(2)*(-1+sigma))+(-2j)+q**(2)*(omega*(4+6*kappa-6*kappa*sigma)+(2j)))+q*h[2]*(48j)*(-2*kappa*omega*q+m*(-1+kappa+q**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-1+q**(2))**(-2)*(1/8)*(12*(sigma*h[4]*(1j)*(-1+sigma)*(-1+q**(2))+2*h[3]*(kappa*m*q*sigma+(-1j)-2*sigma*(omega+kappa*omega+(1j))+q**(2)*((1j)+2*sigma*(omega+(1j))))+h[2]*(-8*omega*(1+kappa)+4*kappa*m*q+(-2j)+q**(2)*(8*omega+(2j))))+sigma*(-1+sigma)*(sigma*h[6]*(1j)*(-1+sigma)*(-1+q**(2))+2*sigma*h[5]*(kappa*m*q+2*q**(2)*(omega+(2j))-2*(omega+kappa*omega+(2j)))+4*h[4]*(-4*omega*(1+kappa)+2*kappa*m*q+(-3j)+q**(2)*(4*omega+(3j))))+6*h[2]*(2*m*q*(-1*kappa+omega*(-4j)*(1+kappa))+q**(2)*(2j)*(m**(2)+2*omega*(-2*omega+(1j)))+4*omega*(1+kappa)*(1+omega*(4j)))+8*h[3]*(m*q*(-1*kappa*(-2+sigma)+omega*sigma*(-4j)*(1+kappa))+q**(2)*(2*(1+omega*sigma*(-1j))*(2*omega+(-1j))+sigma*m**(2)*(1j))+(2j)+2*omega*(1+kappa)*(-2+sigma+omega*sigma*(4j)))+h[4]*((-6j)+m*q*sigma*(-1*kappa*(-4+sigma)+omega*sigma*(-4j)*(1+kappa))+q**(2)*(1j)*(6+sigma*(sigma*m**(2)-2*(2*omega+(-1j))*(omega*sigma+(2j))))+2*sigma*((2j)+omega*(1+kappa)*(-4+sigma+omega*sigma*(4j))))+4*(-1+2*sigma)*(sigma*h[5]*(1j)*(-1+sigma)*(-1+q**(2))+6*h[3]*(-2*omega*(1+kappa)+kappa*m*q+(-1j)+q**(2)*(2*omega+(1j)))+h[4]*(-4*omega*sigma*(1+kappa)+2*kappa*m*q*sigma+(-1j)*(1+6*sigma)+q**(2)*(4*omega*sigma+(1j)+sigma*(6j)))))

	return terms

def metric_IRG_22_nodagger_couplings(lmax, m):
	dlmax = 4
	nterm = 9
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 1)*muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 2, 0, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 3, 0, 2)
				coupling[5, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 4, 0, 2)
				coupling[6, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*muCoupling(l1, 2)
				coupling[7, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*muCoupling(l1, 2)
				coupling[8, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*muCoupling(l1, 2)

	return coupling

def metric_IRG_22_dagger_couplings(lmax, m):
	dlmax = 4
	nterm = 9
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 1)*muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 0, 2, 0, -2, l1, m, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[5, i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[6, i, i + dl] = C3Product(l, m, 0, 3, 0, -2, l1, m, 2)
				coupling[7, i, i + dl] = C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[8, i, i + dl] = C3Product(l, m, 0, 4, 0, -2, l1, m, 2)

	return coupling

def metric_IRG_23_couplings(lmax, m):
	dlmax = 3
	nterm = 5
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 1, 1, 0, -1, l1, m, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 1, 2, 0, -1, l1, m, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*muCoupling(l1, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 1, 3, 0, -1, l1, m, 2)

	return coupling

def metric_IRG_24_couplings(lmax, m):
	dlmax = 3
	nterm = 5
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 1, 0, 1)
				coupling[2, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 2, 0, 1)
				coupling[3, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 3, 0, 1)
				coupling[4, i, i + dl] = C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*muCoupling(l1, 2)

	return coupling

def metric_IRG_33_couplings(lmax, m):
	dlmax = 2
	nterm = 2
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = 1
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 2, 1, 0, 0, l1, m, 2)

	return coupling

def metric_IRG_44_couplings(lmax, m):
	dlmax = 1
	nterm = 2
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = 1
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, -2, l1, m, -2, 1, 0, 0)

	return coupling

def metric_IRG_22_nodagger_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_nodagger_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_nodagger_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_nodagger_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_nodagger_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_nodagger_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_nodagger_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_nodagger_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_nodagger_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_nodagger_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_dagger_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_dagger_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_dagger_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_dagger_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_dagger_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_dagger_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_dagger_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_dagger_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_22_dagger_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_22_dagger_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_22_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_23_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_23_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_23_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_23_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_23_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_23_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_23_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_23_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_23_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_23_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_23_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_23_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_23_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_23_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_23_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_24_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_24_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_24_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_24_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_24_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_24_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_24_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_24_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_24_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_24_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_24_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_24_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_24_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_24_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_24_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_33_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_33_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_33_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_33_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_33_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_33_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_33_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_33_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_33_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_33_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_44_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_44_radial(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_44_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_44_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_44_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_44_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_44_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_44_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_44_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_IRG_44_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_IRG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_IRG_ab_dagger_modes(kappa, sigma, m, omega, h):
	h22_dagger = metric_IRG_22_dagger_modes(kappa, sigma, m, omega, h)
	h23 = metric_IRG_23_modes(kappa, sigma, m, omega, h)
	h33 = metric_IRG_33_modes(kappa, sigma, m, omega, h)
	return [h22_dagger, h23, h33]

def metric_IRG_ab_dagger_modes_deriv1(kappa, sigma, m, omega, h):
	h22_dagger = metric_IRG_22_dagger_modes_deriv1(kappa, sigma, m, omega, h)
	h23 = metric_IRG_23_modes_deriv1(kappa, sigma, m, omega, h)
	h33 = metric_IRG_33_modes_deriv1(kappa, sigma, m, omega, h)
	return [h22_dagger, h23, h33]

def metric_IRG_ab_dagger_modes_deriv2(kappa, sigma, m, omega, h):
	h22_dagger = metric_IRG_22_dagger_modes_deriv2(kappa, sigma, m, omega, h)
	h23 = metric_IRG_23_modes_deriv2(kappa, sigma, m, omega, h)
	h33 = metric_IRG_33_modes_deriv2(kappa, sigma, m, omega, h)
	return [h22_dagger, h23, h33]

def metric_IRG_ab_dagger_modes_deriv3(kappa, sigma, m, omega, h):
	h22_dagger = metric_IRG_22_dagger_modes_deriv3(kappa, sigma, m, omega, h)
	h23 = metric_IRG_23_modes_deriv3(kappa, sigma, m, omega, h)
	h33 = metric_IRG_33_modes_deriv3(kappa, sigma, m, omega, h)
	return [h22_dagger, h23, h33]

def metric_IRG_ab_dagger_modes_deriv4(kappa, sigma, m, omega, h):
	h22_dagger = metric_IRG_22_dagger_modes_deriv4(kappa, sigma, m, omega, h)
	h23 = metric_IRG_23_modes_deriv4(kappa, sigma, m, omega, h)
	h33 = metric_IRG_33_modes_deriv4(kappa, sigma, m, omega, h)
	return [h22_dagger, h23, h33]

def metric_IRG_ab_nodagger_modes(kappa, sigma, m, omega, h):
	h22_nodagger = metric_IRG_22_nodagger_modes(kappa, sigma, m, omega, h)
	h24 = metric_IRG_24_modes(kappa, sigma, m, omega, h)
	h44 = metric_IRG_44_modes(kappa, sigma, m, omega, h)
	return [h22_nodagger, h24, h44]

def metric_IRG_ab_nodagger_modes_deriv1(kappa, sigma, m, omega, h):
	h22_nodagger = metric_IRG_22_nodagger_modes_deriv1(kappa, sigma, m, omega, h)
	h24 = metric_IRG_24_modes_deriv1(kappa, sigma, m, omega, h)
	h44 = metric_IRG_44_modes_deriv1(kappa, sigma, m, omega, h)
	return [h22_nodagger, h24, h44]

def metric_IRG_ab_nodagger_modes_deriv2(kappa, sigma, m, omega, h):
	h22_nodagger = metric_IRG_22_nodagger_modes_deriv2(kappa, sigma, m, omega, h)
	h24 = metric_IRG_24_modes_deriv2(kappa, sigma, m, omega, h)
	h44 = metric_IRG_44_modes_deriv2(kappa, sigma, m, omega, h)
	return [h22_nodagger, h24, h44]

def metric_IRG_ab_nodagger_modes_deriv3(kappa, sigma, m, omega, h):
	h22_nodagger = metric_IRG_22_nodagger_modes_deriv3(kappa, sigma, m, omega, h)
	h24 = metric_IRG_24_modes_deriv3(kappa, sigma, m, omega, h)
	h44 = metric_IRG_44_modes_deriv3(kappa, sigma, m, omega, h)
	return [h22_nodagger, h24, h44]

def metric_IRG_ab_nodagger_modes_deriv4(kappa, sigma, m, omega, h):
	h22_nodagger = metric_IRG_22_nodagger_modes_deriv4(kappa, sigma, m, omega, h)
	h24 = metric_IRG_24_modes_deriv4(kappa, sigma, m, omega, h)
	h44 = metric_IRG_44_modes_deriv4(kappa, sigma, m, omega, h)
	return [h22_nodagger, h24, h44]

def metric_IRG_ab_modes_deriv0(kappa, sigma, m, omega, h):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h22_nodagger, h24, h44 = metric_IRG_ab_nodagger_modes(kappa, sigma, m, omega, h)
    h22_dagger, h23, h33 = metric_IRG_ab_dagger_modes(kappa, sigma, m, omega, hd)
    return [h22_dagger + h22_nodagger, h23, h24, h33, h44]

def metric_IRG_ab_modes_deriv1(kappa, sigma, m, omega, h):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h22_nodagger, h24, h44 = metric_IRG_ab_nodagger_modes_deriv1(kappa, sigma, m, omega, h)
    h22_dagger, h23, h33 = metric_IRG_ab_dagger_modes_deriv1(kappa, sigma, m, omega, hd)
    return [h22_dagger + h22_nodagger, h23, h24, h33, h44]

def metric_IRG_ab_modes_deriv2(kappa, sigma, m, omega, h):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h22_nodagger, h24, h44 = metric_IRG_ab_nodagger_modes_deriv2(kappa, sigma, m, omega, h)
    h22_dagger, h23, h33 = metric_IRG_ab_dagger_modes_deriv2(kappa, sigma, m, omega, hd)
    return [h22_dagger + h22_nodagger, h23, h24, h33, h44]

def metric_IRG_ab_modes_deriv3(kappa, sigma, m, omega, h):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h22_nodagger, h24, h44 = metric_IRG_ab_nodagger_modes_deriv3(kappa, sigma, m, omega, h)
    h22_dagger, h23, h33 = metric_IRG_ab_dagger_modes_deriv3(kappa, sigma, m, omega, hd)
    return [h22_dagger + h22_nodagger, h23, h24, h33, h44]

def metric_IRG_ab_modes_deriv4(kappa, sigma, m, omega, h):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h22_nodagger, h24, h44 = metric_IRG_ab_nodagger_modes_deriv4(kappa, sigma, m, omega, h)
    h22_dagger, h23, h33 = metric_IRG_ab_dagger_modes_deriv4(kappa, sigma, m, omega, hd)
    return [h22_dagger + h22_nodagger, h23, h24, h33, h44]

def metric_IRG_ab_modes(kapppa, sigmma, m, omega, h):
	return [
		metric_IRG_ab_modes_deriv0(kapppa, sigmma, m, omega, h),
		metric_IRG_ab_modes_deriv1(kapppa, sigmma, m, omega, h),
		metric_IRG_ab_modes_deriv2(kapppa, sigmma, m, omega, h),
		metric_IRG_ab_modes_deriv3(kapppa, sigmma, m, omega, h),
		metric_IRG_ab_modes_deriv4(kapppa, sigmma, m, omega, h)
    ]

'''
Metric perturbations in ORG
'''

def metric_scalings_ORG(kappa, sigma):
    pref = 1/(2.*kappa)**5
    sigma = np.asarray(sigma)
    ones = np.zeros(len(sigma)) + 1.
    return pref*np.array([[sigma**3/(1. - sigma)**2, ones, 1/sigma/(1. - sigma), 1/sigma/(1. - sigma)], 
						  [ones, ones, ones, ones], 
						  [1/sigma/(1. - sigma), ones, 1/sigma**3, ones], 
						  [1/sigma/(1. - sigma), ones, ones, 1/sigma**3]])

def metric_rho_factors_ORG(kappa, sigma, z):
    sigma = np.asarray(sigma)
    r = r_sigma(sigma, kappa)
    a = np.sqrt(1. - kappa**2)
    rho = -1/(r - 1j*a*z)
    rhob = -1/(r + 1j*a*z)
    ones = np.zeros(len(sigma)) + 1
    return np.array([[ones,    ones, rhob**3, rho**3],
					 [ones,    ones, ones,    ones],
                     [rhob**3, ones, rhob**4, ones],
                     [rho**3,  ones, ones,    rho**4]])

def hab_rescaled_ORG(kappa, sigma, z, h11, h13, h33):
    scalings = metric_scalings_ORG(kappa, sigma)
    rho_factors = metric_rho_factors_ORG(kappa, sigma, z)
    rescaling = rho_factors*scalings
    return rescaling[0,0]*h11, rescaling[0,2]*h13, rescaling[2,2]*h33

def metric_ORG_11_nodagger_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = h[0]*(1/12)*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))
	terms[1] = q*h[0]*np.sqrt(np.pi*(1/6))*(1/5)*(2*omega*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+3*sigma)))+sigma*(5j)*(kappa*(-2+sigma)-1*sigma))
	terms[2] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(kappa*(-2+sigma)-1*sigma)
	terms[3] = omega*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+sigma**(2)*(-7j)+7*kappa*sigma*(-2+sigma)*(omega+(1j)))
	terms[4] = sigma*h[0]*np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))
	terms[5] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/6)
	terms[6] = omega*sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))
	terms[7] = omega*h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(2/5)
	terms[8] = h[0]*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*sigma**(2)*(1/21)

	return terms

def metric_ORG_11_nodagger_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(6*(-1+kappa)*(-1+sigma)+q**(2)*(-6+4*sigma))+h[1]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(2*h[0]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[1]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(2*h[0]*(kappa*(-1+sigma)-1*sigma)+sigma*h[1]*(kappa*(-2+sigma)-1*sigma))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[1]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(h[0]*(-4*kappa*omega+sigma*(4*omega*(-1+kappa)+(-2j)))+sigma*h[1]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[5] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*h[1])
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(sigma*h[1]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+2*h[0]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))
	terms[7] = omega*sigma*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(2*h[0]+sigma*h[1])
	terms[8] = sigma*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*h[1])

	return terms

def metric_ORG_11_nodagger_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(-6+6*kappa+4*q**(2))+4*h[1]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[2]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(2*h[0]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+4*h[1]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[2]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(2*h[0]*(-1+kappa)+4*h[1]*(kappa*(-1+sigma)-1*sigma)+sigma*h[2]*(kappa*(-2+sigma)-1*sigma))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+4*h[1]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[2]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(h[1]*(-8*kappa*omega+sigma*(8*omega*(-1+kappa)+(-4j)))+h[0]*(4*omega*(-1+kappa)+(-2j))+sigma*h[2]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(2*h[0]*(omega*(-1+kappa)+(-1j))+sigma*h[2]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+4*h[1]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))

	return terms

def metric_ORG_11_nodagger_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(6*h[1]*(-3+3*kappa+2*q**(2))+6*h[2]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[3]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(6*h[1]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+6*h[2]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[3]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(6*h[1]*(-1+kappa)+6*h[2]*(kappa*(-1+sigma)-1*sigma)+sigma*h[3]*(kappa*(-2+sigma)-1*sigma))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(6*h[1]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+6*h[2]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[3]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(6*h[1]*(2*omega*(-1+kappa)+(-1j))+sigma*h[3]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))+6*h[2]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(6*h[1]*(omega*(-1+kappa)+(-1j))+sigma*h[3]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+6*h[2]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))

	return terms

def metric_ORG_11_nodagger_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (h[2]*(-3+3*kappa+2*q**(2))+h[4]*(1/12)*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2)))+h[3]*(2/3)*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma)))
	terms[1] = q*np.sqrt(np.pi*(1/6))*(1/5)*(12*h[2]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+8*h[3]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[4]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[2] = q*np.sqrt(np.pi*(1/3))*(-1j)/(2)*(12*h[2]*(-1+kappa)+8*h[3]*(kappa*(-1+sigma)-1*sigma)+sigma*h[4]*(kappa*(-2+sigma)-1*sigma))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(12*h[2]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+8*h[3]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[4]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(12*h[2]*(2*omega*(-1+kappa)+(-1j))+sigma*h[4]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))+8*h[3]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[5] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[6] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(-1j)*(12*h[2]*(omega*(-1+kappa)+(-1j))+sigma*h[4]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+8*h[3]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))
	terms[7] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(2/5)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[8] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))

	return terms

def metric_ORG_11_dagger_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = h[0]*(1/12)*(-6+3*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(3+sigma*(-3+sigma)))
	terms[1] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(1j)/(2)*(kappa*(-2+sigma)-1*sigma)
	terms[2] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/6)
	terms[3] = omega*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(-7*omega*(2+sigma*(-2+sigma))+2*omega*q**(2)*(7+sigma*(-7+2*sigma))+sigma**(2)*(-7j)+7*kappa*sigma*(-2+sigma)*(omega+(1j)))
	terms[4] = omega*sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(sigma+kappa*omega*(-2j)+omega*sigma*(1j)*(-1+kappa))
	terms[5] = h[0]*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*sigma**(2)*(1/21)
	terms[6] = q*h[0]*np.sqrt(np.pi*(1/6))*(1/5)*(2*omega*(10-5*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(-10+sigma*(10-3*sigma)))+sigma*(-5j)*(kappa*(-2+sigma)-1*sigma))
	terms[7] = sigma*h[0]*np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))
	terms[8] = omega*h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(-2/5)

	return terms

def metric_ORG_11_dagger_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(6*(-1+kappa)*(-1+sigma)+q**(2)*(-6+4*sigma))+h[1]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(2*h[0]*(kappa*(-1+sigma)-1*sigma)+sigma*h[1]*(kappa*(-2+sigma)-1*sigma))
	terms[2] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*h[1])
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[1]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*(1j)*(sigma*h[1]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+2*h[0]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))
	terms[5] = sigma*np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*h[1])
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(2*h[0]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[1]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(h[0]*(-4*kappa*omega+sigma*(4*omega*(-1+kappa)+(-2j)))+sigma*h[1]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[8] = omega*sigma*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(2*h[0]+sigma*h[1])

	return terms

def metric_ORG_11_dagger_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(h[0]*(-6+6*kappa+4*q**(2))+4*h[1]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[2]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(2*h[0]*(-1+kappa)+4*h[1]*(kappa*(-1+sigma)-1*sigma)+sigma*h[2]*(kappa*(-2+sigma)-1*sigma))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(2*h[0]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+4*h[1]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[2]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*((1j)*(sigma*h[2]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+4*h[1]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))+h[0]*(2+omega*(2j)*(-1+kappa)))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(2*h[0]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+4*h[1]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[2]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(h[1]*(-8*kappa*omega+sigma*(8*omega*(-1+kappa)+(-4j)))+h[0]*(4*omega*(-1+kappa)+(-2j))+sigma*h[2]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(2*h[0]+sigma*(4*h[1]+sigma*h[2]))

	return terms

def metric_ORG_11_dagger_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/12)*(6*h[1]*(-3+3*kappa+2*q**(2))+6*h[2]*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma))+h[3]*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(6*h[1]*(-1+kappa)+6*h[2]*(kappa*(-1+sigma)-1*sigma)+sigma*h[3]*(kappa*(-2+sigma)-1*sigma))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(6*h[1]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+6*h[2]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[3]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*((1j)*(sigma*h[3]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+6*h[2]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))+h[1]*(6+omega*(6j)*(-1+kappa)))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(6*h[1]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+6*h[2]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[3]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(6*h[1]*(2*omega*(-1+kappa)+(-1j))+sigma*h[3]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))+6*h[2]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(6*h[1]+sigma*(6*h[2]+sigma*h[3]))

	return terms

def metric_ORG_11_dagger_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 9
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (h[2]*(-3+3*kappa+2*q**(2))+h[4]*(1/12)*(-6-6*sigma*(-1+kappa)+3*sigma**(2)*(-1+kappa)+2*q**(2)*(3-3*sigma+sigma**(2)))+h[3]*(2/3)*(3*(-1+kappa)*(-1+sigma)+q**(2)*(-3+2*sigma)))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(1j)/(2)*(12*h[2]*(-1+kappa)+8*h[3]*(kappa*(-1+sigma)-1*sigma)+sigma*h[4]*(kappa*(-2+sigma)-1*sigma))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(1/6)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[3] = omega*np.sqrt(np.pi*(2/15))*q**(2)*(1/7)*(12*h[2]*(omega*(-7+4*q**(2))+(-7j)+7*kappa*(omega+(1j)))+8*h[3]*(-7*omega*(-1+q**(2))+sigma*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*(-1+sigma)*(omega+(1j)))+h[4]*(14*omega*(-1+q**(2))-14*omega*sigma*(-1+q**(2))+sigma**(2)*(-7*omega+4*omega*q**(2)+(-7j))+7*kappa*sigma*(-2+sigma)*(omega+(1j))))
	terms[4] = omega*np.sqrt(np.pi*(2/105))*q**(3)*((1j)*(sigma*h[4]*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-1j)))+8*h[3]*(-1*kappa*omega+sigma*(omega*(-1+kappa)+(-1j))))+12*h[2]*(1+omega*(1j)*(-1+kappa)))
	terms[5] = np.sqrt(np.pi*(2/5))*omega**(2)*q**(4)*(1/21)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))
	terms[6] = q*np.sqrt(np.pi*(1/6))*(-1/5)*(12*h[2]*(2*omega*(-5+3*q**(2))+(-5j)+5*kappa*(2*omega+(1j)))+8*h[3]*(-10*omega*(-1+q**(2))+sigma*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*(-1+sigma)*(2*omega+(1j)))+h[4]*(20*omega*(-1+q**(2))-20*omega*sigma*(-1+q**(2))+sigma**(2)*(-10*omega+6*omega*q**(2)+(-5j))+5*kappa*sigma*(-2+sigma)*(2*omega+(1j))))
	terms[7] = np.sqrt(np.pi*(1/30))*q**(2)*(-1j)*(12*h[2]*(2*omega*(-1+kappa)+(-1j))+sigma*h[4]*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j)))+8*h[3]*(-2*kappa*omega+sigma*(2*omega*(-1+kappa)+(-1j))))
	terms[8] = omega*np.sqrt(np.pi*(1/21))*q**(3)*(-2/5)*(12*h[2]+sigma*(8*h[3]+sigma*h[4]))

	return terms

def metric_ORG_13_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(h[1]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+h[0]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[1]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+h[0]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*sigma**(2)*(2/7)*(h[1]*sigma**(2)*(-7*kappa*(2+sigma*(-2+sigma))-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7+sigma*(-7+sigma)))+h[0]*(sigma**(3)*(1+omega*(-4j))*(7-7*kappa+q**(2)*(-7+2*kappa))+14*sigma*(-1+q**(2))*(kappa+omega*(-4j)*(-1+2*kappa+q**(2)))+2*sigma**(2)*(-1+q**(2))*(7-7*kappa+omega*(2j)*(21*(-1+kappa)+2*q**(2)))+omega*(-1+q**(2))**(2)*(56j)))
	terms[3] = np.sqrt(np.pi*(2/5))*q**(2)*sigma**(2)*(-2/21)*(h[1]*sigma**(2)*(-7*kappa*(2+sigma*(-2+sigma))-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7+sigma*(-7+sigma)))+h[0]*(kappa*sigma*(1+omega*(-4j))*(-7*(4+sigma*(-3+sigma))+q**(2)*(28+sigma*(-21+2*sigma)))+(-1+q**(2))*(-7*sigma**(2)*(-3+sigma)+omega*(4j)*(7*(-2+sigma*(-2+sigma)*(-1+sigma))+2*q**(2)*(7+sigma*(-7+sigma))))))
	terms[4] = omega*np.sqrt(np.pi*(2/21))*q**(3)*sigma**(2)*(-4/15)*(h[1]*sigma**(2)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[0]*(kappa*sigma*(1+omega*(-4j))*(-3*(4+sigma*(-3+sigma))+q**(2)*(12+sigma*(-9+sigma)))+(-1+q**(2))*(-3*sigma**(2)*(-3+sigma)+omega*(4j)*(-6+3*sigma*(-2+sigma)*(-1+sigma)+q**(2)*(6+sigma*(-6+sigma))))))
	terms[5] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(4)*(-2/21)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))
	terms[6] = np.sqrt(2*np.pi)*q**(4)*sigma**(4)*(2/105)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))
	terms[7] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(4)*(4/21)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))

	return terms

def metric_ORG_13_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(2*sigma*h[1]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+h[2]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*sigma**(2)*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3))))+h[0]*(5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+h[1]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(2*sigma*h[1]*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+h[2]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+4*h[1]*sigma**(2)*(70*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(7*(10-30*sigma+21*sigma**(2)-4*sigma**(3))+kappa*(-70+98*sigma-42*sigma**(2)+6*sigma**(3)))-7*q**(2)*(20-60*sigma+51*sigma**(2)-14*sigma**(3)+kappa*(-20+44*sigma-36*sigma**(2)+9*sigma**(3))))+h[0]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[1]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(2/7)*(h[2]*sigma**(3)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))+h[0]*(1j)*((-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+8*sigma**(2)*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(3)*(4*omega+(1j)))+kappa*sigma*(-2*q**(2)*(5*sigma**(2)*(4*omega+(1j))-28*sigma*(6*omega+(1j))+21*(8*omega+(1j)))+7*(48*omega+5*sigma**(2)*(4*omega+(1j))-8*sigma*(6*omega+(1j))+(6j))))+sigma*h[1]*(1j)*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-84j))+7*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(7*(-12*sigma*(omega+(1j))+2*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))-2*q**(2)*(-42*sigma*(omega+(1j))+7*(8*omega+(5j))+sigma**(2)*(4*omega+(7j))))))
	terms[3] = sigma*np.sqrt(np.pi*(2/5))*q**(2)*(-2j)/(21)*(h[2]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))+h[0]*(-1*kappa*sigma*(4*omega+(1j))*(-84+84*sigma-35*sigma**(2)+2*q**(2)*(42-42*sigma+5*sigma**(2)))+(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(3)*(4*omega+(1j))))+sigma*h[1]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-91j))+7*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(-1*q**(2)*(28*(4*omega+(3j))+2*sigma**(2)*(4*omega+(7j))-7*sigma*(12*omega+(13j)))+7*(4*(4*omega+(3j))+sigma**(2)*(4*omega+(7j))-1*sigma*(12*omega+(13j))))))
	terms[4] = omega*sigma*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(h[2]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))+h[0]*(-1*kappa*sigma*(4*omega+(1j))*(-36+36*sigma-15*sigma**(2)+q**(2)*(36-36*sigma+5*sigma**(2)))+(-1+q**(2))*(48*omega*(-1+q**(2))-72*omega*sigma*(-1+q**(2))+4*sigma**(2)*(4*omega*(-9+q**(2))+(-9j))+15*sigma**(3)*(4*omega+(1j))))+sigma*h[1]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-39j))+3*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(3*(4*(4*omega+(3j))+sigma**(2)*(4*omega+(7j))-1*sigma*(12*omega+(13j)))-1*q**(2)*(48*omega+sigma**(2)*(4*omega+(7j))-3*sigma*(12*omega+(13j))+(36j)))))
	terms[5] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(3)*(-2j)/(21)*(kappa*h[2]*sigma**(3)*(-1j)-1*h[0]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))-1*sigma*h[1]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(7j))))
	terms[6] = np.sqrt(2*np.pi)*q**(4)*sigma**(3)*(2/105)*(kappa*h[2]*sigma**(3)+sigma*h[1]*(kappa*sigma*(7+omega*(-4j))+omega*(4j)*(-1+q**(2)))+h[0]*(5*kappa*(sigma+omega*sigma*(-4j))+omega*(16j)*(-1+q**(2))))
	terms[7] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(3)*(4/21)*(kappa*h[2]*sigma**(3)+sigma*h[1]*(kappa*sigma*(7+omega*(-4j))+omega*(4j)*(-1+q**(2)))+h[0]*(5*kappa*(sigma+omega*sigma*(-4j))+omega*(16j)*(-1+q**(2))))

	return terms

def metric_ORG_13_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(h[3]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*sigma*h[2]*(-10*sigma*(-1+q**(2))*(18-36*sigma+30*sigma**(2)-9*sigma**(3)+q**(2)*(-18+36*sigma-20*sigma**(2)+3*sigma**(3)))+kappa*(2*q**(4)*(30-90*sigma+80*sigma**(2)-25*sigma**(3)+3*sigma**(4))+30*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))-5*q**(2)*(24-72*sigma+104*sigma**(2)-70*sigma**(3)+15*sigma**(4))))+10*h[1]*(-2*sigma*(-1+q**(2))*(36-108*sigma+120*sigma**(2)-45*sigma**(3)+q**(2)*(-36+108*sigma-80*sigma**(2)+15*sigma**(3)))+kappa*(q**(2)*(-24+144*sigma-312*sigma**(2)+280*sigma**(3)-75*sigma**(4))+2*q**(4)*(6-36*sigma+48*sigma**(2)-20*sigma**(3)+3*sigma**(4))+6*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))))+4*h[0]*(5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+2*h[1]*(5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+h[2]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[3]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+4*sigma*h[2]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+2*h[1]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+4*h[0]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+2*h[1]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[2]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(2j)/(7)*(-4*h[0]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+6*sigma**(2)*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(3)*(4*omega+(1j)))+kappa*sigma*(q**(2)*(10*sigma**(2)*(4*omega+(1j))-42*sigma*(6*omega+(1j))+21*(8*omega+(1j)))-7*(24*omega+5*sigma**(2)*(4*omega+(1j))-6*sigma*(6*omega+(1j))+(3j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-4*h[1]*(-2*(-1+q**(2))*(28*omega*(-1+q**(2))-42*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-49j))+35*sigma**(3)*(omega+(1j)))+kappa*sigma*(q**(2)*(20*sigma**(2)*(omega+(1j))+21*(8*omega+(3j))-14*sigma*(12*omega+(7j)))-7*(24*omega+10*sigma**(2)*(omega+(1j))-2*sigma*(12*omega+(7j))+(9j))))+sigma*h[2]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+2*sigma**(2)*(-42*omega+4*omega*q**(2)+(-77j))+7*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(-2*q**(2)*(7*(8*omega+(9j))-7*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))+7*(2*(8*omega+(9j))-2*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))))))
	terms[3] = np.sqrt(np.pi*(2/5))*q**(2)*(-2j)/(21)*(-4*h[0]*(kappa*sigma*(4*omega+(1j))*(-7*(6-9*sigma+5*sigma**(2))+q**(2)*(42-63*sigma+10*sigma**(2)))-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+3*sigma**(2)*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(3)*(4*omega+(1j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-8*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-42*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-56j))+35*sigma**(3)*(omega+(1j)))+kappa*sigma*(-7*(12*omega+5*sigma**(2)*(omega+(1j))-4*sigma*(3*omega+(2j))+(6j))+2*q**(2)*(42*omega+5*sigma**(2)*(omega+(1j))-14*sigma*(3*omega+(2j))+(21j))))+sigma*h[2]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-161j))+7*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(7*(4*(4*omega+(5j))+sigma**(2)*(4*omega+(13j))-1*sigma*(12*omega+(23j)))-1*q**(2)*(28*(4*omega+(5j))-7*sigma*(12*omega+(23j))+sigma**(2)*(8*omega+(26j)))))))
	terms[4] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(-4*h[0]*(kappa*sigma*(4*omega+(1j))*(q**(2)*(18-27*sigma+5*sigma**(2))-3*(6-9*sigma+5*sigma**(2)))-3*(-1+q**(2))*(4*omega*(-1+q**(2))-12*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-9j))+5*sigma**(3)*(4*omega+(1j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))-8*h[1]*(-1*(-1+q**(2))*(12*omega*(-1+q**(2))-18*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-9+q**(2))+(-6j))+15*sigma**(3)*(omega+(1j)))+kappa*sigma*(-3*(12*omega+5*sigma**(2)*(omega+(1j))-4*sigma*(3*omega+(2j))+(6j))+q**(2)*(36*omega+5*sigma**(2)*(omega+(1j))-12*sigma*(3*omega+(2j))+(18j))))+sigma*h[2]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-69j))+3*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(48*omega+3*sigma**(2)*(4*omega+(13j))-3*sigma*(12*omega+(23j))+(60j)-1*q**(2)*(48*omega+sigma**(2)*(4*omega+(13j))-3*sigma*(12*omega+(23j))+(60j))))))
	terms[5] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(2)*(-2j)/(21)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[6] = np.sqrt(2*np.pi)*q**(4)*sigma**(2)*(2j)/(105)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[7] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(2)*(4j)/(21)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))

	return terms

def metric_ORG_13_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(h[4]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+120*h[1]*(6*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(3-18*sigma+20*sigma**(2)-5*sigma**(3)+kappa*(-3+8*sigma-5*sigma**(2)+sigma**(3)))+q**(2)*(kappa*(12-52*sigma+70*sigma**(2)-25*sigma**(3))+4*(-3+18*sigma-25*sigma**(2)+10*sigma**(3))))+6*sigma*h[3]*(-10*sigma*(-1+q**(2))*(18-36*sigma+30*sigma**(2)-9*sigma**(3)+q**(2)*(-18+36*sigma-20*sigma**(2)+3*sigma**(3)))+kappa*(2*q**(4)*(30-90*sigma+80*sigma**(2)-25*sigma**(3)+3*sigma**(4))+30*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))-5*q**(2)*(24-72*sigma+104*sigma**(2)-70*sigma**(3)+15*sigma**(4))))+30*h[2]*(-2*sigma*(-1+q**(2))*(36-108*sigma+120*sigma**(2)-45*sigma**(3)+q**(2)*(-36+108*sigma-80*sigma**(2)+15*sigma**(3)))+kappa*(q**(2)*(-24+144*sigma-312*sigma**(2)+280*sigma**(3)-75*sigma**(4))+2*q**(4)*(6-36*sigma+48*sigma**(2)-20*sigma**(3)+3*sigma**(4))+6*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))))+h[0]*(60*sigma**(2)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+240*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+48*sigma*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+12*h[1]*(5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+3*h[2]*(5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+h[3]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[4]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+24*h[1]*(70*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(-35*(-1+6*sigma-7*sigma**(2)+2*sigma**(3))+kappa*(-35+98*sigma-70*sigma**(2)+15*sigma**(3)))-7*q**(2)*(10*(2-12*sigma+17*sigma**(2)-7*sigma**(3))+kappa*(-20+88*sigma-120*sigma**(2)+45*sigma**(3))))+6*sigma*h[3]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+6*h[2]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+h[0]*(60*sigma**(2)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*sigma*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+336*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+12*h[1]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+3*h[2]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[3]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(2j)/(7)*(-12*h[0]*(-1*(-1+q**(2))*(-28*omega*(-1+q**(2))+4*sigma*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(2)*(4*omega+(1j)))+kappa*(-7*(8*omega+(1j)+5*sigma**(2)*(4*omega+(1j))-4*sigma*(6*omega+(1j)))+q**(2)*(56*omega+10*sigma**(2)*(4*omega+(1j))-28*sigma*(6*omega+(1j))+(7j))))-12*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-63*omega+6*omega*q**(2)+(-28j))+35*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-7*(24*omega+5*sigma**(2)*(4*omega+(3j))-4*sigma*(9*omega+(4j))+(7j))+q**(2)*(10*sigma**(2)*(4*omega+(3j))-28*sigma*(9*omega+(4j))+7*(24*omega+(7j)))))+sigma*(h[4]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+16*sigma**(2)*(-21*omega+2*omega*q**(2)+(-21j))+35*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(-7*(-48*sigma*(omega+(1j))+6*(8*omega+(5j))+5*sigma**(2)*(4*omega+(7j)))+q**(2)*(-336*sigma*(omega+(1j))+42*(8*omega+(5j))+10*sigma**(2)*(4*omega+(7j)))))+sigma*h[3]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-21*omega+2*omega*q**(2)+(-56j))+7*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(7*(16*omega-4*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(26j))-2*q**(2)*(56*omega-14*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(91j))))))
	terms[3] = np.sqrt(np.pi*(2/5))*q**(2)*(-2j)/(21)*(-12*h[0]*(kappa*(4*omega+(1j))*(2*q**(2)*(7-21*sigma+5*sigma**(2))-7*(2-6*sigma+5*sigma**(2)))-1*(-1+q**(2))*(-28*omega*(-1+q**(2))+2*sigma*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(2)*(4*omega+(1j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-84*omega+8*omega*q**(2)+(-91j))+35*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(q**(2)*(84*(4*omega+(3j))+10*sigma**(2)*(4*omega+(7j))-28*sigma*(12*omega+(13j)))-7*(48*omega+5*sigma**(2)*(4*omega+(7j))-4*sigma*(12*omega+(13j))+(36j))))+sigma*h[3]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-231j))+7*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(7*(4*(4*omega+(7j))-3*sigma*(4*omega+(11j))+sigma**(2)*(4*omega+(19j)))-1*q**(2)*(28*(4*omega+(7j))-21*sigma*(4*omega+(11j))+sigma**(2)*(8*omega+(38j))))))-12*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+sigma**(2)*(-252*omega+24*omega*q**(2)+(-133j))+35*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-7*(24*omega+5*sigma**(2)*(4*omega+(3j))+(10j)-1*sigma*(36*omega+(19j)))+q**(2)*(168*omega+10*sigma**(2)*(4*omega+(3j))-7*sigma*(36*omega+(19j))+(70j)))))
	terms[4] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(-12*h[0]*(kappa*(4*omega+(1j))*(q**(2)*(6-18*sigma+5*sigma**(2))-3*(2-6*sigma+5*sigma**(2)))-1*(-1+q**(2))*(-12*omega*(-1+q**(2))+2*sigma*(4*omega*(-9+q**(2))+(-9j))+15*sigma**(2)*(4*omega+(1j))))-12*h[1]*(-3*(-1+q**(2))*(4*omega*(-1+q**(2))-12*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-19j))+5*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-3*(24*omega+5*sigma**(2)*(4*omega+(3j))+(10j)-1*sigma*(36*omega+(19j)))+q**(2)*(72*omega+5*sigma**(2)*(4*omega+(3j))-3*sigma*(36*omega+(19j))+(30j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(48*omega*(-1+q**(2))-72*omega*sigma*(-1+q**(2))+4*sigma**(2)*(4*omega*(-9+q**(2))+(-39j))+15*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(q**(2)*(36*(4*omega+(3j))+5*sigma**(2)*(4*omega+(7j))-12*sigma*(12*omega+(13j)))-3*(48*omega+5*sigma**(2)*(4*omega+(7j))-4*sigma*(12*omega+(13j))+(36j))))+sigma*h[3]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-99j))+3*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(48*omega-9*sigma*(4*omega+(11j))+3*sigma**(2)*(4*omega+(19j))+(84j)-1*q**(2)*(48*omega-9*sigma*(4*omega+(11j))+sigma**(2)*(4*omega+(19j))+(84j))))))
	terms[5] = sigma*np.sqrt(np.pi*(2/5))*q**(4)*(-2j)/(21)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[6] = sigma*np.sqrt(2*np.pi)*q**(4)*(2j)/(105)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[7] = omega*sigma*np.sqrt(np.pi*(1/165))*q**(5)*(4j)/(21)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))

	return terms

def metric_ORG_13_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(h[5]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+120*h[1]*(6*(-1+kappa)*(6-20*sigma+15*sigma**(2))+2*q**(4)*(-18+40*sigma-15*sigma**(2)+kappa*(8-10*sigma+3*sigma**(2)))+q**(2)*(kappa*(-52+140*sigma-75*sigma**(2))+8*(9-25*sigma+15*sigma**(2))))+480*h[2]*(6*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(3-18*sigma+20*sigma**(2)-5*sigma**(3)+kappa*(-3+8*sigma-5*sigma**(2)+sigma**(3)))+q**(2)*(kappa*(12-52*sigma+70*sigma**(2)-25*sigma**(3))+4*(-3+18*sigma-25*sigma**(2)+10*sigma**(3))))+8*sigma*h[4]*(-10*sigma*(-1+q**(2))*(18-36*sigma+30*sigma**(2)-9*sigma**(3)+q**(2)*(-18+36*sigma-20*sigma**(2)+3*sigma**(3)))+kappa*(2*q**(4)*(30-90*sigma+80*sigma**(2)-25*sigma**(3)+3*sigma**(4))+30*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))-5*q**(2)*(24-72*sigma+104*sigma**(2)-70*sigma**(3)+15*sigma**(4))))+60*h[3]*(-2*sigma*(-1+q**(2))*(36-108*sigma+120*sigma**(2)-45*sigma**(3)+q**(2)*(-36+108*sigma-80*sigma**(2)+15*sigma**(3)))+kappa*(q**(2)*(-24+144*sigma-312*sigma**(2)+280*sigma**(3)-75*sigma**(4))+2*q**(4)*(6-36*sigma+48*sigma**(2)-20*sigma**(3)+3*sigma**(4))+6*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))))+h[0]*(120*sigma*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+48*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+4*h[1]*(60*sigma**(2)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+240*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+48*sigma*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+24*h[2]*(5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+4*h[3]*(5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+h[4]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(24*h[1]*(70*(-1+kappa)*(6-20*sigma+15*sigma**(2))+2*q**(4)*(-70*(3-7*sigma+3*sigma**(2))+kappa*(98-140*sigma+45*sigma**(2)))-7*q**(2)*(-10*(12-34*sigma+21*sigma**(2))+kappa*(88-240*sigma+135*sigma**(2))))+h[5]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+96*h[2]*(70*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(-35*(-1+6*sigma-7*sigma**(2)+2*sigma**(3))+kappa*(-35+98*sigma-70*sigma**(2)+15*sigma**(3)))-7*q**(2)*(10*(2-12*sigma+17*sigma**(2)-7*sigma**(3))+kappa*(-20+88*sigma-120*sigma**(2)+45*sigma**(3))))+8*sigma*h[4]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+12*h[3]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+h[0]*(120*sigma*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j)))+4*h[1]*(60*sigma**(2)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*sigma*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+336*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+24*h[2]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+4*h[3]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[4]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(2j)/(7)*(672*omega*h[2]-2016*omega*sigma*h[2]+4032*kappa*omega*sigma*h[2]+448*omega*sigma*h[3]-1344*omega*h[2]*q**(2)+4032*omega*sigma*h[2]*q**(2)-4032*kappa*omega*sigma*h[2]*q**(2)-896*omega*sigma*h[3]*q**(2)+672*omega*h[2]*q**(4)-2016*omega*sigma*h[2]*q**(4)+448*omega*sigma*h[3]*q**(4)+6048*omega*h[2]*sigma**(2)-6048*kappa*omega*h[2]*sigma**(2)-672*omega*h[3]*sigma**(2)+1344*kappa*omega*h[3]*sigma**(2)+56*omega*h[4]*sigma**(2)-6624*omega*h[2]*q**(2)*sigma**(2)+6048*kappa*omega*h[2]*q**(2)*sigma**(2)+1344*omega*h[3]*q**(2)*sigma**(2)-1344*kappa*omega*h[3]*q**(2)*sigma**(2)-112*omega*h[4]*q**(2)*sigma**(2)+576*omega*h[2]*q**(4)*sigma**(2)-672*omega*h[3]*q**(4)*sigma**(2)+56*omega*h[4]*q**(4)*sigma**(2)-3360*omega*h[2]*sigma**(3)+3360*kappa*omega*h[2]*sigma**(3)+1344*omega*h[3]*sigma**(3)-1344*kappa*omega*h[3]*sigma**(3)-56*omega*h[4]*sigma**(3)+112*kappa*omega*h[4]*sigma**(3)+3360*omega*h[2]*q**(2)*sigma**(3)-960*kappa*omega*h[2]*q**(2)*sigma**(3)-1472*omega*h[3]*q**(2)*sigma**(3)+1344*kappa*omega*h[3]*q**(2)*sigma**(3)+112*omega*h[4]*q**(2)*sigma**(3)-112*kappa*omega*h[4]*q**(2)*sigma**(3)+128*omega*h[3]*q**(4)*sigma**(3)-56*omega*h[4]*q**(4)*sigma**(3)-560*omega*h[3]*sigma**(4)+560*kappa*omega*h[3]*sigma**(4)+84*omega*h[4]*sigma**(4)-84*kappa*omega*h[4]*sigma**(4)+560*omega*h[3]*q**(2)*sigma**(4)-160*kappa*omega*h[3]*q**(2)*sigma**(4)-92*omega*h[4]*q**(2)*sigma**(4)+84*kappa*omega*h[4]*q**(2)*sigma**(4)+8*omega*h[4]*q**(4)*sigma**(4)-28*omega*h[4]*sigma**(5)+28*kappa*omega*h[4]*sigma**(5)+28*omega*h[4]*q**(2)*sigma**(5)-8*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-4368j)+h[2]*q**(2)*sigma**(2)*(-4368j)+h[2]*sigma**(3)*(-4200j)+kappa*h[3]*sigma**(3)*(-1904j)+h[3]*q**(2)*sigma**(3)*(-1904j)+kappa*sigma*h[2]*q**(2)*(-1848j)+h[3]*sigma**(4)*(-1400j)+kappa*h[2]*q**(2)*sigma**(3)*(-1200j)+kappa*h[3]*q**(2)*sigma**(2)*(-1176j)+kappa*h[3]*q**(2)*sigma**(4)*(-400j)+kappa*h[4]*sigma**(4)*(-294j)+h[4]*q**(2)*sigma**(4)*(-294j)+kappa*h[4]*q**(2)*sigma**(3)*(-238j)+h[4]*sigma**(5)*(-175j)+kappa*h[4]*q**(2)*sigma**(5)*(-50j)+kappa*h[5]*q**(2)*sigma**(4)*(-14j)+kappa*h[5]*sigma**(5)*(-14j)+h[5]*q**(2)*sigma**(5)*(-14j)+h[5]*sigma**(6)*(-7j)+kappa*h[5]*q**(2)*sigma**(6)*(-2j)+kappa*h[5]*sigma**(6)*(7j)+h[5]*q**(2)*sigma**(6)*(7j)+kappa*h[5]*sigma**(4)*(14j)+h[5]*sigma**(5)*(14j)+kappa*h[5]*q**(2)*sigma**(5)*(14j)-24*h[1]*(-1*(-1+q**(2))*(-56*omega*(-1+q**(2))+2*sigma*(8*omega*(-21+2*q**(2))+(-63j))+35*sigma**(2)*(8*omega+(5j)))+kappa*(-7*(16*omega-6*sigma*(8*omega+(3j))+(4j)+5*sigma**(2)*(8*omega+(5j)))+2*q**(2)*(56*omega-21*sigma*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))+(14j))))+24*h[0]*(8*omega*q**(4)+7*(-1+kappa)*(-12*omega+20*omega*sigma+(-2j)+sigma*(5j))+q**(2)*(-92*omega+140*omega*sigma+(-14j)-2*kappa*(-42*omega+20*omega*sigma+(-7j)+sigma*(5j))+sigma*(35j)))+kappa*h[4]*sigma**(5)*(175j)+h[4]*q**(2)*sigma**(5)*(175j)+kappa*h[4]*sigma**(3)*(238j)+h[4]*sigma**(4)*(294j)+kappa*h[4]*q**(2)*sigma**(4)*(294j)+kappa*h[3]*sigma**(2)*(1176j)+kappa*h[3]*sigma**(4)*(1400j)+h[3]*q**(2)*sigma**(4)*(1400j)+kappa*sigma*h[2]*(1848j)+h[3]*sigma**(3)*(1904j)+kappa*h[3]*q**(2)*sigma**(3)*(1904j)+kappa*h[2]*sigma**(3)*(4200j)+h[2]*q**(2)*sigma**(3)*(4200j)+h[2]*sigma**(2)*(4368j)+kappa*h[2]*q**(2)*sigma**(2)*(4368j))
	terms[3] = np.sqrt(np.pi*(2/5))*q**(2)*(-2j)/(21)*(672*omega*h[2]-2016*omega*sigma*h[2]+4032*kappa*omega*sigma*h[2]+448*omega*sigma*h[3]-1344*omega*h[2]*q**(2)+4032*omega*sigma*h[2]*q**(2)-4032*kappa*omega*sigma*h[2]*q**(2)-896*omega*sigma*h[3]*q**(2)+672*omega*h[2]*q**(4)-2016*omega*sigma*h[2]*q**(4)+448*omega*sigma*h[3]*q**(4)+6048*omega*h[2]*sigma**(2)-6048*kappa*omega*h[2]*sigma**(2)-672*omega*h[3]*sigma**(2)+1344*kappa*omega*h[3]*sigma**(2)+56*omega*h[4]*sigma**(2)-6624*omega*h[2]*q**(2)*sigma**(2)+6048*kappa*omega*h[2]*q**(2)*sigma**(2)+1344*omega*h[3]*q**(2)*sigma**(2)-1344*kappa*omega*h[3]*q**(2)*sigma**(2)-112*omega*h[4]*q**(2)*sigma**(2)+576*omega*h[2]*q**(4)*sigma**(2)-672*omega*h[3]*q**(4)*sigma**(2)+56*omega*h[4]*q**(4)*sigma**(2)-3360*omega*h[2]*sigma**(3)+3360*kappa*omega*h[2]*sigma**(3)+1344*omega*h[3]*sigma**(3)-1344*kappa*omega*h[3]*sigma**(3)-56*omega*h[4]*sigma**(3)+112*kappa*omega*h[4]*sigma**(3)+3360*omega*h[2]*q**(2)*sigma**(3)-960*kappa*omega*h[2]*q**(2)*sigma**(3)-1472*omega*h[3]*q**(2)*sigma**(3)+1344*kappa*omega*h[3]*q**(2)*sigma**(3)+112*omega*h[4]*q**(2)*sigma**(3)-112*kappa*omega*h[4]*q**(2)*sigma**(3)+128*omega*h[3]*q**(4)*sigma**(3)-56*omega*h[4]*q**(4)*sigma**(3)-560*omega*h[3]*sigma**(4)+560*kappa*omega*h[3]*sigma**(4)+84*omega*h[4]*sigma**(4)-84*kappa*omega*h[4]*sigma**(4)+560*omega*h[3]*q**(2)*sigma**(4)-160*kappa*omega*h[3]*q**(2)*sigma**(4)-92*omega*h[4]*q**(2)*sigma**(4)+84*kappa*omega*h[4]*q**(2)*sigma**(4)+8*omega*h[4]*q**(4)*sigma**(4)-28*omega*h[4]*sigma**(5)+28*kappa*omega*h[4]*sigma**(5)+28*omega*h[4]*q**(2)*sigma**(5)-8*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-4872j)+h[2]*q**(2)*sigma**(2)*(-4872j)+h[2]*sigma**(3)*(-4200j)+kappa*sigma*h[2]*q**(2)*(-2352j)+kappa*h[3]*sigma**(3)*(-2016j)+h[3]*q**(2)*sigma**(3)*(-2016j)+h[3]*sigma**(4)*(-1400j)+kappa*h[3]*q**(2)*sigma**(2)*(-1344j)+kappa*h[2]*q**(2)*sigma**(3)*(-1200j)+kappa*h[3]*q**(2)*sigma**(4)*(-400j)+kappa*h[4]*sigma**(4)*(-301j)+h[4]*q**(2)*sigma**(4)*(-301j)+kappa*h[4]*q**(2)*sigma**(3)*(-252j)+h[4]*sigma**(5)*(-175j)+kappa*h[4]*q**(2)*sigma**(5)*(-50j)+kappa*h[5]*q**(2)*sigma**(4)*(-14j)+kappa*h[5]*sigma**(5)*(-14j)+h[5]*q**(2)*sigma**(5)*(-14j)+h[5]*sigma**(6)*(-7j)+kappa*h[5]*q**(2)*sigma**(6)*(-2j)+24*h[0]*(8*omega*q**(4)+7*(-3+5*sigma)*(-1+kappa)*(4*omega+(1j))+q**(2)*(-92*omega+(-21j)+35*sigma*(4*omega+(1j))-1*kappa*(-21+10*sigma)*(4*omega+(1j))))+kappa*h[5]*sigma**(6)*(7j)+h[5]*q**(2)*sigma**(6)*(7j)-24*h[1]*(-1*(-1+q**(2))*(-56*omega*(-1+q**(2))+2*sigma*(8*omega*(-21+2*q**(2))+(-77j))+35*sigma**(2)*(8*omega+(5j)))+kappa*(2*q**(2)*(7*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))-7*sigma*(24*omega+(11j)))-7*(16*omega+5*sigma**(2)*(8*omega+(5j))+(6j)-2*sigma*(24*omega+(11j)))))+kappa*h[5]*sigma**(4)*(14j)+h[5]*sigma**(5)*(14j)+kappa*h[5]*q**(2)*sigma**(5)*(14j)+kappa*h[4]*sigma**(5)*(175j)+h[4]*q**(2)*sigma**(5)*(175j)+kappa*h[4]*sigma**(3)*(252j)+h[4]*sigma**(4)*(301j)+kappa*h[4]*q**(2)*sigma**(4)*(301j)+kappa*h[3]*sigma**(2)*(1344j)+kappa*h[3]*sigma**(4)*(1400j)+h[3]*q**(2)*sigma**(4)*(1400j)+h[3]*sigma**(3)*(2016j)+kappa*h[3]*q**(2)*sigma**(3)*(2016j)+kappa*sigma*h[2]*(2352j)+kappa*h[2]*sigma**(3)*(4200j)+h[2]*q**(2)*sigma**(3)*(4200j)+h[2]*sigma**(2)*(4872j)+kappa*h[2]*q**(2)*sigma**(2)*(4872j))
	terms[4] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(288*omega*h[2]-864*omega*sigma*h[2]+1728*kappa*omega*sigma*h[2]+192*omega*sigma*h[3]-576*omega*h[2]*q**(2)+1728*omega*sigma*h[2]*q**(2)-1728*kappa*omega*sigma*h[2]*q**(2)-384*omega*sigma*h[3]*q**(2)+288*omega*h[2]*q**(4)-864*omega*sigma*h[2]*q**(4)+192*omega*sigma*h[3]*q**(4)+2592*omega*h[2]*sigma**(2)-2592*kappa*omega*h[2]*sigma**(2)-288*omega*h[3]*sigma**(2)+576*kappa*omega*h[3]*sigma**(2)+24*omega*h[4]*sigma**(2)-2880*omega*h[2]*q**(2)*sigma**(2)+2592*kappa*omega*h[2]*q**(2)*sigma**(2)+576*omega*h[3]*q**(2)*sigma**(2)-576*kappa*omega*h[3]*q**(2)*sigma**(2)-48*omega*h[4]*q**(2)*sigma**(2)+288*omega*h[2]*q**(4)*sigma**(2)-288*omega*h[3]*q**(4)*sigma**(2)+24*omega*h[4]*q**(4)*sigma**(2)-1440*omega*h[2]*sigma**(3)+1440*kappa*omega*h[2]*sigma**(3)+576*omega*h[3]*sigma**(3)-576*kappa*omega*h[3]*sigma**(3)-24*omega*h[4]*sigma**(3)+48*kappa*omega*h[4]*sigma**(3)+1440*omega*h[2]*q**(2)*sigma**(3)-480*kappa*omega*h[2]*q**(2)*sigma**(3)-640*omega*h[3]*q**(2)*sigma**(3)+576*kappa*omega*h[3]*q**(2)*sigma**(3)+48*omega*h[4]*q**(2)*sigma**(3)-48*kappa*omega*h[4]*q**(2)*sigma**(3)+64*omega*h[3]*q**(4)*sigma**(3)-24*omega*h[4]*q**(4)*sigma**(3)-240*omega*h[3]*sigma**(4)+240*kappa*omega*h[3]*sigma**(4)+36*omega*h[4]*sigma**(4)-36*kappa*omega*h[4]*sigma**(4)+240*omega*h[3]*q**(2)*sigma**(4)-80*kappa*omega*h[3]*q**(2)*sigma**(4)-40*omega*h[4]*q**(2)*sigma**(4)+36*kappa*omega*h[4]*q**(2)*sigma**(4)+4*omega*h[4]*q**(4)*sigma**(4)-12*omega*h[4]*sigma**(5)+12*kappa*omega*h[4]*sigma**(5)+12*omega*h[4]*q**(2)*sigma**(5)-4*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-2088j)+h[2]*q**(2)*sigma**(2)*(-2088j)+h[2]*sigma**(3)*(-1800j)+kappa*sigma*h[2]*q**(2)*(-1008j)+kappa*h[3]*sigma**(3)*(-864j)+h[3]*q**(2)*sigma**(3)*(-864j)+kappa*h[2]*q**(2)*sigma**(3)*(-600j)+h[3]*sigma**(4)*(-600j)+kappa*h[3]*q**(2)*sigma**(2)*(-576j)+kappa*h[3]*q**(2)*sigma**(4)*(-200j)+kappa*h[4]*sigma**(4)*(-129j)+h[4]*q**(2)*sigma**(4)*(-129j)+kappa*h[4]*q**(2)*sigma**(3)*(-108j)+h[4]*sigma**(5)*(-75j)+kappa*h[4]*q**(2)*sigma**(5)*(-25j)+kappa*h[5]*q**(2)*sigma**(4)*(-6j)+kappa*h[5]*sigma**(5)*(-6j)+h[5]*q**(2)*sigma**(5)*(-6j)+h[5]*sigma**(6)*(-3j)+kappa*h[5]*q**(2)*sigma**(6)*(-1j)+kappa*h[5]*sigma**(6)*(3j)+h[5]*q**(2)*sigma**(6)*(3j)+kappa*h[5]*sigma**(4)*(6j)+h[5]*sigma**(5)*(6j)+kappa*h[5]*q**(2)*sigma**(5)*(6j)-24*h[1]*(-1*(-1+q**(2))*(-24*omega*(-1+q**(2))+2*sigma*(8*omega*(-9+q**(2))+(-33j))+15*sigma**(2)*(8*omega+(5j)))+kappa*(q**(2)*(6*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))-6*sigma*(24*omega+(11j)))-3*(16*omega+5*sigma**(2)*(8*omega+(5j))+(6j)-2*sigma*(24*omega+(11j)))))+24*h[0]*(4*omega*q**(4)+3*(-3+5*sigma)*(-1+kappa)*(4*omega+(1j))+q**(2)*(-40*omega+60*omega*sigma+(-9j)-1*kappa*(-9+5*sigma)*(4*omega+(1j))+sigma*(15j)))+kappa*h[4]*sigma**(5)*(75j)+h[4]*q**(2)*sigma**(5)*(75j)+kappa*h[4]*sigma**(3)*(108j)+h[4]*sigma**(4)*(129j)+kappa*h[4]*q**(2)*sigma**(4)*(129j)+kappa*h[3]*sigma**(2)*(576j)+kappa*h[3]*sigma**(4)*(600j)+h[3]*q**(2)*sigma**(4)*(600j)+h[3]*sigma**(3)*(864j)+kappa*h[3]*q**(2)*sigma**(3)*(864j)+kappa*sigma*h[2]*(1008j)+kappa*h[2]*sigma**(3)*(1800j)+h[2]*q**(2)*sigma**(3)*(1800j)+h[2]*sigma**(2)*(2088j)+kappa*h[2]*q**(2)*sigma**(2)*(2088j))
	terms[5] = np.sqrt(np.pi*(2/5))*q**(4)*(-2j)/(21)*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[6] = np.sqrt(2*np.pi)*q**(4)*(2j)/(105)*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[7] = omega*np.sqrt(np.pi*(1/165))*q**(5)*(-4j)/(21)*(24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(1j)+8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))+sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))

	return terms

def metric_ORG_14_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(-1*h[1]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))-1*h[0]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[1]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+h[0]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*sigma**(2)*(-2/7)*(h[1]*sigma**(2)*(-7*kappa*(2+sigma*(-2+sigma))-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7+sigma*(-7+sigma)))+h[0]*(sigma**(3)*(1+omega*(-4j))*(7-7*kappa+q**(2)*(-7+2*kappa))+14*sigma*(-1+q**(2))*(kappa+omega*(-4j)*(-1+2*kappa+q**(2)))+2*sigma**(2)*(-1+q**(2))*(7-7*kappa+omega*(2j)*(21*(-1+kappa)+2*q**(2)))+omega*(-1+q**(2))**(2)*(56j)))
	terms[3] = omega*np.sqrt(np.pi*(2/21))*q**(3)*sigma**(2)*(-4/15)*(h[1]*sigma**(2)*(-3*kappa*(2+sigma*(-2+sigma))-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6+sigma*(-6+sigma)))+h[0]*(kappa*sigma*(1+omega*(-4j))*(-3*(4+sigma*(-3+sigma))+q**(2)*(12+sigma*(-9+sigma)))+(-1+q**(2))*(-3*sigma**(2)*(-3+sigma)+omega*(4j)*(-6+3*sigma*(-2+sigma)*(-1+sigma)+q**(2)*(6+sigma*(-6+sigma))))))
	terms[4] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(4)*(2/21)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))
	terms[5] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(4)*(4/21)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))
	terms[6] = np.sqrt(np.pi*(2/5))*q**(2)*sigma**(2)*(2/21)*(h[1]*sigma**(2)*(-7*kappa*(2+sigma*(-2+sigma))-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7+sigma*(-7+sigma)))+h[0]*(kappa*sigma*(1+omega*(-4j))*(-7*(4+sigma*(-3+sigma))+q**(2)*(28+sigma*(-21+2*sigma)))+(-1+q**(2))*(-7*sigma**(2)*(-3+sigma)+omega*(4j)*(7*(-2+sigma*(-2+sigma)*(-1+sigma))+2*q**(2)*(7+sigma*(-7+sigma))))))
	terms[7] = np.sqrt(2*np.pi)*q**(4)*sigma**(4)*(-2/105)*(kappa*h[1]*sigma**(2)+h[0]*(kappa*sigma+omega*(4j)*(-1-1*kappa*sigma+q**(2))))

	return terms

def metric_ORG_14_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(-2*sigma*h[1]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))-1*h[2]*sigma**(2)*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))-4*h[1]*sigma**(2)*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3))))-1*h[0]*(5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))-1*h[1]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(2*sigma*h[1]*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+h[2]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+4*h[1]*sigma**(2)*(70*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(7*(10-30*sigma+21*sigma**(2)-4*sigma**(3))+kappa*(-70+98*sigma-42*sigma**(2)+6*sigma**(3)))-7*q**(2)*(20-60*sigma+51*sigma**(2)-14*sigma**(3)+kappa*(-20+44*sigma-36*sigma**(2)+9*sigma**(3))))+h[0]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[1]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(-2j)/(7)*(h[2]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))+h[0]*((-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+8*sigma**(2)*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(3)*(4*omega+(1j)))+kappa*sigma*(-2*q**(2)*(5*sigma**(2)*(4*omega+(1j))-28*sigma*(6*omega+(1j))+21*(8*omega+(1j)))+7*(48*omega+5*sigma**(2)*(4*omega+(1j))-8*sigma*(6*omega+(1j))+(6j))))+sigma*h[1]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-84j))+7*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(7*(-12*sigma*(omega+(1j))+2*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))-2*q**(2)*(-42*sigma*(omega+(1j))+7*(8*omega+(5j))+sigma**(2)*(4*omega+(7j))))))
	terms[3] = omega*sigma*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(h[2]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))+h[0]*(-1*kappa*sigma*(4*omega+(1j))*(-36+36*sigma-15*sigma**(2)+q**(2)*(36-36*sigma+5*sigma**(2)))+(-1+q**(2))*(48*omega*(-1+q**(2))-72*omega*sigma*(-1+q**(2))+4*sigma**(2)*(4*omega*(-9+q**(2))+(-9j))+15*sigma**(3)*(4*omega+(1j))))+sigma*h[1]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-39j))+3*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(3*(4*(4*omega+(3j))+sigma**(2)*(4*omega+(7j))-1*sigma*(12*omega+(13j)))-1*q**(2)*(48*omega+sigma**(2)*(4*omega+(7j))-3*sigma*(12*omega+(13j))+(36j)))))
	terms[4] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(3)*(2/21)*(kappa*h[2]*sigma**(3)+sigma*h[1]*(kappa*sigma*(7+omega*(-4j))+omega*(4j)*(-1+q**(2)))+h[0]*(5*kappa*(sigma+omega*sigma*(-4j))+omega*(16j)*(-1+q**(2))))
	terms[5] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(3)*(4/21)*(kappa*h[2]*sigma**(3)+sigma*h[1]*(kappa*sigma*(7+omega*(-4j))+omega*(4j)*(-1+q**(2)))+h[0]*(5*kappa*(sigma+omega*sigma*(-4j))+omega*(16j)*(-1+q**(2))))
	terms[6] = sigma*np.sqrt(np.pi*(2/5))*q**(2)*(2/21)*(h[2]*sigma**(3)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))+h[0]*(1j)*(-1*kappa*sigma*(4*omega+(1j))*(-84+84*sigma-35*sigma**(2)+2*q**(2)*(42-42*sigma+5*sigma**(2)))+(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(3)*(4*omega+(1j))))+sigma*h[1]*(1j)*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-91j))+7*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(-1*q**(2)*(28*(4*omega+(3j))+2*sigma**(2)*(4*omega+(7j))-7*sigma*(12*omega+(13j)))+7*(4*(4*omega+(3j))+sigma**(2)*(4*omega+(7j))-1*sigma*(12*omega+(13j))))))
	terms[7] = np.sqrt(2*np.pi)*q**(4)*sigma**(3)*(-2j)/(105)*(kappa*h[2]*sigma**(3)*(-1j)-1*h[0]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))-1*sigma*h[1]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(7j))))

	return terms

def metric_ORG_14_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(-2*h[1]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))-4*sigma*(h[2]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))-1*sigma**(2)*(h[3]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+8*h[2]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))+4*h[0]*(-5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+2*h[1]*(-5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))-120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))-1*h[2]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[3]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+4*sigma*h[2]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+2*h[1]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+4*h[0]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+2*h[1]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[2]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-2j)/(7)*(-4*h[0]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+6*sigma**(2)*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(3)*(4*omega+(1j)))+kappa*sigma*(q**(2)*(10*sigma**(2)*(4*omega+(1j))-42*sigma*(6*omega+(1j))+21*(8*omega+(1j)))-7*(24*omega+5*sigma**(2)*(4*omega+(1j))-6*sigma*(6*omega+(1j))+(3j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-4*h[1]*(-2*(-1+q**(2))*(28*omega*(-1+q**(2))-42*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-49j))+35*sigma**(3)*(omega+(1j)))+kappa*sigma*(q**(2)*(20*sigma**(2)*(omega+(1j))+21*(8*omega+(3j))-14*sigma*(12*omega+(7j)))-7*(24*omega+10*sigma**(2)*(omega+(1j))-2*sigma*(12*omega+(7j))+(9j))))+sigma*h[2]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+2*sigma**(2)*(-42*omega+4*omega*q**(2)+(-77j))+7*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(-2*q**(2)*(7*(8*omega+(9j))-7*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))+7*(2*(8*omega+(9j))-2*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))))))
	terms[3] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(-4*h[0]*(kappa*sigma*(4*omega+(1j))*(q**(2)*(18-27*sigma+5*sigma**(2))-3*(6-9*sigma+5*sigma**(2)))-3*(-1+q**(2))*(4*omega*(-1+q**(2))-12*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-9j))+5*sigma**(3)*(4*omega+(1j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))-8*h[1]*(-1*(-1+q**(2))*(12*omega*(-1+q**(2))-18*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-9+q**(2))+(-6j))+15*sigma**(3)*(omega+(1j)))+kappa*sigma*(-3*(12*omega+5*sigma**(2)*(omega+(1j))-4*sigma*(3*omega+(2j))+(6j))+q**(2)*(36*omega+5*sigma**(2)*(omega+(1j))-12*sigma*(3*omega+(2j))+(18j))))+sigma*h[2]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-69j))+3*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(48*omega+3*sigma**(2)*(4*omega+(13j))-3*sigma*(12*omega+(23j))+(60j)-1*q**(2)*(48*omega+sigma**(2)*(4*omega+(13j))-3*sigma*(12*omega+(23j))+(60j))))))
	terms[4] = np.sqrt(np.pi*(2/5))*q**(4)*sigma**(2)*(2j)/(21)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[5] = omega*np.sqrt(np.pi*(1/165))*q**(5)*sigma**(2)*(4j)/(21)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[6] = np.sqrt(np.pi*(2/5))*q**(2)*(2j)/(21)*(-4*h[0]*(kappa*sigma*(4*omega+(1j))*(-7*(6-9*sigma+5*sigma**(2))+q**(2)*(42-63*sigma+10*sigma**(2)))-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+3*sigma**(2)*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(3)*(4*omega+(1j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-8*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-42*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-56j))+35*sigma**(3)*(omega+(1j)))+kappa*sigma*(-7*(12*omega+5*sigma**(2)*(omega+(1j))-4*sigma*(3*omega+(2j))+(6j))+2*q**(2)*(42*omega+5*sigma**(2)*(omega+(1j))-14*sigma*(3*omega+(2j))+(21j))))+sigma*h[2]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-161j))+7*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(7*(4*(4*omega+(5j))+sigma**(2)*(4*omega+(13j))-1*sigma*(12*omega+(23j)))-1*q**(2)*(28*(4*omega+(5j))-7*sigma*(12*omega+(23j))+sigma**(2)*(8*omega+(26j)))))))
	terms[7] = np.sqrt(2*np.pi)*q**(4)*sigma**(2)*(-2j)/(105)*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))

	return terms

def metric_ORG_14_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(-6*(h[2]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))-6*sigma*(h[3]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+8*h[2]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))-1*sigma**(2)*(24*h[1]*(30*(-1+kappa)*(-1+sigma)-5*q**(2)*(10-8*sigma+kappa*(-7+5*sigma))+q**(4)*(20-5*kappa-10*sigma+2*kappa*sigma))+h[4]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+12*h[2]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+12*h[3]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))+h[0]*(-60*sigma**(2)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-240*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-48*sigma*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+12*h[1]*(-5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+3*h[2]*(-5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))-120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))-1*h[3]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(h[4]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+24*h[1]*(70*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(-35*(-1+6*sigma-7*sigma**(2)+2*sigma**(3))+kappa*(-35+98*sigma-70*sigma**(2)+15*sigma**(3)))-7*q**(2)*(10*(2-12*sigma+17*sigma**(2)-7*sigma**(3))+kappa*(-20+88*sigma-120*sigma**(2)+45*sigma**(3))))+6*sigma*h[3]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+6*h[2]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+h[0]*(60*sigma**(2)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*sigma*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+336*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+12*h[1]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+3*h[2]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[3]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-2j)/(7)*(-12*h[0]*(-1*(-1+q**(2))*(-28*omega*(-1+q**(2))+4*sigma*(-42*omega+4*omega*q**(2)+(-7j))+35*sigma**(2)*(4*omega+(1j)))+kappa*(-7*(8*omega+(1j)+5*sigma**(2)*(4*omega+(1j))-4*sigma*(6*omega+(1j)))+q**(2)*(56*omega+10*sigma**(2)*(4*omega+(1j))-28*sigma*(6*omega+(1j))+(7j))))-12*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-63*omega+6*omega*q**(2)+(-28j))+35*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-7*(24*omega+5*sigma**(2)*(4*omega+(3j))-4*sigma*(9*omega+(4j))+(7j))+q**(2)*(10*sigma**(2)*(4*omega+(3j))-28*sigma*(9*omega+(4j))+7*(24*omega+(7j)))))+sigma*(h[4]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+16*sigma**(2)*(-21*omega+2*omega*q**(2)+(-21j))+35*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(-7*(-48*sigma*(omega+(1j))+6*(8*omega+(5j))+5*sigma**(2)*(4*omega+(7j)))+q**(2)*(-336*sigma*(omega+(1j))+42*(8*omega+(5j))+10*sigma**(2)*(4*omega+(7j)))))+sigma*h[3]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-21*omega+2*omega*q**(2)+(-56j))+7*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(7*(16*omega-4*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(26j))-2*q**(2)*(56*omega-14*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(91j))))))
	terms[3] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(-12*h[0]*(kappa*(4*omega+(1j))*(q**(2)*(6-18*sigma+5*sigma**(2))-3*(2-6*sigma+5*sigma**(2)))-1*(-1+q**(2))*(-12*omega*(-1+q**(2))+2*sigma*(4*omega*(-9+q**(2))+(-9j))+15*sigma**(2)*(4*omega+(1j))))-12*h[1]*(-3*(-1+q**(2))*(4*omega*(-1+q**(2))-12*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-19j))+5*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-3*(24*omega+5*sigma**(2)*(4*omega+(3j))+(10j)-1*sigma*(36*omega+(19j)))+q**(2)*(72*omega+5*sigma**(2)*(4*omega+(3j))-3*sigma*(36*omega+(19j))+(30j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-3*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(6-6*sigma+sigma**(2))-3*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(48*omega*(-1+q**(2))-72*omega*sigma*(-1+q**(2))+4*sigma**(2)*(4*omega*(-9+q**(2))+(-39j))+15*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(q**(2)*(36*(4*omega+(3j))+5*sigma**(2)*(4*omega+(7j))-12*sigma*(12*omega+(13j)))-3*(48*omega+5*sigma**(2)*(4*omega+(7j))-4*sigma*(12*omega+(13j))+(36j))))+sigma*h[3]*((-1+q**(2))*(24*omega*(-1+q**(2))-24*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-9+q**(2))+(-99j))+3*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(48*omega-9*sigma*(4*omega+(11j))+3*sigma**(2)*(4*omega+(19j))+(84j)-1*q**(2)*(48*omega-9*sigma*(4*omega+(11j))+sigma**(2)*(4*omega+(19j))+(84j))))))
	terms[4] = sigma*np.sqrt(np.pi*(2/5))*q**(4)*(2j)/(21)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[5] = omega*sigma*np.sqrt(np.pi*(1/165))*q**(5)*(4j)/(21)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[6] = np.sqrt(np.pi*(2/5))*q**(2)*(2j)/(21)*(-12*h[0]*(kappa*(4*omega+(1j))*(2*q**(2)*(7-21*sigma+5*sigma**(2))-7*(2-6*sigma+5*sigma**(2)))-1*(-1+q**(2))*(-28*omega*(-1+q**(2))+2*sigma*(omega*(-84+8*q**(2))+(-21j))+35*sigma**(2)*(4*omega+(1j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-7*sigma*(-2+sigma)*(-1+q**(2))+2*kappa*q**(2)*(7-7*sigma+sigma**(2))-7*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(112*omega*(-1+q**(2))-168*omega*sigma*(-1+q**(2))+4*sigma**(2)*(-84*omega+8*omega*q**(2)+(-91j))+35*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(q**(2)*(84*(4*omega+(3j))+10*sigma**(2)*(4*omega+(7j))-28*sigma*(12*omega+(13j)))-7*(48*omega+5*sigma**(2)*(4*omega+(7j))-4*sigma*(12*omega+(13j))+(36j))))+sigma*h[3]*((-1+q**(2))*(56*omega*(-1+q**(2))-56*omega*sigma*(-1+q**(2))+sigma**(2)*(-84*omega+8*omega*q**(2)+(-231j))+7*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(7*(4*(4*omega+(7j))-3*sigma*(4*omega+(11j))+sigma**(2)*(4*omega+(19j)))-1*q**(2)*(28*(4*omega+(7j))-21*sigma*(4*omega+(11j))+sigma**(2)*(8*omega+(38j))))))-12*h[1]*(-1*(-1+q**(2))*(28*omega*(-1+q**(2))-84*omega*sigma*(-1+q**(2))+sigma**(2)*(-252*omega+24*omega*q**(2)+(-133j))+35*sigma**(3)*(4*omega+(3j)))+kappa*sigma*(-7*(24*omega+5*sigma**(2)*(4*omega+(3j))+(10j)-1*sigma*(36*omega+(19j)))+q**(2)*(168*omega+10*sigma**(2)*(4*omega+(3j))-7*sigma*(36*omega+(19j))+(70j)))))
	terms[7] = sigma*np.sqrt(2*np.pi)*q**(4)*(-2j)/(105)*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))

	return terms

def metric_ORG_14_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 8
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = 2**((-1/2))*(1/15)*(-12*(h[3]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+4*h[1]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+8*h[2]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))-8*sigma*(24*h[1]*(30*(-1+kappa)*(-1+sigma)-5*q**(2)*(10-8*sigma+kappa*(-7+5*sigma))+q**(4)*(20-5*kappa-10*sigma+2*kappa*sigma))+h[4]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+12*h[2]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+12*h[3]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))-1*sigma**(2)*(24*h[1]*(30*(-1+kappa)-5*q**(2)*(-8+5*kappa)+2*q**(4)*(-5+kappa))+96*h[2]*(30*(-1+kappa)*(-1+sigma)-5*q**(2)*(10-8*sigma+kappa*(-7+5*sigma))+q**(4)*(20-5*kappa-10*sigma+2*kappa*sigma))+h[5]*(-10*sigma*(-2+sigma)*(-1+q**(2))*(-3*(2+sigma*(-2+sigma))+q**(2)*(6+sigma*(-6+sigma)))+kappa*(30*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))-5*q**(2)*(24+sigma*(-48+sigma*(52+sigma*(-28+5*sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))))+24*h[3]*(90*(-1+sigma)**(2)*(-1+kappa)+2*q**(4)*(-15*(3-4*sigma+sigma**(2))+kappa*(20-15*sigma+3*sigma**(2)))-5*q**(2)*(-12*(3-5*sigma+2*sigma**(2))+kappa*(26-42*sigma+15*sigma**(2))))+16*h[4]*(30*(-1+sigma)**(3)*(-1+kappa)+q**(4)*(-10*(-3+9*sigma-6*sigma**(2)+sigma**(3))+kappa*(-30+40*sigma-15*sigma**(2)+2*sigma**(3)))-5*q**(2)*(12-36*sigma+30*sigma**(2)-8*sigma**(3)+kappa*(-12+26*sigma-21*sigma**(2)+5*sigma**(3)))))+h[0]*(-120*sigma*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-48*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+4*h[1]*(-60*sigma**(2)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-240*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-48*sigma*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+24*h[2]*(-5*sigma**(3)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*sigma*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-10*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-6*sigma**(2)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))+4*h[3]*(-5*sigma**(4)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))-60*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))-120*sigma**(2)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))-40*sigma*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))-8*sigma**(3)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j)))-1*h[4]*(sigma**(5)*(1+omega*(-4j))*(30*(-1+kappa)+5*q**(2)*(8-5*kappa)+2*q**(4)*(-5+kappa))+60*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+40*sigma**(3)*(-1+q**(2))*(q**(2)*(-9+4*kappa+omega*(-16j)*(-2+kappa))+omega*q**(4)*(-2j)+(3j)*(-1+kappa)*(10*omega+(3j)))+20*sigma**(2)*(-1+q**(2))**(2)*(-15*(-1+kappa)+omega*(4j)*(15*(-1+kappa)+4*q**(2)))+2*sigma**(4)*(-1+q**(2))*(30*(-1+kappa)*(3+omega*(-10j))+15*q**(2)*(4-1*kappa+omega*(2j)*(-7+2*kappa))+omega*q**(4)*(4j))+omega*(-1+q**(2))**(3)*(240j)))
	terms[1] = omega*q*np.sqrt(np.pi*(1/3))*(2/35)*(24*h[1]*(70*(-1+kappa)*(6-20*sigma+15*sigma**(2))+2*q**(4)*(-70*(3-7*sigma+3*sigma**(2))+kappa*(98-140*sigma+45*sigma**(2)))-7*q**(2)*(-10*(12-34*sigma+21*sigma**(2))+kappa*(88-240*sigma+135*sigma**(2))))+h[5]*sigma**(2)*(-14*sigma*(-2+sigma)*(-1+q**(2))*(-5*(2+sigma*(-2+sigma))+2*q**(2)*(5+sigma*(-5+sigma)))+kappa*(70*(2+sigma*(-2+sigma)*(2+sigma*(-2+sigma)))+2*q**(4)*(70+sigma*(-140+sigma*(98+sigma*(-28+3*sigma))))-7*q**(2)*(40+sigma*(-80+sigma*(88-48*sigma+9*sigma**(2))))))+96*h[2]*(70*(-1+kappa)*(-1+6*sigma-10*sigma**(2)+5*sigma**(3))+2*q**(4)*(-35*(-1+6*sigma-7*sigma**(2)+2*sigma**(3))+kappa*(-35+98*sigma-70*sigma**(2)+15*sigma**(3)))-7*q**(2)*(10*(2-12*sigma+17*sigma**(2)-7*sigma**(3))+kappa*(-20+88*sigma-120*sigma**(2)+45*sigma**(3))))+8*sigma*h[4]*(-14*sigma*(-1+q**(2))*(30-60*sigma+50*sigma**(2)-15*sigma**(3)+q**(2)*(-30+60*sigma-35*sigma**(2)+6*sigma**(3)))+kappa*(70*(2-6*sigma+12*sigma**(2)-10*sigma**(3)+3*sigma**(4))+2*q**(4)*(70-210*sigma+196*sigma**(2)-70*sigma**(3)+9*sigma**(4))-7*q**(2)*(40-120*sigma+176*sigma**(2)-120*sigma**(3)+27*sigma**(4))))+12*h[3]*(-70*sigma*(-1+q**(2))*(12-36*sigma+40*sigma**(2)-15*sigma**(3)+2*q**(2)*(-6+18*sigma-14*sigma**(2)+3*sigma**(3)))+kappa*(70*(2-12*sigma+36*sigma**(2)-40*sigma**(3)+15*sigma**(4))+2*q**(4)*(70-420*sigma+588*sigma**(2)-280*sigma**(3)+45*sigma**(4))-7*q**(2)*(40-240*sigma+528*sigma**(2)-480*sigma**(3)+135*sigma**(4))))+h[0]*(120*sigma*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j)))+4*h[1]*(60*sigma**(2)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+48*sigma*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+336*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+24*h[2]*(5*sigma**(3)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+14*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+6*sigma**(2)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+84*sigma*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+4*h[3]*(5*sigma**(4)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+56*sigma*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+8*sigma**(3)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+168*sigma**(2)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j))))+h[4]*(sigma**(5)*(1+omega*(-4j))*(70*(-1+kappa)+q**(2)*(98-63*kappa)+q**(4)*(-28+6*kappa))+140*sigma*(-1+q**(2))**(2)*(3*kappa+omega*(-4j)*(-2+3*kappa+2*q**(2)))+28*sigma**(2)*(-1+q**(2))**(2)*(-25*(-1+kappa)+omega*(4j)*(25*(-1+kappa)+7*q**(2)))+2*sigma**(4)*(-1+q**(2))*(70*(-1+kappa)*(3+omega*(-10j))+7*q**(2)*(21-6*kappa+omega*(2j)*(-37+12*kappa))+omega*q**(4)*(12j))+56*sigma**(3)*(-1+q**(2))*(omega*q**(4)*(-4j)+(5j)*(-1+kappa)*(10*omega+(3j))+q**(2)*(-15+7*kappa*(1+omega*(-4j))+omega*(54j)))+omega*(-1+q**(2))**(3)*(560j)))
	terms[2] = np.sqrt(np.pi*(1/15))*q**(2)*(-2j)/(7)*(672*omega*h[2]-2016*omega*sigma*h[2]+4032*kappa*omega*sigma*h[2]+448*omega*sigma*h[3]-1344*omega*h[2]*q**(2)+4032*omega*sigma*h[2]*q**(2)-4032*kappa*omega*sigma*h[2]*q**(2)-896*omega*sigma*h[3]*q**(2)+672*omega*h[2]*q**(4)-2016*omega*sigma*h[2]*q**(4)+448*omega*sigma*h[3]*q**(4)+6048*omega*h[2]*sigma**(2)-6048*kappa*omega*h[2]*sigma**(2)-672*omega*h[3]*sigma**(2)+1344*kappa*omega*h[3]*sigma**(2)+56*omega*h[4]*sigma**(2)-6624*omega*h[2]*q**(2)*sigma**(2)+6048*kappa*omega*h[2]*q**(2)*sigma**(2)+1344*omega*h[3]*q**(2)*sigma**(2)-1344*kappa*omega*h[3]*q**(2)*sigma**(2)-112*omega*h[4]*q**(2)*sigma**(2)+576*omega*h[2]*q**(4)*sigma**(2)-672*omega*h[3]*q**(4)*sigma**(2)+56*omega*h[4]*q**(4)*sigma**(2)-3360*omega*h[2]*sigma**(3)+3360*kappa*omega*h[2]*sigma**(3)+1344*omega*h[3]*sigma**(3)-1344*kappa*omega*h[3]*sigma**(3)-56*omega*h[4]*sigma**(3)+112*kappa*omega*h[4]*sigma**(3)+3360*omega*h[2]*q**(2)*sigma**(3)-960*kappa*omega*h[2]*q**(2)*sigma**(3)-1472*omega*h[3]*q**(2)*sigma**(3)+1344*kappa*omega*h[3]*q**(2)*sigma**(3)+112*omega*h[4]*q**(2)*sigma**(3)-112*kappa*omega*h[4]*q**(2)*sigma**(3)+128*omega*h[3]*q**(4)*sigma**(3)-56*omega*h[4]*q**(4)*sigma**(3)-560*omega*h[3]*sigma**(4)+560*kappa*omega*h[3]*sigma**(4)+84*omega*h[4]*sigma**(4)-84*kappa*omega*h[4]*sigma**(4)+560*omega*h[3]*q**(2)*sigma**(4)-160*kappa*omega*h[3]*q**(2)*sigma**(4)-92*omega*h[4]*q**(2)*sigma**(4)+84*kappa*omega*h[4]*q**(2)*sigma**(4)+8*omega*h[4]*q**(4)*sigma**(4)-28*omega*h[4]*sigma**(5)+28*kappa*omega*h[4]*sigma**(5)+28*omega*h[4]*q**(2)*sigma**(5)-8*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-4368j)+h[2]*q**(2)*sigma**(2)*(-4368j)+h[2]*sigma**(3)*(-4200j)+kappa*h[3]*sigma**(3)*(-1904j)+h[3]*q**(2)*sigma**(3)*(-1904j)+kappa*sigma*h[2]*q**(2)*(-1848j)+h[3]*sigma**(4)*(-1400j)+kappa*h[2]*q**(2)*sigma**(3)*(-1200j)+kappa*h[3]*q**(2)*sigma**(2)*(-1176j)+kappa*h[3]*q**(2)*sigma**(4)*(-400j)+kappa*h[4]*sigma**(4)*(-294j)+h[4]*q**(2)*sigma**(4)*(-294j)+kappa*h[4]*q**(2)*sigma**(3)*(-238j)+h[4]*sigma**(5)*(-175j)+kappa*h[4]*q**(2)*sigma**(5)*(-50j)+kappa*h[5]*q**(2)*sigma**(4)*(-14j)+kappa*h[5]*sigma**(5)*(-14j)+h[5]*q**(2)*sigma**(5)*(-14j)+h[5]*sigma**(6)*(-7j)+kappa*h[5]*q**(2)*sigma**(6)*(-2j)+kappa*h[5]*sigma**(6)*(7j)+h[5]*q**(2)*sigma**(6)*(7j)+kappa*h[5]*sigma**(4)*(14j)+h[5]*sigma**(5)*(14j)+kappa*h[5]*q**(2)*sigma**(5)*(14j)-24*h[1]*(-1*(-1+q**(2))*(-56*omega*(-1+q**(2))+2*sigma*(8*omega*(-21+2*q**(2))+(-63j))+35*sigma**(2)*(8*omega+(5j)))+kappa*(-7*(16*omega-6*sigma*(8*omega+(3j))+(4j)+5*sigma**(2)*(8*omega+(5j)))+2*q**(2)*(56*omega-21*sigma*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))+(14j))))+24*h[0]*(8*omega*q**(4)+7*(-1+kappa)*(-12*omega+20*omega*sigma+(-2j)+sigma*(5j))+q**(2)*(-92*omega+140*omega*sigma+(-14j)-2*kappa*(-42*omega+20*omega*sigma+(-7j)+sigma*(5j))+sigma*(35j)))+kappa*h[4]*sigma**(5)*(175j)+h[4]*q**(2)*sigma**(5)*(175j)+kappa*h[4]*sigma**(3)*(238j)+h[4]*sigma**(4)*(294j)+kappa*h[4]*q**(2)*sigma**(4)*(294j)+kappa*h[3]*sigma**(2)*(1176j)+kappa*h[3]*sigma**(4)*(1400j)+h[3]*q**(2)*sigma**(4)*(1400j)+kappa*sigma*h[2]*(1848j)+h[3]*sigma**(3)*(1904j)+kappa*h[3]*q**(2)*sigma**(3)*(1904j)+kappa*h[2]*sigma**(3)*(4200j)+h[2]*q**(2)*sigma**(3)*(4200j)+h[2]*sigma**(2)*(4368j)+kappa*h[2]*q**(2)*sigma**(2)*(4368j))
	terms[3] = omega*np.sqrt(np.pi*(2/21))*q**(3)*(-4j)/(15)*(288*omega*h[2]-864*omega*sigma*h[2]+1728*kappa*omega*sigma*h[2]+192*omega*sigma*h[3]-576*omega*h[2]*q**(2)+1728*omega*sigma*h[2]*q**(2)-1728*kappa*omega*sigma*h[2]*q**(2)-384*omega*sigma*h[3]*q**(2)+288*omega*h[2]*q**(4)-864*omega*sigma*h[2]*q**(4)+192*omega*sigma*h[3]*q**(4)+2592*omega*h[2]*sigma**(2)-2592*kappa*omega*h[2]*sigma**(2)-288*omega*h[3]*sigma**(2)+576*kappa*omega*h[3]*sigma**(2)+24*omega*h[4]*sigma**(2)-2880*omega*h[2]*q**(2)*sigma**(2)+2592*kappa*omega*h[2]*q**(2)*sigma**(2)+576*omega*h[3]*q**(2)*sigma**(2)-576*kappa*omega*h[3]*q**(2)*sigma**(2)-48*omega*h[4]*q**(2)*sigma**(2)+288*omega*h[2]*q**(4)*sigma**(2)-288*omega*h[3]*q**(4)*sigma**(2)+24*omega*h[4]*q**(4)*sigma**(2)-1440*omega*h[2]*sigma**(3)+1440*kappa*omega*h[2]*sigma**(3)+576*omega*h[3]*sigma**(3)-576*kappa*omega*h[3]*sigma**(3)-24*omega*h[4]*sigma**(3)+48*kappa*omega*h[4]*sigma**(3)+1440*omega*h[2]*q**(2)*sigma**(3)-480*kappa*omega*h[2]*q**(2)*sigma**(3)-640*omega*h[3]*q**(2)*sigma**(3)+576*kappa*omega*h[3]*q**(2)*sigma**(3)+48*omega*h[4]*q**(2)*sigma**(3)-48*kappa*omega*h[4]*q**(2)*sigma**(3)+64*omega*h[3]*q**(4)*sigma**(3)-24*omega*h[4]*q**(4)*sigma**(3)-240*omega*h[3]*sigma**(4)+240*kappa*omega*h[3]*sigma**(4)+36*omega*h[4]*sigma**(4)-36*kappa*omega*h[4]*sigma**(4)+240*omega*h[3]*q**(2)*sigma**(4)-80*kappa*omega*h[3]*q**(2)*sigma**(4)-40*omega*h[4]*q**(2)*sigma**(4)+36*kappa*omega*h[4]*q**(2)*sigma**(4)+4*omega*h[4]*q**(4)*sigma**(4)-12*omega*h[4]*sigma**(5)+12*kappa*omega*h[4]*sigma**(5)+12*omega*h[4]*q**(2)*sigma**(5)-4*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-2088j)+h[2]*q**(2)*sigma**(2)*(-2088j)+h[2]*sigma**(3)*(-1800j)+kappa*sigma*h[2]*q**(2)*(-1008j)+kappa*h[3]*sigma**(3)*(-864j)+h[3]*q**(2)*sigma**(3)*(-864j)+kappa*h[2]*q**(2)*sigma**(3)*(-600j)+h[3]*sigma**(4)*(-600j)+kappa*h[3]*q**(2)*sigma**(2)*(-576j)+kappa*h[3]*q**(2)*sigma**(4)*(-200j)+kappa*h[4]*sigma**(4)*(-129j)+h[4]*q**(2)*sigma**(4)*(-129j)+kappa*h[4]*q**(2)*sigma**(3)*(-108j)+h[4]*sigma**(5)*(-75j)+kappa*h[4]*q**(2)*sigma**(5)*(-25j)+kappa*h[5]*q**(2)*sigma**(4)*(-6j)+kappa*h[5]*sigma**(5)*(-6j)+h[5]*q**(2)*sigma**(5)*(-6j)+h[5]*sigma**(6)*(-3j)+kappa*h[5]*q**(2)*sigma**(6)*(-1j)+kappa*h[5]*sigma**(6)*(3j)+h[5]*q**(2)*sigma**(6)*(3j)+kappa*h[5]*sigma**(4)*(6j)+h[5]*sigma**(5)*(6j)+kappa*h[5]*q**(2)*sigma**(5)*(6j)-24*h[1]*(-1*(-1+q**(2))*(-24*omega*(-1+q**(2))+2*sigma*(8*omega*(-9+q**(2))+(-33j))+15*sigma**(2)*(8*omega+(5j)))+kappa*(q**(2)*(6*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))-6*sigma*(24*omega+(11j)))-3*(16*omega+5*sigma**(2)*(8*omega+(5j))+(6j)-2*sigma*(24*omega+(11j)))))+24*h[0]*(4*omega*q**(4)+3*(-3+5*sigma)*(-1+kappa)*(4*omega+(1j))+q**(2)*(-40*omega+60*omega*sigma+(-9j)-1*kappa*(-9+5*sigma)*(4*omega+(1j))+sigma*(15j)))+kappa*h[4]*sigma**(5)*(75j)+h[4]*q**(2)*sigma**(5)*(75j)+kappa*h[4]*sigma**(3)*(108j)+h[4]*sigma**(4)*(129j)+kappa*h[4]*q**(2)*sigma**(4)*(129j)+kappa*h[3]*sigma**(2)*(576j)+kappa*h[3]*sigma**(4)*(600j)+h[3]*q**(2)*sigma**(4)*(600j)+h[3]*sigma**(3)*(864j)+kappa*h[3]*q**(2)*sigma**(3)*(864j)+kappa*sigma*h[2]*(1008j)+kappa*h[2]*sigma**(3)*(1800j)+h[2]*q**(2)*sigma**(3)*(1800j)+h[2]*sigma**(2)*(2088j)+kappa*h[2]*q**(2)*sigma**(2)*(2088j))
	terms[4] = np.sqrt(np.pi*(2/5))*q**(4)*(2j)/(21)*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[5] = omega*np.sqrt(np.pi*(1/165))*q**(5)*(-4j)/(21)*(24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(1j)+8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))+sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[6] = np.sqrt(np.pi*(2/5))*q**(2)*(2j)/(21)*(672*omega*h[2]-2016*omega*sigma*h[2]+4032*kappa*omega*sigma*h[2]+448*omega*sigma*h[3]-1344*omega*h[2]*q**(2)+4032*omega*sigma*h[2]*q**(2)-4032*kappa*omega*sigma*h[2]*q**(2)-896*omega*sigma*h[3]*q**(2)+672*omega*h[2]*q**(4)-2016*omega*sigma*h[2]*q**(4)+448*omega*sigma*h[3]*q**(4)+6048*omega*h[2]*sigma**(2)-6048*kappa*omega*h[2]*sigma**(2)-672*omega*h[3]*sigma**(2)+1344*kappa*omega*h[3]*sigma**(2)+56*omega*h[4]*sigma**(2)-6624*omega*h[2]*q**(2)*sigma**(2)+6048*kappa*omega*h[2]*q**(2)*sigma**(2)+1344*omega*h[3]*q**(2)*sigma**(2)-1344*kappa*omega*h[3]*q**(2)*sigma**(2)-112*omega*h[4]*q**(2)*sigma**(2)+576*omega*h[2]*q**(4)*sigma**(2)-672*omega*h[3]*q**(4)*sigma**(2)+56*omega*h[4]*q**(4)*sigma**(2)-3360*omega*h[2]*sigma**(3)+3360*kappa*omega*h[2]*sigma**(3)+1344*omega*h[3]*sigma**(3)-1344*kappa*omega*h[3]*sigma**(3)-56*omega*h[4]*sigma**(3)+112*kappa*omega*h[4]*sigma**(3)+3360*omega*h[2]*q**(2)*sigma**(3)-960*kappa*omega*h[2]*q**(2)*sigma**(3)-1472*omega*h[3]*q**(2)*sigma**(3)+1344*kappa*omega*h[3]*q**(2)*sigma**(3)+112*omega*h[4]*q**(2)*sigma**(3)-112*kappa*omega*h[4]*q**(2)*sigma**(3)+128*omega*h[3]*q**(4)*sigma**(3)-56*omega*h[4]*q**(4)*sigma**(3)-560*omega*h[3]*sigma**(4)+560*kappa*omega*h[3]*sigma**(4)+84*omega*h[4]*sigma**(4)-84*kappa*omega*h[4]*sigma**(4)+560*omega*h[3]*q**(2)*sigma**(4)-160*kappa*omega*h[3]*q**(2)*sigma**(4)-92*omega*h[4]*q**(2)*sigma**(4)+84*kappa*omega*h[4]*q**(2)*sigma**(4)+8*omega*h[4]*q**(4)*sigma**(4)-28*omega*h[4]*sigma**(5)+28*kappa*omega*h[4]*sigma**(5)+28*omega*h[4]*q**(2)*sigma**(5)-8*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-4872j)+h[2]*q**(2)*sigma**(2)*(-4872j)+h[2]*sigma**(3)*(-4200j)+kappa*sigma*h[2]*q**(2)*(-2352j)+kappa*h[3]*sigma**(3)*(-2016j)+h[3]*q**(2)*sigma**(3)*(-2016j)+h[3]*sigma**(4)*(-1400j)+kappa*h[3]*q**(2)*sigma**(2)*(-1344j)+kappa*h[2]*q**(2)*sigma**(3)*(-1200j)+kappa*h[3]*q**(2)*sigma**(4)*(-400j)+kappa*h[4]*sigma**(4)*(-301j)+h[4]*q**(2)*sigma**(4)*(-301j)+kappa*h[4]*q**(2)*sigma**(3)*(-252j)+h[4]*sigma**(5)*(-175j)+kappa*h[4]*q**(2)*sigma**(5)*(-50j)+kappa*h[5]*q**(2)*sigma**(4)*(-14j)+kappa*h[5]*sigma**(5)*(-14j)+h[5]*q**(2)*sigma**(5)*(-14j)+h[5]*sigma**(6)*(-7j)+kappa*h[5]*q**(2)*sigma**(6)*(-2j)+24*h[0]*(8*omega*q**(4)+7*(-3+5*sigma)*(-1+kappa)*(4*omega+(1j))+q**(2)*(-92*omega+(-21j)+35*sigma*(4*omega+(1j))-1*kappa*(-21+10*sigma)*(4*omega+(1j))))+kappa*h[5]*sigma**(6)*(7j)+h[5]*q**(2)*sigma**(6)*(7j)-24*h[1]*(-1*(-1+q**(2))*(-56*omega*(-1+q**(2))+2*sigma*(8*omega*(-21+2*q**(2))+(-77j))+35*sigma**(2)*(8*omega+(5j)))+kappa*(2*q**(2)*(7*(8*omega+(3j))+5*sigma**(2)*(8*omega+(5j))-7*sigma*(24*omega+(11j)))-7*(16*omega+5*sigma**(2)*(8*omega+(5j))+(6j)-2*sigma*(24*omega+(11j)))))+kappa*h[5]*sigma**(4)*(14j)+h[5]*sigma**(5)*(14j)+kappa*h[5]*q**(2)*sigma**(5)*(14j)+kappa*h[4]*sigma**(5)*(175j)+h[4]*q**(2)*sigma**(5)*(175j)+kappa*h[4]*sigma**(3)*(252j)+h[4]*sigma**(4)*(301j)+kappa*h[4]*q**(2)*sigma**(4)*(301j)+kappa*h[3]*sigma**(2)*(1344j)+kappa*h[3]*sigma**(4)*(1400j)+h[3]*q**(2)*sigma**(4)*(1400j)+h[3]*sigma**(3)*(2016j)+kappa*h[3]*q**(2)*sigma**(3)*(2016j)+kappa*sigma*h[2]*(2352j)+kappa*h[2]*sigma**(3)*(4200j)+h[2]*q**(2)*sigma**(3)*(4200j)+h[2]*sigma**(2)*(4872j)+kappa*h[2]*q**(2)*sigma**(2)*(4872j))
	terms[7] = np.sqrt(2*np.pi)*q**(4)*(-2j)/(105)*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))

	return terms

def metric_ORG_33_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[2]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))-2*h[0]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+24*kappa*omega+(-27j)+kappa*(6j)))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+h[1]*sigma**(2)*(-2j)*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*sigma**(2)*(4/5)*(-1+q**(2))*(h[1]*sigma**(2)*(-1j)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10+sigma*(-10+sigma)))+h[0]*(40*omega*(-1+q**(2))**(2)-1*sigma**(3)*(4*omega+(1j))*(5-5*kappa+q**(2)*(-5+kappa))-10*sigma*(-1+q**(2))*(4*omega*(-1+2*kappa+q**(2))+kappa*(1j))+2*sigma**(2)*(-1+q**(2))*(2*omega*(15*(-1+kappa)+q**(2))+(5j)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(-4/21)*(-1+q**(2))*(h[2]*sigma**(4)*(7*(-2+sigma*(-2+sigma)*(-1+kappa))+2*q**(2)*(7+sigma*(-7+sigma)))+h[1]*sigma**(2)*(kappa*omega*(-112j)*(-1+q**(2))+4*sigma**(3)*(1+omega*(-2j))*(-7+7*kappa+2*q**(2))+14*sigma*(-1+q**(2))*(5+omega*(8j)*(-2+kappa))+sigma**(2)*(q**(2)*(-63+omega*(-8j)*(-21+2*kappa))+21*(-3+omega*(8j))*(-1+kappa)))+h[0]*(224*omega**(2)*(-1+q**(2))**(2)-1*sigma**(3)*(4*omega+(1j))*(16*omega*(7-7*kappa+q**(2)*(-7+kappa))+(-35j)*(-1+kappa+q**(2)))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+3*kappa+q**(2))+kappa*(3j))-2*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))+2*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(4)*(4/5)*(-1+q**(2))*(kappa*h[1]*sigma**(2)*(1j)+h[0]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(1j))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(4)*(4/105)*(-1+q**(2))*(h[2]*sigma**(4)+4*h[1]*sigma**(2)*(sigma+omega*(-2j)*(kappa+sigma))-2*h[0]*(-1*sigma**(2)+8*omega**(2)*(1+2*kappa*sigma-1*q**(2)+sigma**(2))+omega*sigma*(2j)*(2*kappa+3*sigma)))

	return terms

def metric_ORG_33_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(240*h[2]*sigma**(3)+20*h[2]*q**(2)*sigma**(3)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+8*h[2]*q**(4)*sigma**(3)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))-150*h[2]*sigma**(4)*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+60*h[3]*sigma**(4)+5*h[3]*q**(2)*sigma**(4)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*h[3]*q**(4)*sigma**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))+4*h[2]*q**(4)*sigma**(4)*(-30+40*sigma-15*sigma**(2)+2*sigma**(3))+20*h[2]*q**(2)*sigma**(4)*(12-26*sigma+21*sigma**(2)-5*sigma**(3)+2*kappa*(-3+9*sigma-6*sigma**(2)+sigma**(3)))-60*h[2]*sigma**(5)*(-2+sigma)*(-1+kappa)*(-1+sigma)-30*h[2]*sigma**(5)*(-1+kappa)*(2+sigma*(-2+sigma))-30*h[3]*sigma**(5)*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+960*h[1]*omega**(2)*(-1+q**(2))**(3)+kappa*omega*sigma*h[1]*(-1+q**(2))**(2)*(-960j)+kappa*omega*h[2]*sigma**(2)*(-1+q**(2))**(2)*(-480j)+28*h[1]*sigma**(6)*(1+omega*(-2j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+4*h[2]*sigma**(7)*(1+omega*(-2j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-10*h[0]*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-2*h[1]*sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-960*omega*h[0]*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-960*omega*sigma*h[1]*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))+160*sigma*h[0]*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+80*h[1]*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+360*h[1]*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(4j)*(-3+2*kappa))+120*h[2]*sigma**(3)*(-1+q**(2))**(2)*(3+omega*(4j)*(-3+2*kappa))-12*h[0]*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-2*h[1]*sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-60*h[0]*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))-20*h[1]*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+h[1]*sigma**(3)*(-80j)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+h[2]*sigma**(4)*(-20j)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))-16*h[0]*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa))-4*h[1]*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa))+h[1]*sigma**(4)*(-100j)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+h[2]*sigma**(5)*(-20j)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+h[1]*sigma**(5)*(-12j)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))+h[2]*sigma**(6)*(-2j)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))
	terms[1] = q*sigma*np.sqrt(np.pi*(1/3))*(4/5)*(-1+q**(2))*(h[2]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))+h[0]*((-1+q**(2))*(80*omega*(-1+q**(2))-120*omega*sigma*(-1+q**(2))+8*sigma**(2)*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(3)*(4*omega+(1j)))-5*kappa*sigma*(-5*sigma**(2)*(4*omega+(1j))+8*sigma*(6*omega+(1j))-6*(8*omega+(1j))+q**(2)*(48*omega+sigma**(2)*(4*omega+(1j))-8*sigma*(6*omega+(1j))+(6j))))+sigma*h[1]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-15+q**(2))+(-15j))+5*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(5*(-12*sigma*(omega+(1j))+2*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))-1*q**(2)*(80*omega-60*sigma*(omega+(1j))+sigma**(2)*(4*omega+(7j))+(50j)))))
	terms[2] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(h[3]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[0]*(448*omega**(2)*(-1+q**(2))**(2)-168*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))-5*sigma**(3)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))-12*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))+8*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))+h[2]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-11j))+4*sigma**(3)*(2*omega+(3j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-161j))-7*(-1+kappa)*(24*omega+(23j))))+sigma*h[1]*(224*omega**(2)*(-1+q**(2))**(2)+4*sigma**(2)*(-1+q**(2))*(98+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-53+25*kappa))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(11j)))-2*sigma**(4)*(-15+8*omega**(2)+omega*(34j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-413-64*omega**(2)*(-7+kappa)+omega*(-28j)*(-45+4*kappa))+7*(-1+kappa)*(-59+64*omega**(2)+omega*(180j)))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(-4/5)*(-1+q**(2))*(kappa*h[2]*sigma**(3)*(-1j)-1*h[0]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))-1*sigma*h[1]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(7j))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(3)*(4/105)*(-1+q**(2))*(h[3]*sigma**(5)+4*h[2]*sigma**(3)*(kappa*omega*(-2j)+sigma*(3+omega*(-2j)))-4*h[0]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(1j))+3*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))-2*sigma*h[1]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(7j))+sigma**(2)*(-15+8*omega**(2)+omega*(34j))))

	return terms

def metric_ORG_33_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[4]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+8*h[3]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+8*h[2]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))-2*h[0]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-4*h[1]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[2]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+h[1]*(-4j)*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+sigma*(-8j)*(h[1]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[2]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(4/5)*(-1+q**(2))*(-4*h[0]*(-1*(-1+q**(2))*(20*omega*(-1+q**(2))-60*omega*sigma*(-1+q**(2))+6*sigma**(2)*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(3)*(4*omega+(1j)))+5*kappa*sigma*(-5*sigma**(2)*(4*omega+(1j))+6*sigma*(6*omega+(1j))-3*(8*omega+(1j))+q**(2)*(24*omega+sigma**(2)*(4*omega+(1j))-6*sigma*(6*omega+(1j))+(3j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))-4*h[1]*(-2*(-1+q**(2))*(20*omega*(-1+q**(2))-30*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-15+q**(2))+(-35j))+25*sigma**(3)*(omega+(1j)))+5*kappa*sigma*(-10*sigma**(2)*(omega+(1j))-3*(8*omega+(3j))+2*sigma*(12*omega+(7j))+q**(2)*(24*omega+2*sigma**(2)*(omega+(1j))-2*sigma*(12*omega+(7j))+(9j))))+sigma*h[2]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+2*sigma**(2)*(2*omega*(-15+q**(2))+(-55j))+5*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(5*(2*(8*omega+(9j))-2*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))-1*q**(2)*(80*omega-10*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j))+(90j))))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(-4*h[0]*(-112*omega**(2)*(-1+q**(2))**(2)+84*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))+5*sigma**(3)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))+15*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))-6*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))+sigma*(h[4]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[3]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-17j))+4*sigma**(3)*(2*omega+(5j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-259j))-7*(-1+kappa)*(24*omega+(37j))))-8*h[1]*(-112*omega**(2)*(-1+q**(2))**(2)+12*sigma**(4)*(-7+7*kappa+2*q**(2))*(-2+2*omega**(2)+omega*(5j))+42*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(7j)))+5*sigma**(3)*(q**(2)*(56+16*omega**(2)*(-7+kappa)+omega*(1j)*(-189+16*kappa))-7*(-1+kappa)*(-8+16*omega**(2)+omega*(27j)))-1*sigma**(2)*(-1+q**(2))*(217+16*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(84j)*(-11+5*kappa)))+sigma*h[2]*(224*omega**(2)*(-1+q**(2))**(2)-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(19j)))+2*sigma**(2)*(-1+q**(2))*(581+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(42j)*(-31+15*kappa))-2*sigma**(4)*(-57+8*omega**(2)+omega*(62j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-1379-64*omega**(2)*(-7+kappa)+omega*(-4j)*(-567+52*kappa))+7*(-1+kappa)*(-197+64*omega**(2)+omega*(324j))))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(2)*(-4/5)*(-1+q**(2))*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(2)*(4/105)*(-1+q**(2))*(-4*h[0]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(1j))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(h[4]*sigma**(5)+4*h[3]*sigma**(3)*(kappa*omega*(-2j)+sigma*(5+omega*(-2j)))-32*h[1]*(-4*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(omega+(1j))+3*sigma**(2)*(-2+2*omega**(2)+omega*(5j)))-2*sigma*h[2]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(13j))+sigma**(2)*(-57+8*omega**(2)+omega*(62j)))))

	return terms

def metric_ORG_33_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[5]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+12*h[4]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+24*h[3]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))+24*sigma*h[2]*(q**(4)*(60-300*sigma+400*sigma**(2)-175*sigma**(3)+28*sigma**(4))-30*(-2-10*sigma*(-1+kappa)+30*sigma**(2)*(-1+kappa)-35*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-60*sigma*(-2+kappa)+20*sigma**(2)*(-13+9*kappa)-35*sigma**(3)*(-7+4*kappa)+14*sigma**(4)*(-5+2*kappa)))-24*h[0]*(5*sigma**(2)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+10*sigma**(3)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+5*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+4*sigma*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-6*h[1]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-6*h[2]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[3]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+(-12j)*(h[1]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[2]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma*(-12j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(h[1]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[2]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[3]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[4]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(4/5)*(-1+q**(2))*(-12*h[0]*(-1*(-1+q**(2))*(-20*omega*(-1+q**(2))+4*sigma*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(2)*(4*omega+(1j)))+5*kappa*(-8*omega+(-1j)-5*sigma**(2)*(4*omega+(1j))+4*sigma*(6*omega+(1j))+q**(2)*(8*omega+(1j)+sigma**(2)*(4*omega+(1j))-4*sigma*(6*omega+(1j)))))-12*h[1]*(-1*(-1+q**(2))*(20*omega*(-1+q**(2))-60*omega*sigma*(-1+q**(2))+4*sigma**(2)*(3*omega*(-15+q**(2))+(-20j))+25*sigma**(3)*(4*omega+(3j)))+5*kappa*sigma*(-24*omega+(-7j)-5*sigma**(2)*(4*omega+(3j))+4*sigma*(9*omega+(4j))+q**(2)*(24*omega+sigma**(2)*(4*omega+(3j))-4*sigma*(9*omega+(4j))+(7j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(80*omega*(-1+q**(2))-120*omega*sigma*(-1+q**(2))+16*sigma**(2)*(omega*(-15+q**(2))+(-15j))+25*sigma**(3)*(4*omega+(7j)))+5*kappa*sigma*(48*sigma*(omega+(1j))-6*(8*omega+(5j))-5*sigma**(2)*(4*omega+(7j))+q**(2)*(-48*sigma*(omega+(1j))+6*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))))+sigma*h[3]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-15+q**(2))+(-40j))+5*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(5*(16*omega-4*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(26j))-1*q**(2)*(80*omega-20*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(130j))))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(-12*h[0]*(28*omega*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))+5*sigma**(2)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))+20*sigma**(3)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))-4*sigma*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))-12*h[1]*(-112*omega**(2)*(-1+q**(2))**(2)-4*sigma**(2)*(-1+q**(2))*(119+12*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-79+35*kappa))+28*omega*sigma*(-1+q**(2))*(12*omega*(-1+q**(2))+kappa*(36*omega+(17j)))+5*sigma**(4)*(-17+24*omega**(2)+omega*(46j))*(-7+7*kappa+2*q**(2))+5*sigma**(3)*(q**(2)*(161+64*omega**(2)*(-7+kappa)+omega*(12j)*(-49+4*kappa))-7*(-1+kappa)*(-23+64*omega**(2)+omega*(84j))))+sigma*(h[5]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[4]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-23j))+4*sigma**(3)*(2*omega+(7j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-357j))-21*(-1+kappa)*(8*omega+(17j))))-3*h[2]*(-448*omega**(2)*(-1+q**(2))**(2)-16*sigma**(2)*(-1+q**(2))*(133+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-53+25*kappa))+168*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(11j)))+4*sigma**(4)*(-73+24*omega**(2)+omega*(102j))*(-7+7*kappa+2*q**(2))+5*sigma**(3)*(q**(2)*(609+64*omega**(2)*(-7+kappa)+omega*(28j)*(-45+4*kappa))-7*(-1+kappa)*(-87+64*omega**(2)+omega*(180j))))+sigma*h[3]*(224*omega**(2)*(-1+q**(2))**(2)+4*sigma**(2)*(-1+q**(2))*(588+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-133+65*kappa))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(9j)))-2*sigma**(4)*(-127+8*omega**(2)+omega*(90j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-2933-64*omega**(2)*(-7+kappa)+omega*(-4j)*(-819+76*kappa))+7*(-1+kappa)*(-419+64*omega**(2)+omega*(468j))))))
	terms[3] = sigma*np.sqrt(np.pi*(1/7))*q**(3)*(-4/5)*(-1+q**(2))*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[4] = sigma*np.sqrt(np.pi)*q**(4)*(4/105)*(-1+q**(2))*(-48*h[0]*(-8*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(1j))+5*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(-12*h[1]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(3j))+5*sigma**(2)*(-17+24*omega**(2)+omega*(46j)))+sigma*(h[5]*sigma**(5)+4*h[4]*sigma**(3)*(kappa*omega*(-2j)+sigma*(7+omega*(-2j)))-2*sigma*h[3]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(19j))+sigma**(2)*(-127+8*omega**(2)+omega*(90j)))-12*h[2]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(7j))+sigma**(2)*(-73+24*omega**(2)+omega*(102j))))))

	return terms

def metric_ORG_33_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[6]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+16*h[5]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+48*h[4]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))+96*sigma*h[3]*(q**(4)*(60-300*sigma+400*sigma**(2)-175*sigma**(3)+28*sigma**(4))-30*(-2-10*sigma*(-1+kappa)+30*sigma**(2)*(-1+kappa)-35*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-60*sigma*(-2+kappa)+20*sigma**(2)*(-13+9*kappa)-35*sigma**(3)*(-7+4*kappa)+14*sigma**(4)*(-5+2*kappa)))+240*h[2]*(2*q**(4)*(3-30*sigma+60*sigma**(2)-35*sigma**(3)+7*sigma**(4))+6*(1+10*sigma*(-1+kappa)-45*sigma**(2)*(-1+kappa)+70*sigma**(3)*(-1+kappa)-35*sigma**(4)*(-1+kappa))+q**(2)*(-12-60*sigma*(-2+kappa)+30*sigma**(2)*(-13+9*kappa)-70*sigma**(3)*(-7+4*kappa)+35*sigma**(4)*(-5+2*kappa)))-48*h[0]*(5*sigma*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+2*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-96*h[1]*(5*sigma**(2)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+10*sigma**(3)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+5*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+4*sigma*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-12*h[2]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-8*h[3]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[4]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+(-24j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma*(-16j)*(h[1]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[2]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[3]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[4]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(24*h[1]*(q**(4)*(8*omega*(-15+kappa)+(-45j))+10*sigma*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))+4*h[2]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+6*h[3]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+4*h[4]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[5]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(4/5)*(-1+q**(2))*(480*omega*h[2]-1440*omega*sigma*h[2]+2880*kappa*omega*sigma*h[2]+320*omega*sigma*h[3]-960*omega*h[2]*q**(2)+2880*omega*sigma*h[2]*q**(2)-2880*kappa*omega*sigma*h[2]*q**(2)-640*omega*sigma*h[3]*q**(2)+480*omega*h[2]*q**(4)-1440*omega*sigma*h[2]*q**(4)+320*omega*sigma*h[3]*q**(4)+4320*omega*h[2]*sigma**(2)-4320*kappa*omega*h[2]*sigma**(2)-480*omega*h[3]*sigma**(2)+960*kappa*omega*h[3]*sigma**(2)+40*omega*h[4]*sigma**(2)-4608*omega*h[2]*q**(2)*sigma**(2)+4320*kappa*omega*h[2]*q**(2)*sigma**(2)+960*omega*h[3]*q**(2)*sigma**(2)-960*kappa*omega*h[3]*q**(2)*sigma**(2)-80*omega*h[4]*q**(2)*sigma**(2)+288*omega*h[2]*q**(4)*sigma**(2)-480*omega*h[3]*q**(4)*sigma**(2)+40*omega*h[4]*q**(4)*sigma**(2)-2400*omega*h[2]*sigma**(3)+2400*kappa*omega*h[2]*sigma**(3)+960*omega*h[3]*sigma**(3)-960*kappa*omega*h[3]*sigma**(3)-40*omega*h[4]*sigma**(3)+80*kappa*omega*h[4]*sigma**(3)+2400*omega*h[2]*q**(2)*sigma**(3)-480*kappa*omega*h[2]*q**(2)*sigma**(3)-1024*omega*h[3]*q**(2)*sigma**(3)+960*kappa*omega*h[3]*q**(2)*sigma**(3)+80*omega*h[4]*q**(2)*sigma**(3)-80*kappa*omega*h[4]*q**(2)*sigma**(3)+64*omega*h[3]*q**(4)*sigma**(3)-40*omega*h[4]*q**(4)*sigma**(3)-400*omega*h[3]*sigma**(4)+400*kappa*omega*h[3]*sigma**(4)+60*omega*h[4]*sigma**(4)-60*kappa*omega*h[4]*sigma**(4)+400*omega*h[3]*q**(2)*sigma**(4)-80*kappa*omega*h[3]*q**(2)*sigma**(4)-64*omega*h[4]*q**(2)*sigma**(4)+60*kappa*omega*h[4]*q**(2)*sigma**(4)+4*omega*h[4]*q**(4)*sigma**(4)-20*omega*h[4]*sigma**(5)+20*kappa*omega*h[4]*sigma**(5)+20*omega*h[4]*q**(2)*sigma**(5)-4*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-3120j)+h[2]*q**(2)*sigma**(2)*(-3120j)+h[2]*sigma**(3)*(-3000j)+kappa*h[3]*sigma**(3)*(-1360j)+h[3]*q**(2)*sigma**(3)*(-1360j)+kappa*sigma*h[2]*q**(2)*(-1320j)+h[3]*sigma**(4)*(-1000j)+kappa*h[3]*q**(2)*sigma**(2)*(-840j)+kappa*h[2]*q**(2)*sigma**(3)*(-600j)+kappa*h[4]*sigma**(4)*(-210j)+h[4]*q**(2)*sigma**(4)*(-210j)+kappa*h[3]*q**(2)*sigma**(4)*(-200j)+kappa*h[4]*q**(2)*sigma**(3)*(-170j)+h[4]*sigma**(5)*(-125j)+kappa*h[4]*q**(2)*sigma**(5)*(-25j)+kappa*h[5]*q**(2)*sigma**(4)*(-10j)+kappa*h[5]*sigma**(5)*(-10j)+h[5]*q**(2)*sigma**(5)*(-10j)+h[5]*sigma**(6)*(-5j)+kappa*h[5]*q**(2)*sigma**(6)*(-1j)+kappa*h[5]*sigma**(6)*(5j)+h[5]*q**(2)*sigma**(6)*(5j)-24*h[1]*(-1*(-1+q**(2))*(-40*omega*(-1+q**(2))+2*sigma*(8*omega*(-15+q**(2))+(-45j))+25*sigma**(2)*(8*omega+(5j)))+5*kappa*(-4*(4*omega+(1j))+6*sigma*(8*omega+(3j))-5*sigma**(2)*(8*omega+(5j))+q**(2)*(16*omega-6*sigma*(8*omega+(3j))+(4j)+sigma**(2)*(8*omega+(5j)))))+kappa*h[5]*sigma**(4)*(10j)+h[5]*sigma**(5)*(10j)+kappa*h[5]*q**(2)*sigma**(5)*(10j)+24*h[0]*(4*omega*q**(4)+5*(-1+kappa)*(-12*omega+20*omega*sigma+(-2j)+sigma*(5j))+q**(2)*(-64*omega+100*omega*sigma+(-10j)-5*kappa*(-12*omega+4*omega*sigma+(-2j)+sigma*(1j))+sigma*(25j)))+kappa*h[4]*sigma**(5)*(125j)+h[4]*q**(2)*sigma**(5)*(125j)+kappa*h[4]*sigma**(3)*(170j)+h[4]*sigma**(4)*(210j)+kappa*h[4]*q**(2)*sigma**(4)*(210j)+kappa*h[3]*sigma**(2)*(840j)+kappa*h[3]*sigma**(4)*(1000j)+h[3]*q**(2)*sigma**(4)*(1000j)+kappa*sigma*h[2]*(1320j)+h[3]*sigma**(3)*(1360j)+kappa*h[3]*q**(2)*sigma**(3)*(1360j)+kappa*h[2]*sigma**(3)*(3000j)+h[2]*q**(2)*sigma**(3)*(3000j)+h[2]*sigma**(2)*(3120j)+kappa*h[2]*q**(2)*sigma**(2)*(3120j))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(2688*h[2]*omega**(2)-8064*sigma*h[2]*omega**(2)+24192*kappa*sigma*h[2]*omega**(2)+1792*sigma*h[3]*omega**(2)-5376*h[2]*omega**(2)*q**(2)+16128*sigma*h[2]*omega**(2)*q**(2)-24192*kappa*sigma*h[2]*omega**(2)*q**(2)-3584*sigma*h[3]*omega**(2)*q**(2)+2688*h[2]*omega**(2)*q**(4)-8064*sigma*h[2]*omega**(2)*q**(4)+1792*sigma*h[3]*omega**(2)*q**(4)-24864*h[2]*sigma**(2)+56448*h[2]*omega**(2)*sigma**(2)-40320*kappa*h[2]*omega**(2)*sigma**(2)-2688*h[3]*omega**(2)*sigma**(2)+8064*kappa*h[3]*omega**(2)*sigma**(2)+224*h[4]*omega**(2)*sigma**(2)+24864*h[2]*q**(2)*sigma**(2)-58752*h[2]*omega**(2)*q**(2)*sigma**(2)+40320*kappa*h[2]*omega**(2)*q**(2)*sigma**(2)+5376*h[3]*omega**(2)*q**(2)*sigma**(2)-8064*kappa*h[3]*omega**(2)*q**(2)*sigma**(2)-448*h[4]*omega**(2)*q**(2)*sigma**(2)+2304*h[2]*omega**(2)*q**(4)*sigma**(2)-2688*h[3]*omega**(2)*q**(4)*sigma**(2)+224*h[4]*omega**(2)*q**(4)*sigma**(2)+46200*h[2]*sigma**(3)-46200*kappa*h[2]*sigma**(3)-15792*h[3]*sigma**(3)-53760*h[2]*omega**(2)*sigma**(3)+53760*kappa*h[2]*omega**(2)*sigma**(3)+12544*h[3]*omega**(2)*sigma**(3)-8960*kappa*h[3]*omega**(2)*sigma**(3)-224*h[4]*omega**(2)*sigma**(3)+672*kappa*h[4]*omega**(2)*sigma**(3)-46200*h[2]*q**(2)*sigma**(3)+15792*h[3]*q**(2)*sigma**(3)+53760*h[2]*omega**(2)*q**(2)*sigma**(3)-7680*kappa*h[2]*omega**(2)*q**(2)*sigma**(3)-13056*h[3]*omega**(2)*q**(2)*sigma**(3)+8960*kappa*h[3]*omega**(2)*q**(2)*sigma**(3)+448*h[4]*omega**(2)*q**(2)*sigma**(3)-672*kappa*h[4]*omega**(2)*q**(2)*sigma**(3)+512*h[3]*omega**(2)*q**(4)*sigma**(3)-224*h[4]*omega**(2)*q**(4)*sigma**(3)-37800*h[2]*sigma**(4)+37800*kappa*h[2]*sigma**(4)+23800*h[3]*sigma**(4)-23800*kappa*h[3]*sigma**(4)-3962*h[4]*sigma**(4)+20160*h[2]*omega**(2)*sigma**(4)-20160*kappa*h[2]*omega**(2)*sigma**(4)-8960*h[3]*omega**(2)*sigma**(4)+8960*kappa*h[3]*omega**(2)*sigma**(4)+784*h[4]*omega**(2)*sigma**(4)-560*kappa*h[4]*omega**(2)*sigma**(4)+10800*h[2]*q**(2)*sigma**(4)-23800*h[3]*q**(2)*sigma**(4)+3962*h[4]*q**(2)*sigma**(4)-5760*h[2]*omega**(2)*q**(2)*sigma**(4)+8960*h[3]*omega**(2)*q**(2)*sigma**(4)-1280*kappa*h[3]*omega**(2)*q**(2)*sigma**(4)-816*h[4]*omega**(2)*q**(2)*sigma**(4)+560*kappa*h[4]*omega**(2)*q**(2)*sigma**(4)+32*h[4]*omega**(2)*q**(4)*sigma**(4)-16800*h[3]*sigma**(5)+16800*kappa*h[3]*sigma**(5)+5075*h[4]*sigma**(5)-5075*kappa*h[4]*sigma**(5)-406*h[5]*sigma**(5)+2688*h[3]*omega**(2)*sigma**(5)-2688*kappa*h[3]*omega**(2)*sigma**(5)-448*h[4]*omega**(2)*sigma**(5)+448*kappa*h[4]*omega**(2)*sigma**(5)+4800*h[3]*q**(2)*sigma**(5)-5075*h[4]*q**(2)*sigma**(5)+406*h[5]*q**(2)*sigma**(5)-768*h[3]*omega**(2)*q**(2)*sigma**(5)+448*h[4]*omega**(2)*q**(2)*sigma**(5)-64*kappa*h[4]*omega**(2)*q**(2)*sigma**(5)-3150*h[4]*sigma**(6)+3150*kappa*h[4]*sigma**(6)+455*h[5]*sigma**(6)-455*kappa*h[5]*sigma**(6)-14*h[6]*sigma**(6)+112*h[4]*omega**(2)*sigma**(6)-112*kappa*h[4]*omega**(2)*sigma**(6)+900*h[4]*q**(2)*sigma**(6)-455*h[5]*q**(2)*sigma**(6)+14*h[6]*q**(2)*sigma**(6)-32*h[4]*omega**(2)*q**(2)*sigma**(6)-252*h[5]*sigma**(7)+252*kappa*h[5]*sigma**(7)+14*h[6]*sigma**(7)-14*kappa*h[6]*sigma**(7)+72*h[5]*q**(2)*sigma**(7)-14*h[6]*q**(2)*sigma**(7)-7*h[6]*sigma**(8)+7*kappa*h[6]*sigma**(8)+2*h[6]*q**(2)*sigma**(8)+omega*h[2]*sigma**(3)*(-110880j)+omega*h[2]*q**(2)*sigma**(2)*(-79968j)+kappa*omega*h[2]*sigma**(4)*(-62160j)+kappa*omega*h[2]*sigma**(2)*(-36960j)+omega*h[3]*sigma**(4)*(-35280j)+omega*h[3]*q**(2)*sigma**(3)*(-32704j)+omega*h[2]*q**(2)*sigma**(4)*(-17760j)+kappa*omega*sigma*h[2]*q**(2)*(-16800j)+kappa*omega*h[3]*sigma**(5)*(-16128j)+kappa*omega*h[3]*sigma**(3)*(-15680j)+kappa*omega*h[3]*q**(2)*sigma**(2)*(-10080j)+kappa*omega*h[2]*q**(2)*sigma**(3)*(-9600j)+omega*h[4]*q**(2)*sigma**(4)*(-4844j)+omega*h[3]*q**(2)*sigma**(5)*(-4608j)+omega*h[4]*sigma**(5)*(-4284j)+kappa*omega*h[3]*q**(2)*sigma**(4)*(-3200j)+kappa*omega*h[4]*sigma**(4)*(-2380j)+kappa*omega*h[4]*q**(2)*sigma**(3)*(-1960j)+kappa*omega*h[4]*sigma**(6)*(-1652j)+omega*h[4]*q**(2)*sigma**(6)*(-472j)+kappa*omega*h[4]*q**(2)*sigma**(5)*(-400j)+omega*h[5]*q**(2)*sigma**(5)*(-224j)+omega*h[5]*sigma**(6)*(-168j)+kappa*omega*h[5]*q**(2)*sigma**(4)*(-112j)+kappa*omega*h[5]*sigma**(5)*(-112j)+kappa*omega*h[5]*sigma**(7)*(-56j)+kappa*omega*h[5]*q**(2)*sigma**(6)*(-16j)+omega*h[5]*q**(2)*sigma**(7)*(-16j)+24*h[0]*(32*omega**(2)*q**(4)-7*(4*omega+(1j))*(4*omega*(-7+5*kappa)+(-6j)+30*sigma**(2)*(-1+kappa)*(2*omega+(1j))-5*sigma*(-1+kappa)*(16*omega+(5j)))+q**(2)*(42+16*omega**(2)*(-51+35*kappa)-5*sigma*(4*omega+(1j))*(16*omega*(-7+kappa)+(-35j))-60*sigma**(2)*(-1+8*omega**(2)+omega*(6j))+omega*(28j)*(-13+5*kappa)))+omega*h[5]*sigma**(7)*(56j)+kappa*omega*h[5]*sigma**(4)*(112j)+kappa*omega*h[5]*q**(2)*sigma**(5)*(112j)-24*h[1]*(56*omega*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(5j)))+20*sigma**(3)*(-9+16*omega**(2)+omega*(26j))*(-7+7*kappa+2*q**(2))-2*sigma*(-1+q**(2))*(259+32*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(56j)*(-23+10*kappa))+5*sigma**(2)*(q**(2)*(259+128*omega**(2)*(-7+kappa)+omega*(16j)*(-63+5*kappa))-7*(-1+kappa)*(-37+128*omega**(2)+omega*(144j))))+kappa*omega*h[5]*sigma**(6)*(168j)+omega*h[5]*q**(2)*sigma**(6)*(168j)+omega*h[5]*sigma**(5)*(224j)+omega*h[4]*sigma**(6)*(1652j)+kappa*omega*h[4]*sigma**(3)*(1960j)+kappa*omega*h[4]*q**(2)*sigma**(4)*(2380j)+kappa*omega*h[4]*sigma**(5)*(4284j)+omega*h[4]*q**(2)*sigma**(5)*(4284j)+omega*h[4]*sigma**(4)*(4844j)+kappa*omega*h[3]*sigma**(2)*(10080j)+kappa*omega*h[3]*q**(2)*sigma**(3)*(15680j)+omega*h[3]*sigma**(5)*(16128j)+kappa*omega*sigma*h[2]*(16800j)+omega*h[3]*sigma**(3)*(32704j)+kappa*omega*h[3]*sigma**(4)*(35280j)+omega*h[3]*q**(2)*sigma**(4)*(35280j)+kappa*omega*h[2]*q**(2)*sigma**(2)*(36960j)+omega*h[2]*sigma**(4)*(62160j)+omega*h[2]*sigma**(2)*(79968j)+kappa*omega*h[2]*sigma**(3)*(110880j)+omega*h[2]*q**(2)*sigma**(3)*(110880j))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*(-4/5)*(-1+q**(2))*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[4] = np.sqrt(np.pi)*q**(4)*(4/105)*(-1+q**(2))*(-48*h[0]*(-8*omega**(2)*(-1+q**(2))+20*kappa*omega*sigma*(4*omega+(1j))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(-96*h[1]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(8*omega+(5j))+5*sigma**(2)*(-9+16*omega**(2)+omega*(26j)))+sigma*(-24*h[2]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(5j))+5*sigma**(2)*(-45+24*omega**(2)+omega*(74j)))+sigma*(h[6]*sigma**(5)+4*h[5]*sigma**(3)*(kappa*omega*(-2j)+sigma*(9+omega*(-2j)))-32*h[3]*(-8*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(2*omega+(5j))+3*sigma**(2)*(-25+4*omega**(2)+omega*(24j)))-2*sigma*h[4]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(25j))+sigma**(2)*(-225+8*omega**(2)+omega*(118j)))))))

	return terms

def metric_ORG_44_radial(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[2]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))-2*h[0]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+24*kappa*omega+(-27j)+kappa*(6j)))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+h[1]*sigma**(2)*(-2j)*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*sigma**(2)*(4/5)*(-1+q**(2))*(h[1]*sigma**(2)*(1j)*(-5*kappa*(2+sigma*(-2+sigma))-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10+sigma*(-10+sigma)))+h[0]*(-40*omega*(-1+q**(2))**(2)+sigma**(3)*(4*omega+(1j))*(5-5*kappa+q**(2)*(-5+kappa))+10*sigma*(-1+q**(2))*(4*omega*(-1+2*kappa+q**(2))+kappa*(1j))-2*sigma**(2)*(-1+q**(2))*(2*omega*(15*(-1+kappa)+q**(2))+(5j)*(-1+kappa))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(-4/21)*(-1+q**(2))*(h[2]*sigma**(4)*(7*(-2+sigma*(-2+sigma)*(-1+kappa))+2*q**(2)*(7+sigma*(-7+sigma)))+h[1]*sigma**(2)*(kappa*omega*(-112j)*(-1+q**(2))+4*sigma**(3)*(1+omega*(-2j))*(-7+7*kappa+2*q**(2))+14*sigma*(-1+q**(2))*(5+omega*(8j)*(-2+kappa))+sigma**(2)*(q**(2)*(-63+omega*(-8j)*(-21+2*kappa))+21*(-3+omega*(8j))*(-1+kappa)))+h[0]*(224*omega**(2)*(-1+q**(2))**(2)-1*sigma**(3)*(4*omega+(1j))*(16*omega*(7-7*kappa+q**(2)*(-7+kappa))+(-35j)*(-1+kappa+q**(2)))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+3*kappa+q**(2))+kappa*(3j))-2*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))+2*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(4)*(4/5)*(-1+q**(2))*(kappa*h[1]*sigma**(2)*(-1j)-1*h[0]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(1j))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(4)*(4/105)*(-1+q**(2))*(h[2]*sigma**(4)+4*h[1]*sigma**(2)*(sigma+omega*(-2j)*(kappa+sigma))-2*h[0]*(-1*sigma**(2)+8*omega**(2)*(1+2*kappa*sigma-1*q**(2)+sigma**(2))+omega*sigma*(2j)*(2*kappa+3*sigma)))

	return terms

def metric_ORG_44_radial_deriv1(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(240*h[2]*sigma**(3)+20*h[2]*q**(2)*sigma**(3)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+8*h[2]*q**(4)*sigma**(3)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))-150*h[2]*sigma**(4)*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+60*h[3]*sigma**(4)+5*h[3]*q**(2)*sigma**(4)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*h[3]*q**(4)*sigma**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma))))+4*h[2]*q**(4)*sigma**(4)*(-30+40*sigma-15*sigma**(2)+2*sigma**(3))+20*h[2]*q**(2)*sigma**(4)*(12-26*sigma+21*sigma**(2)-5*sigma**(3)+2*kappa*(-3+9*sigma-6*sigma**(2)+sigma**(3)))-60*h[2]*sigma**(5)*(-2+sigma)*(-1+kappa)*(-1+sigma)-30*h[2]*sigma**(5)*(-1+kappa)*(2+sigma*(-2+sigma))-30*h[3]*sigma**(5)*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+960*h[1]*omega**(2)*(-1+q**(2))**(3)+kappa*omega*sigma*h[1]*(-1+q**(2))**(2)*(-960j)+kappa*omega*h[2]*sigma**(2)*(-1+q**(2))**(2)*(-480j)+28*h[1]*sigma**(6)*(1+omega*(-2j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+4*h[2]*sigma**(7)*(1+omega*(-2j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-10*h[0]*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-2*h[1]*sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-960*omega*h[0]*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-960*omega*sigma*h[1]*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))+160*sigma*h[0]*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+80*h[1]*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+360*h[1]*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(4j)*(-3+2*kappa))+120*h[2]*sigma**(3)*(-1+q**(2))**(2)*(3+omega*(4j)*(-3+2*kappa))-12*h[0]*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-2*h[1]*sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))-60*h[0]*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))-20*h[1]*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+h[1]*sigma**(3)*(-80j)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+h[2]*sigma**(4)*(-20j)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))-16*h[0]*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa))-4*h[1]*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa))+h[1]*sigma**(4)*(-100j)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+h[2]*sigma**(5)*(-20j)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+h[1]*sigma**(5)*(-12j)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))+h[2]*sigma**(6)*(-2j)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))
	terms[1] = q*sigma*np.sqrt(np.pi*(1/3))*(-4/5)*(-1+q**(2))*(h[2]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))+h[0]*((-1+q**(2))*(80*omega*(-1+q**(2))-120*omega*sigma*(-1+q**(2))+8*sigma**(2)*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(3)*(4*omega+(1j)))-5*kappa*sigma*(-5*sigma**(2)*(4*omega+(1j))+8*sigma*(6*omega+(1j))-6*(8*omega+(1j))+q**(2)*(48*omega+sigma**(2)*(4*omega+(1j))-8*sigma*(6*omega+(1j))+(6j))))+sigma*h[1]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-15+q**(2))+(-15j))+5*sigma**(3)*(4*omega+(7j)))+kappa*sigma*(5*(-12*sigma*(omega+(1j))+2*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))-1*q**(2)*(80*omega-60*sigma*(omega+(1j))+sigma**(2)*(4*omega+(7j))+(50j)))))
	terms[2] = sigma*np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(h[3]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[0]*(448*omega**(2)*(-1+q**(2))**(2)-168*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))-5*sigma**(3)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))-12*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))+8*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))+h[2]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-11j))+4*sigma**(3)*(2*omega+(3j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-161j))-7*(-1+kappa)*(24*omega+(23j))))+sigma*h[1]*(224*omega**(2)*(-1+q**(2))**(2)+4*sigma**(2)*(-1+q**(2))*(98+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-53+25*kappa))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(11j)))-2*sigma**(4)*(-15+8*omega**(2)+omega*(34j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-413-64*omega**(2)*(-7+kappa)+omega*(-28j)*(-45+4*kappa))+7*(-1+kappa)*(-59+64*omega**(2)+omega*(180j)))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(4/5)*(-1+q**(2))*(kappa*h[2]*sigma**(3)*(-1j)-1*h[0]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))-1*sigma*h[1]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(7j))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(3)*(4/105)*(-1+q**(2))*(h[3]*sigma**(5)+4*h[2]*sigma**(3)*(kappa*omega*(-2j)+sigma*(3+omega*(-2j)))-4*h[0]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(1j))+3*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))-2*sigma*h[1]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(7j))+sigma**(2)*(-15+8*omega**(2)+omega*(34j))))

	return terms

def metric_ORG_44_radial_deriv2(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[4]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+8*h[3]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+8*h[2]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))-2*h[0]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-4*h[1]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[2]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+h[1]*(-4j)*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+sigma*(-8j)*(h[1]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[2]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-4/5)*(-1+q**(2))*(-4*h[0]*(-1*(-1+q**(2))*(20*omega*(-1+q**(2))-60*omega*sigma*(-1+q**(2))+6*sigma**(2)*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(3)*(4*omega+(1j)))+5*kappa*sigma*(-5*sigma**(2)*(4*omega+(1j))+6*sigma*(6*omega+(1j))-3*(8*omega+(1j))+q**(2)*(24*omega+sigma**(2)*(4*omega+(1j))-6*sigma*(6*omega+(1j))+(3j))))+sigma*(h[3]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))-4*h[1]*(-2*(-1+q**(2))*(20*omega*(-1+q**(2))-30*omega*sigma*(-1+q**(2))+sigma**(2)*(4*omega*(-15+q**(2))+(-35j))+25*sigma**(3)*(omega+(1j)))+5*kappa*sigma*(-10*sigma**(2)*(omega+(1j))-3*(8*omega+(3j))+2*sigma*(12*omega+(7j))+q**(2)*(24*omega+2*sigma**(2)*(omega+(1j))-2*sigma*(12*omega+(7j))+(9j))))+sigma*h[2]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+2*sigma**(2)*(2*omega*(-15+q**(2))+(-55j))+5*sigma**(3)*(4*omega+(13j)))+kappa*sigma*(5*(2*(8*omega+(9j))-2*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j)))-1*q**(2)*(80*omega-10*sigma*(6*omega+(11j))+sigma**(2)*(4*omega+(13j))+(90j))))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(-4*h[0]*(-112*omega**(2)*(-1+q**(2))**(2)+84*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))+5*sigma**(3)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))+15*sigma**(4)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))-6*sigma**(2)*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))+sigma*(h[4]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[3]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-17j))+4*sigma**(3)*(2*omega+(5j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-259j))-7*(-1+kappa)*(24*omega+(37j))))-8*h[1]*(-112*omega**(2)*(-1+q**(2))**(2)+12*sigma**(4)*(-7+7*kappa+2*q**(2))*(-2+2*omega**(2)+omega*(5j))+42*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(7j)))+5*sigma**(3)*(q**(2)*(56+16*omega**(2)*(-7+kappa)+omega*(1j)*(-189+16*kappa))-7*(-1+kappa)*(-8+16*omega**(2)+omega*(27j)))-1*sigma**(2)*(-1+q**(2))*(217+16*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(84j)*(-11+5*kappa)))+sigma*h[2]*(224*omega**(2)*(-1+q**(2))**(2)-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(19j)))+2*sigma**(2)*(-1+q**(2))*(581+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(42j)*(-31+15*kappa))-2*sigma**(4)*(-57+8*omega**(2)+omega*(62j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-1379-64*omega**(2)*(-7+kappa)+omega*(-4j)*(-567+52*kappa))+7*(-1+kappa)*(-197+64*omega**(2)+omega*(324j))))))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(2)*(4/5)*(-1+q**(2))*(-4*h[0]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(kappa*h[3]*sigma**(3)*(-1j)-8*h[1]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(omega+(1j)))-1*sigma*h[2]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(13j)))))
	terms[4] = np.sqrt(np.pi)*q**(4)*sigma**(2)*(4/105)*(-1+q**(2))*(-4*h[0]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(1j))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(h[4]*sigma**(5)+4*h[3]*sigma**(3)*(kappa*omega*(-2j)+sigma*(5+omega*(-2j)))-32*h[1]*(-4*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(omega+(1j))+3*sigma**(2)*(-2+2*omega**(2)+omega*(5j)))-2*sigma*h[2]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(13j))+sigma**(2)*(-57+8*omega**(2)+omega*(62j)))))

	return terms

def metric_ORG_44_radial_deriv3(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[5]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+12*h[4]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+24*h[3]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))+24*sigma*h[2]*(q**(4)*(60-300*sigma+400*sigma**(2)-175*sigma**(3)+28*sigma**(4))-30*(-2-10*sigma*(-1+kappa)+30*sigma**(2)*(-1+kappa)-35*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-60*sigma*(-2+kappa)+20*sigma**(2)*(-13+9*kappa)-35*sigma**(3)*(-7+4*kappa)+14*sigma**(4)*(-5+2*kappa)))-24*h[0]*(5*sigma**(2)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+10*sigma**(3)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+5*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+4*sigma*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-6*h[1]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-6*h[2]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[3]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+(-12j)*(h[1]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[2]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma*(-12j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(h[1]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[2]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[3]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[4]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-4/5)*(-1+q**(2))*(-12*h[0]*(-1*(-1+q**(2))*(-20*omega*(-1+q**(2))+4*sigma*(2*omega*(-15+q**(2))+(-5j))+25*sigma**(2)*(4*omega+(1j)))+5*kappa*(-8*omega+(-1j)-5*sigma**(2)*(4*omega+(1j))+4*sigma*(6*omega+(1j))+q**(2)*(8*omega+(1j)+sigma**(2)*(4*omega+(1j))-4*sigma*(6*omega+(1j)))))-12*h[1]*(-1*(-1+q**(2))*(20*omega*(-1+q**(2))-60*omega*sigma*(-1+q**(2))+4*sigma**(2)*(3*omega*(-15+q**(2))+(-20j))+25*sigma**(3)*(4*omega+(3j)))+5*kappa*sigma*(-24*omega+(-7j)-5*sigma**(2)*(4*omega+(3j))+4*sigma*(9*omega+(4j))+q**(2)*(24*omega+sigma**(2)*(4*omega+(3j))-4*sigma*(9*omega+(4j))+(7j))))+sigma*(h[4]*sigma**(3)*(-1j)*(-5*sigma*(-2+sigma)*(-1+q**(2))+kappa*q**(2)*(10-10*sigma+sigma**(2))-5*kappa*(2-2*sigma+sigma**(2)))-3*h[2]*(-1*(-1+q**(2))*(80*omega*(-1+q**(2))-120*omega*sigma*(-1+q**(2))+16*sigma**(2)*(omega*(-15+q**(2))+(-15j))+25*sigma**(3)*(4*omega+(7j)))+5*kappa*sigma*(48*sigma*(omega+(1j))-6*(8*omega+(5j))-5*sigma**(2)*(4*omega+(7j))+q**(2)*(-48*sigma*(omega+(1j))+6*(8*omega+(5j))+sigma**(2)*(4*omega+(7j)))))+sigma*h[3]*((-1+q**(2))*(40*omega*(-1+q**(2))-40*omega*sigma*(-1+q**(2))+4*sigma**(2)*(omega*(-15+q**(2))+(-40j))+5*sigma**(3)*(4*omega+(19j)))+kappa*sigma*(5*(16*omega-4*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(26j))-1*q**(2)*(80*omega-20*sigma*(3*omega+(8j))+sigma**(2)*(4*omega+(19j))+(130j))))))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(-12*h[0]*(28*omega*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(1j)))+5*sigma**(2)*(4*omega+(1j))*(q**(2)*(16*omega*(-7+kappa)+(-35j))-7*(-1+kappa)*(16*omega+(5j)))+20*sigma**(3)*(-7+7*kappa+2*q**(2))*(-1+8*omega**(2)+omega*(6j))-4*sigma*(-1+q**(2))*(21+8*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(14j)*(-13+5*kappa)))-12*h[1]*(-112*omega**(2)*(-1+q**(2))**(2)-4*sigma**(2)*(-1+q**(2))*(119+12*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-79+35*kappa))+28*omega*sigma*(-1+q**(2))*(12*omega*(-1+q**(2))+kappa*(36*omega+(17j)))+5*sigma**(4)*(-17+24*omega**(2)+omega*(46j))*(-7+7*kappa+2*q**(2))+5*sigma**(3)*(q**(2)*(161+64*omega**(2)*(-7+kappa)+omega*(12j)*(-49+4*kappa))-7*(-1+kappa)*(-23+64*omega**(2)+omega*(84j))))+sigma*(h[5]*sigma**(5)*(2*q**(2)*(7-7*sigma+sigma**(2))+7*(-2-2*sigma*(-1+kappa)+sigma**(2)*(-1+kappa)))+h[4]*sigma**(3)*(-1j)*(112*kappa*omega*(-1+q**(2))-14*sigma*(-1+q**(2))*(8*omega*(-2+kappa)+(-23j))+4*sigma**(3)*(2*omega+(7j))*(-7+7*kappa+2*q**(2))+sigma**(2)*(q**(2)*(8*omega*(-21+2*kappa)+(-357j))-21*(-1+kappa)*(8*omega+(17j))))-3*h[2]*(-448*omega**(2)*(-1+q**(2))**(2)-16*sigma**(2)*(-1+q**(2))*(133+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-53+25*kappa))+168*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(11j)))+4*sigma**(4)*(-73+24*omega**(2)+omega*(102j))*(-7+7*kappa+2*q**(2))+5*sigma**(3)*(q**(2)*(609+64*omega**(2)*(-7+kappa)+omega*(28j)*(-45+4*kappa))-7*(-1+kappa)*(-87+64*omega**(2)+omega*(180j))))+sigma*h[3]*(224*omega**(2)*(-1+q**(2))**(2)+4*sigma**(2)*(-1+q**(2))*(588+4*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(7j)*(-133+65*kappa))-56*omega*sigma*(-1+q**(2))*(4*omega*(-1+q**(2))+3*kappa*(4*omega+(9j)))-2*sigma**(4)*(-127+8*omega**(2)+omega*(90j))*(-7+7*kappa+2*q**(2))+sigma**(3)*(q**(2)*(-2933-64*omega**(2)*(-7+kappa)+omega*(-4j)*(-819+76*kappa))+7*(-1+kappa)*(-419+64*omega**(2)+omega*(468j))))))
	terms[3] = sigma*np.sqrt(np.pi*(1/7))*q**(3)*(4/5)*(-1+q**(2))*(-12*h[0]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-12*h[1]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(3j)))+sigma*(kappa*h[4]*sigma**(3)*(-1j)-3*h[2]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(7j)))-1*sigma*h[3]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(19j))))))
	terms[4] = sigma*np.sqrt(np.pi)*q**(4)*(4/105)*(-1+q**(2))*(-48*h[0]*(-8*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(1j))+5*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(-12*h[1]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(3j))+5*sigma**(2)*(-17+24*omega**(2)+omega*(46j)))+sigma*(h[5]*sigma**(5)+4*h[4]*sigma**(3)*(kappa*omega*(-2j)+sigma*(7+omega*(-2j)))-2*sigma*h[3]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(19j))+sigma**(2)*(-127+8*omega**(2)+omega*(90j)))-12*h[2]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(4*omega+(7j))+sigma**(2)*(-73+24*omega**(2)+omega*(102j))))))

	return terms

def metric_ORG_44_radial_deriv4(kappa, sigma, m, omega, h):
	q = np.sqrt(1. - kappa**2)
	nterm = 5
	terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

	terms[0] = (1/15)*(-1+q**(2))*(h[6]*sigma**(4)*(60-30*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+5*q**(2)*(-24+sigma*(48+sigma*(-52+sigma*(28-5*sigma))+2*kappa*(-2+sigma)*(6+sigma*(-6+sigma))))+2*q**(4)*(30+sigma*(-60+sigma*(40+sigma*(-10+sigma)))))+16*h[5]*sigma**(3)*(q**(4)*(60-150*sigma+120*sigma**(2)-35*sigma**(3)+4*sigma**(4))-30*(-2-5*sigma*(-1+kappa)+9*sigma**(2)*(-1+kappa)-7*sigma**(3)*(-1+kappa)+2*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-30*sigma*(-2+kappa)+6*sigma**(2)*(-13+9*kappa)-7*sigma**(3)*(-7+4*kappa)+2*sigma**(4)*(-5+2*kappa)))+48*h[4]*sigma**(2)*(q**(4)*(90-300*sigma+300*sigma**(2)-105*sigma**(3)+14*sigma**(4))-15*(-6-20*sigma*(-1+kappa)+45*sigma**(2)*(-1+kappa)-42*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-36-60*sigma*(-2+kappa)+15*sigma**(2)*(-13+9*kappa)-21*sigma**(3)*(-7+4*kappa)+7*sigma**(4)*(-5+2*kappa)))+96*sigma*h[3]*(q**(4)*(60-300*sigma+400*sigma**(2)-175*sigma**(3)+28*sigma**(4))-30*(-2-10*sigma*(-1+kappa)+30*sigma**(2)*(-1+kappa)-35*sigma**(3)*(-1+kappa)+14*sigma**(4)*(-1+kappa))+5*q**(2)*(-24-60*sigma*(-2+kappa)+20*sigma**(2)*(-13+9*kappa)-35*sigma**(3)*(-7+4*kappa)+14*sigma**(4)*(-5+2*kappa)))+240*h[2]*(2*q**(4)*(3-30*sigma+60*sigma**(2)-35*sigma**(3)+7*sigma**(4))+6*(1+10*sigma*(-1+kappa)-45*sigma**(2)*(-1+kappa)+70*sigma**(3)*(-1+kappa)-35*sigma**(4)*(-1+kappa))+q**(2)*(-12-60*sigma*(-2+kappa)+30*sigma**(2)*(-13+9*kappa)-70*sigma**(3)*(-7+4*kappa)+35*sigma**(4)*(-5+2*kappa)))-48*h[0]*(5*sigma*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+2*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-96*h[1]*(5*sigma**(2)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+10*sigma**(3)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+5*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+4*sigma*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-12*h[2]*(20*sigma**(3)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-80*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+30*sigma**(4)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*sigma*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+24*sigma**(2)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-8*h[3]*(5*sigma**(4)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-80*sigma*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+6*sigma**(5)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*sigma**(2)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+8*sigma**(3)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))-2*h[4]*(-480*omega**(2)*(-1+q**(2))**(3)+sigma**(5)*(4*omega+(1j))*(4*omega*(90*(-1+kappa)+q**(2)*(110-65*kappa)+2*q**(4)*(-10+kappa))+(-25j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))+480*omega*sigma*(-1+q**(2))**(2)*(2*omega*(-1+2*kappa+q**(2))+kappa*(1j))-40*sigma**(2)*(-1+q**(2))**(2)*(3+omega*(16*omega*(-6+q**(2))+(-36j)+21*kappa*(4*omega+(1j))))+sigma**(6)*(-1+8*omega**(2)+omega*(6j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(3)*(-1+q**(2))*(16*omega**(2)*q**(4)+q**(2)*(21+8*omega*(-62*omega+(-27j)+6*kappa*(4*omega+(1j))))-3*(-7+8*omega*(20*omega+(9j)))*(-1+kappa))+2*sigma**(4)*(-1+q**(2))*(-8*omega**(2)*q**(4)-10*q**(2)*(6+omega*(-106*omega+20*kappa*omega+(-52j)+kappa*(5j)))+15*(-9+4*omega*(30*omega+(17j)))*(-1+kappa)))+(-24j)*(h[1]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+2*h[2]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[3]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma*(-16j)*(h[1]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[2]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+3*h[3]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[4]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))))+sigma**(2)*(-2j)*(24*h[1]*(q**(4)*(8*omega*(-15+kappa)+(-45j))+10*sigma*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j)))+4*h[2]*(120*sigma**(2)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+60*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+24*sigma*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+6*h[3]*(40*sigma**(3)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+60*sigma*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+12*sigma**(2)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+4*h[4]*(-60*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+10*sigma**(4)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+20*sigma*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+30*sigma**(2)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+4*sigma**(3)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))+h[5]*(240*kappa*omega*(-1+q**(2))**(2)-60*sigma*(-1+q**(2))**(2)*(4*omega*(-3+2*kappa)+(-3j))+2*sigma**(5)*(2*omega+(1j))*(-30*(-1+kappa)+5*q**(2)*(-5+2*kappa)+2*q**(4))+10*sigma**(2)*(-1+q**(2))*(q**(2)*(8*omega*(-15+4*kappa)+(-33j))-3*(-1+kappa)*(40*omega+(11j)))+10*sigma**(3)*(-1+q**(2))*(15*(-1+kappa)*(8*omega+(3j))+q**(2)*(-8*omega*(-8+kappa)+(20j)))+sigma**(4)*(q**(4)*(8*omega*(-15+kappa)+(-45j))+30*(-1+kappa)*(20*omega+(9j))+15*q**(2)*(48*omega-4*kappa*(7*omega+(3j))+(21j))))))
	terms[1] = q*np.sqrt(np.pi*(1/3))*(-4/5)*(-1+q**(2))*(480*omega*h[2]-1440*omega*sigma*h[2]+2880*kappa*omega*sigma*h[2]+320*omega*sigma*h[3]-960*omega*h[2]*q**(2)+2880*omega*sigma*h[2]*q**(2)-2880*kappa*omega*sigma*h[2]*q**(2)-640*omega*sigma*h[3]*q**(2)+480*omega*h[2]*q**(4)-1440*omega*sigma*h[2]*q**(4)+320*omega*sigma*h[3]*q**(4)+4320*omega*h[2]*sigma**(2)-4320*kappa*omega*h[2]*sigma**(2)-480*omega*h[3]*sigma**(2)+960*kappa*omega*h[3]*sigma**(2)+40*omega*h[4]*sigma**(2)-4608*omega*h[2]*q**(2)*sigma**(2)+4320*kappa*omega*h[2]*q**(2)*sigma**(2)+960*omega*h[3]*q**(2)*sigma**(2)-960*kappa*omega*h[3]*q**(2)*sigma**(2)-80*omega*h[4]*q**(2)*sigma**(2)+288*omega*h[2]*q**(4)*sigma**(2)-480*omega*h[3]*q**(4)*sigma**(2)+40*omega*h[4]*q**(4)*sigma**(2)-2400*omega*h[2]*sigma**(3)+2400*kappa*omega*h[2]*sigma**(3)+960*omega*h[3]*sigma**(3)-960*kappa*omega*h[3]*sigma**(3)-40*omega*h[4]*sigma**(3)+80*kappa*omega*h[4]*sigma**(3)+2400*omega*h[2]*q**(2)*sigma**(3)-480*kappa*omega*h[2]*q**(2)*sigma**(3)-1024*omega*h[3]*q**(2)*sigma**(3)+960*kappa*omega*h[3]*q**(2)*sigma**(3)+80*omega*h[4]*q**(2)*sigma**(3)-80*kappa*omega*h[4]*q**(2)*sigma**(3)+64*omega*h[3]*q**(4)*sigma**(3)-40*omega*h[4]*q**(4)*sigma**(3)-400*omega*h[3]*sigma**(4)+400*kappa*omega*h[3]*sigma**(4)+60*omega*h[4]*sigma**(4)-60*kappa*omega*h[4]*sigma**(4)+400*omega*h[3]*q**(2)*sigma**(4)-80*kappa*omega*h[3]*q**(2)*sigma**(4)-64*omega*h[4]*q**(2)*sigma**(4)+60*kappa*omega*h[4]*q**(2)*sigma**(4)+4*omega*h[4]*q**(4)*sigma**(4)-20*omega*h[4]*sigma**(5)+20*kappa*omega*h[4]*sigma**(5)+20*omega*h[4]*q**(2)*sigma**(5)-4*kappa*omega*h[4]*q**(2)*sigma**(5)+kappa*h[2]*sigma**(2)*(-3120j)+h[2]*q**(2)*sigma**(2)*(-3120j)+h[2]*sigma**(3)*(-3000j)+kappa*h[3]*sigma**(3)*(-1360j)+h[3]*q**(2)*sigma**(3)*(-1360j)+kappa*sigma*h[2]*q**(2)*(-1320j)+h[3]*sigma**(4)*(-1000j)+kappa*h[3]*q**(2)*sigma**(2)*(-840j)+kappa*h[2]*q**(2)*sigma**(3)*(-600j)+kappa*h[4]*sigma**(4)*(-210j)+h[4]*q**(2)*sigma**(4)*(-210j)+kappa*h[3]*q**(2)*sigma**(4)*(-200j)+kappa*h[4]*q**(2)*sigma**(3)*(-170j)+h[4]*sigma**(5)*(-125j)+kappa*h[4]*q**(2)*sigma**(5)*(-25j)+kappa*h[5]*q**(2)*sigma**(4)*(-10j)+kappa*h[5]*sigma**(5)*(-10j)+h[5]*q**(2)*sigma**(5)*(-10j)+h[5]*sigma**(6)*(-5j)+kappa*h[5]*q**(2)*sigma**(6)*(-1j)+kappa*h[5]*sigma**(6)*(5j)+h[5]*q**(2)*sigma**(6)*(5j)-24*h[1]*(-1*(-1+q**(2))*(-40*omega*(-1+q**(2))+2*sigma*(8*omega*(-15+q**(2))+(-45j))+25*sigma**(2)*(8*omega+(5j)))+5*kappa*(-4*(4*omega+(1j))+6*sigma*(8*omega+(3j))-5*sigma**(2)*(8*omega+(5j))+q**(2)*(16*omega-6*sigma*(8*omega+(3j))+(4j)+sigma**(2)*(8*omega+(5j)))))+kappa*h[5]*sigma**(4)*(10j)+h[5]*sigma**(5)*(10j)+kappa*h[5]*q**(2)*sigma**(5)*(10j)+24*h[0]*(4*omega*q**(4)+5*(-1+kappa)*(-12*omega+20*omega*sigma+(-2j)+sigma*(5j))+q**(2)*(-64*omega+100*omega*sigma+(-10j)-5*kappa*(-12*omega+4*omega*sigma+(-2j)+sigma*(1j))+sigma*(25j)))+kappa*h[4]*sigma**(5)*(125j)+h[4]*q**(2)*sigma**(5)*(125j)+kappa*h[4]*sigma**(3)*(170j)+h[4]*sigma**(4)*(210j)+kappa*h[4]*q**(2)*sigma**(4)*(210j)+kappa*h[3]*sigma**(2)*(840j)+kappa*h[3]*sigma**(4)*(1000j)+h[3]*q**(2)*sigma**(4)*(1000j)+kappa*sigma*h[2]*(1320j)+h[3]*sigma**(3)*(1360j)+kappa*h[3]*q**(2)*sigma**(3)*(1360j)+kappa*h[2]*sigma**(3)*(3000j)+h[2]*q**(2)*sigma**(3)*(3000j)+h[2]*sigma**(2)*(3120j)+kappa*h[2]*q**(2)*sigma**(2)*(3120j))
	terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*(-4/21)*(-1+q**(2))*(2688*h[2]*omega**(2)-8064*sigma*h[2]*omega**(2)+24192*kappa*sigma*h[2]*omega**(2)+1792*sigma*h[3]*omega**(2)-5376*h[2]*omega**(2)*q**(2)+16128*sigma*h[2]*omega**(2)*q**(2)-24192*kappa*sigma*h[2]*omega**(2)*q**(2)-3584*sigma*h[3]*omega**(2)*q**(2)+2688*h[2]*omega**(2)*q**(4)-8064*sigma*h[2]*omega**(2)*q**(4)+1792*sigma*h[3]*omega**(2)*q**(4)-24864*h[2]*sigma**(2)+56448*h[2]*omega**(2)*sigma**(2)-40320*kappa*h[2]*omega**(2)*sigma**(2)-2688*h[3]*omega**(2)*sigma**(2)+8064*kappa*h[3]*omega**(2)*sigma**(2)+224*h[4]*omega**(2)*sigma**(2)+24864*h[2]*q**(2)*sigma**(2)-58752*h[2]*omega**(2)*q**(2)*sigma**(2)+40320*kappa*h[2]*omega**(2)*q**(2)*sigma**(2)+5376*h[3]*omega**(2)*q**(2)*sigma**(2)-8064*kappa*h[3]*omega**(2)*q**(2)*sigma**(2)-448*h[4]*omega**(2)*q**(2)*sigma**(2)+2304*h[2]*omega**(2)*q**(4)*sigma**(2)-2688*h[3]*omega**(2)*q**(4)*sigma**(2)+224*h[4]*omega**(2)*q**(4)*sigma**(2)+46200*h[2]*sigma**(3)-46200*kappa*h[2]*sigma**(3)-15792*h[3]*sigma**(3)-53760*h[2]*omega**(2)*sigma**(3)+53760*kappa*h[2]*omega**(2)*sigma**(3)+12544*h[3]*omega**(2)*sigma**(3)-8960*kappa*h[3]*omega**(2)*sigma**(3)-224*h[4]*omega**(2)*sigma**(3)+672*kappa*h[4]*omega**(2)*sigma**(3)-46200*h[2]*q**(2)*sigma**(3)+15792*h[3]*q**(2)*sigma**(3)+53760*h[2]*omega**(2)*q**(2)*sigma**(3)-7680*kappa*h[2]*omega**(2)*q**(2)*sigma**(3)-13056*h[3]*omega**(2)*q**(2)*sigma**(3)+8960*kappa*h[3]*omega**(2)*q**(2)*sigma**(3)+448*h[4]*omega**(2)*q**(2)*sigma**(3)-672*kappa*h[4]*omega**(2)*q**(2)*sigma**(3)+512*h[3]*omega**(2)*q**(4)*sigma**(3)-224*h[4]*omega**(2)*q**(4)*sigma**(3)-37800*h[2]*sigma**(4)+37800*kappa*h[2]*sigma**(4)+23800*h[3]*sigma**(4)-23800*kappa*h[3]*sigma**(4)-3962*h[4]*sigma**(4)+20160*h[2]*omega**(2)*sigma**(4)-20160*kappa*h[2]*omega**(2)*sigma**(4)-8960*h[3]*omega**(2)*sigma**(4)+8960*kappa*h[3]*omega**(2)*sigma**(4)+784*h[4]*omega**(2)*sigma**(4)-560*kappa*h[4]*omega**(2)*sigma**(4)+10800*h[2]*q**(2)*sigma**(4)-23800*h[3]*q**(2)*sigma**(4)+3962*h[4]*q**(2)*sigma**(4)-5760*h[2]*omega**(2)*q**(2)*sigma**(4)+8960*h[3]*omega**(2)*q**(2)*sigma**(4)-1280*kappa*h[3]*omega**(2)*q**(2)*sigma**(4)-816*h[4]*omega**(2)*q**(2)*sigma**(4)+560*kappa*h[4]*omega**(2)*q**(2)*sigma**(4)+32*h[4]*omega**(2)*q**(4)*sigma**(4)-16800*h[3]*sigma**(5)+16800*kappa*h[3]*sigma**(5)+5075*h[4]*sigma**(5)-5075*kappa*h[4]*sigma**(5)-406*h[5]*sigma**(5)+2688*h[3]*omega**(2)*sigma**(5)-2688*kappa*h[3]*omega**(2)*sigma**(5)-448*h[4]*omega**(2)*sigma**(5)+448*kappa*h[4]*omega**(2)*sigma**(5)+4800*h[3]*q**(2)*sigma**(5)-5075*h[4]*q**(2)*sigma**(5)+406*h[5]*q**(2)*sigma**(5)-768*h[3]*omega**(2)*q**(2)*sigma**(5)+448*h[4]*omega**(2)*q**(2)*sigma**(5)-64*kappa*h[4]*omega**(2)*q**(2)*sigma**(5)-3150*h[4]*sigma**(6)+3150*kappa*h[4]*sigma**(6)+455*h[5]*sigma**(6)-455*kappa*h[5]*sigma**(6)-14*h[6]*sigma**(6)+112*h[4]*omega**(2)*sigma**(6)-112*kappa*h[4]*omega**(2)*sigma**(6)+900*h[4]*q**(2)*sigma**(6)-455*h[5]*q**(2)*sigma**(6)+14*h[6]*q**(2)*sigma**(6)-32*h[4]*omega**(2)*q**(2)*sigma**(6)-252*h[5]*sigma**(7)+252*kappa*h[5]*sigma**(7)+14*h[6]*sigma**(7)-14*kappa*h[6]*sigma**(7)+72*h[5]*q**(2)*sigma**(7)-14*h[6]*q**(2)*sigma**(7)-7*h[6]*sigma**(8)+7*kappa*h[6]*sigma**(8)+2*h[6]*q**(2)*sigma**(8)+omega*h[2]*sigma**(3)*(-110880j)+omega*h[2]*q**(2)*sigma**(2)*(-79968j)+kappa*omega*h[2]*sigma**(4)*(-62160j)+kappa*omega*h[2]*sigma**(2)*(-36960j)+omega*h[3]*sigma**(4)*(-35280j)+omega*h[3]*q**(2)*sigma**(3)*(-32704j)+omega*h[2]*q**(2)*sigma**(4)*(-17760j)+kappa*omega*sigma*h[2]*q**(2)*(-16800j)+kappa*omega*h[3]*sigma**(5)*(-16128j)+kappa*omega*h[3]*sigma**(3)*(-15680j)+kappa*omega*h[3]*q**(2)*sigma**(2)*(-10080j)+kappa*omega*h[2]*q**(2)*sigma**(3)*(-9600j)+omega*h[4]*q**(2)*sigma**(4)*(-4844j)+omega*h[3]*q**(2)*sigma**(5)*(-4608j)+omega*h[4]*sigma**(5)*(-4284j)+kappa*omega*h[3]*q**(2)*sigma**(4)*(-3200j)+kappa*omega*h[4]*sigma**(4)*(-2380j)+kappa*omega*h[4]*q**(2)*sigma**(3)*(-1960j)+kappa*omega*h[4]*sigma**(6)*(-1652j)+omega*h[4]*q**(2)*sigma**(6)*(-472j)+kappa*omega*h[4]*q**(2)*sigma**(5)*(-400j)+omega*h[5]*q**(2)*sigma**(5)*(-224j)+omega*h[5]*sigma**(6)*(-168j)+kappa*omega*h[5]*q**(2)*sigma**(4)*(-112j)+kappa*omega*h[5]*sigma**(5)*(-112j)+kappa*omega*h[5]*sigma**(7)*(-56j)+kappa*omega*h[5]*q**(2)*sigma**(6)*(-16j)+omega*h[5]*q**(2)*sigma**(7)*(-16j)+24*h[0]*(32*omega**(2)*q**(4)-7*(4*omega+(1j))*(4*omega*(-7+5*kappa)+(-6j)+30*sigma**(2)*(-1+kappa)*(2*omega+(1j))-5*sigma*(-1+kappa)*(16*omega+(5j)))+q**(2)*(42+16*omega**(2)*(-51+35*kappa)-5*sigma*(4*omega+(1j))*(16*omega*(-7+kappa)+(-35j))-60*sigma**(2)*(-1+8*omega**(2)+omega*(6j))+omega*(28j)*(-13+5*kappa)))+omega*h[5]*sigma**(7)*(56j)+kappa*omega*h[5]*sigma**(4)*(112j)+kappa*omega*h[5]*q**(2)*sigma**(5)*(112j)-24*h[1]*(56*omega*(-1+q**(2))*(4*omega*(-1+q**(2))+kappa*(12*omega+(5j)))+20*sigma**(3)*(-9+16*omega**(2)+omega*(26j))*(-7+7*kappa+2*q**(2))-2*sigma*(-1+q**(2))*(259+32*omega**(2)*(-49+35*kappa+2*q**(2))+omega*(56j)*(-23+10*kappa))+5*sigma**(2)*(q**(2)*(259+128*omega**(2)*(-7+kappa)+omega*(16j)*(-63+5*kappa))-7*(-1+kappa)*(-37+128*omega**(2)+omega*(144j))))+kappa*omega*h[5]*sigma**(6)*(168j)+omega*h[5]*q**(2)*sigma**(6)*(168j)+omega*h[5]*sigma**(5)*(224j)+omega*h[4]*sigma**(6)*(1652j)+kappa*omega*h[4]*sigma**(3)*(1960j)+kappa*omega*h[4]*q**(2)*sigma**(4)*(2380j)+kappa*omega*h[4]*sigma**(5)*(4284j)+omega*h[4]*q**(2)*sigma**(5)*(4284j)+omega*h[4]*sigma**(4)*(4844j)+kappa*omega*h[3]*sigma**(2)*(10080j)+kappa*omega*h[3]*q**(2)*sigma**(3)*(15680j)+omega*h[3]*sigma**(5)*(16128j)+kappa*omega*sigma*h[2]*(16800j)+omega*h[3]*sigma**(3)*(32704j)+kappa*omega*h[3]*sigma**(4)*(35280j)+omega*h[3]*q**(2)*sigma**(4)*(35280j)+kappa*omega*h[2]*q**(2)*sigma**(2)*(36960j)+omega*h[2]*sigma**(4)*(62160j)+omega*h[2]*sigma**(2)*(79968j)+kappa*omega*h[2]*sigma**(3)*(110880j)+omega*h[2]*q**(2)*sigma**(3)*(110880j))
	terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*(4/5)*(-1+q**(2))*(-24*h[0]*(-4*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(1j)))+sigma*(-24*h[1]*(-16*omega*(-1+q**(2))+5*kappa*sigma*(8*omega+(5j)))+sigma*(-24*h[2]*(-12*omega*(-1+q**(2))+5*kappa*sigma*(4*omega+(5j)))+sigma*(kappa*h[5]*sigma**(3)*(-1j)-8*h[3]*(-8*omega*(-1+q**(2))+5*kappa*sigma*(2*omega+(5j)))-1*sigma*h[4]*(-4*omega*(-1+q**(2))+kappa*sigma*(4*omega+(25j)))))))
	terms[4] = np.sqrt(np.pi)*q**(4)*(4/105)*(-1+q**(2))*(-48*h[0]*(-8*omega**(2)*(-1+q**(2))+20*kappa*omega*sigma*(4*omega+(1j))+15*sigma**(2)*(-1+8*omega**(2)+omega*(6j)))+sigma*(-96*h[1]*(-16*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(8*omega+(5j))+5*sigma**(2)*(-9+16*omega**(2)+omega*(26j)))+sigma*(-24*h[2]*(-48*omega**(2)*(-1+q**(2))+40*kappa*omega*sigma*(4*omega+(5j))+5*sigma**(2)*(-45+24*omega**(2)+omega*(74j)))+sigma*(h[6]*sigma**(5)+4*h[5]*sigma**(3)*(kappa*omega*(-2j)+sigma*(9+omega*(-2j)))-32*h[3]*(-8*omega**(2)*(-1+q**(2))+10*kappa*omega*sigma*(2*omega+(5j))+3*sigma**(2)*(-25+4*omega**(2)+omega*(24j)))-2*sigma*h[4]*(-8*omega**(2)*(-1+q**(2))+4*kappa*omega*sigma*(4*omega+(25j))+sigma**(2)*(-225+8*omega**(2)+omega*(118j)))))))

	return terms

def metric_ORG_11_nodagger_couplings(lmax, m):
	dlmax = 4
	nterm = 9
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 1)*muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 0, 2, 0, -2, l1, m, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[5, i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[6, i, i + dl] = C3Product(l, m, 0, 3, 0, -2, l1, m, 2)
				coupling[7, i, i + dl] = C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*muCoupling(l1, 2)
				coupling[8, i, i + dl] = C3Product(l, m, 0, 4, 0, -2, l1, m, 2)

	return coupling

def metric_ORG_11_dagger_couplings(lmax, m):
	dlmax = 4
	nterm = 9
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 1)*muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 1)*muCoupling(l1, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 2, 0, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 3, 0, 2)
				coupling[5, i, i + dl] = C3Product(l, m, 0, l1, m, -2, 4, 0, 2)
				coupling[6, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*muCoupling(l1, 2)
				coupling[7, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*muCoupling(l1, 2)
				coupling[8, i, i + dl] = C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*muCoupling(l1, 2)

	return coupling

def metric_ORG_13_couplings(lmax, m):
	dlmax = 5
	nterm = 8
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 1, 1, 0, -1, l1, m, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 1, 2, 0, -1, l1, m, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*muCoupling(l1, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 1, 3, 0, -1, l1, m, 2)
				coupling[5, i, i + dl] = C3Product(l, m, 1, 4, 0, -1, l1, m, 2)
				coupling[6, i, i + dl] = C3Product(l, m, 1, 4, 0, 0, l1, m, 1)*muCoupling(l1, 2)
				coupling[7, i, i + dl] = C3Product(l, m, 1, 5, 0, -1, l1, m, 2)

	return coupling

def metric_ORG_14_couplings(lmax, m):
	dlmax = 5
	nterm = 8
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = muCoupling(l, 2)
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 1, 0, 1)
				coupling[2, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 2, 0, 1)
				coupling[3, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 3, 0, 1)
				coupling[4, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 4, 0, 1)
				coupling[5, i, i + dl] = C3Product(l, m, -1, l1, m, -2, 5, 0, 1)
				coupling[6, i, i + dl] = C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*muCoupling(l1, 2)
				coupling[7, i, i + dl] = C3Product(l, m, -1, l1, m, -1, 4, 0, 0)*muCoupling(l1, 2)

	return coupling

def metric_ORG_33_couplings(lmax, m):
	dlmax = 4
	nterm = 5
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = 1
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, 2, 1, 0, 0, l1, m, 2)
				coupling[2, i, i + dl] = C3Product(l, m, 2, 2, 0, 0, l1, m, 2)
				coupling[3, i, i + dl] = C3Product(l, m, 2, 3, 0, 0, l1, m, 2)
				coupling[4, i, i + dl] = C3Product(l, m, 2, 4, 0, 0, l1, m, 2)

	return coupling

def metric_ORG_44_couplings(lmax, m):
	dlmax = 4
	nterm = 5
	lmin = np.abs(m)
	coupling = np.zeros((nterm, lmax-lmin+1, lmax-lmin+1))

	for i, l in enumerate(range(lmin, lmax + 1)):
		coupling[0, i, i] = 1
		for dl in range(-dlmax, dlmax + 1):
			l1 = l + dl
			if l1 >= lmin and l1 <= lmax:
				coupling[1, i, i + dl] = C3Product(l, m, -2, l1, m, -2, 1, 0, 0)
				coupling[2, i, i + dl] = C3Product(l, m, -2, l1, m, -2, 2, 0, 0)
				coupling[3, i, i + dl] = C3Product(l, m, -2, l1, m, -2, 3, 0, 0)
				coupling[4, i, i + dl] = C3Product(l, m, -2, l1, m, -2, 4, 0, 0)

	return coupling

def metric_ORG_11_nodagger_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_nodagger_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_nodagger_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_nodagger_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_nodagger_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_nodagger_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_nodagger_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_nodagger_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_nodagger_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_nodagger_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_nodagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_dagger_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_dagger_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_dagger_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_dagger_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_dagger_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_dagger_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_dagger_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_dagger_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_11_dagger_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_11_dagger_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_11_dagger_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_13_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_13_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_13_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_13_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_13_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_13_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_13_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_13_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_13_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_13_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_13_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_13_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_13_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_13_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_13_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_14_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_14_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_14_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_14_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_14_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_14_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_14_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_14_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_14_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_14_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_14_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_14_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_14_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_14_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_14_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_33_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_33_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_33_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_33_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_33_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_33_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_33_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_33_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_33_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_33_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_33_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_44_modes(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_44_radial(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_44_modes_deriv1(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_44_radial_deriv1(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_44_modes_deriv2(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_44_radial_deriv2(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_44_modes_deriv3(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_44_radial_deriv3(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_44_modes_deriv4(kappa, sigma, m, omega, h):
	lmin = np.abs(m)
	lmax = lmin + h[0].shape[0] - 1
	habmats = metric_ORG_44_radial_deriv4(kappa, sigma, m, omega, h)
	Cmats = metric_ORG_44_couplings(lmax, m)
	hab = Cmats[0] @ habmats[0]
	for habi, Ci in zip(habmats[1:], Cmats[1:]):
		hab += Ci @ habi

	return hab

def metric_ORG_ab_dagger_modes_deriv0(kappa, sigma, m, omega, h):
	h11_dagger = metric_ORG_11_dagger_modes(kappa, sigma, m, omega, h)
	h14 = metric_ORG_14_modes(kappa, sigma, m, omega, h)
	h44 = metric_ORG_44_modes(kappa, sigma, m, omega, h)
	return [h11_dagger, h14, h44]

def metric_ORG_ab_dagger_modes_deriv1(kappa, sigma, m, omega, h):
	h11_dagger = metric_ORG_11_dagger_modes_deriv1(kappa, sigma, m, omega, h)
	h14 = metric_ORG_14_modes_deriv1(kappa, sigma, m, omega, h)
	h44 = metric_ORG_44_modes_deriv1(kappa, sigma, m, omega, h)
	return [h11_dagger, h14, h44]

def metric_ORG_ab_dagger_modes_deriv2(kappa, sigma, m, omega, h):
	h11_dagger = metric_ORG_11_dagger_modes_deriv2(kappa, sigma, m, omega, h)
	h14 = metric_ORG_14_modes_deriv2(kappa, sigma, m, omega, h)
	h44 = metric_ORG_44_modes_deriv2(kappa, sigma, m, omega, h)
	return [h11_dagger, h14, h44]

def metric_ORG_ab_dagger_modes_deriv3(kappa, sigma, m, omega, h):
	h11_dagger = metric_ORG_11_dagger_modes_deriv3(kappa, sigma, m, omega, h)
	h14 = metric_ORG_14_modes_deriv3(kappa, sigma, m, omega, h)
	h44 = metric_ORG_44_modes_deriv3(kappa, sigma, m, omega, h)
	return [h11_dagger, h14, h44]

def metric_ORG_ab_dagger_modes_deriv4(kappa, sigma, m, omega, h):
	h11_dagger = metric_ORG_11_dagger_modes_deriv4(kappa, sigma, m, omega, h)
	h14 = metric_ORG_14_modes_deriv4(kappa, sigma, m, omega, h)
	h44 = metric_ORG_44_modes_deriv4(kappa, sigma, m, omega, h)
	return [h11_dagger, h14, h44]

def metric_ORG_ab_nodagger_modes_deriv0(kappa, sigma, m, omega, h):
	h11_nodagger = metric_ORG_11_nodagger_modes(kappa, sigma, m, omega, h)
	h13 = metric_ORG_13_modes(kappa, sigma, m, omega, h)
	h33 = metric_ORG_33_modes(kappa, sigma, m, omega, h)
	return [h11_nodagger, h13, h33]

def metric_ORG_ab_nodagger_modes_deriv1(kappa, sigma, m, omega, h):
	h11_nodagger = metric_ORG_11_nodagger_modes_deriv1(kappa, sigma, m, omega, h)
	h13 = metric_ORG_13_modes_deriv1(kappa, sigma, m, omega, h)
	h33 = metric_ORG_33_modes_deriv1(kappa, sigma, m, omega, h)
	return [h11_nodagger, h13, h33]

def metric_ORG_ab_nodagger_modes_deriv2(kappa, sigma, m, omega, h):
	h11_nodagger = metric_ORG_11_nodagger_modes_deriv2(kappa, sigma, m, omega, h)
	h13 = metric_ORG_13_modes_deriv2(kappa, sigma, m, omega, h)
	h33 = metric_ORG_33_modes_deriv2(kappa, sigma, m, omega, h)
	return [h11_nodagger, h13, h33]

def metric_ORG_ab_nodagger_modes_deriv3(kappa, sigma, m, omega, h):
	h11_nodagger = metric_ORG_11_nodagger_modes_deriv3(kappa, sigma, m, omega, h)
	h13 = metric_ORG_13_modes_deriv3(kappa, sigma, m, omega, h)
	h33 = metric_ORG_33_modes_deriv3(kappa, sigma, m, omega, h)
	return [h11_nodagger, h13, h33]

def metric_ORG_ab_nodagger_modes_deriv4(kappa, sigma, m, omega, h):
	h11_nodagger = metric_ORG_11_nodagger_modes_deriv4(kappa, sigma, m, omega, h)
	h13 = metric_ORG_13_modes_deriv4(kappa, sigma, m, omega, h)
	h33 = metric_ORG_33_modes_deriv4(kappa, sigma, m, omega, h)
	return [h11_nodagger, h13, h33]

def metric_ORG_ab_modes_deriv0(kappa, sigma, m, omega, h, parity_func = None):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    if parity_func is None:
        parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    else:
        parity = parity_func(np.arange(lmin, lmax + 1), m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h11_nodagger, h13, h33 = metric_ORG_ab_nodagger_modes(kappa, sigma, m, omega, h)
    h11_dagger, h14, h44 = metric_ORG_ab_dagger_modes(kappa, sigma, m, omega, hd)
    return [h11_dagger + h11_nodagger, h13, h14, h33, h44]

def metric_ORG_ab_modes_deriv1(kappa, sigma, m, omega, h, parity_func = None):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    if parity_func is None:
        parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    else:
        parity = parity_func(np.arange(lmin, lmax + 1), m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h11_nodagger, h13, h33 = metric_ORG_ab_nodagger_modes_deriv1(kappa, sigma, m, omega, h)
    h11_dagger, h14, h44 = metric_ORG_ab_dagger_modes_deriv1(kappa, sigma, m, omega, hd)
    return [h11_dagger + h11_nodagger, h13, h14, h33, h44]

def metric_ORG_ab_modes_deriv2(kappa, sigma, m, omega, h, parity_func = None):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    if parity_func is None:
        parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    else:
        parity = parity_func(np.arange(lmin, lmax + 1), m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h11_nodagger, h13, h33 = metric_ORG_ab_nodagger_modes_deriv2(kappa, sigma, m, omega, h)
    h11_dagger, h14, h44 = metric_ORG_ab_dagger_modes_deriv2(kappa, sigma, m, omega, hd)
    return [h11_dagger + h11_nodagger, h13, h14, h33, h44]

def metric_ORG_ab_modes_deriv3(kappa, sigma, m, omega, h, parity_func = None):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    if parity_func is None:
        parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    else:
        parity = parity_func(np.arange(lmin, lmax + 1), m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h11_nodagger, h13, h33 = metric_ORG_ab_nodagger_modes_deriv3(kappa, sigma, m, omega, h)
    h11_dagger, h14, h44 = metric_ORG_ab_dagger_modes_deriv3(kappa, sigma, m, omega, hd)
    return [h11_dagger + h11_nodagger, h13, h14, h33, h44]

def metric_ORG_ab_modes_deriv4(kappa, sigma, m, omega, h, parity_func = None):
    lmax = np.abs(m) + h[0].shape[0] - 1
    lmin = np.abs(m)
    if parity_func is None:
        parity = (-1)**(np.arange(lmin, lmax + 1) + m)
    else:
        parity = parity_func(np.arange(lmin, lmax + 1), m)
    hd = [(parity*hi.swapaxes(0,1)).swapaxes(0,1) for hi in h]
    h11_nodagger, h13, h33 = metric_ORG_ab_nodagger_modes_deriv4(kappa, sigma, m, omega, h)
    h11_dagger, h14, h44 = metric_ORG_ab_dagger_modes_deriv4(kappa, sigma, m, omega, hd)
    return [h11_dagger + h11_nodagger, h13, h14, h33, h44]

def metric_ORG_ab_nodagger_modes(kapppa, sigmma, m, omega, h):
	return [
		metric_ORG_ab_nodagger_modes_deriv0(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_nodagger_modes_deriv1(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_nodagger_modes_deriv2(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_nodagger_modes_deriv3(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_nodagger_modes_deriv4(kapppa, sigmma, m, omega, h)
    ]

def metric_ORG_ab_dagger_modes(kapppa, sigmma, m, omega, h):
	return [
		metric_ORG_ab_dagger_modes_deriv0(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_dagger_modes_deriv1(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_dagger_modes_deriv2(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_dagger_modes_deriv3(kapppa, sigmma, m, omega, h),
		metric_ORG_ab_dagger_modes_deriv4(kapppa, sigmma, m, omega, h)
    ]

def metric_ORG_ab_modes(kapppa, sigmma, m, omega, h, parity_func = None):
	return [
		metric_ORG_ab_modes_deriv0(kapppa, sigmma, m, omega, h, parity_func),
		metric_ORG_ab_modes_deriv1(kapppa, sigmma, m, omega, h, parity_func),
		metric_ORG_ab_modes_deriv2(kapppa, sigmma, m, omega, h, parity_func),
		metric_ORG_ab_modes_deriv3(kapppa, sigmma, m, omega, h, parity_func),
		metric_ORG_ab_modes_deriv4(kapppa, sigmma, m, omega, h, parity_func)
    ]

def huu_reg(geo):
    L = geo.constants[1]
    r0 = geo.p
    q = geo.a
    kappa = L**2 + q**2 + 2*q**2/r0
    k = kappa/(r0**2 + kappa)
    K = ellipk(k)
    return 4.*K/(np.pi*np.sqrt(kappa + r0**2))

def ut_circ(geo):
    v = 1./np.sqrt(geo.p)
    q = geo.a
    return (1. + q*v**3)/(np.sqrt(1 - 3*v**2 + 2.*q*v**3))

"""
Note that the notation is a bit confusing below. u2 refers to u^2 = u^\\alpha e^2_\\alpha.
Recall that e^2_\\alpha = - e_{1\\alpha} = - l_\\alpha. Thus u^2 = - u_1 = - u_1^\\alpha e_{1\\alpha}.
"""
def u2_circ_norm(geo):
    Omega = geo.frequencies[-1]
    q = geo.a
    return (1. - q*Omega)

def u4_circ_norm(geo):
    Omega = geo.frequencies[-1]
    q = geo.a
    r0 = geo.p
    rhobar = -1./(r0)
    return 1.j*rhobar*(q - (r0**2 + q**2)*Omega)/np.sqrt(2.)

def u1_circ_norm(geo):
    r0 = geo.p
    q = geo.a
    return 0.5*(r0**2 - 2.*r0 + q**2)/r0**2*u2_circ_norm(geo)

def u3_circ_norm(geo):
    Omega = geo.frequencies[-1]
    q = geo.a
    r0 = geo.p
    rho = -1./(r0)
    return -1.j*rho*(q - (r0**2 + q**2)*Omega)/np.sqrt(2.)

def huu_IRG(geo, h22, h24, h44):
    '''
	Function for calculating the circular redshift given a circular geodesic and the normalized Kinnersley-tetrad-projected metric perturbation components h22, h24, and h44 in IRG. The metric components are normalized to maintain regularity at the horizon and infinity.
	
	Parameters
	----------
	geo : MiniGeo
        An instance of the MiniGeo class
    h11 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors n^\\alpha n^\\beta
    h13 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors n^\\alpha mbar^\\beta
    h33 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors mbar^\\alpha mbar^\\beta

    Returns
    -------
    np.ndarray(float)
        The l-modes of the circular orbit redshift  
	'''
    kappa = np.sqrt(1-geo.a**2)
    sigma = np.array([sigma_r(geo.p, kappa)])
    z = 0.
    hab = hab_rescaled_IRG(kappa, sigma, z, h22, h24, h44)
    u2 = u2_circ_norm(geo)
    u4 = u4_circ_norm(geo)
    ut = ut_circ(geo)
    # We take the two times the real part of h13 and h33 to account for their complex conjugates
    huu = np.real(ut**2*(hab[0]*u2*u2 + 4.*hab[1]*u2*u4 + 2.*hab[2]*u4*u4))
    return huu

def huu_ORG(geo, h11, h13, h33):
    '''
	Function for calculation the circular redshift given a circular geodesic and the normalized Kinnersley-tetrad-projected metric perturbation components h11, h13, and h33 in ORG. The metric components are normalized to maintain regularity at the horizon and infinity.
	
	Parameters
	----------
	geo : MiniGeo
        An instance of the MiniGeo class
    h11 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors l^\\alpha l^\\beta
    h13 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors l^\\alpha m^\\beta
    h33 : np.ndarray(float)
        The l-modes of the normalized metric perturbation projected onto Kinnersley null vectors m^\\alpha m^\\beta

    Returns
    -------
    np.ndarray(float)
        The l-modes of the circular orbit redshift  
	'''
    kappa = np.sqrt(1-geo.a**2)
    sigma = np.array([sigma_r(geo.p, kappa)])
    z = 0.
	# The metric perturbation needs to be rescaled due to the normalizations of the metric components to maintain regularity at the horizon and infinity
    hab = hab_rescaled_ORG(kappa, sigma, z, h11, h13, h33)
	# Note that the projected four velocity components are normalized with respect to u^t
    u1 = u1_circ_norm(geo)
    u3 = u3_circ_norm(geo)
    ut = ut_circ(geo)
	# We take the two times the real part of h13 and h33 to account for their complex conjugates
    huu = np.real(ut**2*(hab[0]*u1*u1 + 4.*hab[1]*u1*u3 + 2.*hab[2]*u3*u3))
    return huu