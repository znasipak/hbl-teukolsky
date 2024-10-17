from spherical import Wigner3j
from teuk import sigma_r, r_sigma
import numpy as np
#from numba import njit
from collocode import multi_domain_chebyshev_nodes
from cheby import MultiDomainChebyshev, MultiGridMultiDomainChebyshev
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
    

from scipy.special import ellipk

def huu_reg(geo):
    L = geo.constants[1]
    r0 = geo.p
    q = geo.a
    kappa = L**2 + q**2 + 2*q**2/r0
    k = kappa/(r0**2 + kappa)
    K = ellipk(k)
    return 2.*K/(np.pi*np.sqrt(kappa + r0**2))

def ut_circ(geo):
    v = 1./np.sqrt(geo.p)
    q = geo.a
    return (1. + q*v**3)/(np.sqrt(1 - 3*v**2 + 2.*q*v**3))

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

from scipy.special import factorial
def spin_zero_to_s_coupling(s, j, l, m):
    if s > 0:
        return (-1.)**(m + s)*np.sqrt(4**s*factorial(s)**2*(2*l + 1)*(2*j + 1)/factorial(2*s))*Wigner3j(s, l, j, 0, m, -m)*Wigner3j(s, l, j, s, -s, 0)
    else:
        return (-1.)**(m)*np.sqrt(4**(-s)*factorial(-s)**2*(2*l + 1)*(2*j + 1)/factorial(-2*s))*Wigner3j(-s, l, j, 0, m, -m)*Wigner3j(-s, l, j, s, -s, 0)

def hab_rescaled(kappa, sigma, z, h22, h24, h44):
    scalings = metric_scalings_IRG(kappa, sigma)
    rho_factors = metric_rho_factors_IRG(kappa, sigma, z)
    rescaling = rho_factors*scalings
    return rescaling[1,1]*h22, rescaling[1,3]*h24, rescaling[3,3]*h44

def huu(geo, sigma, z, h22, h24, h44):
    kappa = np.sqrt(1-geo.a**2)
    hab = hab_rescaled(kappa, sigma, z, h22, h24, h44)
    u2 = u2_circ_norm(geo)
    u4 = u4_circ_norm(geo)
    ut = ut_circ(geo)
    huu = np.real(0.5*ut**2*(hab[0]*u2*u2 + 4.*hab[1]*u2*u4 + 2.*hab[2]*u4*u4))
    return huu

def metric_IRG_22_components_nodagger(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = m*omega*h[0]*np.sqrt((2/15)*np.pi)*(1/7)*(-1+kappa**(2))*(2*m*omega*kappa**(2)*(7+sigma*(-7+2*sigma))+sigma**(2)*(3*m*omega+(-7j))-7*kappa*sigma*(-2+sigma)*(m*omega+(-1j)))
    term1 = m*omega*sigma*h[0]*np.sqrt((2/105)*np.pi)*q**(3)*(sigma+m*omega*(-1j)*(kappa*(-2+sigma)-sigma))
    term2 = h[0]*np.sqrt((2/5)*np.pi)*m**(2)*omega**(2)*sigma**(2)*(-1+kappa**(2))**(2)*(1/21)
    term3 = (-1/12)*h[0]*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))
    term4 = q*h[0]*np.sqrt((1/6)*np.pi)*(1/5)*(2*m*omega*kappa**(2)*(10-10*sigma+3*sigma**(2))+sigma**(2)*(4*m*omega+(-5j))-5*kappa*sigma*(-2+sigma)*(2*m*omega+(-1j)))
    term5 = sigma*h[0]*np.sqrt((1/30)*np.pi)*(-1+kappa**(2))*(sigma+m*omega*sigma*(-2j)*(-1+kappa)+kappa*m*omega*(4j))
    term6 = m*omega*q*h[0]*np.sqrt((1/21)*np.pi)*sigma**(2)*(2/5)*(-1+kappa**(2))
    term7 = q*sigma*h[0]*np.sqrt((1/3)*np.pi)*(-1j)/(2)*(kappa*(-2+sigma)-sigma)
    term8 = h[0]*np.sqrt((1/5)*np.pi)*sigma**(2)*(-1/6)*(-1+kappa**(2))
    return np.array([term0, term1, term2, term3, term4, term5, term6, term7, term8])

def metric_IRG_22_components_dagger(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = m*omega*h[0]*np.sqrt((2/15)*np.pi)*(1/7)*(-1+kappa**(2))*(2*m*omega*kappa**(2)*(7+sigma*(-7+2*sigma))+sigma**(2)*(3*m*omega+(-7j))-7*kappa*sigma*(-2+sigma)*(m*omega+(-1j)))
    term1 = m*omega*sigma*h[0]*np.sqrt((2/105)*np.pi)*q**(3)*(-sigma+m*omega*(1j)*(kappa*(-2+sigma)-sigma))
    term2 = h[0]*np.sqrt((2/5)*np.pi)*m**(2)*omega**(2)*sigma**(2)*(-1+kappa**(2))**(2)*(1/21)
    term3 = h[0]*(-1/12)*(-3*kappa*sigma*(-2+sigma)+sigma**(2)+2*kappa**(2)*(3-3*sigma+sigma**(2)))
    term4 = q*h[0]*np.sqrt((1/6)*np.pi)*(-1/5)*(2*m*omega*kappa**(2)*(10-10*sigma+3*sigma**(2))+sigma**(2)*(4*m*omega+(-5j))-5*kappa*sigma*(-2+sigma)*(2*m*omega+(-1j)))
    term5 = sigma*h[0]*np.sqrt((1/30)*np.pi)*(-1+kappa**(2))*(sigma+m*omega*sigma*(-2j)*(-1+kappa)+kappa*m*omega*(4j))
    term6 = m*omega*q*h[0]*np.sqrt((1/21)*np.pi)*sigma**(2)*(-2/5)*(-1+kappa**(2))
    term7 = q*sigma*h[0]*np.sqrt((1/3)*np.pi)*(1j)/(2)*(kappa*(-2+sigma)-sigma)
    term8 = h[0]*np.sqrt((1/5)*np.pi)*sigma**(2)*(-1/6)*(-1+kappa**(2))
    return np.array([term0, term1, term2, term3, term4, term5, term6, term7, term8])

def metric_IRG_22_couplings_nodagger(lmax, m):
    lmin = np.abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling5 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling6 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling7 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling8 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = muCoupling(l, 2)*muCoupling(l, 1)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 2, 0, 2)
                coupling1[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 3, 0, 2)
                coupling2[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 4, 0, 2)
                coupling4[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*muCoupling(l1, 2)
                coupling5[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*muCoupling(l1, 2)
                coupling6[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*muCoupling(l1, 2)
                coupling7[i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l1, 1)
                coupling8[i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l1, 1)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4, coupling5, coupling6, coupling7, coupling8])

def metric_IRG_22_couplings_dagger(lmax, m):
    lmin = np.abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling5 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling6 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling7 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling8 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = (-1.)**(l + m)*muCoupling(l, 2)*muCoupling(l, 1)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, -2, l1, m, 2)
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 3, 0, -2, l1, m, 2)
                coupling2[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)
                coupling4[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling5[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling6[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling7[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l1, 1)
                coupling8[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l1, 1)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4, coupling5, coupling6, coupling7, coupling8])

def metric_IRG_22_modes(kappa, sigma, m, omega, h):
    lmin = np.abs(m)
    lmax = lmin + h[0].shape[0] - 1
    print(lmin, lmax)
    habmats_nodagger = metric_IRG_22_components_nodagger(kappa, sigma, m, omega, h)
    habmats_dagger = metric_IRG_22_components_dagger(kappa, sigma, m, -np.conj(omega), h)
    Cmats_nodagger = metric_IRG_22_couplings_nodagger(lmax, m)
    print(Cmats_nodagger)
    Cmats_dagger = metric_IRG_22_couplings_dagger(lmax, m)
    hab = Cmats_nodagger[0] @ habmats_nodagger[0]
    for habi, Ci in zip(habmats_nodagger[1:], Cmats_nodagger[1:]):
        hab += Ci @ habi
    for habi, Ci in zip(habmats_dagger, Cmats_dagger):
        hab += Ci @ habi
    return hab

def metric_IRG_23_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = np.sqrt((1/15)*np.pi)*kappa**(-2)*sigma**(2)*(1/2)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term1 = m*omega*q*np.sqrt((1/42)*np.pi)*kappa**(-2)*sigma**(2)*(1/5)*(-kappa*sigma*h[1]*(-1+sigma)*(-1+kappa**(2))+h[0]*q**(2)*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term2 = m*omega*q*np.sqrt((1/3)*np.pi)*kappa**(-2)*(1/40)*(4*kappa*sigma*h[1]*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+2*kappa**(2)*(5+sigma*(-5+sigma))+3*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-24*omega+11*q+q**(3))+sigma*kappa**(2)*(m*q*(40+sigma*(-40+9*sigma))+8*m*omega*(-10+3*sigma**(2))+(-20j)*(4+sigma*(-3+sigma)))-4*kappa*sigma**(2)*(-4*m*omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+8*kappa**(3)*(-2*m*omega*sigma*(5+sigma*(-5+sigma))+(1j)*(-5+sigma*(5+sigma*(-3+sigma))))))
    term3 = 2**((-1/2))*kappa**(-2)*(1/48)*(4*kappa*sigma*h[1]*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+kappa**(2)*(6+sigma*(-6+sigma))+2*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-16*omega+7*q+q**(3))+sigma*kappa**(2)*(m*q*(24+sigma*(-24+5*sigma))+16*m*omega*(-3+sigma**(2))+(-12j)*(4+sigma*(-3+sigma)))+4*kappa**(3)*(-2*m*omega*sigma*(6+sigma*(-6+sigma))+(1j)*(-6+sigma*(6+sigma*(-3+sigma))))+4*kappa*sigma**(2)*(2*m*omega*(-6+sigma)-3*m*q*(-2+sigma)+(2j)*(-3+sigma))))
    term4 = np.sqrt((1/10)*np.pi)*kappa**(-2)*sigma**(2)*(-1/6)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    return np.array([term0, term1, term2, term3, term4])

def metric_IRG_23_couplings(lmax, m):
    lmin = abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = (-1.)**(l + m)*muCoupling(l, 2)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 2, 0, -1, l1, m, 2)
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 3, 0, -1, l1, m, 2)
                coupling2[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 1, 0, -1, l1, m, 2)
                coupling4[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*muCoupling(l1, 2)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_IRG_23_modes(kappa, sigma, m, omega, h):
    lmin = abs(m)
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_IRG_23_components(kappa, sigma, m, omega, h)
    Cmats = metric_IRG_23_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_IRG_24_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = np.sqrt((1/15)*np.pi)*kappa**(-2)*sigma**(2)*(-1/2)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term1 = m*omega*q*np.sqrt((1/42)*np.pi)*kappa**(-2)*sigma**(2)*(1/5)*(-kappa*sigma*h[1]*(-1+sigma)*(-1+kappa**(2))+h[0]*q**(2)*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term2 = m*omega*q*np.sqrt((1/3)*np.pi)*kappa**(-2)*(1/40)*(4*kappa*sigma*h[1]*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+2*kappa**(2)*(5+sigma*(-5+sigma))+3*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-24*omega+11*q+q**(3))+sigma*kappa**(2)*(m*q*(40+sigma*(-40+9*sigma))+8*m*omega*(-10+3*sigma**(2))+(-20j)*(4+sigma*(-3+sigma)))-4*kappa*sigma**(2)*(-4*m*omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+8*kappa**(3)*(-2*m*omega*sigma*(5+sigma*(-5+sigma))+(1j)*(-5+sigma*(5+sigma*(-3+sigma))))))
    term3 = 2**((-1/2))*kappa**(-2)*(1/48)*(-4*kappa*sigma*h[1]*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+kappa**(2)*(6+sigma*(-6+sigma))+2*sigma**(2))+h[0]*(-1j)*(m*sigma**(3)*(-16*omega+7*q+q**(3))+sigma*kappa**(2)*(m*q*(24+sigma*(-24+5*sigma))+16*m*omega*(-3+sigma**(2))+(-12j)*(4+sigma*(-3+sigma)))+4*kappa**(3)*(-2*m*omega*sigma*(6+sigma*(-6+sigma))+(1j)*(-6+sigma*(6+sigma*(-3+sigma))))+4*kappa*sigma**(2)*(2*m*omega*(-6+sigma)-3*m*q*(-2+sigma)+(2j)*(-3+sigma))))
    term4 = np.sqrt((1/10)*np.pi)*kappa**(-2)*sigma**(2)*(1/6)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    return np.array([term0, term1, term2, term3, term4])

def metric_IRG_24_couplings(lmax, m):
    lmin = abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = muCoupling(l, 2)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 2, 0, 1)
                coupling1[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 3, 0, 1)
                coupling2[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 1, 0, 1)
                coupling4[i, i + dl] = C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*muCoupling(l1, 2)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_IRG_24_modes(kappa, sigma, m, omega, h):
    lmin = abs(m)
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_IRG_24_components(kappa, sigma, m, omega, h)
    Cmats = metric_IRG_24_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_IRG_33_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = kappa**(-4)*(1/16)*(-sigma*h[2]*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-sigma)+h[0]*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*omega**(2))+m*omega*(-2j)*(4+sigma*(-2+sigma)))+kappa*m*sigma*(m*(-2+8*omega*q+sigma-4*omega**(2)*(2+sigma))+(-1j)*(-4+sigma)*(-2*omega+q))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*omega*q*(-2+sigma)+sigma+4*omega**(2)*(-4+sigma))+m*(1j)*(-4*omega*(2+sigma)+q*(4+sigma*(-2+sigma)))))+kappa*h[1]*(2j)*(-1+sigma)*(m*sigma**(2)*(-2*omega+q)+2*kappa**(2)*(m*omega*sigma*(-2+sigma)+(-1j))-kappa*sigma*(4*m*omega+m*q*(-2+sigma)+(2j))))
    term1 = q*np.sqrt((1/3)*np.pi)*kappa**(-4)*(1/8)*(h[2]*kappa**(2)*sigma**(2)*(-1+sigma)**(2)*(1j)+2*kappa*sigma*h[1]*(-1+sigma)*(2*m*omega*sigma*(1+kappa)-m*q*sigma+kappa*(2j))+h[0]*(1j)*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(2)*(6+sigma*(-4+m**(2)*(sigma-4*sigma*omega**(2))+m*omega*(2j)*(-4+sigma)))+kappa*m*sigma*(-2*omega+q)*(sigma*(4*m*omega+(-1j))+(4j))))
    return np.array([term0, term1])

def metric_IRG_33_couplings(lmax, m):
    dlMax = 5
    lmin = abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling0[i, i] = (-1.)**(l + m)
        for dl in range(-dlMax, dlMax+1):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 2, 1, 0, 0, l1, m, 2)

    return np.array([coupling0, coupling1])

def metric_IRG_33_modes(kappa, sigma, m, omega, h):
    lmin = abs(m)
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_IRG_33_components(kappa, sigma, m, omega, h)
    Cmats = metric_IRG_33_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_IRG_44_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = kappa**(-4)*(1/16)*(-sigma*h[2]*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-sigma)+h[0]*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*omega**(2))+m*omega*(-2j)*(4+sigma*(-2+sigma)))+kappa*m*sigma*(m*(-2+8*omega*q+sigma-4*omega**(2)*(2+sigma))+(-1j)*(-4+sigma)*(-2*omega+q))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*omega*q*(-2+sigma)+sigma+4*omega**(2)*(-4+sigma))+m*(1j)*(-4*omega*(2+sigma)+q*(4+sigma*(-2+sigma)))))+kappa*h[1]*(2j)*(-1+sigma)*(m*sigma**(2)*(-2*omega+q)+2*kappa**(2)*(m*omega*sigma*(-2+sigma)+(-1j))-kappa*sigma*(4*m*omega+m*q*(-2+sigma)+(2j))))
    term1 = q*np.sqrt((1/3)*np.pi)*kappa**(-4)*(1/8)*(2*kappa*sigma*h[1]*(-1+sigma)*(-2*m*omega*sigma*(1+kappa)+m*q*sigma+kappa*(-2j))+h[2]*kappa**(2)*sigma**(2)*(-1+sigma)**(2)*(-1j)+h[0]*(1j)*(m**(2)*sigma**(2)*(1+4*omega*(omega-q))+kappa**(2)*(-6+sigma*(4+sigma*m**(2)*(-1+4*omega**(2))+m*omega*(-2j)*(-4+sigma)))-kappa*m*sigma*(-2*omega+q)*(sigma*(4*m*omega+(-1j))+(4j))))
    return np.array([term0, term1])

def metric_IRG_44_couplings(lmax, m):
    dlMax = 5
    lmin = abs(m)
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling0[i, i] = 1.
        for dl in range(-dlMax, dlMax+1):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling1[i, i + dl] = C3Product(l, m, -2, l1, m, -2, 1, 0, 0)

    return np.array([coupling0, coupling1])

def metric_IRG_44_modes(kappa, sigma, m, omega, h):
    lmin = abs(m)
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_IRG_44_components(kappa, sigma, m, omega, h)
    Cmats = metric_IRG_44_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_IRG_ab_modes(kappa, sigma, m, omega, h):
    h22 = metric_IRG_22_modes(kappa, sigma, m, omega, h)
    h23 = metric_IRG_23_modes(kappa, sigma, m, omega, h)
    h24 = metric_IRG_24_modes(kappa, sigma, m, omega, h)
    h33 = metric_IRG_33_modes(kappa, sigma, m, omega, h)
    h44 = metric_IRG_44_modes(kappa, sigma, m, omega, h)
    return [h22, h23, h24, h33, h44]


'''
Metric perturbations in ORG

WARNING! THE FORMULAS BELOW ARE INCORRECT AND JUST COPIES OF THE IRG RESULTS
'''

def metric_ORG_11_components_nodagger(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 21
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = h[0]*(1/105)*(420*(-2+sigma*(-2+sigma)*(-1+kappa)*(1+sigma*(-1+sigma))*(3+sigma*(-3+sigma)))-35*q**(2)*(-72+sigma*(216+sigma*(-448+sigma*(536+sigma*(-354+sigma*(122-17*sigma))))+kappa*(-2+sigma)*(72+sigma*(-144+sigma*(136+sigma*(-64+11*sigma))))))+14*q**(4)*(-180+sigma*(540+2*kappa*(-2+sigma)*(45+sigma*(-90+sigma*(65+2*sigma*(-10+sigma))))+sigma*(-890+sigma*(880-7*sigma*(69+sigma*(-19+2*sigma))))))+8*q**(6)*(105+sigma*(-315+sigma*(385+sigma*(-7+sigma)*(35+sigma*(-7+sigma))))))
    terms[1] = q*h[0]*np.sqrt(np.pi*(2/3))*(2j)/(105)*(3*kappa*sigma*(-2+sigma)*(700*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+omega*(-280j)*(1+sigma*(-1+sigma))*(3+sigma*(-3+sigma))+2*q**(4)*(5*(70+sigma*(-140+sigma*(112+sigma*(-42+5*sigma))))+omega*(-2j)*(210+sigma*(-420+sigma*(308+sigma*(-98+11*sigma)))))+7*q**(2)*(-5*(40+sigma*(-80+sigma*(112+sigma*(-72+17*sigma))))+omega*(2j)*(120+sigma*(-240+sigma*(228+sigma*(-108+19*sigma))))))+(-1j)*(-1680*omega*kappa**(6)+5040*omega*sigma*kappa**(6)-168*kappa**(4)*sigma**(3)*(4*omega*(-25+6*q**(2))+(-125j))+84*kappa**(4)*sigma**(2)*(2*omega*(-75+37*q**(2))+(-125j))-12*kappa**(2)*sigma**(4)*(120*omega*q**(4)+350*(3*omega+(5j))-21*q**(2)*(49*omega+(65j)))+sigma**(6)*(20*omega*q**(6)-420*(2*omega+(5j))-6*q**(4)*(71*omega+(130j))+21*q**(2)*(58*omega+(135j)))+12*kappa**(2)*sigma**(5)*(420*omega+22*omega*q**(4)-7*q**(2)*(43*omega+(70j))+(875j))))
    terms[2] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(2j)/(35)*(700*sigma*kappa**(4)*(-1+kappa)-280*kappa**(5)+56*kappa**(2)*sigma**(2)*(-25*(-1+kappa)+q**(2)*(-25+11*kappa))-28*kappa**(2)*sigma**(3)*(-50*(-1+kappa)+q**(2)*(-33+8*kappa))+2*sigma**(4)*(-350*(-1+kappa)+7*q**(2)*(-66+41*kappa)-16*q**(4)*(-7+kappa))+sigma**(5)*(140*(-1+kappa)+7*q**(2)*(19-9*kappa)+2*q**(4)*(-8+kappa)))
    terms[3] = h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(4/231)*(-1848*kappa**(6)*omega**(2)-1848*omega*sigma*kappa**(4)*(3*omega*(kappa-1*kappa**(2))+kappa*(5j))+264*kappa**(2)*sigma**(3)*(17*omega**(2)*q**(4)+q**(2)*(98+omega**(2)*(-87+52*kappa)+omega*(5j)*(-35+18*kappa))-7*(-14+5*omega*(2*omega+(5j)))*(-1+kappa))-22*kappa**(2)*sigma**(4)*(74*omega**(2)*q**(4)+3*q**(2)*(336+3*omega**(2)*(-69+34*kappa)+omega*(10j)*(-54+19*kappa))-42*(-42+5*omega*(3*omega+(10j)))*(-1+kappa))+sigma**(6)*(24*omega**(2)*q**(6)-33*q**(2)*(238-140*kappa+omega**(2)*(-41+27*kappa)+omega*(5j)*(-37+23*kappa))+924*(-7+omega*(omega+(5j)))*(-1+kappa)+22*q**(4)*(70+omega**(2)*(-22+7*kappa)+omega*(15j)*(-5+kappa)))+132*kappa**(4)*sigma**(2)*(98+omega*(105*omega*(-1+kappa)+52*omega*q**(2)+(175j)*(-1+kappa)))+22*sigma**(5)*(-14*omega**(2)*q**(6)+3*q**(2)*(532-336*kappa+omega**(2)*(-145+103*kappa)+omega*(5j)*(-108+73*kappa))-42*(-28+omega*(6*omega+(25j)))*(-1+kappa)+q**(4)*(-420+omega*(197*omega-2*kappa*(37*omega+(75j))+(570j)))))
    terms[4] = sigma*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(2/21)*(omega*kappa**(5)*(-336j)-12*kappa**(2)*sigma**(3)*(35*(-1+kappa)*(9+omega*(-4j))+2*q**(2)*(75+omega*(1j)*(-48+13*kappa)))+420*sigma*kappa**(4)*(3+omega*(2j)*(-1+kappa))-24*kappa**(2)*sigma**(2)*(q**(2)*(-105+omega*(-2j)*(-35+16*kappa))+35*(-3+omega*(2j))*(-1+kappa))+sigma**(5)*(2*q**(4)*(25+omega*(2j)*(-7+kappa))+42*(-15+omega*(4j))*(-1+kappa)+3*q**(2)*(-195+30*kappa*(3+omega*(-1j))+omega*(58j)))+4*sigma**(4)*(210*(-1+kappa)*(3+omega*(-1j))+q**(4)*(-135+omega*(2j)*(39-7*kappa))+3*q**(2)*(255+omega*(-96j)+kappa*(-150+omega*(61j)))))
    terms[5] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(-2/21)*(28-14*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+q**(2)*(-56+sigma*(112+sigma*(-132+sigma*(76-17*sigma))+2*kappa*(-2+sigma)*(14+sigma*(-14+5*sigma))))+2*q**(4)*(14+sigma*(-28+sigma*(-6+sigma)*(-4+sigma))))
    terms[6] = sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(4j)/(33)*(kappa*(-2+sigma)*(132*kappa**(4)*omega**(2)-264*sigma*kappa**(4)*omega**(2)+44*kappa**(2)*sigma**(3)*(42+omega*(omega*(-9+q**(2))+(-45j)))-44*kappa**(2)*sigma**(2)*(42+omega*(4*omega*(-3+q**(2))+(-45j)))+sigma**(4)*(4*omega**(2)*q**(4)+66*(omega+(4j))*(2*omega+(7j))-11*q**(2)*(-56+omega*(7*omega+(40j)))))+sigma*(-660*omega*kappa**(4)*(omega+(3j))+1320*omega*sigma*kappa**(4)*(omega+(3j))-44*kappa**(2)*sigma**(3)*(126-15*omega*(omega+(6j))+omega*q**(2)*(6*omega+(20j)))+44*kappa**(2)*sigma**(2)*(126-15*omega*(2*omega+(9j))+omega*q**(2)*(21*omega+(65j)))+sigma**(4)*(-66*(omega+(4j))*(2*omega+(7j))-2*omega*q**(4)*(13*omega+(45j))+11*q**(2)*(-140+omega*(13*omega+(85j))))))
    terms[7] = h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(-8/165)*(omega*(66*(2-1*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+11*q**(2)*(-24+sigma*(48+sigma*(-56+sigma*(32-7*sigma))+4*kappa*(-2+sigma)*(3+sigma*(-3+sigma))))+2*q**(4)*(66+sigma*(-132+sigma*(110+sigma*(-44+5*sigma)))))+sigma*(165j)*(2*kappa*(-2+sigma)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))+sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma)))))
    terms[8] = h[0]*np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(-4j)/(45)*(-18*sigma*(-1+np.sqrt((1-1*q**(2))))*(3+sigma*(-3+sigma))+36*np.sqrt((1-1*q**(2)))+q**(2)*(-36*np.sqrt((1-1*q**(2)))+sigma*(54*(-1+np.sqrt((1-1*q**(2))))+sigma*(54-22*np.sqrt((1-1*q**(2)))+sigma*(-11+2*np.sqrt((1-1*q**(2))))))))
    terms[9] = h[0]*np.sqrt(np.pi*(2/5))*q**(4)*sigma**(2)*(-4/3003)*(572*kappa**(4)*omega**(2)-156*kappa**(2)*sigma**(2)*(-154+omega**(2)*(-11+11*kappa+6*q**(2))+omega*(-55j)*(-1+kappa))+sigma**(4)*(44*omega**(2)*q**(4)+13*q**(2)*(-504+omega**(2)*(-25+14*kappa)+omega*(-10j)*(-12+kappa))-286*(-1+kappa)*(42+omega*(omega+(-10j))))+1144*omega*sigma*kappa**(2)*(omega*(kappa-1*kappa**(2))+kappa*(-5j))+52*sigma**(3)*(-7*omega**(2)*q**(4)+11*(-1+kappa)*(42+omega*(2*omega+(-15j)))+q**(2)*(462+omega**(2)*(29-18*kappa)+omega*(15j)*(-11+4*kappa))))
    terms[10] = h[0]*np.sqrt(np.pi*(1/5))*q**(4)*sigma**(3)*(-8j)/(231)*(2*omega*(2*sigma*(11*(3+sigma*(-3+sigma))+q**(2)*(-33+sigma*(33-7*sigma)))+kappa*(-2+sigma)*(-22*(1+sigma*(-1+sigma))+q**(2)*(22+sigma*(-22+3*sigma))))+sigma*(-5j)*(-22+11*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(22-22*sigma+8*sigma**(2))))
    terms[11] = h[0]*np.sqrt(np.pi)*q**(4)*sigma**(4)*(4/1155)*(22-11*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-11+sigma*(11+sigma)))
    terms[12] = h[0]*np.sqrt(np.pi*(2/1155))*q**(5)*sigma**(3)*(-8j)/(39)*(kappa*(-2+sigma)*(2*omega**(2)*(-13*(1+sigma*(-1+sigma))+q**(2)*(13+sigma*(-13+2*sigma)))-182*sigma**(2)+omega*sigma**(2)*(-65j))+sigma*(26*omega*kappa**(2)*(3*omega+(5j))-26*omega*sigma*kappa**(2)*(3*omega+(5j))+sigma**(2)*(182+omega*(26*omega-1*q**(2)*(17*omega+(45j))+(65j)))))
    terms[13] = h[0]*np.sqrt(np.pi*(2/165))*q**(5)*sigma**(4)*(4/273)*(2*omega*(26-13*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(-26+sigma*(26+sigma)))+sigma*(-195j)*(kappa*(-2+sigma)-1*sigma))
    terms[14] = h[0]*np.sqrt(np.pi*(1/11))*q**(5)*sigma**(5)*(4j)/(63)*(kappa*(-2+sigma)-1*sigma)
    terms[15] = h[0]*np.sqrt(np.pi*(1/1365))*q**(6)*sigma**(4)*(16/33)*(omega**(2)*(2-1*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-1+sigma))+14*sigma**(2)+omega*sigma*(-15j)*(kappa*(-2+sigma)-1*sigma))
    terms[16] = h[0]*np.sqrt(np.pi*(2/273))*q**(6)*sigma**(5)*(4j)/(33)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(5j)))
    terms[17] = h[0]*np.sqrt(np.pi*(1/13))*q**(6)*sigma**(6)*(4/231)
    terms[18] = omega*h[0]*np.sqrt(np.pi*(1/35))*q**(7)*sigma**(5)*(16j)/(429)*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(5j)))
    terms[19] = omega*h[0]*np.sqrt(np.pi*(2/105))*q**(7)*sigma**(6)*(16/429)
    terms[20] = h[0]*np.sqrt(np.pi*(1/595))*omega**(2)*q**(8)*sigma**(6)*(16/429)

    return terms

def metric_ORG_11_components_dagger(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 21
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = h[0]*(1/105)*(420*(-2+sigma*(-2+sigma)*(-1+kappa)*(1+sigma*(-1+sigma))*(3+sigma*(-3+sigma)))-35*q**(2)*(-72+sigma*(216+sigma*(-448+sigma*(536+sigma*(-354+sigma*(122-17*sigma))))+kappa*(-2+sigma)*(72+sigma*(-144+sigma*(136+sigma*(-64+11*sigma))))))+14*q**(4)*(-180+sigma*(540+2*kappa*(-2+sigma)*(45+sigma*(-90+sigma*(65+2*sigma*(-10+sigma))))+sigma*(-890+sigma*(880-7*sigma*(69+sigma*(-19+2*sigma))))))+8*q**(6)*(105+sigma*(-315+sigma*(385+sigma*(-7+sigma)*(35+sigma*(-7+sigma))))))
    terms[1] = q*sigma*h[0]*np.sqrt(np.pi*(1/3))*(-2j)/(35)*(700*sigma*kappa**(4)*(-1+kappa)-280*kappa**(5)+56*kappa**(2)*sigma**(2)*(-25*(-1+kappa)+q**(2)*(-25+11*kappa))-28*kappa**(2)*sigma**(3)*(-50*(-1+kappa)+q**(2)*(-33+8*kappa))+2*sigma**(4)*(-350*(-1+kappa)+7*q**(2)*(-66+41*kappa)-16*q**(4)*(-7+kappa))+sigma**(5)*(140*(-1+kappa)+7*q**(2)*(19-9*kappa)+2*q**(4)*(-8+kappa)))
    terms[2] = h[0]*np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(-2/21)*(28-14*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+q**(2)*(-56+sigma*(112+sigma*(-132+sigma*(76-17*sigma))+2*kappa*(-2+sigma)*(14+sigma*(-14+5*sigma))))+2*q**(4)*(14+sigma*(-28+sigma*(-6+sigma)*(-4+sigma))))
    terms[3] = h[0]*np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(4j)/(45)*(-18*sigma*(-1+np.sqrt((1-1*q**(2))))*(3+sigma*(-3+sigma))+36*np.sqrt((1-1*q**(2)))+q**(2)*(-36*np.sqrt((1-1*q**(2)))+sigma*(54*(-1+np.sqrt((1-1*q**(2))))+sigma*(54-22*np.sqrt((1-1*q**(2)))+sigma*(-11+2*np.sqrt((1-1*q**(2))))))))
    terms[4] = h[0]*np.sqrt(np.pi)*q**(4)*sigma**(4)*(4/1155)*(22-11*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-11+sigma*(11+sigma)))
    terms[5] = h[0]*np.sqrt(np.pi*(1/11))*q**(5)*sigma**(5)*(-4j)/(63)*(kappa*(-2+sigma)-1*sigma)
    terms[6] = h[0]*np.sqrt(np.pi*(1/13))*q**(6)*sigma**(6)*(4/231)
    terms[7] = h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(4/231)*(-1848*kappa**(6)*omega**(2)-1848*omega*sigma*kappa**(4)*(3*omega*(kappa-1*kappa**(2))+kappa*(5j))+264*kappa**(2)*sigma**(3)*(17*omega**(2)*q**(4)+q**(2)*(98+omega**(2)*(-87+52*kappa)+omega*(5j)*(-35+18*kappa))-7*(-14+5*omega*(2*omega+(5j)))*(-1+kappa))-22*kappa**(2)*sigma**(4)*(74*omega**(2)*q**(4)+3*q**(2)*(336+3*omega**(2)*(-69+34*kappa)+omega*(10j)*(-54+19*kappa))-42*(-42+5*omega*(3*omega+(10j)))*(-1+kappa))+sigma**(6)*(24*omega**(2)*q**(6)-33*q**(2)*(238-140*kappa+omega**(2)*(-41+27*kappa)+omega*(5j)*(-37+23*kappa))+924*(-7+omega*(omega+(5j)))*(-1+kappa)+22*q**(4)*(70+omega**(2)*(-22+7*kappa)+omega*(15j)*(-5+kappa)))+132*kappa**(4)*sigma**(2)*(98+omega*(105*omega*(-1+kappa)+52*omega*q**(2)+(175j)*(-1+kappa)))+22*sigma**(5)*(-14*omega**(2)*q**(6)+3*q**(2)*(532-336*kappa+omega**(2)*(-145+103*kappa)+omega*(5j)*(-108+73*kappa))-42*(-28+omega*(6*omega+(25j)))*(-1+kappa)+q**(4)*(-420+omega*(197*omega-2*kappa*(37*omega+(75j))+(570j)))))
    terms[8] = sigma*h[0]*np.sqrt(np.pi*(2/105))*q**(3)*(-4j)/(33)*(kappa*(-2+sigma)*(132*kappa**(4)*omega**(2)-264*sigma*kappa**(4)*omega**(2)+44*kappa**(2)*sigma**(3)*(42+omega*(omega*(-9+q**(2))+(-45j)))-44*kappa**(2)*sigma**(2)*(42+omega*(4*omega*(-3+q**(2))+(-45j)))+sigma**(4)*(4*omega**(2)*q**(4)+66*(omega+(4j))*(2*omega+(7j))-11*q**(2)*(-56+omega*(7*omega+(40j)))))+sigma*(-660*omega*kappa**(4)*(omega+(3j))+1320*omega*sigma*kappa**(4)*(omega+(3j))-44*kappa**(2)*sigma**(3)*(126-15*omega*(omega+(6j))+omega*q**(2)*(6*omega+(20j)))+44*kappa**(2)*sigma**(2)*(126-15*omega*(2*omega+(9j))+omega*q**(2)*(21*omega+(65j)))+sigma**(4)*(-66*(omega+(4j))*(2*omega+(7j))-2*omega*q**(4)*(13*omega+(45j))+11*q**(2)*(-140+omega*(13*omega+(85j))))))
    terms[9] = h[0]*np.sqrt(np.pi*(2/5))*q**(4)*sigma**(2)*(-4/3003)*(572*kappa**(4)*omega**(2)-156*kappa**(2)*sigma**(2)*(-154+omega**(2)*(-11+11*kappa+6*q**(2))+omega*(-55j)*(-1+kappa))+sigma**(4)*(44*omega**(2)*q**(4)+13*q**(2)*(-504+omega**(2)*(-25+14*kappa)+omega*(-10j)*(-12+kappa))-286*(-1+kappa)*(42+omega*(omega+(-10j))))+1144*omega*sigma*kappa**(2)*(omega*(kappa-1*kappa**(2))+kappa*(-5j))+52*sigma**(3)*(-7*omega**(2)*q**(4)+11*(-1+kappa)*(42+omega*(2*omega+(-15j)))+q**(2)*(462+omega**(2)*(29-18*kappa)+omega*(15j)*(-11+4*kappa))))
    terms[10] = h[0]*np.sqrt(np.pi*(2/1155))*q**(5)*sigma**(3)*(8j)/(39)*(kappa*(-2+sigma)*(2*omega**(2)*(-13*(1+sigma*(-1+sigma))+q**(2)*(13+sigma*(-13+2*sigma)))-182*sigma**(2)+omega*sigma**(2)*(-65j))+sigma*(26*omega*kappa**(2)*(3*omega+(5j))-26*omega*sigma*kappa**(2)*(3*omega+(5j))+sigma**(2)*(182+omega*(26*omega-1*q**(2)*(17*omega+(45j))+(65j)))))
    terms[11] = h[0]*np.sqrt(np.pi*(1/1365))*q**(6)*sigma**(4)*(16/33)*(omega**(2)*(2-1*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-1+sigma))+14*sigma**(2)+omega*sigma*(-15j)*(kappa*(-2+sigma)-1*sigma))
    terms[12] = omega*h[0]*np.sqrt(np.pi*(1/35))*q**(7)*sigma**(5)*(16/429)*(sigma*(5+omega*(-1j)*(-1+kappa))+kappa*omega*(2j))
    terms[13] = h[0]*np.sqrt(np.pi*(1/595))*omega**(2)*q**(8)*sigma**(6)*(16/429)
    terms[14] = q*h[0]*np.sqrt(np.pi*(2/3))*(-2j)/(105)*(3*kappa*sigma*(-2+sigma)*(700*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+omega*(-280j)*(1+sigma*(-1+sigma))*(3+sigma*(-3+sigma))+2*q**(4)*(5*(70+sigma*(-140+sigma*(112+sigma*(-42+5*sigma))))+omega*(-2j)*(210+sigma*(-420+sigma*(308+sigma*(-98+11*sigma)))))+7*q**(2)*(-5*(40+sigma*(-80+sigma*(112+sigma*(-72+17*sigma))))+omega*(2j)*(120+sigma*(-240+sigma*(228+sigma*(-108+19*sigma))))))+(-1j)*(-1680*omega*kappa**(6)+5040*omega*sigma*kappa**(6)-168*kappa**(4)*sigma**(3)*(4*omega*(-25+6*q**(2))+(-125j))+84*kappa**(4)*sigma**(2)*(2*omega*(-75+37*q**(2))+(-125j))-12*kappa**(2)*sigma**(4)*(120*omega*q**(4)+350*(3*omega+(5j))-21*q**(2)*(49*omega+(65j)))+sigma**(6)*(20*omega*q**(6)-420*(2*omega+(5j))-6*q**(4)*(71*omega+(130j))+21*q**(2)*(58*omega+(135j)))+12*kappa**(2)*sigma**(5)*(420*omega+22*omega*q**(4)-7*q**(2)*(43*omega+(70j))+(875j))))
    terms[15] = sigma*h[0]*np.sqrt(np.pi*(2/15))*q**(2)*(2/21)*(omega*kappa**(5)*(-336j)-12*kappa**(2)*sigma**(3)*(35*(-1+kappa)*(9+omega*(-4j))+2*q**(2)*(75+omega*(1j)*(-48+13*kappa)))+420*sigma*kappa**(4)*(3+omega*(2j)*(-1+kappa))-24*kappa**(2)*sigma**(2)*(q**(2)*(-105+omega*(-2j)*(-35+16*kappa))+35*(-3+omega*(2j))*(-1+kappa))+sigma**(5)*(2*q**(4)*(25+omega*(2j)*(-7+kappa))+42*(-15+omega*(4j))*(-1+kappa)+3*q**(2)*(-195+30*kappa*(3+omega*(-1j))+omega*(58j)))+4*sigma**(4)*(210*(-1+kappa)*(3+omega*(-1j))+q**(4)*(-135+omega*(2j)*(39-7*kappa))+3*q**(2)*(255+omega*(-96j)+kappa*(-150+omega*(61j)))))
    terms[16] = h[0]*np.sqrt(np.pi*(1/21))*q**(3)*sigma**(2)*(8/165)*(omega*(66*(2-1*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+11*q**(2)*(-24+sigma*(48+sigma*(-56+sigma*(32-7*sigma))+4*kappa*(-2+sigma)*(3+sigma*(-3+sigma))))+2*q**(4)*(66+sigma*(-132+sigma*(110+sigma*(-44+5*sigma)))))+sigma*(165j)*(2*kappa*(-2+sigma)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))+sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma)))))
    terms[17] = h[0]*np.sqrt(np.pi*(1/5))*q**(4)*sigma**(3)*(-8j)/(231)*(2*omega*(2*sigma*(11*(3+sigma*(-3+sigma))+q**(2)*(-33+sigma*(33-7*sigma)))+kappa*(-2+sigma)*(-22*(1+sigma*(-1+sigma))+q**(2)*(22+sigma*(-22+3*sigma))))+sigma*(-5j)*(-22+11*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(22-22*sigma+8*sigma**(2))))
    terms[18] = h[0]*np.sqrt(np.pi*(2/165))*q**(5)*sigma**(4)*(-4/273)*(2*omega*(26-13*sigma*(-2+sigma)*(-1+kappa)+q**(2)*(-26+sigma*(26+sigma)))+sigma*(-195j)*(kappa*(-2+sigma)-1*sigma))
    terms[19] = h[0]*np.sqrt(np.pi*(2/273))*q**(6)*sigma**(5)*(4j)/(33)*(-4*kappa*omega+sigma*(2*omega*(-1+kappa)+(5j)))
    terms[20] = omega*h[0]*np.sqrt(np.pi*(2/105))*q**(7)*sigma**(6)*(-16/429)

    return terms

def metric_ORG_13_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 7
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = 2**((-1/2))*(1/12)*(-2*h[1]*sigma**(2)*(-1+sigma)*(-12-6*sigma*(-1+kappa)*(3+sigma*(-3+sigma))+q**(2)*(24+sigma*(-36+sigma*(26-7*sigma)+2*kappa*(-3+sigma)*(-3+2*sigma)))+q**(4)*(-2+sigma)*(6+sigma*(-6+sigma)))+h[0]*(1j)*(48*omega*kappa**(5)+6*kappa**(2)*sigma**(3)*(-40*omega*(-1+kappa)+3*m*q*(-1+kappa)+q**(2)*(-26*omega+6*kappa*omega+(-6j)))-24*sigma*kappa**(4)*(5*omega*(-1+kappa)+(-2j))+sigma**(5)*(6*m*q*(-1+kappa)-1*m*q**(3)*(-4+kappa)+2*q**(4)*(omega+(-2j))-24*(-1+kappa)*(omega+(-1j))+2*q**(2)*(-11*omega+kappa*(5*omega+(-8j))+(14j)))+2*sigma**(4)*(m*q*(9-9*kappa+q**(2)*(-9+4*kappa))+12*(-1+kappa)*(5*omega+(-2j))+2*q**(4)*(omega*(-9+kappa)+(3j))+q**(2)*(78*omega-48*kappa*omega+(-30j)+kappa*(16j)))-4*kappa**(2)*sigma**(2)*(-60*omega*(-1+kappa)+3*kappa*m*q+q**(2)*(omega*(-60+26*kappa)+(-24j))+(24j))))
    terms[1] = q*np.sqrt(np.pi*(1/3))*(1/10)*(2*kappa*omega*h[1]*sigma**(2)*(-1+sigma)*(sigma*(10*(3+sigma*(-3+sigma))+q**(2)*(-30+sigma*(30-7*sigma)))+2*kappa*(-2+sigma)*(-5*(1+sigma*(-1+sigma))+q**(2)*(5+sigma*(-5+sigma))))+kappa*h[1]*sigma**(3)*(-2j)*(-1+sigma)*(20*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(40+sigma*(-40+7*sigma)))+h[0]*omega**(2)*(2j)*(kappa*(-2+sigma)*(-20*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+q**(2)*(40+sigma*(-80+sigma*(104+sigma*(-64+9*sigma))))+4*q**(4)*(-1+sigma)*(5+sigma*(-5+sigma)))+sigma*(20*(5+sigma*(-10+sigma*(10+sigma*(-5+sigma))))+q**(2)*(-200+sigma*(400+sigma*(-332+sigma*(132-19*sigma))))+2*q**(4)*(50+sigma*(-100+sigma*(66+sigma*(-16+sigma))))))+omega*sigma*h[0]*(-1j)*(-10*m*q*sigma*(-1*sigma*(3+sigma*(-3+sigma))+kappa*(-2+sigma)*(1+sigma*(-1+sigma)))+m*sigma*q**(3)*(sigma*(-30+sigma*(30-7*sigma))+2*kappa*(-2+sigma)*(5+sigma*(-5+sigma)))+q**(2)*(2j)*(240+sigma*(-480+sigma*(364+sigma*(-124+3*sigma))+kappa*(-80+sigma*(80+7*sigma))*(-2+sigma)))+q**(4)*(8j)*(-6+sigma*(6+sigma))*(5+sigma*(-5+sigma))+(40j)*(-6+sigma*(12-1*sigma*(12+sigma*(-6+sigma))+kappa*(-2+sigma)**(3))))+h[0]*sigma**(2)*(kappa*(4j)*(-2+sigma)*(30-10*sigma*(3+2*sigma)+q**(2)*(-30+sigma*(30+7*sigma))+m*q*sigma**(2)*(5j))+sigma*(20*m*q*(2+sigma*(-2+sigma))+m*q**(3)*(-40+sigma*(40-7*sigma))+kappa**(2)*(40j)*(1+sigma*(-1+2*sigma)))))
    terms[2] = q*sigma*np.sqrt(np.pi*(1/6))*(1j)/(10)*(2*kappa*h[1]*sigma**(2)*(-1+sigma)*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+sigma)))+omega*h[0]*(2j)*(10*(-2+sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+q**(2)*(40+sigma*(-80-1*kappa*(-2+sigma)*(20+sigma*(-20+sigma))+2*sigma*(41+3*sigma*(-7+sigma))))+2*q**(4)*(-1+sigma)*(10+sigma*(-10+sigma)))+sigma*h[0]*(sigma*(-20*kappa**(2)*(3+sigma*(-3+sigma))+m*q*(-1j)*(-5*(2+sigma*(-2+sigma))+q**(2)*(10+sigma*(-10+sigma))))-1*kappa*(-2+sigma)*(-20*(1+sigma*(-1+sigma))+4*q**(2)*(5+sigma*(-5+sigma))+m*q*sigma**(2)*(5j))))
    terms[3] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(1/14)*(2*kappa*h[1]*sigma**(2)*(-1+sigma)*(7*sigma*(kappa*(-2+sigma)-1*sigma)+omega*(1j)*(7*(-2+sigma*(-2+sigma)*(-1+kappa))+2*q**(2)*(7+sigma*(-7+sigma))))+h[0]*(2*omega**(2)*(28-14*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))-4*q**(4)*(-1+sigma)*(7+sigma*(-7+sigma))+q**(2)*(-56+sigma*(112+2*kappa*(-2+sigma)*(14+sigma*(-14+sigma))+sigma*(-116+60*sigma-9*sigma**(2)))))+7*sigma**(2)*(40-8*sigma*(5+kappa)+4*q**(2)*(-10+sigma*(10+sigma))+4*sigma**(2)*(-1+kappa)+m*q*sigma*(-1j)*(kappa*(-2+sigma)-1*sigma))+omega*sigma*(-1j)*(168*sigma+sigma*(56*sigma*(-3+sigma)-42*q**(2)*(-2+sigma)**(2)+m*q*(1j)*(-7*(2+sigma*(-2+sigma))+2*q**(2)*(7+sigma*(-7+sigma))))+kappa*(-2+sigma)*(-56*(1+sigma*(-1+sigma))+8*q**(2)*(7+sigma*(-7+sigma))+m*q*sigma**(2)*(7j)))))
    terms[4] = np.sqrt(np.pi*(1/10))*q**(2)*sigma**(2)*(-1/6)*(2*kappa*h[1]*sigma**(2)*(-1+sigma)*(kappa*(-2+sigma)-1*sigma)+h[0]*(-1j)*(-2*omega*sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma)))-1*sigma**(3)*(q*(m+q*(-4j))+(4j))+kappa*(-2+sigma)*(-4*omega-4*omega*(-1+sigma)*(sigma+q**(2))+sigma**(2)*(m*q+(4j)))))
    terms[5] = np.sqrt(np.pi*(1/42))*q**(3)*sigma**(2)*(1/5)*(-2*kappa*h[1]*sigma**(2)*(-1+sigma)*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-6j)))+h[0]*(1j)*(kappa*(-2+sigma)*(-4*omega**(2)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))-24*sigma**(2)+omega*sigma**(2)*(m*q+(-8j)))+2*omega*sigma*(2*(3+sigma*(-3+sigma))*(omega+(2j))+q**(2)*(-1*omega*(6+sigma*(-6+sigma))+(2j)*(-6+sigma*(6+sigma))))-1*m*q*sigma**(3)*(omega+(6j))))
    terms[6] = np.sqrt(np.pi*(1/14))*q**(3)*sigma**(3)*(1/10)*(kappa*h[1]*sigma**(2)*(-2j)*(-1+sigma)+h[0]*(2*omega*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))-1*m*q*sigma**(2)-2*kappa*sigma*(-2+sigma)*(omega+(-2j))))
    terms[7] = omega*np.sqrt(np.pi*(1/10))*q**(4)*sigma**(3)*(1/21)*(kappa*h[1]*sigma**(2)*(-2j)*(-1+sigma)+h[0]*(2*omega*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))-1*m*q*sigma**(2)-2*kappa*sigma*(-2+sigma)*(omega+(-2j))))

    return terms

def metric_ORG_14_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 7
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = 2**((-1/2))*(1/12)*(2*h[1]*sigma**(2)*(-1+sigma)*(-12-6*sigma*(-1+kappa)*(3+sigma*(-3+sigma))+q**(2)*(24+sigma*(-36+sigma*(26-7*sigma)+2*kappa*(-3+sigma)*(-3+2*sigma)))+q**(4)*(-2+sigma)*(6+sigma*(-6+sigma)))+h[0]*(1j)*(-48*omega*kappa**(5)-6*kappa**(2)*sigma**(3)*(-40*omega*(-1+kappa)+3*m*q*(-1+kappa)+q**(2)*(-26*omega+6*kappa*omega+(-6j)))+24*sigma*kappa**(4)*(5*omega*(-1+kappa)+(-2j))+sigma**(5)*(24*omega*(-1+kappa)-2*omega*q**(2)*(-11+5*kappa+q**(2))+m*q*(6-6*kappa+q**(2)*(-4+kappa))+(4j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4)))-2*sigma**(4)*(m*q*(9-9*kappa+q**(2)*(-9+4*kappa))+12*(-1+kappa)*(5*omega+(-2j))+2*q**(4)*(omega*(-9+kappa)+(3j))+q**(2)*(78*omega-48*kappa*omega+(-30j)+kappa*(16j)))+4*kappa**(2)*sigma**(2)*(-60*omega*(-1+kappa)+3*kappa*m*q+q**(2)*(omega*(-60+26*kappa)+(-24j))+(24j))))
    terms[1] = q*np.sqrt(np.pi*(1/3))*(1/10)*(2*kappa*omega*h[1]*sigma**(2)*(-1+sigma)*(sigma*(10*(3+sigma*(-3+sigma))+q**(2)*(-30+sigma*(30-7*sigma)))+2*kappa*(-2+sigma)*(-5*(1+sigma*(-1+sigma))+q**(2)*(5+sigma*(-5+sigma))))+kappa*h[1]*sigma**(3)*(-2j)*(-1+sigma)*(20*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(40+sigma*(-40+7*sigma)))+h[0]*omega**(2)*(2j)*(kappa*(-2+sigma)*(-20*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+q**(2)*(40+sigma*(-80+sigma*(104+sigma*(-64+9*sigma))))+4*q**(4)*(-1+sigma)*(5+sigma*(-5+sigma)))+sigma*(20*(5+sigma*(-10+sigma*(10+sigma*(-5+sigma))))+q**(2)*(-200+sigma*(400+sigma*(-332+sigma*(132-19*sigma))))+2*q**(4)*(50+sigma*(-100+sigma*(66+sigma*(-16+sigma))))))+omega*sigma*h[0]*(-1j)*(-10*m*q*sigma*(-1*sigma*(3+sigma*(-3+sigma))+kappa*(-2+sigma)*(1+sigma*(-1+sigma)))+m*sigma*q**(3)*(sigma*(-30+sigma*(30-7*sigma))+2*kappa*(-2+sigma)*(5+sigma*(-5+sigma)))+q**(2)*(2j)*(240+sigma*(-480+sigma*(364+sigma*(-124+3*sigma))+kappa*(-80+sigma*(80+7*sigma))*(-2+sigma)))+q**(4)*(8j)*(-6+sigma*(6+sigma))*(5+sigma*(-5+sigma))+(40j)*(-6+sigma*(12-1*sigma*(12+sigma*(-6+sigma))+kappa*(-2+sigma)**(3))))+h[0]*sigma**(2)*(kappa*(4j)*(-2+sigma)*(30-10*sigma*(3+2*sigma)+q**(2)*(-30+sigma*(30+7*sigma))+m*q*sigma**(2)*(5j))+sigma*(20*m*q*(2+sigma*(-2+sigma))+m*q**(3)*(-40+sigma*(40-7*sigma))+kappa**(2)*(40j)*(1+sigma*(-1+2*sigma)))))
    terms[2] = sigma*np.sqrt(np.pi*(1/15))*q**(2)*(1/14)*(2*kappa*h[1]*sigma**(2)*(-1+sigma)*(7*sigma*(-1*kappa*(-2+sigma)+sigma)+omega*(-1j)*(7*(-2+sigma*(-2+sigma)*(-1+kappa))+2*q**(2)*(7+sigma*(-7+sigma))))+h[0]*(2*omega**(2)*(14*(-2+sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+4*q**(4)*(-1+sigma)*(7+sigma*(-7+sigma))+q**(2)*(56+sigma*(-112-2*kappa*(-2+sigma)*(14+sigma*(-14+sigma))+sigma*(116-60*sigma+9*sigma**(2)))))+7*sigma**(2)*(-40+4*sigma*(10-1*kappa*(-2+sigma)+sigma)-4*q**(2)*(-10+sigma*(10+sigma))+m*q*sigma*(1j)*(kappa*(-2+sigma)-1*sigma))+omega*sigma*(1j)*(168*sigma+sigma*(56*sigma*(-3+sigma)-42*q**(2)*(-2+sigma)**(2)+m*q*(1j)*(-7*(2+sigma*(-2+sigma))+2*q**(2)*(7+sigma*(-7+sigma))))+kappa*(-2+sigma)*(-56*(1+sigma*(-1+sigma))+8*q**(2)*(7+sigma*(-7+sigma))+m*q*sigma**(2)*(7j)))))
    terms[3] = np.sqrt(np.pi*(1/42))*q**(3)*sigma**(2)*(1/5)*(-2*kappa*h[1]*sigma**(2)*(-1+sigma)*(-2*kappa*omega+sigma*(omega*(-1+kappa)+(-6j)))+h[0]*(1j)*(kappa*(-2+sigma)*(-4*omega**(2)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))-24*sigma**(2)+omega*sigma**(2)*(m*q+(-8j)))+2*omega*sigma*(2*(3+sigma*(-3+sigma))*(omega+(2j))+q**(2)*(-1*omega*(6+sigma*(-6+sigma))+(2j)*(-6+sigma*(6+sigma))))-1*m*q*sigma**(3)*(omega+(6j))))
    terms[4] = omega*np.sqrt(np.pi*(1/10))*q**(4)*sigma**(3)*(1/21)*(h[0]*(-2*omega*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))+m*q*sigma**(2)+2*kappa*sigma*(-2+sigma)*(omega+(-2j)))+kappa*h[1]*sigma**(2)*(2j)*(-1+sigma))
    terms[5] = q*sigma*np.sqrt(np.pi*(1/6))*(-1/10)*(kappa*h[1]*sigma**(2)*(-2j)*(-1+sigma)*(5*(-2+sigma*(-2+sigma)*(-1+kappa))+q**(2)*(10+sigma*(-10+sigma)))+h[0]*(2*omega*(10*(-2+sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+q**(2)*(40+sigma*(-80-1*kappa*(-2+sigma)*(20+sigma*(-20+sigma))+2*sigma*(41+3*sigma*(-7+sigma))))+2*q**(4)*(-1+sigma)*(10+sigma*(-10+sigma)))+sigma*(kappa*(1j)*(-2+sigma)*(-20*(1+sigma*(-1+sigma))+4*q**(2)*(5+sigma*(-5+sigma))+m*q*sigma**(2)*(5j))+sigma*(m*q*(5*(2+sigma*(-2+sigma))-1*q**(2)*(10+sigma*(-10+sigma)))+kappa**(2)*(20j)*(3+sigma*(-3+sigma))))))
    terms[6] = np.sqrt(np.pi*(1/10))*q**(2)*sigma**(2)*(1/6)*(2*kappa*h[1]*sigma**(2)*(-1+sigma)*(kappa*(-2+sigma)-1*sigma)+h[0]*(-1j)*(-2*omega*sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma)))-1*sigma**(3)*(q*(m+q*(-4j))+(4j))+kappa*(-2+sigma)*(-4*omega-4*omega*(-1+sigma)*(sigma+q**(2))+sigma**(2)*(m*q+(4j)))))
    terms[7] = np.sqrt(np.pi*(1/14))*q**(3)*sigma**(3)*(1/10)*(kappa*h[1]*sigma**(2)*(-2j)*(-1+sigma)+h[0]*(2*omega*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))-1*m*q*sigma**(2)-2*kappa*sigma*(-2+sigma)*(omega+(-2j))))

    return terms

def metric_ORG_33_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 4
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = (1/24)*(-4*sigma**(2)*(-1+sigma)*(h[2]*sigma**(2)*(-1+q)*(-1+sigma)*(1+q)*(18*sigma-2*sigma*(-3+sigma)*(-3*sigma+q**(2)*(-3+2*sigma))+kappa*(-2+sigma)*(-6*(1+sigma*(-1+sigma))+q**(2)*(6+sigma*(-6+sigma))))+h[1]*(-1j)*(2*kappa*sigma*(12*omega*(5+sigma*(-10+sigma*(10+sigma*(-5+sigma))))-3*m*q*sigma**(2)*(3+sigma*(-3+sigma))+m*q**(3)*sigma**(2)*(-3+sigma)*(-3+2*sigma)+(-6j)*(-4+sigma*(9+sigma*(-3+sigma*(-1+sigma))))+q**(4)*(omega*(60+sigma*(-120+sigma*(78+sigma*(-18+sigma))))+(-1j)*(-4+sigma*(5+sigma))*(6+sigma*(-6+sigma)))+q**(2)*(omega*(-120+sigma*(240+sigma*(-198+sigma*(78-11*sigma))))+(1j)*(-48+sigma*(108+sigma*(-46+7*sigma*(-1+sigma))))))-1*kappa**(2)*(-2*omega*(-2+sigma)*(-12*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+q**(2)*(24+sigma*(-48+sigma*(62+sigma*(-38+5*sigma))))+2*q**(4)*(-1+sigma)*(6+sigma*(-6+sigma)))+sigma**(2)*(m*q*(-2+sigma)*(-6*(1+sigma*(-1+sigma))+q**(2)*(6+sigma*(-6+sigma)))+(4j)*(-3+3*sigma*(3+sigma-1*sigma**(2))+q**(2)*(3-9*sigma+2*sigma**(3)))))))+h[0]*(-192*kappa**(7)*omega**(2)+96*omega*sigma*kappa**(6)*(7*omega*(-1+kappa)+(-2j))-16*kappa**(4)*sigma**(3)*(24+omega*q**(2)*(35*omega*(-4+kappa)+(-26j))+3*m*q*(5*omega*(-1+kappa)+(-2j))-15*omega*(-1+kappa)*(14*omega+(1j)))+32*omega*kappa**(4)*sigma**(2)*(7*omega*(9-9*kappa+q**(2)*(-9+4*kappa))+(-3j)*(-6+kappa+6*q**(2)+kappa*m*q*(1j)))+sigma**(7)*(m**(2)*q**(2)*(6*(-1+kappa)-1*q**(2)*(-4+kappa))+4*m*omega*q*(-12*(-1+kappa)+q**(2)*(-11+5*kappa)+q**(4))+m*q*(-6j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4))+4*q**(4)*(8-2*kappa+omega**(2)*(-6+kappa)+omega*(-3j)*(-5+kappa))+48*(-1+kappa)*(-1+omega*(2*omega+(-3j)))+4*q**(2)*(-20+14*kappa+4*omega**(2)*(7-4*kappa)+omega*(3j)*(-17+11*kappa)))+2*sigma**(6)*(24*kappa**(2)*(1-1*kappa+q**(2)*(-3+kappa))+m**(2)*q**(2)*(9-9*kappa+q**(2)*(-9+4*kappa))+4*m*omega*q*(30*(-1+kappa)+q**(2)*(39-24*kappa)+q**(4)*(-9+kappa))-4*omega**(2)*(84*(-1+kappa)+q**(2)*(133-91*kappa)+q**(4)*(-51+16*kappa)+2*q**(6))+m*q*(2j)*(-15*(-1+kappa)+q**(2)*(-20+9*kappa)+5*q**(4))+omega*(4j)*(78*(-1+kappa)+q**(2)*(136-97*kappa)+q**(4)*(-62+25*kappa)+4*q**(6)))-4*kappa**(2)*sigma**(4)*(3*kappa*m**(2)*q**(2)-8*(24-9*kappa+4*q**(2)*(-6+kappa))+4*m*omega*q*(-30*(-1+kappa)+q**(2)*(-30+13*kappa))+20*omega**(2)*(42*(-1+kappa)+7*q**(2)*(8-5*kappa)+2*q**(4)*(-7+kappa))+m*q*(-6j)*(-8+kappa+8*q**(2))+omega*(8j)*(15+5*kappa*(-3+q**(2))-19*q**(2)+4*q**(4)))-2*kappa**(2)*sigma**(5)*(-9*m**(2)*q**(2)*(-1+kappa)+4*q**(2)*(-66+24*kappa+7*omega**(2)*(-35+17*kappa)+omega*(-60j)*(-2+kappa))-24*(-1+kappa)*(9+42*omega**(2)+omega*(-20j))-8*omega*q**(4)*(omega*(-15+kappa)+(12j))+2*m*q*(3*(-1+kappa)*(40*omega+(3j))+q**(2)*(78*omega-18*kappa*omega+(19j))))))
    terms[1] = q*sigma*np.sqrt(np.pi*(1/3))*(1/20)*(4*kappa*sigma**(2)*(-1+sigma)*(h[2]*sigma**(2)*(1j)*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))+5*sigma*kappa**(2)*(-2+sigma)+kappa*q**(2)*(10+sigma*(-10+sigma)))+h[1]*(2*omega*(20-10*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))+q**(2)*(-40+sigma*(80+kappa*(-2+sigma)*(20+sigma*(-20+sigma))-2*sigma*(41+3*sigma*(-7+sigma))))-2*q**(4)*(-1+sigma)*(10+sigma*(-10+sigma)))+sigma*(sigma*(-5*m*q*(2+sigma*(-2+sigma))+m*q**(3)*(10+sigma*(-10+sigma))+kappa**(2)*(-10j)*(-3+sigma)*(-2+sigma))+kappa*(5*m*q*sigma**(2)*(-2+sigma)+q**(2)*(-2j)*(-3+sigma)*(10+sigma*(-10+sigma))+(10j)*(-3+sigma)*(2+sigma*(-2+sigma))))))+h[0]*(-1j)*(-160*kappa**(6)*omega**(2)-160*omega*sigma*kappa**(4)*(3*omega*(kappa-1*kappa**(2))+kappa*(-4j))+sigma**(6)*(m**(2)*q**(2)*(-5+5*kappa+q**(2))+4*m*omega*q*(-10*(-1+kappa)+q**(2)*(-6+kappa))+m*q*(-6j)*(5-5*kappa+q**(2)*(-5+kappa))-4*q**(4)*(-2+omega*(omega+(-3j)))+40*(-1+kappa)*(-1+omega*(2*omega+(-3j)))+4*q**(2)*(2*(-6+5*kappa)+omega**(2)*(17-7*kappa)+omega*(3j)*(-11+6*kappa)))-2*kappa**(2)*sigma**(4)*(5*m**(2)*q**(2)+8*omega**(2)*q**(4)+4*q**(2)*(-64+3*omega**(2)*(-37+12*kappa)+omega*(4j)*(48-13*kappa))+40*(4+9*kappa-15*omega**(2)*(-1+kappa)+omega*(25j)*(-1+kappa))+2*m*q*(-60*omega+22*omega*q**(2)+15*kappa*(4*omega+(-3j))+(30j)))+16*kappa**(4)*sigma**(2)*(-40+omega*(-75*omega+5*m*q+31*omega*q**(2)+25*kappa*(3*omega+(-4j))+(75j)))-8*kappa**(2)*sigma**(3)*(-4*omega**(2)*(-50*(-1+kappa)+q**(2)*(-56+31*kappa)+6*q**(4))+5*(-4*(8+3*kappa)+32*q**(2)+kappa*m*q*(3j))-4*omega*(q*(5*m*(kappa-1*kappa**(2))+q*(3j)*(25-14*kappa))+(75j)*(-1+kappa)))+2*sigma**(5)*(-5*m**(2)*q**(2)*(kappa-1*kappa**(2))+80*(-1+kappa)*(2+omega*(-3*omega+(5j)))+4*m*q*(-1*omega*q**(4)+5*(-1+kappa)*(4*omega+(-3j))+q**(2)*(21*omega-11*kappa*omega+(-15j)+kappa*(9j)))-4*q**(4)*(16+omega*(-19*omega+2*kappa*(omega+(-2j))+(42j)))+4*q**(2)*(56+kappa*(-40+omega*(49*omega+(-92j)))+omega*(-79*omega+(142j))))))
    terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/12)*(4*kappa*sigma**(2)*(-1+sigma)*(h[2]*sigma**(2)*(-1+sigma)*(2+sigma*(-1+kappa)+q**(2)*(-2+sigma))+h[1]*(-1j)*(2*omega*(2*kappa*(-2+sigma)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))+sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma))))+sigma*(m*q*sigma*(-1*kappa*(-2+sigma)+sigma)+(-2j)*(4+sigma*(-5+kappa*(-3+sigma)-1*sigma))+q**(2)*(-2j)*(-4+sigma*(5+sigma)))))+h[0]*(kappa*(-2+sigma)*(16*kappa**(4)*omega**(2)-32*sigma*kappa**(4)*omega**(2)+8*kappa**(2)*sigma**(3)*(-14+omega*(-6*omega+m*q+(1j)))-8*kappa**(2)*sigma**(2)*(-14+omega*(m*q+2*omega*(-4+q**(2))+(1j)))+sigma**(4)*(m**(2)*q**(2)-4*(omega+(-1j))*(-4*omega+q**(2)*(omega+(-2j))+(2j))+2*m*q*(-4*omega+(3j))))+sigma*(-16*omega*kappa**(4)*(5*omega+(2j))+32*omega*sigma*kappa**(4)*(5*omega+(2j))+8*kappa**(2)*sigma**(2)*(6+m*q*(3*omega+(2j))+omega*(4*omega*(-5+3*q**(2))+(3j)))-1*sigma**(4)*(-8+4*omega**(2)*(4-3*q**(2))+omega*(-24j)+4*omega*q*(m*(-2+q**(2))+q*(6j))+q*(8*q+m*(q*(m+q*(-6j))+(6j))))-8*kappa**(2)*sigma**(3)*(6+m*q*(3*omega+(2j))+omega*(-10*omega+2*q**(2)*(omega+(-2j))+(7j))))))
    terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(1/20)*(4*kappa*sigma**(2)*(-1+sigma)*(h[2]*np.sqrt((1-1*q**(2)))*sigma**(2)*(-1j)*(-1+sigma)+h[1]*(2*omega*(2-1*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-1+sigma))+sigma*(-1*m*q*sigma+kappa*(2j)*(-3+sigma))))+h[0]*(1j)*(4*omega**(2)*(4-2*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))-1*q**(2)*(8+sigma*(-16+4*kappa*(-2+sigma)*(-1+sigma)+sigma*(-4+sigma)**(2)))+4*q**(4)*(-1+sigma)**(2))+sigma**(2)*(-8*(6+sigma*(-6+sigma))+q**(2)*(48+sigma*(-48+sigma*(8+m**(2))))+kappa*m*q*sigma*(-6j)*(-2+sigma))+omega*sigma*(4j)*(kappa*(-2+sigma)*(8+sigma*(-8+3*sigma)+8*q**(2)*(-1+sigma)+m*q*sigma**(2)*(-1j))+sigma*(-1*kappa**(2)*(14+sigma*(-14+3*sigma))+m*q*(1j)*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))))))

    return terms

def metric_ORG_44_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    nterm = 4
    terms = np.zeros((nterm,) + h[0].shape, dtype=np.complex128)

    terms[0] = (1/24)*(-4*sigma**(2)*(-1+sigma)*(-1*h[2]*kappa**(2)*sigma**(2)*(-1+sigma)*(18*sigma-2*sigma*(-3+sigma)*(-3*sigma+q**(2)*(-3+2*sigma))+kappa*(-2+sigma)*(-6*(1+sigma*(-1+sigma))+q**(2)*(6+sigma*(-6+sigma))))+h[1]*(-1j)*(2*kappa*sigma*(12*omega*(5+sigma*(-10+sigma*(10+sigma*(-5+sigma))))-3*m*q*sigma**(2)*(3+sigma*(-3+sigma))+m*q**(3)*sigma**(2)*(-3+sigma)*(-3+2*sigma)+(-6j)*(-4+sigma*(9+sigma*(-3+sigma*(-1+sigma))))+q**(4)*(omega*(60+sigma*(-120+sigma*(78+sigma*(-18+sigma))))+(-1j)*(-4+sigma*(5+sigma))*(6+sigma*(-6+sigma)))+q**(2)*(omega*(-120+sigma*(240+sigma*(-198+sigma*(78-11*sigma))))+(1j)*(-48+sigma*(108+sigma*(-46+7*sigma*(-1+sigma))))))-1*kappa**(2)*(-2*omega*(-2+sigma)*(-12*(1+sigma*(-1+sigma)*(2+sigma*(-2+sigma)))+q**(2)*(24+sigma*(-48+sigma*(62+sigma*(-38+5*sigma))))+2*q**(4)*(-1+sigma)*(6+sigma*(-6+sigma)))+sigma**(2)*(m*q*(-2+sigma)*(-6*(1+sigma*(-1+sigma))+q**(2)*(6+sigma*(-6+sigma)))+(4j)*(-3+3*sigma*(3+sigma-1*sigma**(2))+q**(2)*(3-9*sigma+2*sigma**(3)))))))+h[0]*(-192*kappa**(7)*omega**(2)+96*omega*sigma*kappa**(6)*(7*omega*(-1+kappa)+(-2j))-16*kappa**(4)*sigma**(3)*(24+omega*q**(2)*(35*omega*(-4+kappa)+(-26j))+3*m*q*(5*omega*(-1+kappa)+(-2j))-15*omega*(-1+kappa)*(14*omega+(1j)))+32*omega*kappa**(4)*sigma**(2)*(7*omega*(9-9*kappa+q**(2)*(-9+4*kappa))+(-3j)*(-6+kappa+6*q**(2)+kappa*m*q*(1j)))+sigma**(7)*(m**(2)*q**(2)*(6*(-1+kappa)-1*q**(2)*(-4+kappa))+4*m*omega*q*(-12*(-1+kappa)+q**(2)*(-11+5*kappa)+q**(4))+m*q*(-6j)*(6-6*kappa+q**(2)*(-7+4*kappa)+q**(4))+4*q**(4)*(8-2*kappa+omega**(2)*(-6+kappa)+omega*(-3j)*(-5+kappa))+48*(-1+kappa)*(-1+omega*(2*omega+(-3j)))+4*q**(2)*(-20+14*kappa+4*omega**(2)*(7-4*kappa)+omega*(3j)*(-17+11*kappa)))+2*sigma**(6)*(24*kappa**(2)*(1-1*kappa+q**(2)*(-3+kappa))+m**(2)*q**(2)*(9-9*kappa+q**(2)*(-9+4*kappa))+4*m*omega*q*(30*(-1+kappa)+q**(2)*(39-24*kappa)+q**(4)*(-9+kappa))-4*omega**(2)*(84*(-1+kappa)+q**(2)*(133-91*kappa)+q**(4)*(-51+16*kappa)+2*q**(6))+m*q*(2j)*(-15*(-1+kappa)+q**(2)*(-20+9*kappa)+5*q**(4))+omega*(4j)*(78*(-1+kappa)+q**(2)*(136-97*kappa)+q**(4)*(-62+25*kappa)+4*q**(6)))-4*kappa**(2)*sigma**(4)*(3*kappa*m**(2)*q**(2)-8*(24-9*kappa+4*q**(2)*(-6+kappa))+4*m*omega*q*(-30*(-1+kappa)+q**(2)*(-30+13*kappa))+20*omega**(2)*(42*(-1+kappa)+7*q**(2)*(8-5*kappa)+2*q**(4)*(-7+kappa))+m*q*(-6j)*(-8+kappa+8*q**(2))+omega*(8j)*(15+5*kappa*(-3+q**(2))-19*q**(2)+4*q**(4)))-2*kappa**(2)*sigma**(5)*(-9*m**(2)*q**(2)*(-1+kappa)+4*q**(2)*(-66+24*kappa+7*omega**(2)*(-35+17*kappa)+omega*(-60j)*(-2+kappa))-24*(-1+kappa)*(9+42*omega**(2)+omega*(-20j))-8*omega*q**(4)*(omega*(-15+kappa)+(12j))+2*m*q*(3*(-1+kappa)*(40*omega+(3j))+q**(2)*(78*omega-18*kappa*omega+(19j))))))
    terms[1] = q*sigma*np.sqrt(np.pi*(1/3))*(1/20)*(4*kappa*sigma**(2)*(-1+sigma)*(h[2]*sigma**(2)*(-1j)*(-1+sigma)*(-5*kappa*(2+sigma*(-2+sigma))+5*sigma*kappa**(2)*(-2+sigma)+kappa*q**(2)*(10+sigma*(-10+sigma)))+h[1]*(2*omega*(10*(-2+sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma)))+q**(2)*(40+sigma*(-80-1*kappa*(-2+sigma)*(20+sigma*(-20+sigma))+2*sigma*(41+3*sigma*(-7+sigma))))+2*q**(4)*(-1+sigma)*(10+sigma*(-10+sigma)))+sigma*(kappa*(-5*m*q*sigma**(2)*(-2+sigma)+(-10j)*(-3+sigma)*(2+sigma*(-2+sigma))+q**(2)*(2j)*(-3+sigma)*(10+sigma*(-10+sigma)))+sigma*(5*m*q*(2+sigma*(-2+sigma))-1*m*q**(3)*(10+sigma*(-10+sigma))+kappa**(2)*(10j)*(-3+sigma)*(-2+sigma)))))+h[0]*(1j)*(-160*kappa**(6)*omega**(2)-160*omega*sigma*kappa**(4)*(3*omega*(kappa-1*kappa**(2))+kappa*(-4j))+sigma**(6)*(m**(2)*q**(2)*(-5+5*kappa+q**(2))+4*m*omega*q*(-10*(-1+kappa)+q**(2)*(-6+kappa))+m*q*(-6j)*(5-5*kappa+q**(2)*(-5+kappa))-4*q**(4)*(-2+omega*(omega+(-3j)))+40*(-1+kappa)*(-1+omega*(2*omega+(-3j)))+4*q**(2)*(2*(-6+5*kappa)+omega**(2)*(17-7*kappa)+omega*(3j)*(-11+6*kappa)))-2*kappa**(2)*sigma**(4)*(5*m**(2)*q**(2)+8*omega**(2)*q**(4)+4*q**(2)*(-64+3*omega**(2)*(-37+12*kappa)+omega*(4j)*(48-13*kappa))+40*(4+9*kappa-15*omega**(2)*(-1+kappa)+omega*(25j)*(-1+kappa))+2*m*q*(-60*omega+22*omega*q**(2)+15*kappa*(4*omega+(-3j))+(30j)))+16*kappa**(4)*sigma**(2)*(-40+omega*(-75*omega+5*m*q+31*omega*q**(2)+25*kappa*(3*omega+(-4j))+(75j)))-8*kappa**(2)*sigma**(3)*(-4*omega**(2)*(-50*(-1+kappa)+q**(2)*(-56+31*kappa)+6*q**(4))+5*(-4*(8+3*kappa)+32*q**(2)+kappa*m*q*(3j))-4*omega*(q*(5*m*(kappa-1*kappa**(2))+q*(3j)*(25-14*kappa))+(75j)*(-1+kappa)))+2*sigma**(5)*(-5*m**(2)*q**(2)*(kappa-1*kappa**(2))+80*(-1+kappa)*(2+omega*(-3*omega+(5j)))+4*m*q*(-1*omega*q**(4)+5*(-1+kappa)*(4*omega+(-3j))+q**(2)*(21*omega-11*kappa*omega+(-15j)+kappa*(9j)))-4*q**(4)*(16+omega*(-19*omega+2*kappa*(omega+(-2j))+(42j)))+4*q**(2)*(56+kappa*(-40+omega*(49*omega+(-92j)))+omega*(-79*omega+(142j))))))
    terms[2] = np.sqrt(np.pi*(1/5))*q**(2)*sigma**(2)*(1/12)*(4*kappa*sigma**(2)*(-1+sigma)*(h[2]*sigma**(2)*(-1+sigma)*(2+sigma*(-1+kappa)+q**(2)*(-2+sigma))+h[1]*(-1j)*(2*omega*(2*kappa*(-2+sigma)*(1+sigma*(-1+sigma)+q**(2)*(-1+sigma))+sigma*(-2*(3+sigma*(-3+sigma))+q**(2)*(6+sigma*(-6+sigma))))+sigma*(m*q*sigma*(-1*kappa*(-2+sigma)+sigma)+(-2j)*(4+sigma*(-5+kappa*(-3+sigma)-1*sigma))+q**(2)*(-2j)*(-4+sigma*(5+sigma)))))+h[0]*(kappa*(-2+sigma)*(16*kappa**(4)*omega**(2)-32*sigma*kappa**(4)*omega**(2)+8*kappa**(2)*sigma**(3)*(-14+omega*(-6*omega+m*q+(1j)))-8*kappa**(2)*sigma**(2)*(-14+omega*(m*q+2*omega*(-4+q**(2))+(1j)))+sigma**(4)*(m**(2)*q**(2)-4*(omega+(-1j))*(-4*omega+q**(2)*(omega+(-2j))+(2j))+2*m*q*(-4*omega+(3j))))+sigma*(-16*omega*kappa**(4)*(5*omega+(2j))+32*omega*sigma*kappa**(4)*(5*omega+(2j))+8*kappa**(2)*sigma**(2)*(6+m*q*(3*omega+(2j))+omega*(4*omega*(-5+3*q**(2))+(3j)))-1*sigma**(4)*(-8+4*omega**(2)*(4-3*q**(2))+omega*(-24j)+4*omega*q*(m*(-2+q**(2))+q*(6j))+q*(8*q+m*(q*(m+q*(-6j))+(6j))))-8*kappa**(2)*sigma**(3)*(6+m*q*(3*omega+(2j))+omega*(-10*omega+2*q**(2)*(omega+(-2j))+(7j))))))
    terms[3] = np.sqrt(np.pi*(1/7))*q**(3)*sigma**(3)*(1/20)*(4*kappa*sigma**(2)*(-1+sigma)*(h[1]*(-2*omega*(2-1*sigma*(-2+sigma)*(-1+kappa)+2*q**(2)*(-1+sigma))+sigma*(m*q*sigma+kappa*(-2j)*(-3+sigma)))+h[2]*np.sqrt((1-1*q**(2)))*sigma**(2)*(1j)*(-1+sigma))+h[0]*(-1j)*(4*omega**(2)*(4-2*sigma*(-2+sigma)*(-1+kappa)*(2+sigma*(-2+sigma))-1*q**(2)*(8+sigma*(-16+4*kappa*(-2+sigma)*(-1+sigma)+sigma*(-4+sigma)**(2)))+4*q**(4)*(-1+sigma)**(2))+sigma**(2)*(-8*(6+sigma*(-6+sigma))+q**(2)*(48+sigma*(-48+sigma*(8+m**(2))))+kappa*m*q*sigma*(-6j)*(-2+sigma))+omega*sigma*(4j)*(kappa*(-2+sigma)*(8+sigma*(-8+3*sigma)+8*q**(2)*(-1+sigma)+m*q*sigma**(2)*(-1j))+sigma*(-1*kappa**(2)*(14+sigma*(-14+3*sigma))+m*q*(1j)*(2+sigma*(-2+sigma)+2*q**(2)*(-1+sigma))))))

    return terms

def metric_ORG_11_couplings_nodagger(lmax, m):
    lmin = np.max([0, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling5 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling6 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling7 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling8 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = muCoupling(l, 2)*muCoupling(l, 1)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 2, 0, 2)
                coupling1[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 3, 0, 2)
                coupling2[i, i + dl] = C3Product(l, m, 0, l1, m, -2, 4, 0, 2)
                coupling4[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 1, 0, 1)*muCoupling(l1, 2)
                coupling5[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 2, 0, 1)*muCoupling(l1, 2)
                coupling6[i, i + dl] = C3Product(l, m, 0, l1, m, -1, 3, 0, 1)*muCoupling(l1, 2)
                coupling7[i, i + dl] = C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l, 1)
                coupling8[i, i + dl] = C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l, 1)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_ORG_11_couplings_dagger(lmax, m):
    lmin = np.max([0, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling5 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling6 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling7 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling8 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = (-1.)**(l + m)*muCoupling(l, 2)*muCoupling(l, 1)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, -2, l1, m, 2)
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 3, 0, -2, l1, m, 2)
                coupling2[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 4, 0, -2, l1, m, 2)
                coupling4[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 1, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling5[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling6[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 3, 0, -1, l1, m, 1)*muCoupling(l1, 2)
                coupling7[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 1, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l, 1)
                coupling8[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 0, 2, 0, 0, l1, m, 0)*muCoupling(l1, 2)*muCoupling(l, 1)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_ORG_11_modes(kappa, sigma, m, omega, h):
    lmin = np.max([0, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats_nodagger = metric_ORG_11_components_nodagger(kappa, sigma, m, omega, h)
    habmats_dagger = metric_ORG_11_components_dagger(kappa, sigma, m, omega, h)
    Cmats_nodagger = metric_ORG_11_couplings_nodagger(lmax, m)
    Cmats_dagger = metric_ORG_11_couplings_dagger(lmax, m)
    hab = Cmats_nodagger[0] @ habmats_nodagger[0]
    for habi, Ci in zip(habmats_nodagger[1:], Cmats_nodagger[1:]):
        hab += Ci @ habi
    for habi, Ci in zip(habmats_dagger, Cmats_dagger):
        hab += Ci @ habi
    return hab

def metric_ORG_11_nodagger_modes(kappa, sigma, m, omega, h):
    lmin = np.max([0, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats_nodagger = metric_ORG_11_components_nodagger(kappa, sigma, m, omega, h)
    Cmats_nodagger = metric_ORG_11_couplings_nodagger(lmax, m)
    hab = Cmats_nodagger[0] @ habmats_nodagger[0]
    for habi, Ci in zip(habmats_nodagger[1:], Cmats_nodagger[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_11_dagger_modes(kappa, sigma, m, omega, h):
    lmin = np.max([0, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats_dagger = metric_ORG_11_components_dagger(kappa, sigma, m, omega, h)
    Cmats_dagger = metric_ORG_11_couplings_dagger(lmax, m)
    hab = Cmats_dagger[0] @ habmats_dagger[0]
    for habi, Ci in zip(habmats_dagger[1:], Cmats_dagger[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_13_couplings(lmax, m):
    lmin = np.max([1, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = (-1.)**(l + m)*muCoupling(l, 2)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 2, 0, -1, l1, m, 2)
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 3, 0, -1, l1, m, 2)
                coupling2[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 1, 0, -1, l1, m, 2)
                coupling4[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 1, 2, 0, 0, l1, m, 1)*muCoupling(l1, 2)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_ORG_13_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_ORG_13_components(kappa, sigma, m, omega, h)
    Cmats = metric_ORG_13_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_14_couplings(lmax, m):
    lmin = np.max([1, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling2 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling3 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling4 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling3[i, i] = muCoupling(l, 2)
        for dl in range(-10, 11):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling0[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 2, 0, 1)
                coupling1[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 3, 0, 1)
                coupling2[i, i + dl] = C3Product(l, m, -1, l1, m, -2, 1, 0, 1)
                coupling4[i, i + dl] = C3Product(l, m, -1, l1, m, -1, 2, 0, 0)*muCoupling(l1, 2)

    return np.array([coupling0, coupling1, coupling2, coupling3, coupling4])

def metric_ORG_14_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_ORG_14_components(kappa, sigma, m, omega, h)
    Cmats = metric_ORG_14_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_33_couplings(lmax, m):
    dlMax = 1
    lmin = np.max([1, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling0[i, i] = (-1.)**(l + m)
        for dl in range(-dlMax, dlMax+1):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling1[i, i + dl] = (-1.)**(l1 + m)*C3Product(l, m, 2, 1, 0, 0, l1, m, 2)

    return np.array([coupling0, coupling1])

def metric_ORG_33_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_ORG_33_components(kappa, sigma, m, omega, h)
    Cmats = metric_ORG_33_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_44_couplings(lmax, m):
    dlMax = 1
    lmin = np.max([1, abs(m)])
    coupling0 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    coupling1 = np.zeros((lmax-lmin+1, lmax-lmin+1))
    for i, l in enumerate(range(lmin, lmax + 1)):
        coupling0[i, i] = 1.
        for dl in range(-dlMax, dlMax+1):
            l1 = l + dl
            if l1 >= lmin and l1 <= lmax:
                coupling1[i, i + dl] = C3Product(l, m, -2, l1, m, -2, 1, 0, 0)

    return np.array([coupling0, coupling1])

def metric_ORG_44_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats = metric_ORG_44_components(kappa, sigma, m, omega, h)
    Cmats = metric_ORG_44_couplings(lmax, m)
    hab = Cmats[0] @ habmats[0]
    for habi, Ci in zip(habmats[1:], Cmats[1:]):
        hab += Ci @ habi
    return hab

def metric_ORG_ab_modes(kappa, sigma, m, omega, h):
    h11 = metric_ORG_11_modes(kappa, sigma, m, omega, h)
    h13 = metric_ORG_13_modes(kappa, sigma, m, omega, h)
    h14 = metric_ORG_14_modes(kappa, sigma, m, omega, h)
    h33 = metric_ORG_33_modes(kappa, sigma, m, omega, h)
    h44 = metric_ORG_44_modes(kappa, sigma, m, omega, h)
    return [h11, h13, h14, h33, h44]
