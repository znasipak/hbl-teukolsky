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

    def hab_coupling(self, l, m, l1):
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

def hab_rescaled(kappa, sigma, z, hab, h24, h44):
    scalings = metric_scalings_IRG(kappa, sigma)
    rho_factors = metric_rho_factors_IRG(kappa, sigma, z)
    rescaling = rho_factors*scalings
    return rescaling[1,1]*hab, rescaling[1,3]*h24, rescaling[3,3]*h44

def huu(geo, sigma, z, hab, h24, h44):
    kappa = np.sqrt(1-geo.a**2)
    hab = hab_rescaled(kappa, sigma, z, hab, h24, h44)
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

def metric_IRG_22_couplings_dagger(lmax, m):
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

def metric_IRG_22_modes(kappa, sigma, m, omega, h):
    lmin = np.max([0, abs(m)])
    lmax = lmin + h[0].shape[0] - 1
    habmats_nodagger = metric_IRG_22_components_nodagger(kappa, sigma, m, omega, h)
    habmats_dagger = metric_IRG_22_components_dagger(kappa, sigma, m, omega, h)
    Cmats_nodagger = metric_IRG_22_couplings_nodagger(lmax, m)
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

def metric_IRG_23_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
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

def metric_IRG_24_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
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

def metric_IRG_33_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
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

def metric_IRG_44_modes(kappa, sigma, m, omega, h):
    lmin = np.max([1, abs(m)])
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
'''

def metric_ORG_11_components_nodagger(kappa, sigma, m, omega, h):
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

def metric_ORG_11_components_dagger(kappa, sigma, m, omega, h):
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

def metric_ORG_13_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = np.sqrt((1/15)*np.pi)*kappa**(-2)*sigma**(2)*(1/2)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term1 = m*omega*q*np.sqrt((1/42)*np.pi)*kappa**(-2)*sigma**(2)*(1/5)*(-kappa*sigma*h[1]*(-1+sigma)*(-1+kappa**(2))+h[0]*q**(2)*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term2 = m*omega*q*np.sqrt((1/3)*np.pi)*kappa**(-2)*(1/40)*(4*kappa*sigma*h[1]*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+2*kappa**(2)*(5+sigma*(-5+sigma))+3*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-24*omega+11*q+q**(3))+sigma*kappa**(2)*(m*q*(40+sigma*(-40+9*sigma))+8*m*omega*(-10+3*sigma**(2))+(-20j)*(4+sigma*(-3+sigma)))-4*kappa*sigma**(2)*(-4*m*omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+8*kappa**(3)*(-2*m*omega*sigma*(5+sigma*(-5+sigma))+(1j)*(-5+sigma*(5+sigma*(-3+sigma))))))
    term3 = 2**((-1/2))*kappa**(-2)*(1/48)*(4*kappa*sigma*h[1]*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+kappa**(2)*(6+sigma*(-6+sigma))+2*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-16*omega+7*q+q**(3))+sigma*kappa**(2)*(m*q*(24+sigma*(-24+5*sigma))+16*m*omega*(-3+sigma**(2))+(-12j)*(4+sigma*(-3+sigma)))+4*kappa**(3)*(-2*m*omega*sigma*(6+sigma*(-6+sigma))+(1j)*(-6+sigma*(6+sigma*(-3+sigma))))+4*kappa*sigma**(2)*(2*m*omega*(-6+sigma)-3*m*q*(-2+sigma)+(2j)*(-3+sigma))))
    term4 = np.sqrt((1/10)*np.pi)*kappa**(-2)*sigma**(2)*(-1/6)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    return np.array([term0, term1, term2, term3, term4])

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

def metric_ORG_14_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = np.sqrt((1/15)*np.pi)*kappa**(-2)*sigma**(2)*(-1/2)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term1 = m*omega*q*np.sqrt((1/42)*np.pi)*kappa**(-2)*sigma**(2)*(1/5)*(-kappa*sigma*h[1]*(-1+sigma)*(-1+kappa**(2))+h[0]*q**(2)*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    term2 = m*omega*q*np.sqrt((1/3)*np.pi)*kappa**(-2)*(1/40)*(4*kappa*sigma*h[1]*(-1+sigma)*(-5*kappa*sigma*(-2+sigma)+2*kappa**(2)*(5+sigma*(-5+sigma))+3*sigma**(2))+h[0]*(1j)*(m*sigma**(3)*(-24*omega+11*q+q**(3))+sigma*kappa**(2)*(m*q*(40+sigma*(-40+9*sigma))+8*m*omega*(-10+3*sigma**(2))+(-20j)*(4+sigma*(-3+sigma)))-4*kappa*sigma**(2)*(-4*m*omega*(-5+sigma)+5*m*q*(-2+sigma)+(-3j)*(-3+sigma))+8*kappa**(3)*(-2*m*omega*sigma*(5+sigma*(-5+sigma))+(1j)*(-5+sigma*(5+sigma*(-3+sigma))))))
    term3 = 2**((-1/2))*kappa**(-2)*(1/48)*(-4*kappa*sigma*h[1]*(-1+sigma)*(-3*kappa*sigma*(-2+sigma)+kappa**(2)*(6+sigma*(-6+sigma))+2*sigma**(2))+h[0]*(-1j)*(m*sigma**(3)*(-16*omega+7*q+q**(3))+sigma*kappa**(2)*(m*q*(24+sigma*(-24+5*sigma))+16*m*omega*(-3+sigma**(2))+(-12j)*(4+sigma*(-3+sigma)))+4*kappa**(3)*(-2*m*omega*sigma*(6+sigma*(-6+sigma))+(1j)*(-6+sigma*(6+sigma*(-3+sigma))))+4*kappa*sigma**(2)*(2*m*omega*(-6+sigma)-3*m*q*(-2+sigma)+(2j)*(-3+sigma))))
    term4 = np.sqrt((1/10)*np.pi)*kappa**(-2)*sigma**(2)*(1/6)*(-1+kappa**(2))*(kappa*sigma*h[1]*(-1+sigma)+h[0]*(kappa*(3-sigma+m*omega*sigma*(-2j))+m*sigma*(1j)*(-2*omega+q)))
    return np.array([term0, term1, term2, term3, term4])

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

def metric_ORG_33_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = kappa**(-4)*(1/16)*(-sigma*h[2]*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-sigma)+h[0]*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*omega**(2))+m*omega*(-2j)*(4+sigma*(-2+sigma)))+kappa*m*sigma*(m*(-2+8*omega*q+sigma-4*omega**(2)*(2+sigma))+(-1j)*(-4+sigma)*(-2*omega+q))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*omega*q*(-2+sigma)+sigma+4*omega**(2)*(-4+sigma))+m*(1j)*(-4*omega*(2+sigma)+q*(4+sigma*(-2+sigma)))))+kappa*h[1]*(2j)*(-1+sigma)*(m*sigma**(2)*(-2*omega+q)+2*kappa**(2)*(m*omega*sigma*(-2+sigma)+(-1j))-kappa*sigma*(4*m*omega+m*q*(-2+sigma)+(2j))))
    term1 = q*np.sqrt((1/3)*np.pi)*kappa**(-4)*(1/8)*(h[2]*kappa**(2)*sigma**(2)*(-1+sigma)**(2)*(1j)+2*kappa*sigma*h[1]*(-1+sigma)*(2*m*omega*sigma*(1+kappa)-m*q*sigma+kappa*(2j))+h[0]*(1j)*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(2)*(6+sigma*(-4+m**(2)*(sigma-4*sigma*omega**(2))+m*omega*(2j)*(-4+sigma)))+kappa*m*sigma*(-2*omega+q)*(sigma*(4*m*omega+(-1j))+(4j))))
    return np.array([term0, term1])

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

def metric_ORG_44_components(kappa, sigma, m, omega, h):
    q = np.sqrt(1. - kappa**2)
    term0 = kappa**(-4)*(1/16)*(-sigma*h[2]*kappa**(2)*(-1+sigma)**(2)*(kappa*(-2+sigma)-sigma)+h[0]*(m**(2)*sigma**(2)*(-1+4*omega*(-omega+q))+kappa**(3)*(2+sigma*m**(2)*(-2+sigma)*(-1+4*omega**(2))+m*omega*(-2j)*(4+sigma*(-2+sigma)))+kappa*m*sigma*(m*(-2+8*omega*q+sigma-4*omega**(2)*(2+sigma))+(-1j)*(-4+sigma)*(-2*omega+q))+kappa**(2)*(6-4*sigma+sigma*m**(2)*(-4*omega*q*(-2+sigma)+sigma+4*omega**(2)*(-4+sigma))+m*(1j)*(-4*omega*(2+sigma)+q*(4+sigma*(-2+sigma)))))+kappa*h[1]*(2j)*(-1+sigma)*(m*sigma**(2)*(-2*omega+q)+2*kappa**(2)*(m*omega*sigma*(-2+sigma)+(-1j))-kappa*sigma*(4*m*omega+m*q*(-2+sigma)+(2j))))
    term1 = q*np.sqrt((1/3)*np.pi)*kappa**(-4)*(1/8)*(2*kappa*sigma*h[1]*(-1+sigma)*(-2*m*omega*sigma*(1+kappa)+m*q*sigma+kappa*(-2j))+h[2]*kappa**(2)*sigma**(2)*(-1+sigma)**(2)*(-1j)+h[0]*(1j)*(m**(2)*sigma**(2)*(1+4*omega*(omega-q))+kappa**(2)*(-6+sigma*(4+sigma*m**(2)*(-1+4*omega**(2))+m*omega*(-2j)*(-4+sigma)))-kappa*m*sigma*(-2*omega+q)*(sigma*(4*m*omega+(-1j))+(4j))))
    return np.array([term0, term1])

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
