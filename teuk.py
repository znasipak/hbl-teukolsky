# teuk.py
import numpy as np
import numpy as xp # This is leaving open the introduction of GPUs in the future
import collocode
import swsh

import numba as nb

@nb.njit
def p_hbl(sigma):
    return sigma**2*(1 - sigma)

@nb.njit
def q_hbl(sigma, kappa, s, ma, omega):
    Ktilde = -((1 + s)*kappa + 1j*ma - 2j*(1 + kappa)*omega)/kappa
    return -4j*kappa*omega + 2.*sigma*(1. + s - 2j*omega*(1 - kappa)) + sigma**2*(Ktilde + 4j*omega - 2)

@nb.njit
def u_hbl(sigma, kappa, s, lam, ma, omega):
    Ktilde = -((1 + s)*kappa + 1j*ma - 2j*(1 + kappa)*omega)/kappa
    return -2j*omega*(2.*kappa*Ktilde + (1 + 2.*s)*(1. + kappa)) + (1 - 4j*omega)*Ktilde*sigma - lam

@nb.njit([nb.complex128[:,:](nb.float64[:], nb.float64, nb.int64, nb.float64, nb.float64, nb.float64), 
          nb.complex128[:,:](nb.float64[:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64),
          nb.complex128[:,:,:](nb.float64[:,:], nb.float64, nb.int64, nb.float64, nb.float64, nb.float64), 
          nb.complex128[:,:,:](nb.float64[:,:], nb.float64, nb.float64, nb.float64, nb.float64, nb.float64)])
def teuk_sys(sigma, kappa, s, lam, ma, omega):
    Ktilde = -((1 + s)*kappa + 1j*ma - 2j*(1 + kappa)*omega)/kappa
    P = sigma**2*(1 - sigma)
    Q = -4j*kappa*omega + 2.*sigma*(1. + s - 2j*omega*(1 - kappa)) + sigma**2*(Ktilde + 4j*omega - 2)
    U = -2j*omega*(2.*kappa*Ktilde + (1 + 2.*s)*(1. + kappa)) + (1 - 4j*omega)*Ktilde*sigma - lam

    PQU = np.empty((3,) + sigma.shape, dtype = xp.complex128)
    PQU[0] = P
    PQU[1] = Q
    PQU[2] = U
    return PQU

def sigma_r(r, kappa):
    return 2.*kappa/(r - (1. - kappa))

def sigma_r_deriv(r, kappa):
    return -2.*kappa/(r - (1. - kappa))**2

def sigma_r_deriv_sigma(sigma, kappa):
    return -sigma**2/(2.*kappa)

def sigma_r_deriv2_sigma(sigma, kappa):
    return sigma**3/(2.*kappa**2)

def sigma_r_deriv_sigma_deriv(sigma, kappa):
    return -sigma/kappa

def r_sigma(sigma, kappa):
    return 1. - kappa + 2.*kappa/sigma

def r_sigma_deriv(sigma, kappa):
    return -2.*kappa/sigma**2

def r_sigma_deriv(sigma, kappa):
    return -2.*kappa/sigma**2

def r_sigma_deriv2(sigma, kappa):
    return 4.*kappa/sigma**3

def r_sigma_deriv_r(r, kappa):
    return -0.5*(r - (1. - kappa))**2/kappa

def Phi_sigma(sigma, kappa):
    return 1./(2.*kappa)*xp.log(1. - sigma)

def h_sigma(sigma, kappa):
    return 2.*kappa/sigma - 2.*xp.log(sigma) - (1. + kappa)/kappa*xp.log(1. - sigma) + (1. - kappa) + 2.*xp.log(kappa)

def Z_sigma(sigma, kappa, s, ma, omega):
    return sigma/(2.*kappa)*(4*kappa**2*(1. - sigma)/sigma**2)**(-s)*xp.exp(1.j*(ma*Phi_sigma(sigma, kappa) + omega*h_sigma(sigma, kappa)))

def Z_sigma_deriv(sigma, kappa, s, ma, omega):
    return Z_sigma(sigma, kappa, s, ma, omega)*((kappa*omega*sigma**(-2)*(-2j)+sigma**(-1)*(-1+sigma)**(-1)*(-1+sigma+s*(-2+sigma)+omega*sigma*(-3j)+omega*(2j))+kappa**(-1)*(-1+sigma)**(-1)*(1j)/(2)*(ma-2*omega)))

def Z_sigma_deriv2(sigma, kappa, s, ma, omega):
    return Z_sigma(sigma, kappa, s, ma, omega)*(kappa**(-2)*sigma**(-4)*(-1+sigma)**(-2)*(-1/4)*((ma)**(2)*sigma**(4)-4*kappa**(2)*sigma**(2)*s**(2)*(-2+sigma)**(2)-2*(ma)*sigma**(2)*(4*omega*kappa**(2)*(-1+sigma)+2*omega*sigma**(2)+kappa*sigma*(-4*omega+6*omega*sigma+(-2j)+sigma*(1j)))+4*omega*(4*omega*kappa**(4)*(-1+sigma)**(2)+4*omega*sigma*kappa**(3)*(2-5*sigma+3*sigma**(2))+omega*sigma**(4)+kappa*sigma**(3)*(-4*omega+(-2j)+sigma*(6*omega+(1j)))+kappa**(2)*sigma**(2)*((2j)-2*sigma*(4*omega+(3j))+sigma**(2)*(9*omega+(3j))))+kappa*sigma*s*(4j)*(-1*sigma**(2)*(-2+sigma)*((ma)-2*omega)+4*omega*kappa**(2)*(2-3*sigma+sigma**(2))+kappa*sigma*(8*omega+sigma**(2)*(6*omega+(1j))-2*sigma*(8*omega+(1j))+(2j)))))

def Z_scale(sigma, kappa, s):
    return sigma/(2.*kappa)*(4*kappa**2*(1. - sigma)/sigma**2)**(-s)

def Z_scale_deriv(sigma, kappa, s):
    return Z_scale(sigma, kappa, s)*(sigma**(-1)*(-1+sigma)**(-1)*(-1+sigma+s*(-2+sigma)))

def Z_scale_deriv2(sigma, kappa, s):
    return Z_scale(sigma, kappa, s)*(s*sigma**(-2)*(-1+sigma)**(-2)*(2+sigma*(-2+sigma)+s*(-2+sigma)**(2)))

def Z_slicing(sigma, kappa, ma, omega):
    return xp.exp(1.j*(ma*Phi_sigma(sigma, kappa) + omega*h_sigma(sigma, kappa)))

def Z_slicing_deriv(sigma, kappa, ma, omega):
    return (1j)*Z_slicing(sigma, kappa, ma, omega)*(ma*Phi_sigma_deriv(sigma, kappa) + omega*h_sigma_deriv(sigma, kappa))

def Z_slicing_deriv2(sigma, kappa, ma, omega):
    return Z_slicing(sigma, kappa, ma, omega)*((1j)*(ma*Phi_sigma_deriv2(sigma, kappa) + omega*h_sigma_deriv2(sigma, kappa))-(ma*Phi_sigma_deriv(sigma, kappa) + omega*h_sigma_deriv(sigma, kappa))**2)

def Phi_sigma_deriv(sigma, kappa):
    return -0.5/(1. - sigma)/kappa

def h_sigma_deriv(sigma, kappa):
    return -2.*kappa/sigma**2 - 2./sigma + (1. + kappa)/kappa/(1. - sigma)

def Phi_sigma_deriv2(sigma, kappa):
    return -0.5/(1. - sigma)**2/kappa

def h_sigma_deriv2(sigma, kappa):
    return -4.*kappa/sigma**3 + 2./sigma**2 + (1. + kappa)/kappa/(1. - sigma)**2

def p_hbl_static(sigma):
    return sigma*(1 - sigma)

def q_hbl_static(sigma, s, l):
    return 2.*(l + 1.) - (2.*l + 3. + s)*sigma

def u_hbl_static(sigma, s, l):
    return -(l + 1.)*(l + s + 1.)

@nb.njit
def teuk_static_sys(sigma, s, l):
    PQU = np.empty((3, len(sigma)), dtype = xp.float64)
    PQU[0] = sigma*(1 - sigma)
    PQU[1] = 2.*(l + 1.) - (2.*l + 3. + s)*sigma
    PQU[2] = -(l + 1.)*(l + s + 1.)
    return PQU

def Z_sigma_static(sigma, kappa, s, l):
    return sigma**(l + s + 1.)/(2.*kappa)

def Z_sigma_static_deriv(sigma, kappa, s, l):
    return (l + s + 1.)*sigma**(l + s)/(2.*kappa)

def Z_sigma_static_deriv2(sigma, kappa, s, l):
    return (l + s + 1.)*(l + s)*sigma**(l + s - 1)/(2.*kappa)

def Delta_sigma(sigma, kappa):
    return (4*kappa**2*(1. - sigma)/sigma**2)

def Delta_sigma_deriv(sigma, kappa):
    return -(4*kappa**2*(2. - sigma)/sigma**3)

def Delta_sigma_deriv2(sigma, kappa):
    return (8.*kappa**2*(3. - sigma)/sigma**4)

def Delta_sigma_s(sigma, kappa, s):
    return 4*kappa**(2.*s)*(1. - sigma)**(s)/sigma**(2.*s)

def Delta_sigma_s_deriv(sigma, kappa, s):
    return -4*kappa**(2.*s)*((2. - sigma)*s*(1. - sigma)**(s-1)/sigma**(2.*s+1))

def Delta_sigma_s_deriv2(sigma, kappa, s):
    return 4*kappa**(2.*s)*(((2. - sigma)**2*s - sigma*(4. - sigma) + 2.)*s*(1. - sigma)**(s-2)/sigma**(2.*s+2))

def boundary_condition_up(kappa, s, eigenvalue, ma, omega, scale = 1):
    y0 = scale
    dy0 = -u_hbl(0, kappa, s, eigenvalue, ma, omega)/q_hbl(0, kappa, s, ma, omega)*y0
    return [y0, dy0]

def boundary_condition_in(kappa, s, eigenvalue, ma, omega, scale = 1):
    y0 = scale
    dy0 = -u_hbl(1, kappa, s, eigenvalue, ma, omega)/q_hbl(1, kappa, s, ma, omega)*y0
    return [y0, dy0]

def static_boundary_condition_up(kappa, s, l, scale = 1):
    y0 = scale
    dy0 = -u_hbl_static(0, s, l)/q_hbl_static(0, s, l)*y0
    return [y0, dy0]

def static_boundary_condition_in(kappa, s, l, scale = 1):
    y0 = scale
    dy0 = -u_hbl_static(1, s, l)/q_hbl_static(1, s, l)*y0
    return [y0, dy0]

class HyperboloidalTeukolskySolver:
    def __init__(self, domains = [[0, 0.1], [1, 0.1]], solver = None, **solver_kwargs):
        if solver is None:
            solver = collocode.CollocationODEMultiDomainFixedStepSolver(n = 64, chtype = 1)
            solver_kwargs = {"subdomains": 10, "tol": 1e-11}

        self.solver = solver
        self.solver_kwargs = {"In": solver_kwargs.copy(), "Up": solver_kwargs.copy()}

        if "subdomains" in solver_kwargs:
            subdomains = solver_kwargs["subdomains"]
            if isinstance(subdomains, dict):
                self.solver_kwargs["Up"]["subdomains"] = subdomains["Up"]
                self.solver_kwargs["In"]["subdomains"] = subdomains["In"]

        self.domain = {"In": domains[1], "Up": domains[0]}

        if self.domain["Up"][0] > self.domain["Up"][1]:
            self.domain["Up"] = [domains[0][1], domains[0][0]]
        if self.domain["In"][0] < self.domain["In"][1]:
            self.domain["In"] = [domains[1][1], domains[1][0]]

    def reduce(self, hbl, n = 64, solver = None, **solver_kwargs):
        if not isinstance(hbl, HyperboloidalTeukolsky):
            ValueError("hbl variable must be an instance of the HyperboloidalTeukolsky class")

        assert (hbl.psi["Up"] is not None)

        if solver is None:
            solver = collocode.CollocationAlgebra(n)

        out = hbl.copy()
        for key in hbl.psi.keys():
            out.psi[key] = solver(hbl.psi[key], domain = hbl.domain[key], **solver_kwargs)

        return out
    
    def rescale(self, hbl):
        if not isinstance(hbl, HyperboloidalTeukolsky):
            ValueError("hbl variable must be an instance of the HyperboloidalTeukolsky class")

        assert (hbl.psi["Up"] is not None)

        if xp.abs(hbl.frequency) > 0.:
            print("Rescaling only affects static modes. This is a radiative mode.")
        else:
            out = HyperboloidalTeukolsky(hbl.a, hbl.s, hbl.l, hbl.m, hbl.omega, hbl.eigenvalue)

            dimensions = hbl.psi["Up"].coeffs.shape
            n = dimensions[-1]
            solver_kwargs = {}

            if len(dimensions) > 1 and dimensions[0] > 1:
                solver = collocode.CollocationAlgebraMultiDomain(n)
            else:
                solver = collocode.CollocationAlgebra(n)

            def psi_rescale(sigma, bc):
                return hbl.Z_static_rescale(sigma)*hbl.psi[bc](sigma)
            
            out.psi["Up"] = solver(psi_rescale, args=("Up",), domain = hbl.domains["Up"], **solver_kwargs)
            out.psi["In"] = solver(psi_rescale, args=("In",), domain = hbl.domains["In"], **solver_kwargs)

            out.Z_sigma = out.Z_radiative
            out.Z_sigma_deriv = out.Z_radiative_deriv
            out.Z_sigma_deriv2 = out.Z_radiative_deriv2

    def __call__(self, a, s, l, m, omega, eigenvalue = None, rescale = True, reduce = False, **reduce_kwargs):
        hbl = HyperboloidalTeukolsky(a, s, l, m, omega, eigenvalue)
        ma = m*a
        psi = {"In": None, "Up": None}

        if np.abs(omega) > 0.:
            y0 = 1
            bcUp = boundary_condition_up(hbl.kappa, s, hbl.eigenvalue, ma, omega, scale = y0)

            y1 = 2.*hbl.kappa*xp.exp(-1.j*(2.*hbl.frequency - 0.5*ma/hbl.horizon)*(1. + hbl.kappa + 2.*xp.log(hbl.kappa)))
            bcIn = boundary_condition_in(hbl.kappa, s, hbl.eigenvalue, ma, omega, scale = y1)

            psi["Up"] = self.solver(teuk_sys, bcUp, args = (hbl.kappa, hbl.s, hbl.eigenvalue, ma, hbl.frequency), domain = self.domain["Up"], **self.solver_kwargs["Up"])
            psi["In"] = self.solver(teuk_sys, bcIn, args = (hbl.kappa, hbl.s, hbl.eigenvalue, ma, hbl.frequency), domain = self.domain["In"], **self.solver_kwargs["In"])
        else:
            y0 = 1
            bcUp = static_boundary_condition_up(np.abs(hbl.s), self.l, scale = y0)

            y1 = 1
            bcIn = static_boundary_condition_in(np.abs(hbl.s), self.l, scale = y1)

            psi["Up"] = self.solver(teuk_static_sys, bcUp, args = (np.abs(hbl.s), hbl.l), domain = self.domain["Up"], **self.solver_kwargs["Up"])
            psi["In"] = self.solver(teuk_static_sys, bcIn, args = (np.abs(hbl.s), hbl.l), domain = self.domain["In"], **self.solver_kwargs["In"])

            if rescale: # rescale hyperboloidal solutions to the static equation to match the scaling of the radiative modes
                dimensions = psi["Up"].coeffs.shape
                n = dimensions[-1]
                solver_kwargs = {}

                if len(dimensions) > 1 and dimensions[0] > 1:
                    solver = collocode.CollocationAlgebraMultiDomain(n)
                else:
                    solver = collocode.CollocationAlgebra(n)

                def psi_rescale(sigma, bc):
                    return hbl.Z_static_rescale(sigma)*psi[bc](sigma)
                
                psi["Up"] = solver(psi_rescale, args=("Up",), domain = hbl.domains["Up"], **solver_kwargs)
                psi["In"] = solver(psi_rescale, args=("In",), domain = hbl.domains["In"], **solver_kwargs)
            else:
                if hbl.s >= 0:
                    hbl.Z_sigma = hbl.Z_static_plus
                    hbl.Z_sigma_deriv = hbl.Z_static_plus_deriv
                    hbl.Z_sigma_deriv2 = hbl.Z_static_plus_deriv2
                else:
                    hbl.Z_sigma = hbl.Z_static_minus
                    hbl.Z_sigma_deriv = hbl.Z_static_minus_deriv
                    hbl.Z_sigma_deriv2 = hbl.Z_static_minus_deriv2

        if reduce:
            if "solver" in reduce_kwargs.keys():
                reduce_solver = reduce_kwargs["solver"]
            else:
                if "n" in reduce_kwargs.keys():
                    reduce_solver = collocode.CollocationAlgebra(reduce_kwargs["n"])
                else:
                    reduce_solver = collocode.CollocationAlgebra()

            for key in psi.keys():
                psi[key] = reduce_solver(psi[key], domain = hbl.domain[key], **reduce_kwargs)

        hbl.set_solutions(psi)

        return hbl
    
class FrequencyDomainMode:
    def __init__(self, a, m, omega):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)
        self.m = m
        self.omega = omega
    
    @property
    def frequency(self):
        return self.omega
    
    @property
    def blackholespin(self):
        return self.a
    
class HarmonicMode:
    def __init__(self, l, m, eigen):
        self.l = l
        self.m = m
        self.eigen = eigen

    @property
    def eigenvalue(self):
        return self.eigen
    
class HyperboloidalSlicing:
    def __init__(self, a):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)

    def height_function(self, sigma):
        return h_sigma(sigma, self.kappa)
    
    def height_deriv(self, sigma):
        return h_sigma_deriv(sigma, self.kappa)
    
    def height_deriv2(self, sigma):
        return h_sigma_deriv2(sigma, self.kappa)
    
    def Phi_function(self, sigma):
        return Phi_sigma(sigma, self.kappa)
    
    def Phi_deriv(self, sigma):
        return Phi_sigma_deriv(sigma, self.kappa)
    
    def Phi_deriv2(self, sigma):
        return Phi_sigma_deriv2(sigma, self.kappa)
    
class FrequencyDomainModeHyperboloidalSlicing(FrequencyDomainMode, HyperboloidalSlicing):    
    def Z_slicing(self, sigma):
        return Z_slicing(sigma, self.kappa, self.a*self.m, self.omega)
    
    def Z_slicing_deriv(self, sigma):
        return Z_slicing_deriv(sigma, self.kappa, self.a*self.m, self.omega)
    
    def Z_slicing_deriv2(self, sigma):
        return Z_slicing_deriv2(sigma, self.kappa, self.a*self.m, self.omega)
    
    def eval_with_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_slicing(sigma)
    
    def deriv_with_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_slicing_deriv(sigma) + self.deriv(sigma, *arg)*self.Z_slicing(sigma)
    
    def deriv2_with_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_slicing_deriv2(sigma) + 2.*self.deriv(sigma, *arg)*self.Z_slicing_deriv(sigma) + self.deriv2(sigma, *arg)*self.Z_slicing(sigma)
    
class RadialCompactification:
    def __init__(self, a):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)

    def sigma_r(self, r):
        return sigma_r(r, self.kappa)
    
    def r_sigma(self, sigma):
        return r_sigma(sigma, self.kappa)
    
    def sigma_r_deriv_sigma(self, sigma):
        return sigma_r_deriv_sigma(sigma, self.kappa)
    
    def sigma_r_deriv2_sigma(self, sigma):
        return sigma_r_deriv2_sigma(sigma, self.kappa)
    
    def eval_with_r(self, r):
        sigma = self.sigma_r(r)
        return self.eval(sigma)
    
    def deriv_with_r(self, r):
        sigma = self.sigma_r(r)
        return self.sigma_r_deriv_sigma(sigma)*self.deriv(sigma)
    
    def deriv2_with_r(self, r):
        sigma = self.sigma_r(r)
        return self.sigma_r_deriv2_sigma(sigma)*self.deriv(sigma) + self.sigma_r_deriv_sigma(sigma)**2*self.deriv2(sigma)
    
class ScaledTeukolsky:
    def __init__(self, a, s):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)
        self.s = s

    def Z_scale(self, sigma):
        return Z_scale(sigma, self.kappa, self.s)
    
    def Z_scale_deriv(self, sigma):
        return Z_scale_deriv(sigma, self.kappa, self.s)
    
    def Z_scale_deriv2(self, sigma):
        return Z_scale_deriv2(sigma, self.kappa, self.s)
    
    def eval_with_rescaling(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_scale(sigma)
    
    def deriv_with_rescaling(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_scale_deriv(sigma) + self.deriv(sigma, *arg)*self.Z_scale(sigma)
    
    def deriv2_with_rescaling(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_scale_deriv2(sigma) + 2.*self.deriv(sigma, *arg)*self.Z_scale_deriv(sigma) + self.deriv2(sigma, *arg)*self.Z_scale(sigma)
    
class ScaledTeukolskyHyperboloidalSlicingRadialCompactification(ScaledTeukolsky, HyperboloidalSlicing, RadialCompactification):
    pass
    
class ScaledTeukolskyFrequencyDomainModeHyperboloidalSlicingRadialCompactification(FrequencyDomainModeHyperboloidalSlicing, ScaledTeukolsky, RadialCompactification):
    def __init__(self, a, s, m, omega):
        FrequencyDomainModeHyperboloidalSlicing.__init__(self, a, m, omega)
        ScaledTeukolsky.__init__(self, a, s)

    def Z_sigma(self, sigma):
        return Z_sigma(sigma, self.kappa, self.s, self.a*self.m, self.omega)
    
    def Z_sigma_deriv(self, sigma):
        return Z_sigma_deriv(sigma, self.kappa, self.s, self.a*self.m, self.omega)
    
    def Z_sigma_deriv2(self, sigma):
        return Z_sigma_deriv2(sigma, self.kappa, self.s, self.a*self.m, self.omega)
    
    def eval_with_rescaling_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_sigma(sigma)
    
    def deriv_with_rescaling_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_sigma_deriv(sigma) + self.deriv(sigma, *arg)*self.Z_sigma(sigma)
    
    def deriv2_with_rescaling_slicing(self, sigma, *arg):
        return self.eval(sigma, *arg)*self.Z_sigma_deriv2(sigma) + 2.*self.deriv(sigma, *arg)*self.Z_sigma_deriv(sigma) + self.deriv2(sigma, *arg)*self.Z_sigma(sigma)
    
class ScaledTeukolskyModeHyperboloidalSlicingRadialCompactification(ScaledTeukolskyFrequencyDomainModeHyperboloidalSlicingRadialCompactification, HarmonicMode):
    def __init__(self, a, s, l, m, omega, eigen):
        ScaledTeukolskyFrequencyDomainModeHyperboloidalSlicingRadialCompactification.__init__(self, a, s, m, omega)
        HarmonicMode.__init__(self, l, m, eigen)

class HyperboloidalTeukolskySolution(ScaledTeukolskyModeHyperboloidalSlicingRadialCompactification):
    def __init__(self, a, s, l, m, omega, eigen, sol = None):
        ScaledTeukolskyModeHyperboloidalSlicingRadialCompactification.__init__(self, a, s, l, m, omega, eigen)
        self._sol = sol

    @property
    def domain(self):
        return self._sol.domain
    
    @property
    def domains(self):
        return self._sol.domains
    
    @property
    def coeffs(self):
        return self._sol.coeffs

    def eval(self, sigma, deriv = 0):
        return self._sol(sigma, deriv)

    def deriv(self, sigma):
        return self._sol(sigma, deriv = 1)
    
    def deriv2(self, sigma):
        return self._sol(sigma, deriv = 2)
    
    def eval_rescaling(self, sigma, deriv = 0):
        if deriv == 0:
            return self.eval_with_rescaling(sigma)
        elif deriv == 1:
            return self.deriv_with_rescaling(sigma)
        elif deriv == 2:
            return self.deriv2_with_rescaling(sigma)
        else:
            return ValueError("Only supports up to second-derivatives")

    def eval_slicing(self, sigma, deriv = 0):
        if deriv == 0:
            return self.eval_with_slicing(sigma)
        elif deriv == 1:
            return self.deriv_with_slicing(sigma)
        elif deriv == 2:
            return self.deriv2_with_slicing(sigma)
        else:
            return ValueError("Only supports up to second-derivatives")

    def eval_rescaling_slicing(self, sigma, deriv = 0):
        if deriv == 0:
            return self.eval_with_rescaling_slicing(sigma)
        elif deriv == 1:
            return self.deriv_with_rescaling_slicing(sigma)
        elif deriv == 2:
            return self.deriv2_with_rescaling_slicing(sigma)
        else:
            return ValueError("Only supports up to second-derivatives")

    def __call__(self, sigma, deriv = 0, slicing = "hyperboloidal", scaled = True, compactification = True):
        if slicing == "hyperboloidal" or slicing == "hbl":
            if scaled:
                return self.eval(sigma, deriv = deriv)
            else:
                return self.eval_with_rescaling(sigma, deriv = deriv)
        elif slicing == "time" or slicing == "t":
            if scaled:
                return self.eval_slicing(sigma, deriv = deriv)
            else:
                return self.eval_rescaling_slicing(sigma, deriv = deriv)
        else:
            return ValueError(f"Only hyperboloidal and t-slicing are supported, not {slicing}")

class HyperboloidalTeukolsky(ScaledTeukolskyModeHyperboloidalSlicingRadialCompactification):
    def __init__(self, a, s, l, m, omega, eigenvalue = None, psi = {"In": None, "Up": None}):
        ScaledTeukolskyModeHyperboloidalSlicingRadialCompactification.__init__(self, a, s, l, m, omega, eigenvalue)
        if eigenvalue is None:
            self.eigenvalue = swsh.swsh_eigenvalue(s, l, m, a*omega)
        self.horizon = 1 + self.kappa
        self.psi = {"In": HyperboloidalTeukolskySolution(a, s, l, m, omega, eigenvalue, psi["In"]), "Up": HyperboloidalTeukolskySolution(a, s, l, m, omega, eigenvalue, psi["Up"])}

    @property
    def domain(self):
        return {"In": self.psi["In"].domain, "Up": self.psi["Up"].domain}
    
    @property
    def domains(self):
        return {"In": self.psi["In"].domains, "Up": self.psi["Up"].domains}
    
    def __getitem__(self, key):
        return self.psi[key]
    
    def set_solutions(self, psi):
        self.psi = {"In": HyperboloidalTeukolskySolution(self.a, self.s, self.l, self.m, self.omega, self.eigenvalue, psi["In"]), "Up": HyperboloidalTeukolskySolution(self.a, self.s, self.l, self.m, self.omega, self.eigenvalue, psi["Up"])}
    
    def eval(self, bc, sigma, deriv = 0):
        return self.psi[bc].eval(sigma, deriv)

    def deriv(self, bc, sigma):
        return self.psi[bc].deriv(sigma)
    
    def deriv2(self, bc, sigma):
        return self.psi[bc].deriv2(sigma)

    def __call__(self, bc, sigma, deriv = 0, slicing = "hyperboloidal", scaled = True, compactification = True):
        return self.psi[bc](sigma, deriv, slicing, scaled, compactification)

# Real part of the Teukolsky-Starobinsky constant given in terms of (a, m, omega) 
# and Chandrasekhar's spin-invariant eigenvalue lambdaCH = lambda_s + s(s+1) for s = pm 2
def teukolsky_starobinsky_constant_D(m, a, omega, lambdaCH):
	return np.sqrt((pow(lambdaCH, 2) + 4*m*a*omega - pow(2*a*omega, 2))*(pow(lambdaCH - 2., 2) + 36.*m*a*omega - pow(6.*a*omega, 2)) + (2.*lambdaCH - 1.)*(96.*pow(a*omega, 2) - 48.*m*a*omega) - pow(12.*a*omega, 2))

# Complex Teukolsky-Starobinsky constant given in terms of (a, m, omega) 
# and Chandrasekhar's spin-invariant eigenvalue lambdaCH = lambda_s + s(s+1) for s = pm 2
def teukolsky_starobinsky_complex_constant(j, m, a, omega, lambdaCH):
	return teukolsky_starobinsky_constant_D(m, a, omega, lambdaCH) + pow(-1., j + m)*12.*1j*omega

class ScaledStaticPlusTeukolsky(ScaledTeukolsky):
    def __init__(self, a, s, l):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)
        self.s = s
        self.l = l

    def Z_scale(self, sigma):
        return Z_sigma_static(sigma, self.kappa, self.s, self.l)
    
    def Z_scale_deriv(self, sigma):
        return Z_sigma_static_deriv(sigma, self.kappa, self.s, self.l)
    
    def Z_scale_deriv2(self, sigma):
        return Z_sigma_static_deriv2(sigma, self.kappa, self.s, self.l)
    
    def Z_rescale(self, sigma):
        return self.Z_scale(sigma)/Z_scale(sigma, self.kappa, self.s)
    
class ScaledStaticMinusTeukolsky(ScaledTeukolsky):
    def __init__(self, a, s, l):
        self.a = a
        self.kappa = np.sqrt(1 - a**2)
        self.s = s
        self.l = l

    def Z_scale(self, sigma):
        return Delta_sigma_s(sigma, self.kappa, -self.s)*Z_sigma_static(sigma, self.kappa, -self.s, self.l)
    
    def Z_scale_deriv(self, sigma):
        return Delta_sigma_s(sigma, self.kappa, -self.s)*Z_sigma_static_deriv(sigma, self.kappa, -self.s, self.l) + Delta_sigma_s_deriv(sigma, self.kappa, -self.s)*Z_sigma_static(sigma, self.kappa, -self.s, self.l)
    
    def Z_scale_deriv2(self, sigma):
        return Delta_sigma_s(sigma, self.kappa, -self.s)*Z_sigma_static_deriv2(sigma, self.kappa, -self.s, self.l) + Delta_sigma_s_deriv2(sigma, self.kappa, -self.s)*Z_sigma_static(sigma, self.kappa, -self.s, self.l) + 2.*Delta_sigma_s_deriv(sigma, self.kappa, -self.s)*Z_sigma_static_deriv(sigma, self.kappa, -self.s, self.l)

    def Z_rescale(self, sigma):
        return self.Z_scale(sigma)/Z_scale(sigma, self.kappa, self.s)
    
class ScaledStaticTeukolskyHyperboloidalSlicingRadialCompactification(ScaledStaticPlusTeukolsky, ScaledStaticMinusTeukolsky, HyperboloidalSlicing, RadialCompactification):
    def __init__(self, a, s, l):
        if s >= 0:
            ScaledStaticPlusTeukolsky.__init__(self, a, s, l)
        else:
            ScaledStaticMinusTeukolsky.__init__(self, a, s, l)
        HyperboloidalSlicing.__init__(self, a)
        RadialCompactification.__init__(self, a)