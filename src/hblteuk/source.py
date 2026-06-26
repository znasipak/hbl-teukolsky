# source.py

import numpy as np
import numpy as xp
# from spheroidal import SpinWeightedSpheroidalHarmonic
from .swsh import SpinWeightedSpheroidalHarmonic
from .teuk import (sigma_r, 
                  sigma_r_deriv,
                  sigma_r_deriv_sigma, 
                  sigma_r_deriv_sigma_deriv,
                  HyperboloidalTeukolsky,
                  teuk_sys
                )

def integrate_source(s, l, m, k, n, geo, Sslm = None, Rslm = None, **kwargs):
    if (s == -2):
        if geo.e == 0.:
            return gravitational_source_circular_minus2(l, m, geo, Sslm = Sslm, Rslm = Rslm, **kwargs)
        elif np.abs(geo.x) == 1.:
            return gravitational_source_eccentric(l, m, n, geo, Sslm = Sslm, Rslm = Rslm, **kwargs)
    elif (s == 2):
        if geo.e == 0.: 
            return gravitational_source_circular_plus2(l, m, geo, Sslm = Sslm, Rslm = Rslm, **kwargs)
    else:
        return ValueError(f"Source integration not implemented for spin-weight s = {s} and geodesic parameters (p, e, x) = ({geo.p}, {geo.e}, {geo.x})")

def u_13_coeffs(geo, signUr, signUth, rp, zp):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    a = geo.blackholespin
    rhobar = -1./(rp + 1j*a*zp)

    ur = geo.ur(0, rp)
    uth = geo.uth(0, zp)
    
    u1 = -(En*(rp**2 + a**2) - a*Lz - signUr*ur)/(rp**2 - 2.*rp + a**2)
    u3 = rhobar/(np.sqrt(2))*(1j*np.sqrt(1 - zp**2)*(a*En - Lz/(1 - zp**2)) - signUth*uth)

    return u1, u3

def u_24_coeffs(geo, signUr, signUth, rp, zp):
    En = geo.orbitalenergy
    Lz = geo.orbitalangularmomentum
    a = geo.blackholespin
    rho = -1./(rp - 1j*a*zp)

    ur = geo.ur(0, rp)
    uth = geo.uth(0, zp)
    
    u2 = -0.5*(En*(rp**2 + a**2) - a*Lz + signUr*ur)/(rp**2 + a**2*zp**2)
    u4 = -rho/(np.sqrt(2))*(1j*np.sqrt(1 - zp**2)*(a*En - Lz/(1 - zp**2)) + signUth*uth)

    return u2, u4


def gravitational_source_circular_minus2(l, m, geo, Sslm = None, Rslm = None, **kwargs):
    omega = geo.mode_frequency(m, 0, 0)
    a = geo.a
    kappa = np.sqrt(1 - a**2)
    En, Lz, _ = geo.constants
    s = -2

    if Sslm is None:
        Sslm = SpinWeightedSpheroidalHarmonic(-2, l, m, a*omega)
    
    th = 0.5*np.pi
    uth = 0.
    sth = np.sin(th)
    cth = np.cos(th)

    # print(l, m)
    Seq = Sslm.eval(th)
    dSeq = Sslm.deriv(th)
    d2Seq = Sslm.deriv2(th)
    # Seq = Sslm(cth)
    # dSeq = Sslm.deriv(cth)
    # d2Seq = Sslm.deriv2(cth)

    L1 = -m/sth + a*omega*sth + cth/sth
    L2 = -m/sth + a*omega*sth + 2.*cth/sth
    L2S = dSeq + L2*Seq
    L2p = m*cth/sth**2 + a*omega*cth - 2./sth**2
    L1Sp = d2Seq + L1*dSeq
    L1L2S = L1Sp + L2p*Seq + L2*dSeq + L1*L2*Seq

    rp = geo.radial_roots[0]
    urp = 0
    sp = sigma_r(rp, kappa)

    if Rslm is None:
        Rslm = HyperboloidalTeukolsky(a, s, l, m, omega, domains=[[0, sp], [1, sp]])
    
    Psihbl = Rslm.psi

    sp = Rslm.sigma_r(rp)
    PsiUp = Psihbl["Up"](sp)
    PsiIn = Psihbl["In"](sp)
    dPsiUp = Psihbl["Up"](sp, deriv=1)
    dPsiIn = Psihbl["In"](sp, deriv=1)
    # d2PsiUpTemp = Psihbl["Up"](sp, deriv=2)
    # d2PsiInTemp = Psihbl["In"](sp, deriv=2)

    P,Q,U = teuk_sys(np.array([sp]), Rslm.kappa, s, Rslm.eigenvalue, m*a, omega)
    d2PsiUp = (-Q[0]*dPsiUp - U[0]*PsiUp)/P[0]
    d2PsiIn = (-Q[0]*dPsiIn - U[0]*PsiIn)/P[0]

    normUp = np.sqrt(np.max(np.abs([PsiUp, dPsiUp, d2PsiUp])))
    normIn = np.sqrt(np.max(np.abs([PsiIn, dPsiIn, d2PsiIn])))

    PsiUp *= 1/normUp
    PsiIn *= 1/normIn
    dPsiUp *= 1/normUp
    dPsiIn *= 1/normIn
    d2PsiUp *= 1/normUp
    d2PsiIn *= 1/normIn

    Delta = rp**2 - 2.*rp + a**2
    DeltaSqrt2 = np.sqrt(2)*Delta
    Kt = (rp**2 + a**2)*omega - m*a
    rho = -1./(rp - 1j*a*cth)
    rhobar = -1./(rp + 1j*a*cth)
    Sigma = 1/rho/rhobar

    Ann0 = -rho**(-2)*rhobar**(-1)*DeltaSqrt2**(-2)*(rho**(-1)*L1L2S + 3j*a*sth*L1*Seq + 3j*a*cth*Seq + 2j*a*sth*dSeq - 1j*a*sth*L2*Seq)
    Anmbar0 = rho**(-3)*DeltaSqrt2**(-1)*((rho + rhobar - 1j*Kt/Delta)*L2S + (rho - rhobar)*a*sth*Kt/Delta*Seq)
    Anmbar1 = -rho**(-3)*DeltaSqrt2**(-1)*( L2S + 1j*(rho - rhobar)*a*sth*Seq)
    Ambarmbar0 = rho**(-3)*rhobar*DeltaSqrt2**(-2)*(0.5*Kt**2 + 1j*Kt*(1. - rp + Delta*rho) + 1j*rp*omega*Delta)*Seq
    Ambarmbar1 = -0.5*(rho)**(-3)*rhobar*(1j*Kt/Delta - rho)*Seq
    Ambarmbar2 = -0.25*(rho)**(-3)*rhobar*Seq

    Cr = (En*(rp**2 + a**2) - a*Lz + urp)/(2*Sigma)
    Cth = rho*(1j*sth*(a*En - Lz/sth**2) + uth)/np.sqrt(2)

    Cnn = Cr*Cr
    Cnmbar = Cr*Cth
    Cmbarmbar = Cth*Cth

    A0 = Ann0*Cnn + Anmbar0*Cnmbar + Ambarmbar0*Cmbarmbar
    A1 = -(Anmbar1*Cnmbar + Ambarmbar1*Cmbarmbar)
    A2 = Ambarmbar2*Cmbarmbar

    Zs = Rslm.Z_sigma(sp)
    dZs = Rslm.Z_sigma_deriv(sp)
    d2Zs = Rslm.Z_sigma_deriv2(sp)

    gs = sigma_r_deriv_sigma(sp, kappa)
    dgs = sigma_r_deriv_sigma_deriv(sp, kappa)

    B0 = A0*Zs + A1*gs*dZs + A2*gs*(dgs*dZs + gs*d2Zs)
    B1 = A1*gs*Zs + A2*gs*(dgs*Zs + 2.*gs*dZs)
    B2 = A2*gs*gs*Zs

    integrand = np.array([(B0*PsiUp + B1*dPsiUp + B2*d2PsiUp), (B0*PsiIn + B1*dPsiIn + B2*d2PsiIn)])

    Gamma = geo.mino_frequencies[0]
    wronskian = PsiIn*dPsiUp - PsiUp*dPsiIn
    rescale = Rslm.Z_sigma(sp)**2*(rp**2 - 2.*rp + a**2)**(-1)*sigma_r_deriv(rp, kappa)
    const_wronskian = np.array([normIn, normUp])*rescale*wronskian

    amplitudes = -8.*np.pi*integrand/const_wronskian/Gamma
    amplitudes_max_error = 0.*amplitudes

    return amplitudes, amplitudes_max_error

def gravitational_source_circular_plus2(l, m, geo, Sslm = None, Rslm = None, **kwargs):
    omega = geo.mode_frequency(m, 0, 0)
    a = geo.a
    kappa = np.sqrt(1 - a**2)
    s = 2

    if Sslm is None:
        Sslm = SpinWeightedSpheroidalHarmonic(s, l, m, a*omega)

    rp = geo.radial_roots[0]
    urp = 0
    sp = sigma_r(rp, kappa)
    
    thp = 0.5*np.pi
    uth = 0.
    sthp = np.sin(thp)
    cthp = np.cos(thp)

    Delta = rp**2 - 2.*rp + a**2
    DeltaP = 2.*rp - 2.
    DeltaPP = 2.
    K = (rp**2 + a**2)*omega - m*a
    Kp = 2*omega*rp
    rho = -1./(rp - 1j*a*cthp)
    rhobar = -1./(rp + 1j*a*cthp)

    # print(l, m)
    Slm = Sslm.eval(thp)
    SlmP = Sslm.deriv(thp)
    SlmPP = Sslm.deriv2(thp)

    dLd1 = m/sthp - a*omega*sthp + cthp/sthp
    dLd2 = m/sthp - a*omega*sthp + 2.*cthp/sthp
    dLd2p = -m*cthp/sthp/sthp - a*omega*cthp - 2./sthp/sthp
    rhoOverRhoP = 1j*a*rho*sthp
    rhoOverRhoPP = 1j*a*rho*(cthp + 2.*sthp*rhoOverRhoP)

    if Rslm is None:
        Rslm = HyperboloidalTeukolsky(a, s, l, m, omega, domains=[[0, sp], [1, sp]])
    
    Psihbl = Rslm.psi

    sp = Rslm.sigma_r(rp)
    PsiUp = Psihbl["Up"](sp)
    PsiIn = Psihbl["In"](sp)
    dPsiUp = Psihbl["Up"](sp, deriv=1)
    dPsiIn = Psihbl["In"](sp, deriv=1)

    P,Q,U = teuk_sys(np.array([sp]), Rslm.kappa, s, Rslm.eigenvalue, m*a, omega)
    d2PsiUp = (-Q[0]*dPsiUp - U[0]*PsiUp)/P[0]
    d2PsiIn = (-Q[0]*dPsiIn - U[0]*PsiIn)/P[0]

    normUp = np.sqrt(np.max(np.abs([PsiUp, dPsiUp, d2PsiUp])))
    normIn = np.sqrt(np.max(np.abs([PsiIn, dPsiIn, d2PsiIn])))

    PsiUp *= 1/normUp
    PsiIn *= 1/normIn
    dPsiUp *= 1/normUp
    dPsiIn *= 1/normIn
    d2PsiUp *= 1/normUp
    d2PsiIn *= 1/normIn

    All0 = -0.5*pow(rho, -1)*rhobar*(SlmPP + (dLd1 + dLd2 + 2.*rhoOverRhoP)*SlmP + (dLd2p + dLd1*dLd2 - 6.*pow(rhoOverRhoP, 2) + 3.*rhoOverRhoPP + (3.*dLd1 - dLd2)*rhoOverRhoP)*Slm)
    Alm0 = 2.*pow(np.sqrt(2.)*rho, -1)*( -1.*(rho + rhobar + 1j*K/Delta)*(SlmP + dLd2*Slm) + (rho - rhobar)*a*sthp*K/Delta*Slm)
    Alm1 = 2.*pow(np.sqrt(2.)*rho, -1)*( SlmP + dLd2*Slm + 1j*(rho - rhobar)*a*sthp*Slm )
    Amm0 = -pow(rho, -1)*pow(rhobar, -1)*Slm*( 1j*(Kp/Delta - DeltaP/Delta*K/Delta) - K/Delta*K/Delta + 2.*1j*rho*K/Delta )
    Amm1 = 2.*pow(rho, -1)*pow(rhobar, -1)*Slm*( 1j*K/Delta + rho )
    Amm2 = -pow(rho, -1)*pow(rhobar, -1)*Slm

    u1, u3 = u_13_coeffs(geo, 1, 1, rp, cthp)

    Cll = u1*u1
    Clm = u1*u3
    Cmm = u3*u3

    A0 = All0*Cll + Alm0*Clm + Amm0*Cmm
    A1 = -(Alm1*Clm + Amm1*Cmm)
    A2 = Amm2*Cmm

    gs = sigma_r_deriv_sigma(sp, kappa)
    dgs = sigma_r_deriv_sigma_deriv(sp, kappa)

    Zs = Rslm.Z_sigma(sp)*Delta**2
    dZs = Rslm.Z_sigma_deriv(sp)*Delta**2 + 2.*Rslm.Z_sigma(sp)*Delta*DeltaP/gs
    d2Zs = Rslm.Z_sigma_deriv2(sp)*Delta**2 + 4.*Delta*DeltaP/gs*Rslm.Z_sigma_deriv(sp) + 2.*Rslm.Z_sigma(sp)*(DeltaPP*Delta + DeltaP**2 - Delta*DeltaP*dgs)/gs**2

    B0 = A0*Zs + A1*gs*dZs + A2*gs*(dgs*dZs + gs*d2Zs)
    B1 = A1*gs*Zs + A2*gs*(dgs*Zs + 2.*gs*dZs)
    B2 = A2*gs*gs*Zs

    integrand = np.array([(B0*PsiUp + B1*dPsiUp + B2*d2PsiUp), (B0*PsiIn + B1*dPsiIn + B2*d2PsiIn)])

    Gamma = geo.mino_frequencies[0]
    wronskian = PsiIn*dPsiUp - PsiUp*dPsiIn
    rescale = Rslm.Z_sigma(sp)**2*(rp**2 - 2.*rp + a**2)**(s+1)*gs
    const_wronskian = np.array([normIn, normUp])*rescale*wronskian

    amplitudes = -8.*np.pi*integrand/const_wronskian/Gamma
    amplitudes_max_error = 0.*amplitudes

    return amplitudes, amplitudes_max_error

def gravitational_source_eccentric(l, m, n, geo, Sslm = None, Rslm = None, **kwargs):
    omega = geo.mode_frequency(m, 0, n)
    a = geo.a
    kappa = np.sqrt(1 - a**2)
    En, Lz, _ = geo.constants

    if Sslm is None:
        Sslm = SpinWeightedSpheroidalHarmonic(-2, l, m, a*omega)

    if "nsamples" in kwargs.keys():
        nsamples = kwargs["nsamples"]
    else:
        nsamples = 2**9
    
    th = 0.5*np.pi
    uth = 0.
    sth = np.sin(th)
    cth = np.cos(th)

    Seq = Sslm.eval(th)
    dSeq = Sslm.deriv(th)
    d2Seq = Sslm.deriv2(th)

    L1 = -m/sth + a*omega*sth + cth/sth
    L2 = -m/sth + a*omega*sth + 2.*cth/sth
    L2S = dSeq + L2*Seq
    L2p = m*cth/sth**2 + a*omega*cth - 2./sth**2
    L1Sp = d2Seq + L1*dSeq
    L1L2S = L1Sp + L2p*Seq + L2*dSeq + L1*L2*Seq

    rmin = geo.radial_roots[1]
    rmax = geo.radial_roots[0]
    smin = sigma_r(rmax, kappa)
    smax = sigma_r(rmin, kappa)

    if Rslm is None:
        Rslm = HyperboloidalTeukolsky(a, -2, l, m, omega, domains=[[0, smax], [1, smin]])
    
    Psihbl = Rslm.psi
    qr = np.linspace(0, np.pi, nsamples + 1)
    rp = geo.r(qr)
    urp = geo.ur(qr)
    urp[0] = 0.
    urp[-1] = 0. # we set this to zero to avoid rounding errors near the turning points

    # start = time.time()
    # sp = Rslm.sigma_r(rp)
    # PsiUp = Psihbl["Up"](sp)
    # PsiIn = Psihbl["In"](sp)
    # print(time.time() - start)
    # start = time.time()
    # dPsiUp = Psihbl["Up"](sp, deriv=1)
    # dPsiIn = Psihbl["In"](sp, deriv=1)
    # print(time.time() - start)
    # start = time.time()
    # d2PsiUp = Psihbl["Up"](sp, deriv=2)
    # d2PsiIn = Psihbl["In"](sp, deriv=2)
    # print(time.time() - start)

    sp = Rslm.sigma_r(rp)
    PsiUp = Psihbl["Up"](sp)
    PsiIn = Psihbl["In"](sp)
    dPsiUp = Psihbl["Up"](sp, deriv=1)
    dPsiIn = Psihbl["In"](sp, deriv=1)
    d2PsiUp = Psihbl["Up"](sp, deriv=2)
    d2PsiIn = Psihbl["In"](sp, deriv=2)

    normUp = np.max(np.abs([PsiUp, dPsiUp, d2PsiUp]))
    normIn = np.max(np.abs([PsiIn, dPsiIn, d2PsiIn]))

    PsiUp *= 1/normUp
    PsiIn *= 1/normIn
    dPsiUp *= 1/normUp
    dPsiIn *= 1/normIn
    d2PsiUp *= 1/normUp
    d2PsiIn *= 1/normIn

    Delta = rp**2 - 2.*rp + a**2
    DeltaSqrt2 = np.sqrt(2)*Delta
    Kt = (rp**2 + a**2)*omega - m*a
    rho = -1./(rp - 1j*a*cth)
    rhobar = -1./(rp + 1j*a*cth)
    Sigma = 1/rho/rhobar

    Ann0 = -rho**(-2)*rhobar**(-1)*DeltaSqrt2**(-2)*(rho**(-1)*L1L2S + 3j*a*sth*L1*Seq + 3j*a*cth*Seq + 2j*a*sth*dSeq - 1j*a*sth*L2*Seq)
    Anmbar0 = rho**(-3)*DeltaSqrt2**(-1)*((rho + rhobar - 1j*Kt/Delta)*L2S + (rho - rhobar)*a*sth*Kt/Delta*Seq)
    Anmbar1 = -rho**(-3)*DeltaSqrt2**(-1)*( L2S + 1j*(rho - rhobar)*a*sth*Seq)
    Ambarmbar0 = rho**(-3)*rhobar*DeltaSqrt2**(-2)*(0.5*Kt**2 + 1j*Kt*(1. - rp + Delta*rho) + 1j*rp*omega*Delta)*Seq
    Ambarmbar1 = -0.5*(rho)**(-3)*rhobar*(1j*Kt/Delta - rho)*Seq
    Ambarmbar2 = -0.25*(rho)**(-3)*rhobar*Seq

    # print(np.array([Ann0, Anmbar0, Anmbar1, Ambarmbar0, Ambarmbar1, Ambarmbar2])[:, 0])

    Cr = (En*(rp**2 + a**2) - a*Lz + np.array([urp, -urp]))/(2*Sigma)
    Cth = rho*(1j*sth*(a*En - Lz/sth**2) + uth)/np.sqrt(2)

    # print(np.array([Cr, Cth])[:,0])

    Cnn = Cr*Cr
    Cnmbar = Cr*Cth
    Cmbarmbar = Cth*Cth

    rphase = omega*geo.tr(qr) - m*geo.phir(qr) + n*qr

    A0 = Ann0*Cnn + Anmbar0*Cnmbar + Ambarmbar0*Cmbarmbar
    A1 = -(Anmbar1*Cnmbar + Ambarmbar1*Cmbarmbar)
    A2 = Ambarmbar2*Cmbarmbar

    Zs = Rslm.Z_sigma(sp)
    dZs = Rslm.Z_sigma_deriv(sp)
    d2Zs = Rslm.Z_sigma_deriv2(sp)

    gs = sigma_r_deriv_sigma(sp, kappa)
    dgs = sigma_r_deriv_sigma_deriv(sp, kappa)

    B0 = A0*Zs + A1*gs*dZs + A2*gs*(dgs*dZs + gs*d2Zs)
    B1 = A1*gs*Zs + A2*gs*(dgs*Zs + 2.*gs*dZs)
    B2 = A2*gs*gs*Zs

    integrand_base_P = np.array([(B0[0]*PsiUp + B1[0]*dPsiUp + B2*d2PsiUp), (B0[0]*PsiIn + B1[0]*dPsiIn + B2*d2PsiIn)])
    integrand_base_M = np.array([(B0[1]*PsiUp + B1[1]*dPsiUp + B2*d2PsiUp), (B0[1]*PsiIn + B1[1]*dPsiIn + B2*d2PsiIn)])

    # qr = 0 comes from plus and qr = pi comes from minus to avoid double counting
    integrand_P = integrand_base_P*np.exp(1j*rphase) 
    integrand_M = integrand_base_M*np.exp(-1j*rphase)

    integrand_half0 = integrand_P[:, ::2][:, 1:] + integrand_M[:, ::2][:, :-1]
    integrand_half1 = integrand_P[:, 1:-1][:, ::2] + integrand_M[:, 1:-1][:, ::2]

    # plt.plot(np.abs(integrand_half0[0].real))
    # plt.yscale('log')
    # plt.show()

    integrate0 = 0.5*np.mean(integrand_half0, axis = 1)
    integrate1 = 0.5*np.sum(integrand_half1, axis = 1)/integrand_half0.shape[1]

    amplitudes_unscaled = 0.5*(integrate0 + integrate1)
    integration_error = 0.5*np.abs(1. - integrate1/integrate0)
    amplitudes_cancellation_error = (1e-15)/np.abs(amplitudes_unscaled/np.mean(integrand_base_P + integrand_base_M, axis = 1))
    amplitudes_max_error = np.max([amplitudes_cancellation_error, integration_error], axis = 0)

    Gamma = geo.mino_frequencies[0]
    wronskian = PsiIn[0]*dPsiUp[0] - PsiUp[0]*dPsiIn[0]
    rescale = Rslm.Z_sigma(smax)**2*(rmin**2 - 2.*rmin + a**2)**(-1)*sigma_r_deriv(rmin, kappa)
    const_wronskian = np.array([normIn, normUp])*rescale*wronskian

    amplitudes = -8.*np.pi*amplitudes_unscaled/const_wronskian/Gamma

    return amplitudes, amplitudes_max_error