"""Old-API-compatible RadialTeukolsky / HyperboloidalTeukolsky classes,
mirroring the pre-refactor tutorial API (per-boundary-condition solve(),
callable access, linspace helpers, In/Up cross-conversion) on top of the
current hblteuk.teuk primitives. See legacy_tutorial.ipynb.
"""
import numpy as np
from hblteuk import teuk as _teuk
from hblteuk import collocode as _collocode
from hblteuk import swsh as _swsh


def _solve_branch(base_solver, subdomains, spacing, kappa, s, ma, omega,
                   eigenvalue, bc, domain, use_ts_transform, horizon):
    solver_kwargs = {"subdomains": subdomains, "spacing": spacing}
    lam_ch = eigenvalue + s * (s + 1)
    if bc == "Up":
        y0 = 1
        stab = use_ts_transform and s == -2
        if stab:
            lam_flip = eigenvalue - 4.
            bc0 = _teuk.boundary_condition_up(kappa, 2, lam_flip, ma, omega, scale=y0)
            psi_flip = base_solver(_teuk.teuk_sys, bc0, args=(kappa, 2, lam_flip, ma, omega),
                                    domain=domain, **solver_kwargs)
            psi = _teuk.ts_up_plus_to_minus_2(psi_flip, kappa, lam_ch, ma, omega)
        else:
            bc0 = _teuk.boundary_condition_up(kappa, s, eigenvalue, ma, omega, scale=y0)
            psi = base_solver(_teuk.teuk_sys, bc0, args=(kappa, s, eigenvalue, ma, omega),
                               domain=domain, **solver_kwargs)
    else:
        y1 = 2. * kappa * np.exp(-1.j * (2. * omega - 0.5 * ma / horizon) * (1. + kappa + 2. * np.log(kappa)))
        stab = use_ts_transform and s == 2
        if stab:
            lam_flip = eigenvalue + 4.
            bc0 = _teuk.boundary_condition_in(kappa, -2, lam_flip, ma, omega, scale=y1)
            psi_flip = base_solver(_teuk.teuk_sys, bc0, args=(kappa, -2, lam_flip, ma, omega),
                                    domain=domain, **solver_kwargs)
            psi = _teuk.ts_in_minus_to_plus_2(psi_flip, kappa, lam_ch, ma, omega)
        else:
            bc0 = _teuk.boundary_condition_in(kappa, s, eigenvalue, ma, omega, scale=y1)
            psi = base_solver(_teuk.teuk_sys, bc0, args=(kappa, s, eigenvalue, ma, omega),
                               domain=domain, **solver_kwargs)
    return psi


class _TeukolskySolver:
    def __init__(self, a, s, l, m, omega, eigenvalue=None):
        self.a = a
        self.s = s
        self.l = l
        self.m = m
        self.frequency = omega
        self.kappa = np.sqrt(1. - a**2)
        self.horizon = 1. + self.kappa
        self.eigenvalue = eigenvalue if eigenvalue is not None else _swsh.swsh_eigenvalue(s, l, m, a * omega)
        self.bcs = ["In", "Up"]
        self._psi = dict.fromkeys(self.bcs)
        self.domain = {"In": [self.horizon, np.inf], "Up": [0., np.inf]}
        self.solution = dict.fromkeys(self.bcs)

    def sigma_of_r(self, r):
        return _teuk.sigma_r(r, self.kappa)

    def r_of_sigma(self, sigma):
        return _teuk.r_sigma(sigma, self.kappa)

    def _default_cutoff(self):
        # sigma bounds: In's far (small-sigma) edge scales with omega, Up's
        # near-horizon (large-sigma) edge sits a fixed proper distance out.
        smin = 0.03 * np.abs(self.frequency)**(2. / 3.)
        smax = self.sigma_of_r(2. * self.horizon - self.a)
        return smin, smax

    def _solve_one(self, bc, use_ts_transform, cutoff, subdomains, chebyshev_samples):
        smin_default, smax_default = self._default_cutoff()
        if cutoff is None:
            smin, smax = smin_default, smax_default
        else:
            if isinstance(cutoff, (list, np.ndarray, tuple)):
                rmin, rmax = cutoff
            else:
                rmin = rmax = cutoff
            smax = self.sigma_of_r(rmin)
            smin = self.sigma_of_r(rmax) if np.isfinite(rmax) else 0.

        # In's boundary condition anchors at sigma=1 (the horizon), so the
        # solve must march from domain[0]=1 down to smin -- domain[0] is
        # where the collocation solver applies the BC (see
        # HyperboloidalTeukolskySolver.__init__'s own reversal of "In").
        domain = [0., smax] if bc == "Up" else [1., smin]

        boundaryNum = subdomains if subdomains else max(16, 4 * self.l)

        base_solver = _collocode.CollocationODEFixedMultiDomainFixedStepSolver(n=chebyshev_samples)
        psi = _solve_branch(base_solver, boundaryNum, "arcsinh7", self.kappa, self.s,
                             self.m * self.a, self.frequency, self.eigenvalue, bc, domain,
                             use_ts_transform, self.horizon)
        self._psi[bc] = psi
        return psi


class RadialTeukolskySolution:
    def __init__(self, a, s, l, m, omega, eigenvalue, bc, psi):
        self._sol = _teuk.HyperboloidalTeukolskySolution(a, s, l, m, omega, eigenvalue, psi)
        self.a, self.s, self.l, self.m = a, s, l, m
        self.frequency = omega
        self.eigenvalue = eigenvalue
        self.bc = bc
        self.kappa = np.sqrt(1. - a**2)
        self.horizon = 1. + self.kappa

        smin, smax = min(psi.domain), max(psi.domain)
        r_lo = self.horizon if smax >= 1. else _teuk.r_sigma(smax, self.kappa)
        r_hi = np.inf if smin <= 0. else _teuk.r_sigma(smin, self.kappa)
        self.domain = sorted([r_lo, r_hi])

    def linspace_var(self, cutoff, n=100):
        rmin, rmax = self.domain
        if rmax == np.inf:
            rmax = cutoff
        if rmin == self.horizon:
            rmin = cutoff
        return np.linspace(rmin, rmax, n)

    def linspace(self, cutoff, n=100):
        r = self.linspace_var(cutoff, n)
        return [r, self.sol(r)]

    def sol(self, r):
        return self._sol(r, deriv=0, slicing="time", scaled=False, compactification=False)

    def deriv(self, r):
        return self._sol(r, deriv=1, slicing="time", scaled=False, compactification=False)

    def __call__(self, r, deriv=0):
        return self._sol(r, deriv=deriv, slicing="time", scaled=False, compactification=False)

    def _repr_latex_(self):
        sub_str = "{}, {}, {}, {:.2f}, {:.2f}".format(self.s, self.l, self.m, self.a, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$R^\mathrm{' + self.bc + '}_{slm\\omega}(r; ' + sub_str + ')$, $r \\in ' + domain_str + ' $}'


class HyperboloidalTeukolskySolution:
    def __init__(self, a, s, l, m, omega, eigenvalue, bc, psi):
        self._sol = _teuk.HyperboloidalTeukolskySolution(a, s, l, m, omega, eigenvalue, psi)
        self.a, self.s, self.l, self.m = a, s, l, m
        self.frequency = omega
        self.eigenvalue = eigenvalue
        self.bc = bc
        self.domain = sorted([min(psi.domain), max(psi.domain)])

    def linspace_var(self, n=100):
        return np.linspace(self.domain[0], self.domain[1], n)

    def linspace(self, n=100):
        sigma = self.linspace_var(n)
        return [sigma, self.sol(sigma)]

    def sol(self, sigma):
        return self._sol(sigma, deriv=0)

    def deriv(self, sigma):
        return self._sol(sigma, deriv=1)

    def __call__(self, sigma, deriv=0):
        return self._sol(sigma, deriv=deriv)

    def _repr_latex_(self):
        sub_str = "{}, {}, {}, {}, {:.2f}".format(self.a, self.s, self.l, self.m, self.frequency)
        domain_str = "[{:.2f}, {:.2f}]".format(self.domain[0], self.domain[1])
        return r'{$\psi^\mathrm{' + self.bc + '}_{aslm\\omega}(\\sigma; ' + sub_str + ')$, $\\sigma \\in ' + domain_str + ' $}'


class RadialTeukolsky(_TeukolskySolver):
    # chebyshev_samples default is 64, not the tutorial-documented 16: the
    # non-adaptive Fixed multi-domain solver gives no warning when
    # under-resolved, and at n=16 the "In" branch is not just less accurate
    # but wrong by an O(1) factor for l=16 (the amplitude grows ~1e28 across
    # the chain; n needs to be >~48 before results stop changing). n=64
    # matches the real HyperboloidalTeukolskySolver's own default n.
    def solve(self, bc=["In", "Up"], use_ts_transform=True, cutoff=None, subdomains=0, chebyshev_samples=64):
        if isinstance(bc, (list, np.ndarray)):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff,
                           subdomains=subdomains, chebyshev_samples=chebyshev_samples)
            return None
        psi = self._solve_one(bc, use_ts_transform, cutoff, subdomains, chebyshev_samples)
        self.solution[bc] = RadialTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                                      self.eigenvalue, bc, psi)
        self.domain[bc] = self.solution[bc].domain

    def get_hyperboloidal(self, bc=None):
        if bc is None:
            return {k: HyperboloidalTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                                        self.eigenvalue, k, self._psi[k]) for k in self.bcs}
        return HyperboloidalTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                               self.eigenvalue, bc, self._psi[bc])

    def __call__(self, bc=None):
        if bc is None:
            return self.solution
        return self.solution[bc]

    def _repr_latex_(self):
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        return r'$R^\mathrm{In/Up}_{sl\omega}(r; ' + sub_str + ')$'


class HyperboloidalTeukolsky(_TeukolskySolver):
    # chebyshev_samples default is 64, not the tutorial-documented 16: the
    # non-adaptive Fixed multi-domain solver gives no warning when
    # under-resolved, and at n=16 the "In" branch is not just less accurate
    # but wrong by an O(1) factor for l=16 (the amplitude grows ~1e28 across
    # the chain; n needs to be >~48 before results stop changing). n=64
    # matches the real HyperboloidalTeukolskySolver's own default n.
    def solve(self, bc=["In", "Up"], use_ts_transform=True, cutoff=None, subdomains=0, chebyshev_samples=64):
        if isinstance(bc, (list, np.ndarray)):
            for condition in bc:
                self.solve(condition, use_ts_transform=use_ts_transform, cutoff=cutoff,
                           subdomains=subdomains, chebyshev_samples=chebyshev_samples)
            return None
        psi = self._solve_one(bc, use_ts_transform, cutoff, subdomains, chebyshev_samples)
        self.solution[bc] = HyperboloidalTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                                             self.eigenvalue, bc, psi)
        self.domain[bc] = self.solution[bc].domain

    def get_teukolsky(self, bc=None):
        if bc is None:
            return {k: RadialTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                                self.eigenvalue, k, self._psi[k]) for k in self.bcs}
        return RadialTeukolskySolution(self.a, self.s, self.l, self.m, self.frequency,
                                        self.eigenvalue, bc, self._psi[bc])

    def __call__(self, bc=None):
        if bc is None:
            return self.solution
        return self.solution[bc]

    def _repr_latex_(self):
        sub_str = "{}, {}, {:.2f}".format(self.s, self.l, self.frequency)
        return r'$\Psi^\mathrm{In/Up}_{sl\omega}(\sigma; ' + sub_str + ')$'
