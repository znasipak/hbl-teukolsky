import scipy.sparse
import scipy.sparse.linalg
import numpy as np

"""
Wigner 3j-symbol and Clebsch-Gordon coefficients
"""

def fac(n):
    if n < 0:
        return 0;
    return float(np.math.factorial(n));

def clebsch(l1, l2, l3, m1, m2, m3):
    return (-1)**(l1 - l2 + m3)*np.sqrt(2*l3 + 1)*w3j(l1, l2, l3, m1, m2, -m3);

def w3j(l1, l2, l3, m1, m2, m3):
    if m1 + m2 + m3 != 0:
        return 0;
    elif abs(l1 - l2) > l3:
        return 0;
    elif l1 + l2 < l3:
        return 0;
    
    if abs(m1) > l1:
        return 0;
    elif abs(m2) > l2:
        return 0;
    elif abs(m3) > l3:
        return 0;
    
    sumTerm = w3j_tsum(l1, l2, l3, m1, m2, m3);
    if sumTerm == 0:
        return 0;
    sumSign = np.sign(sumTerm);
    tempLog = 0.5*(np.log(fac(l1 + m1)) + np.log(fac(l2 + m2)) + np.log(fac(l3 + m3)));
    tempLog += 0.5*(np.log(fac(l1 - m1)) + np.log(fac(l2 - m2)) + np.log(fac(l3 - m3)));
    tempLog += np.log(triangle_coeff(l1, l2, l3));
    tempLog += np.log(abs(sumTerm));
    
    temp = sumSign*np.exp(tempLog);
    temp *= (-1)**(l1-l2-m3);
    
    return temp;
    
def w3j_tsum(l1, l2, l3, m1, m2, m3):
    t_min_num = w3j_t_min(l1, l2, l3, m1, m2, m3);
    t_max_num = w3j_t_max(l1, l2, l3, m1, m2, m3);
    x = 0;
    if t_max_num < t_min_num:
        t_max_num = t_min_num;

    for t in range(t_min_num - 1, t_max_num + 2):
        term = (fac(t)*fac(l3 - l2 + m1 + t)*fac(l3 - l1 - m2 + t)
            *fac(l1 + l2 - l3 - t)*fac(l1 - t - m1)*fac(l2 - t + m2));
        if term > 0:
            x += (-1)**t/term;
    
    return x;

def w3j_t_min(l1, l2, l3, m1, m2, m3):
    temp = 0;
    
    comp = l3 - l2 + m1;
    if temp + comp < 0:
        temp = -comp;
    comp = l3 - l1 - m2;
    if temp + comp < 0:
        temp = -comp;
        
    return temp;

def w3j_t_max(l1, l2, l3, m1, m2, m3):
    temp = 1;
    comp = l1 + l2 - l3;
    if comp - temp > 0:
        temp = comp;
    comp = l1 - m1;
    if comp - temp > 0:
        temp = comp;
    comp = l2 + m2;
    if comp - temp > 0:
        temp = comp;
        
    return temp;

def triangle_coeff(l1, l2, l3):
    return np.sqrt(fac(l1 + l2 - l3)*fac(l3 + l1 - l2)*fac(l2 + l3 - l1)/fac(l1 + l2 + l3 + 1));

"""
SWSH Eigenvalue Functions
"""

def k1(s, l, j, m):
    return np.sqrt((2*l + 1)/(2*j + 1))*clebsch(l, 1, j, m, 0, m)*clebsch(l, 1, j, -s, 0, -s);

def k2(s, l, j, m):
    ktemp = 2./3.*np.sqrt((2*l + 1)/(2*j + 1))*clebsch(l, 2, j, m, 0, m)*clebsch(l, 2, j, -s, 0, -s);
    if j == l:
        ktemp += 1/3.;
    return ktemp;

def k2m2(s, l, m):
    temp = (l - m - 1.)/(l - 1.)
    temp *= (l + m - 1.)/(l - 1.)
    temp *= np.float64(l - m)/l
    temp *= np.float64(l + m)/l
    temp *= (l - s)/(2.*l - 1.)
    temp *= (l + s)/(2.*l - 1.)
    temp *= (l - s - 1.)/(2.*l + 1.)
    temp *= (l + s - 1.)/(2.*l - 3.)
    return np.sqrt(temp)

def k2m1(s, l, m):
    temp = np.float64(l - m)*np.float64(l + m)/(2.*l - 1.)
    temp *= np.float64(l - s)*np.float64(l + s)/(2.*l + 1.)
    return -2.*m*s*np.sqrt(temp)/l/(l - 1.)/(l + 1.)

def k2p0(s, l, m):
    temp = np.float64(l*(l + 1.) - 3.*m*m)/(2.*l - 1.)/l
    temp *= np.float64(l*(l + 1.) - 3.*s*s)/(2.*l + 3.)/(l + 1.)
    return 1./3.*(1. + 2.*temp)

def k2p1(s, l, m):
    temp = np.float64(l - m + 1.)*np.float64(l + m + 1.)/(2.*l + 1.)
    temp *= np.float64(l - s + 1.)*np.float64(l + s + 1.)/(2.*l + 3.)
    return -2.*m*s*np.sqrt(temp)/l/(l + 1.)/(l + 2.)

def k2p2(s, l, m):
    temp = (l - m + 1.)/(l + 1.)
    temp *= (l + m + 1.)/(l + 1.)
    temp *= (l - m + 2.)/(l + 2.)
    temp *= (l + m + 2.)/(l + 2.)
    temp *= (l - s + 2.)/(2.*l + 3.)
    temp *= (l + s + 2.)/(2.*l + 3.)
    temp *= (l - s + 1.)/(2.*l + 1.)
    temp *= (l + s + 1.)/(2.*l + 5.)
    return np.sqrt(temp)

def k1m1(s, l, m):
    temp = np.float64(l - m)*np.float64(l + m)/(2.*l - 1.)
    temp *= np.float64(l - s)*np.float64(l + s)/(2.*l + 1.)
    return np.sqrt(temp)/l

def k1p0(s, l, m):
    return -np.float64(m*s)/l/(l + 1.)

def k1p1(s, l, m):
    temp = np.float64(l - m + 1.)*np.float64(l + m + 1.)/(2.*l + 3.)
    temp *= np.float64(l - s + 1.)*np.float64(l + s + 1.)/(2.*l + 1.)
    return np.sqrt(temp)/(l + 1.)

def akm2(s, l, m, g):
    return -g*g*k2m2(s, l, m)

def akm1(s, l, m, g):
    return -g*g*k2m1(s, l, m) + 2.*s*g*k1m1(s, l, m)

def akp0(s, l, m, g):
    return -g*g*k2p0(s, l, m) + 2.*s*g*k1p0(s, l, m) + l*(l + 1.) - s*(s + 1.) - 2.*m*g + g*g

def akp1(s, l, m, g):
    return -g*g*k2p1(s, l, m) + 2.*s*g*k1p1(s, l, m)

def akp2(s, l, m, g):
    return -g*g*k2p2(s, l, m)
    

# def akm2(s, l, m, g):
#     if l < 0 or abs(m) > l or abs(s) > l:
#         return 0;
#     return -g*g*k2(s, l - 2, l, m);

# def akm1(s, l, m, g):
#     if l < 0 or abs(m) > l or abs(s) > l:
#         return 0;
#     return -g*g*k2(s, l - 1, l, m) + 2*s*g*k1(s, l - 1, l, m);

# def akp0(s, l, m, g):
#     if l < 0 or abs(m) > l + 1 or abs(s) > l + 1:
#         return 0;
#     return -g*g*k2(s, l, l, m) + 2*s*g*k1(s, l, l, m) + l*(l + 1) - s*(s + 1) - 2*m*g + g*g;

# def akp1(s, l, m, g):
#     if l < 0 or abs(m) > l + 2 or abs(s) > l + 2:
#         return 0;
#     return -g*g*k2(s, l + 1, l, m) + 2*s*g*k1(s, l + 1, l, m);

# def akp2(s, l, m, g):
#     if l < 0 or abs(m) > l or abs(s) > l:
#         return 0;
#     return -g*g*k2(s, l + 2, l, m);

# def akm2_vec(s, l, m, g):
#     aarray = np.empty(l.shape[0])
#     for i, li in enumerate(l):
#         aarray[i] = akm2(s, li, m, g)
#     return aarray

# def akm1_vec(s, l, m, g):
#     aarray = np.empty(l.shape[0])
#     for i, li in enumerate(l):
#         aarray[i] = akm1(s, li, m, g)
#     return aarray

# def akp0_vec(s, l, m, g):
#     aarray = np.empty(l.shape[0])
#     for i, li in enumerate(l):
#         aarray[i] = akp0(s, li, m, g)
#     return aarray

# def akp1_vec(s, l, m, g):
#     aarray = np.empty(l.shape[0])
#     for i, li in enumerate(l):
#         aarray[i] = akp1(s, li, m, g)
#     return aarray

# def akp2_vec(s, l, m, g):
#     aarray = np.empty(l.shape[0])
#     for i, li in enumerate(l):
#         aarray[i] = akp2(s, li, m, g)
#     return aarray

def spectral_sparse_matrix(s, m, g, nmax):
    lmin = max(abs(s), abs(m));
    larray = np.arange(lmin, lmin + nmax)
    return scipy.sparse.diags([akm2(s, larray[2:], m, g), akm1(s, larray[1:], m, g), akp0(s, larray, m, g), akp1(s, larray[:-1], m, g), akp2(s, larray[:-2], m, g)], [-2, -1, 0, 1, 2])
    

def sYlm_eigenvalue(s, l, m):
    return l*(l + 1.) - s*(s + 1.)

def swsh_eigenvalue(s, l, m, g, nmax=None):
    if g == 0.:
        return sYlm_eigenvalue(s, l, m)
    
    lmin = max(abs(s), abs(m))
    kval = l - lmin
    
    if nmax is None:
        buffer = round(20 + 2*g)
        Nmax = kval + buffer + 2
    else:
        if nmax < kval:
            Nmax = kval + 5
        else:
            Nmax = nmax
    
    mat = spectral_sparse_matrix(s, m, g, Nmax)
    las = scipy.sparse.linalg.eigs(mat, k=Nmax-2, which='SM', return_eigenvectors=False)
    
    return np.real(las[::-1][kval])