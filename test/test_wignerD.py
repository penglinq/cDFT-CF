'''
constrained dft with given magnetic quantum number
'''
import pyscf
from functools import reduce
from pyscf import scf
from pyscf.scf.hf import RHF
from pyscf.dft.uks import UKS
from pyscf.dft.gks import GKS
from pyscf import lib
import os, sys
import numpy as np
import numpy
from numpy import sqrt
import scipy.linalg as la
from scipy.special import factorial
import scipy
from scipy.linalg import sqrtm, eigh
from scipy.optimize import root
np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=190, suppress=True)
       
def dmatrix(l, beta, reorder_p=False):
    '''Wigner small-d matrix (in z-y-z convention) from PySCF'''
    c = numpy.cos(beta/2)
    s = numpy.sin(beta/2)
    if l == 0:
        return numpy.eye(1)
    elif l == 1:
        mat = numpy.array(((c**2        , sqrt(2)*c*s , s**2       ), \
                           (-sqrt(2)*c*s, c**2-s**2   , sqrt(2)*c*s), \
                           (s**2        , -sqrt(2)*c*s, c**2       )))
        if reorder_p:
            mat = mat[[2,0,1]][:,[2,0,1]]
        return mat
    elif l == 2:
        c3s = c**3*s
        s3c = s**3*c
        c2s2 = (c*s)**2
        c4 = c**4
        s4 = s**4
        s631 = sqrt(6)*(c3s-s3c)
        s622 = sqrt(6)*c2s2
        c4s2 = c4-3*c2s2
        c2s4 = 3*c2s2-s4
        c4s4 = c4-4*c2s2+s4
        return numpy.array((( c4    , 2*c3s, s622, 2*s3c, s4   ),
                            (-2*c3s , c4s2 , s631, c2s4 , 2*s3c),
                            ( s622  ,-s631 , c4s4, s631 , s622 ),
                            (-2*s3c , c2s4 ,-s631, c4s2 , 2*c3s),
                            ( s4    ,-2*s3c, s622,-2*c3s, c4   )))
    else:
        facs = factorial(numpy.arange(2*l+1))
        cs = c**numpy.arange(2*l+1)
        ss = s**numpy.arange(2*l+1)

        mat = numpy.zeros((2*l+1,2*l+1))
        for i,m1 in enumerate(range(-l, l+1)):
            for j,m2 in enumerate(range(-l, l+1)):
                #fac = sqrt( factorial(l+m1)*factorial(l-m1) \
                #           *factorial(l+m2)*factorial(l-m2))
                #for k in range(max(m2-m1,0), min(l+m2, l-m1)+1):
                #    mat[i,j] += (-1)**(m1+m2+k) \
                #            * c**(2*l+m2-m1-2*k) * s**(m1-m2+2*k) \
                #            / (factorial(l+m2-k) * factorial(k) \
                #               * factorial(m1-m2+k) * factorial(l-m1-k))
                #mat[i,j] *= fac
                k = numpy.arange(max(m2-m1,0), min(l+m2, l-m1)+1)
                tmp = (cs[2*l+m2-m1-2*k] * ss[m1-m2+2*k] /
                       (facs[l+m2-k] * facs[k] * facs[m1-m2+k] * facs[l-m1-k]))

                mask = ((m1+m2+k) & 0b1).astype(bool)
                mat[i,j] -= tmp[ mask].sum()
                mat[i,j] += tmp[~mask].sum()

        ms = numpy.arange(-l, l+1)
        msfac = numpy.sqrt(facs[l+ms] * facs[l-ms])
        mat *= numpy.einsum('i,j->ij', msfac, msfac)
    return mat

def get_wigner_d(J, m, k, beta):
    """
    Wigner small d-matrix.
    Copied from Shu Fay and Xing.
    """
    nmin = int(max(0, k-m)) 
    nmax = int(min(J+k, J-m))
    sum_arr = []
    num = np.sqrt(
        scipy.special.factorial(J+k) * scipy.special.factorial(J-k) *
        scipy.special.factorial(J+m) * scipy.special.factorial(J-m))

    for n in range(nmin, nmax+1):
        denom = (scipy.special.factorial(J+k-n) * scipy.special.factorial(J-n-m) * 
                 scipy.special.factorial(n-k+m) * scipy.special.factorial(n))
        
        sign = (-1.)**(n - k + m) 
        cos = (np.cos(beta/2))**(2*J - 2*n + k - m)
        sin = (np.sin(beta/2))**(2*n - k + m)
        sum_arr.append(sign * num / denom * cos * sin)
        
    return sum(sum_arr)

def Sx_matrix(S):
    '''
        m in ascending order.
    '''
    dim = int(2*S + 1)
    Sx = np.zeros((dim, dim))

    for m in np.arange(-S, S+1):
        if m < S:
            Sx[int(S-m), int(S-(m+1))] = 0.5 * np.sqrt(S*(S+1) - m*(m+1))
        if m > -S:
            Sx[int(S-m), int(S-(m-1))] = 0.5 * np.sqrt(S*(S+1) - m*(m-1))
                                                                        
    return Sx[::-1][:, ::-1]

def Sy_matrix(S):
    '''
        m in ascending order.
    '''
    dim = int(2 * S + 1)
    Sy = np.zeros((dim, dim), dtype=complex)

    for m in np.arange(-S, S + 1):
        if m < S:
            Sy[int(S - m), int(S - (m + 1))] = 0.5j * np.sqrt(S * (S + 1) - m * (m + 1))
        if m > -S: 
            Sy[int(S - m), int(S - (m - 1))] = -0.5j * np.sqrt(S * (S + 1) - m * (m - 1))

    return Sy[::-1][:, ::-1]

def Sz_matrix(S):
    '''
        m in ascending order.
    '''
    dim = int(2 * S + 1)
    Sz = np.diag(np.arange(-S, S+1, dtype=complex))
    return Sz

if __name__ == "__main__":
    # calculate zeeman effect
    # test wigner matrix to perform rotations when B field is not along z axis
    np.set_printoptions(precision=4, suppress=True)
    J = 8
    g = 1.24
    de = 1.8
    de2 = 13
    Jmol2cm = 11.96265919
    kbT = 1.85 *1.3806*6.02214/Jmol2cm # in cm-1
    mu_Na = 0.9274 * 6.02214
    # two-state model
    H = np.zeros((17,17))
    H[4, 4] = -10000000
    H[-5, -5] = -10000000
    H[4,-5] = - de/2 # determine energy splitting of ground doublets at zero field
    H[-5, 4] = de/2
    Hzee = np.diag(np.arange(-8,9)*g*mu_Na/Jmol2cm)
    Sz = Sz_matrix(S=J)
    for k in range(20, 40, 40): # change k range here to change rotation angle
        wig_d = np.zeros((17, 17))
        for i in range(17):
            for j in range(17):
                wig_d[i,j] = get_wigner_d(J, i-8, j-8, k*np.pi/40)
        #print(wig_d)
        H_tmp = wig_d.T @ H @ wig_d
        #print(H_tmp)
        for B in np.arange(0, 5, 0.1): # before saturation
        #for B in np.arange(20, 500, 1000): # at saturation
            H_tot = H_tmp + Hzee * B
            e, v = la.eigh(H_tot)
            #print(- g*np.diag(v[:,:2].T @ Sz @ v[:,:2])[0].real,',')
            #print(np.diag(v[:,:2].T @ (Hzee * B) @ v[:,:2])) # check delta_E
            p0 = np.exp(-(e[0] - e[0])/kbT)/(np.exp(-(e[0] - e[0])/kbT) + np.exp(-(e[1] - e[0])/kbT))
            p1 = np.exp(-(e[1] - e[0])/kbT)/(np.exp(-(e[0] - e[0])/kbT) + np.exp(-(e[1] - e[0])/kbT))
            print(- g*np.diag(v[:,:2].T @ Sz @ v[:,:2]).real.dot([p0, p1]), ',')

    # Test dmatrix for integer J by comparing with pyscf's implementation
    # Half integer J was not implemented in PySCF
    J = 8 
    N_beta = 2
    xs, ws = np.polynomial.legendre.leggauss(N_beta)
    betas = np.arccos(xs)
    sorted_inds = np.argsort(betas)
    betas.sort()
    ws = ws[sorted_inds] / (- np.sin(betas))
    
    for m in np.arange(-J, J+1):
        for k in np.arange(-J, J+1):
            for beta in betas: 
                print("\n### beta ###", beta)
                #a = dmatrix(J, beta)[J+m, J+k]
                a = pyscf.symm.Dmatrix.dmatrix(J, beta)[J+m, J+k]
                print("pyscf wd", a)
                b = get_wigner_d(J,m,k, beta)
                print("shufay wd", b)
                assert abs(a - b) < 1e-6
    
