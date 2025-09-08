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
import scipy.linalg as la
import scipy
from scipy.linalg import sqrtm, eigh
from scipy.optimize import root
np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=190, suppress=True)

def func(beta):
    return np.sin(beta)

# Quadrature points and weights.
# x = cos(beta) in [-1, 1]
N_beta = 10
xs, ws = np.polynomial.legendre.leggauss(N_beta)
betas = np.arccos(xs)
sorted_inds = np.argsort(betas)
betas.sort()
ws = ws[sorted_inds] / (- np.sin(betas))

print("without rescaling weight (just f(x_i)w_i), correct")
count = 0
for n, beta in enumerate(betas):
    count += func(beta) * ws[n] * np.sin(beta)
print(count)

print("with rescaling weight, wrong")
count = 0
for n, beta in enumerate(betas):
    count += func(beta) * ws[n] 
print(count)

