'''
constrained dft with given magnetic quantum number
'''
import pyscf
from pyscf import scf
from pyscf.scf.hf import RHF
from pyscf.dft.uks import UKS
from pyscf.dft.gks import GKS
from pyscf import lib
import os, sys
import numpy as np
import scipy.linalg as la
from scipy.linalg import sqrtm, eigh
from scipy.optimize import root
np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=190, suppress=True)
log = lib.logger.Logger(sys.stdout, 4)

if __name__ == "__main__":
    from pyscf import gto, scf, dft, lib
    from pyscf.lib import chkfile
    from stevens.momentum import get_mj

    # system input
    geometry = 'ErCO34' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 3 # number of unpaired electrons in a unit cell
    charge = -5 # charge of the molecule or a unit cell
    basis = {'default': '631g', 'Er': 'sarc-dkh'} # specify basis for all elements here
    max_memory = 40000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    qmmm = True # add environment with classical charges
    atom_index = 0 # the index of central lanthanide atom

    # (c)HF/DFT parameters
    chkfname = 'constrained_chkfile.chk' 
    xc_lst = ['R2SCAN', 'PBE', 'B3LYP', 'TPSS', 'M06',  'B97-D']
    # END USER INPUT

    # set up system
    if os.path.isfile('../../' + geometry + ".xyz"):
        mol = gto.M(
            atom = '../../' + geometry + ".xyz",
           basis = basis,
            verbose = 4,
            spin = spin,
            charge = charge,
            max_memory = max_memory,
            )
        mol.set_common_origin(mol._atom[atom_index][1])
    else:
        raise ValueError("No .xyz file available!")
    log.info("N_elec %s, N_ao %s"%(mol.nelec, mol.nao))

    assert chkfname is not None
    mf = scf.GKS(mol).x2c()
    data = chkfile.load(chkfname, 'scf')
    mf.__dict__.update(data)
    dm = mf.make_rdm1()
    for xc in xc_lst:
        mf = scf.GKS(mol).x2c()
        if qmmm: # add the precomputed potential from environmental classical charges 
            get_hcore_original = mf.get_hcore
            hcore_mm = np.load('../../mm_hcore.npy')
            def get_hcore_mm(self, mol=None):
                return get_hcore_original(mol=mol) + la.block_diag(hcore_mm, hcore_mm)
            import types
            mf.get_hcore = types.MethodType(get_hcore_mm, mf)
        mf.collinear = 'mcol'
        mf.grids.level = 6
        mf.xc = xc
        print("energy with functional %s"%mf.xc, mf.energy_tot(dm=dm))
    print("S vector", get_mj(mol, dm, include_L=False))
    print("L vector", get_mj(mol, dm, include_S=False))
    print("J vector", get_mj(mol, dm))
