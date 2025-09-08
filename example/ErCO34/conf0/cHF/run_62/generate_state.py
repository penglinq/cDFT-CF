'''
Example input to calculate the ground state of a molecular system
Include x2c relativistic effect
'''
import pyscf
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
    from stevens import cdft, utils
    from stevens.project import get_mj

    # system input
    geometry = 'ErCO34' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 3 # number of unpaired electrons in a unit cell
    charge = -5 # charge of the molecule or a unit cell
    basis = {'default': '631g', 'Er': 'sarc-dkh'} # specify basis for all elements here
    max_memory = 40000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    qmmm = True # add environment with classical charges

    # (c)HF/DFT parameters
    xc = 'HF' # functional
    smearing = None # To help HF/DFT converge, try e.g. 5e-2
    chkfname_guess = '../../gs_%s.chk'%geometry # initial guess. 'minao' for default PySCF guess; None for constrained HF; 
                # path to a checkpoint file to restart from a SCF solution.
                # 'project' to project from the SCF solution in a small basis to generate the initial guess
    chkfname_small = '../gs_small_%s.chk'%geometry # checkpoint file in a small basis. Used when chkfname_guess == 'project'
    basis_small = {'default': 'minao', 'Er': 'cc-x2c-minao'}# the small basis associated with chkfname_small
    change_dm_mode = 'rotate' # None: no change
                          # 'positive_z': set the 4f block of dm so that J~(0,0,J)
                          # 'rotate': rotate the 4f block of dm by some angles
    atom_index = 0 # the index of central lanthanide atom
    noise = 0 # max noise on density matrix initial guess

    # constrained HF parameters for the initial guess
    ao_shell = 'f'
    lo_chkfname = '../../UHF/uhf_gs_%s_lo.h5'%geometry  
    multiplier0 = 0.1
    dir_idx = 62
    plane = 'random'
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

    # generate HF/DFT initial guess
    if chkfname_guess == 'minao':
        mf = scf.GKS(mol).x2c()
        dm0 = mf.get_init_guess()
        print("Use PySCF default init guess")
    elif chkfname_guess == 'project':
        # project dm0 in a small basis to the larger basis 
        dm0 = utils.project_dm_from_small_basis(mol, basis_small, chkfname_small, change_dm_mode, atom_index=atom_index,
                occ_idx=0)
        print("Project wavefunction in a smaller basis to the target basis as init guess")
    else:
        mf = scf.GKS(mol).x2c()
        data = chkfile.load(chkfname_guess, 'scf')
        mf.__dict__.update(data)
        dm0 = mf.make_rdm1()
        mf.analyze(with_meta_lowdin=False)
        print("Load .chk file for init guess")
    angles_from_z, direction = cdft.sample_rotation(plane, dir_idx, dir_init=(0,0,1), unit=20, return_direction=True) # the rotation angles from +z
    if change_dm_mode is not None:  
        dm0  = utils.change_dm_guess(mol, dm0, change_dm_mode, atom_index=atom_index, angles_from_z=angles_from_z)
    dm0 += (np.random.rand(*(dm0.shape)) - 0.5) * noise  
    print("J vector of initial guess", get_mj(mol, dm0, analyze=True))
    print("target direction", direction)

    # run constrained HF/DFT
    cmf = cdft.CGKS(mol, direction, multiplier_guess=multiplier0, max_cycle=500,
            atom_index=0, ao_shell=ao_shell, lo_chkfname=lo_chkfname, xc=xc, smearing=smearing)
    if qmmm: # add the precomputed potential from environmental classical charges 
        mf = cmf._scf
        get_hcore_original = mf.get_hcore
        hcore_mm = np.load('../../mm_hcore.npy')
        def get_hcore_mm(self, mol=None):
            return get_hcore_original(mol=mol) + la.block_diag(hcore_mm, hcore_mm)
        import types
        mf.get_hcore = types.MethodType(get_hcore_mm, mf)
        cmf._scf = mf
    cmf.kernel(dm0=dm0)

    # calculate DFT energy
    dm = cmf._scf.make_rdm1()
    gmf = scf.GKS(mol).x2c()
    if xc.upper() != 'HF':
        gmf.grids.level = 6
        gmf.collinear = 'mcol'
    gmf.xc = xc
    if qmmm: # add the precomputed potential from environmental classical charges 
        get_hcore_original = gmf.get_hcore
        hcore_mm = np.load('../../mm_hcore.npy')
        def get_hcore_mm(self, mol=None):
            return get_hcore_original(mol=mol) + la.block_diag(hcore_mm, hcore_mm)
        import types
        gmf.get_hcore = types.MethodType(get_hcore_mm, gmf)
    print("energy with functional %s"%gmf.xc, gmf.energy_tot(dm=dm))
    gmf = scf.GKS(mol).x2c()
    gmf.grids.level = 6
    gmf.collinear = 'mcol'
    gmf.xc = 'PBE0'
    if qmmm: # add the precomputed potential from environmental classical charges 
        get_hcore_original = gmf.get_hcore
        hcore_mm = np.load('../../mm_hcore.npy')
        def get_hcore_mm(self, mol=None):
            return get_hcore_original(mol=mol) + la.block_diag(hcore_mm, hcore_mm)
        import types
        gmf.get_hcore = types.MethodType(get_hcore_mm, gmf)
    print("energy with functional %s"%gmf.xc, gmf.energy_tot(dm=dm))

    print("S vector", get_mj(mol, dm, include_L=False))
    print("L vector", get_mj(mol, dm, include_S=False))
    print("J vector", get_mj(mol, dm))
