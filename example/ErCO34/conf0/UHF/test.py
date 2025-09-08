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
    from stevens import cdft, project, utils
    from stevens.project import get_mj
    from stevens.localize import plot_lo, get_iao

    # system input
    geometry = 'ErCO34' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 3 # number of unpaired electrons in a unit cell
    charge = -5 # charge of the molecule or a unit cell
    basis = {'default': 'cc-pvdz', 'Er': 'cc-pvtz-x2c'} # specify basis for all elements here
    max_memory = 80000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    qmmm = True ###True # add environment with classical charges

    # (c)HF/DFT parameters
    smearing = None # To help HF/DFT converge, try e.g. 5e-2
    chkfname_guess = 'project' # initial guess. 'minao' for default PySCF guess; None for constrained HF; 
                # path to a checkpoint file to restart from a SCF solution.
                # 'project' to project from the SCF solution in a small basis to generate the initial guess
    chkfname_small = '../GS/gs_small_%s.chk'%geometry # checkpoint file in a small basis. Used when chkfname_guess == 'project'
    basis_small = {'default': 'minao', 'Er': 'cc-x2c-minao'}# the small basis associated with chkfname_small
    change_dm_mode = None # None: no change
                          # 'positive_z': set the 4f block of dm so that J~(0,0,J)
                          # 'rotate': rotate the 4f block of dm by some angles
    atom_index = 0 # the index of central lanthanide atom
    occ_idx = 1 # index of angles in a sequence to sample xz plane and xy plane
    noise = 0 # max noise on density matrix initial guess

    # localization
    ao_shell = 'f'
    loc_method = 'iao'
    # END USER INPUT

    # set up system
    if os.path.isfile('../' + geometry + ".xyz"):
        mol = gto.M(
            atom = '../' + geometry + ".xyz",
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
        mf = scf.UKS(mol).x2c()
        dm0 = mf.get_init_guess()
        print("Use PySCF default init guess")
    elif chkfname_guess == 'project':
        # project dm0 in a small basis to the larger basis 
        dm0 = utils.project_dm_from_small_basis(mol, basis_small, chkfname_small, change_dm_mode, atom_index=atom_index, occ_idx=occ_idx)
        print("Project wavefunction in a smaller basis to the target basis as init guess")
    else:
        mf = scf.UKS(mol).x2c()
        data = chkfile.load(chkfname_guess, 'scf')
        mf.__dict__.update(data)
        dm0 = mf.make_rdm1()
        mf.analyze(with_meta_lowdin=False)
        print("Load .chk file for init guess")
    if change_dm_mode is not None and not chkfname_guess == 'project': # when chkfname_guess == 'project', change in small basis
        dm0 = utils.change_dm_guess(mol, dm0, change_dm_mode, atom_index=atom_index, occ_idx=occ_idx)
    dm0 += (np.random.rand(*(dm0.shape)) - 0.5) * noise  
    print("J vector of initial guess", get_mj(mol, dm0, analyze=True))
    # change GHF dm0 to UHF format
    if dm0.ndim == 2:
        dm0 = np.array([dm0[:mol.nao, :mol.nao], dm0[mol.nao:, mol.nao:]]).real

    # run HF/DFT
    chkfname = 'uhf_gs_' + geometry + '.chk' 
    mf = scf.UHF(mol).sfx2c1e()
    mf.chkfile = chkfname
    mf.max_cycle = 1000 
    if smearing is not None:
        mf = scf.addons.smearing_(mf, sigma=smearing, method="fermi")
    if qmmm: # add the precomputed potential from environmental classical charges 
        get_hcore_original = mf.get_hcore
        hcore_mm = np.load('../mm_hcore.npy')
        def get_hcore_mm(self, mol=None):
            return get_hcore_original(mol=mol) + hcore_mm
        import types
        mf.get_hcore = types.MethodType(get_hcore_mm, mf)
    ###mf.kernel(dm0=dm0)
    data = chkfile.load(chkfname, 'scf')
    mf.__dict__.update(data)
    dm = mf.make_rdm1()
    S = mf.get_ovlp()
    print("nelec", np.einsum('sij,ji->s', dm, S))
    '''
    mo_i, mo_e, status_i, status_e = mf.stability(verbose=4, return_status=True)
    if not status_i:
        dm_new = mf.make_rdm1(mo_coeff=mo_i)
        log.info("Stability analysis changed dm by %s" % la.norm(dm - dm_new))
        mf.kernel(dm0=dm_new)
        dm = mf.make_rdm1()
        mo_i, mo_e, status_i, status_e = mf.stability(verbose=4, return_status=True)
        log.info("Final stability: %s" % status_i)
    '''

    log.info("S vector %s" % get_mj(mol, dm, include_L=False))
    log.info("L vector %s" % get_mj(mol, dm, include_S=False))
    log.info("J vector %s" % get_mj(mol, dm))
    c_lo = get_iao(mol, chkfname, from_umf=True, ao_ovlp=None)  
    import h5py
    feri = h5py.File(chkfname.replace(".chk", "_lo.h5"), 'w')
    feri['c_lo'] = np.array(c_lo)
    feri.close()
    plot_lo(mol, c_lo, ao_shell, atom_index=atom_index, loc_method=loc_method, alpha_only=True)

