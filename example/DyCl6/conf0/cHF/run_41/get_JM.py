'''
Expand a Slater Determinant in eigenfunctions of J amd Jz
'''
import pyscf
from functools import reduce
from pyscf import scf, lib, lo, symm, gto
import os, sys
import numpy as np
import scipy.linalg as la
import scipy
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
log = lib.logger.Logger(sys.stdout, 4)
np.set_printoptions(precision=4, threshold=sys.maxsize, linewidth=380, suppress=True)

if __name__ == "__main__":
    from pyscf import gto, scf, dft, lib
    from pyscf.lib import chkfile
    from stevens import project
    from stevens.momentum import get_mj

    # system input
    geometry = 'DyCl6' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 5 # number of unpaired electrons in a unit cell
    charge = -3 # charge of the molecule or a unit cell
    basis = {'default': '631g', 'Dy': 'sarc-dkh'} # specify basis for all elements here
    max_memory = 6000 # in the unit of MB. Recommend: memory available on a node - 10000 MB

    # other parameter
    chkfname = 'constrained_chkfile.chk'  # load PySCF cHF/DFT solution 
    lo_chkfname = '../../UHF/uhf_gs_%s_lo.h5'%geometry  
    atom_index = 0 # the index of central lanthanide atom
    ao_shell = 'f'
    N_grid = (19,10,19) 
    max_J = 7.5
    max_J_only = True
    project_type = 'full' # "spin" (S), "orb" (L), or "full" (J)
    diag_only = False

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

    # load cHF solution
    gmf = scf.GHF(mol).x2c()
    if chkfname is not None and os.path.isfile(chkfname):
        data = chkfile.load(chkfname, 'scf')
        gmf.__dict__.update(data)
    else: 
        gmf.chkfile = geometry + '.chk'
        gmf.kernel()
    if mpirank == 0:
        gmf.analyze(with_meta_lowdin=True)
        dm = gmf.make_rdm1()
        print("S vector", get_mj(mol, dm, include_L=False))
        print("L vector", get_mj(mol, dm, include_S=False))
        print("J vector", get_mj(mol, dm))
    
    # calculate JM projection
    comm.Barrier()
    proj = project.Project(mol, gmf, atom_index=atom_index, ao_shell=ao_shell, spin_separate=False, \
            N_grid=N_grid, lo_chkfname=lo_chkfname)

    c2_sum = 0
    if mpirank == 0:
        print("  %s     M     c^2  "%({"spin": "S", "orb": "L", "full": "J"}[project_type]))
    if project_type in ["spin", "full"] and abs(mol.spin%2) > 1e-4: 
        # Scan half-integer J. The rest is always 0.
        J_range = np.arange(int(np.floor(max_J))+0.5, 0, -1)
    else:
        # Scan integer J. The rest is always 0.
        J_range = np.arange(int(np.floor(max_J)), -0.5, -1)
    if max_J_only:
        J_range = J_range[:1]
    c2_Jmk = [] 
    for J in J_range:
        for m in np.arange(J,-J-1,-1):
            for k in np.arange(J,-J-1,-1):
                if (diag_only and np.abs(k-m) > 1e-10) or k-m > 1e-10: 
                    res = 0
                else:
                    # S: proj='spin'
                    # L: proj='orb'
                    # J: proj='full'
                    res = proj.get_proj_ovlp(J=J, m=m, k=k, proj=project_type, N_alpha=N_grid[0], N_beta=N_grid[1], \
                            N_gamma=N_grid[2], verbose=0)
                if J == max(J_range):
                    c2_Jmk.append(res) 
                if mpirank == 0 and la.norm(res) > 1e-5:
                    if abs(res.imag) > 1e-6:
                        print(" %3.1f   %4.1f   %4.1f   %8.5f   %8.5f"%(J,m,k, res.real, res.imag))
                    else:
                        print(" %3.1f   %4.1f   %4.1f   %8.5f"%(J,m,k, res.real))
                    if np.abs(m-k) < 1e-6:
                        c2_sum += res.real
    if mpirank == 0:
        c2_Jmk = np.array(c2_Jmk, dtype=complex).reshape((int(2*max(J_range)+1), int(2*max(J_range)+1)))
        print('triangle', c2_Jmk)
        c2_Jmk[np.tril_indices(len(c2_Jmk), -1)] = c2_Jmk.conj().T[np.tril_indices(len(c2_Jmk), -1)]
        c2_Jmk[np.diag_indices(len(c2_Jmk))] = c2_Jmk[np.diag_indices(len(c2_Jmk))].real # c*c is real 
        print('full', c2_Jmk)
        np.save('c2_Jmk.npy', c2_Jmk)
        print("Sum of c^2", c2_sum)



    
    
