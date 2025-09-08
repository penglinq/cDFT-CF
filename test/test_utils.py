'''
constrained dft with given magnetic quantum number
'''
import pyscf
from pyscf.scf.hf import RHF
from pyscf.dft.uks import UKS
from pyscf.dft.gks import GKS
from pyscf import lib
import os, sys
import numpy as np
import scipy.linalg as la
from stevens import utils
np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=190, suppress=True)

def test_find_close_atom():
    from libdmet.utils.misc import read_poscar

    # Ground-state HF input
    geometry = 'HoPy' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 4 # number of unpaired electrons in a unit cell
    from_poscar = True
    max_memory = 300000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    charge = 0  # use 0 for charge neutral unit cell
    basis = {'default': '631g', 'Dy': 'sarc-dkh', 'Ho': 'sarc-dkh', 'Y': 'cc-pvtz-dk'} # specify basis for other lanthanide here

    if from_poscar and os.path.isfile(geometry + ".vasp"):
        cell = read_poscar(fname=geometry + ".vasp")
        cell.basis = basis
        cell.spin = spin
        cell.charge = charge
        cell.max_memory = max_memory
        cell.verbose = 4
        cell.build()
        cell.set_common_origin(cell._atom[0][1])
    else:
        raise ValueError("No .xyz file available!")
    cell.build()

    # find O attached to Cl
    Cl_idx = np.where([cell._atom[i][0] == 'Cl' for i in range(cell.natm)])[0]
    Cl_coord = np.array(cell.atom_coords(unit='ANG')[Cl_idx])
    O_idx = utils.find_close_atom(cell, Cl_coord, 1.5, None)
    assert len(O_idx) == 48

def test_find_fragment():
    from libdmet.utils.misc import read_poscar

    # Ground-state HF input
    geometry = 'HoPy' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 4 # number of unpaired electrons in a unit cell
    from_poscar = True
    max_memory = 300000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    charge = 0  # use 0 for charge neutral unit cell
    basis = {'default': '631g', 'Dy': 'sarc-dkh', 'Ho': 'sarc-dkh', 'Y': 'cc-pvtz-dkminao'} # specify basis for other lanthanide here

    if from_poscar and os.path.isfile(geometry + ".vasp"):
        cell = read_poscar(fname=geometry + ".vasp")
        cell.basis = basis
        cell.spin = spin
        cell.charge = charge
        cell.max_memory = max_memory
        cell.verbose = 4
        cell.build()
        cell.set_common_origin(cell._atom[0][1])
    else:
        raise ValueError("No .xyz file available!")
    cell.build()
    assert len(utils.find_fragment(cell, 2)) == 32

def test_get_init_guess_fragment():
    from libdmet.utils.misc import read_poscar

    # Ground-state HF input
    geometry = 'HoPy' # The name of geometry file minus ".xyz"
                       # Make sure first atom is the central magnetic atom 
    spin = 4 # number of unpaired electrons in a unit cell
    from_poscar = True
    max_memory = 300000 # in the unit of MB. Recommend: memory available on a node - 10000 MB
    charge = 0  # use 0 for charge neutral unit cell
    basis = {'default': 'minao', 'Dy': 'sarc-dkh', 'Ho': 'sarcdkhminao', 'Y': 'cc-pvtz-dkminao'} # specify basis for other lanthanide here

    if from_poscar and os.path.isfile(geometry + ".vasp"):
        cell = read_poscar(fname=geometry + ".vasp")
        cell.basis = basis
        cell.spin = spin
        cell.charge = charge
        cell.max_memory = max_memory
        cell.verbose = 4
        cell.build()
        cell.set_common_origin(cell._atom[0][1])
    else:
        raise ValueError("No .xyz file available!")
    cell.build()
    fragment = utils.find_fragment(cell, 2)
    frag_charge = [3] * 4 + [0] * 16 + [-1] * 12
    frag_spin = [4] + [0] * 31
    
    dm = utils.get_init_guess_fragment_GHF(cell, fragment, frag_charge, frag_spin)
    np.save('dm.npy', dm)
    from pyscf.scf.ghf import mulliken_meta
    mulliken_meta(cell, dm)
    return 

def test_sph_dm_rotate():
    dma = np.diag(np.random.rand(7))
    u = pyscf.symm.sph.sph_pure2real(3)
    dmb = u.conj().T @ np.diag(np.random.rand(7)) @ u
    dm = la.block_diag(dma, dmb)
    assert np.max(np.abs(dm - utils.sph_dm_rotate(dm, (0,0,np.pi*2))[0])) < 1e-6
    assert np.max(np.abs(dm - utils.sph_dm_rotate(dm, (0,np.pi*2,0))[0])) < 1e-6
    assert np.max(np.abs(dm - utils.sph_dm_rotate(dm, (np.pi*2,0,0))[0])) < 1e-6
    assert np.max(np.abs(la.block_diag(u.conj().T @ ((u @ dmb @ u.conj().T)[::-1,::-1]) @ u,
                    u.conj().T @ ((u @ dma @ u.conj().T)[::-1,::-1]) @ u) - utils.sph_dm_rotate(dm, (0,np.pi,0))[0])) < 1e-6
    return 

def test_dm_rotate():
    # test inversion
    raise NotImplementedError
    u = pyscf.symm.sph.sph_pure2real(l=3)
    dm = dm.reshape(2, 7, 2, 7)
    dm = np.einsum('ipjq,rp,sq->irjs', dm, u, u.conj(), optimize=True) 
    dm = dm[:, ::-1, :, ::-1]
    dm = np.einsum('irjs,r,s->irjs', dm, (-1)**np.arange(7), (-1)**np.arange(7), optimize=True)
    dm = dm.conj()
    dm = np.einsum('irjs,rp,sq->ipjq', dm, u.conj(), u, optimize=True) 
    dm = dm.reshape(14,14)
    dm = dm.reshape(2, nao, 2, nao)[::-1, :, ::-1]
    dm[[0,1], :, [1,0]] = - dm[[0,1], :, [1,0]] 
    dm = dm.reshape(2*nao, 2*nao)

def test_get_rotation_to_z_from_vector():
    v = np.random.rand(3)
    angles = utils.get_rotation_to_z_from_vector(v)
    import test_rot
    v_final =  test_rot.R_cartesian_new(v, (0,0,0), angles)
    assert np.abs(la.norm(v_final) - la.norm(v)) < 1e-6
    v_final = v_final / la.norm(v_final)
    assert np.max(np.abs(v_final - [0,0,1])) < 1e-6


if __name__ == "__main__":
    #test_find_close_atom() 
    #test_find_fragment()
    #test_get_init_guess_fragment()
    #test_sph_dm_rotate()
    test_get_rotation_to_z_from_vector()
