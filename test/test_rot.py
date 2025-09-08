import pyscf
from stevens.project import Project
from functools import reduce
from pyscf import gto, scf, dft, lib, lo, symm
import os, sys
import numpy as np
import scipy.linalg as la
import scipy
mpirank = 0


def Ry_cartesian(r, center, beta):
    Ry = np.array([[np.cos(beta), 0, np.sin(beta)], 
                   [0, 1, 0],
                   [-np.sin(beta), 0, np.cos(beta)]])
    return Ry @ (r - center) + center

def Rz_cartesian(r, center, gamma):
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0], 
                   [np.sin(gamma), np.cos(gamma), 0],
                   [0, 0, 1]])
    return Rz @ (r - center) + center

def R_cartesian(r, center, angles): # deprecate
    alpha, beta, gamma = angles
    z = Rz_cartesian(r, center, gamma)
    yz = Ry_cartesian(z, center, beta)
    zyz = Rz_cartesian(yz, center, alpha)
    return zyz

def R_cartesian_new(r, center, angles): # alpha first, then beta, finally gamma
    alpha, beta, gamma = angles
    z = Rz_cartesian(r, center, alpha)
    yz = Ry_cartesian(z, center, beta)
    zyz = Rz_cartesian(yz, center, gamma)
    return zyz

    
def get_mo_ovlp(mo_coeff1, mo_coeff2, gmf):
    """
    Calculates the MO overlap matrix.
    """
    ao_ovlp = gmf.get_ovlp()
    s = reduce(lib.dot, (mo_coeff1.T.conj(), ao_ovlp, mo_coeff2))
    return s

def test_rot_DFT():
    from stevens import utils, project
    # User input
    geometry = 'Cl' # To start with, try 'H' 'C' 'Ho'
    angles = (0, np.pi/2, 0)

    # input a molecule
    mol = gto.M(
        atom = '''Cl 0 0 0''',
        symmetry = False,
        basis = 'cc-pvtz', 
        spin = 1, 
        charge = 0, 
        verbose = 4)
    mf = scf.UHF(mol) 
    mf.kernel()
    # Convert the UHF solution to GHF 
    gmf = scf.addons.convert_to_ghf(mf)
    dm = gmf.make_rdm1()
    print("S vector init", stevens.get_mj(mol, dm, include_L=False))
    dm_new = utils.spin_dm_rotate(dm, angles)[0]
    print("S vector after rotation", stevens.get_mj(mol, dm_new, include_L=False))

    print("***** Test HF *****")
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("init energy", en0) 
    print("energy after rotation", en1)
    assert np.abs(en0 - en1) < 1e-6

    # test DFT
    print("***** Test LDA *****")
    gmf = dft.GKS(mol)
    gmf.collinear = 'ncol'
    gmf.xc = 'lda'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("init energy", en0) 
    print("energy after rotation", en1)
    assert np.abs(en0 - en1) < 1e-6

    print("***** Test PBE *****")
    gmf = dft.GKS(mol)
    gmf.collinear = 'mcol'
    gmf.xc = 'lda'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("init energy", en0) 
    print("energy after rotation", en1)
    assert np.abs(en0 - en1) < 1e-6

    print("***** Test PBE0 *****")
    gmf = dft.GKS(mol)
    gmf.collinear = 'mcol'
    gmf.xc = 'PBE0'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("init energy", en0) 
    print("energy after rotation", en1)
    assert np.abs(en0 - en1) < 1e-6

def test_rot_PBCDFT():
    from pyscf.pbc import gto, scf, dft
    import utils, stevens
    # User input
    geometry = 'Cl' # To start with, try 'H' 'C' 'Ho'
    angles = (0, np.pi/2, 0)

    # input a molecule
    mol = gto.M(
        atom = '''Cl 0 0 0''',
        symmetry = False,
        basis = 'cc-pvtz', 
        a = np.diag([10, 10, 10]),
        spin = 1, 
        charge = 0, 
        verbose = 4)
    mf = scf.UHF(mol).density_fit()
    mf.kernel()
    # Convert the UHF solution to GHF 
    gmf = scf.addons.convert_to_ghf(mf)
    dm = gmf.make_rdm1()
    print("S vector init", stevens.get_mj(mol, dm, include_L=False))
    dm_new = utils.spin_dm_rotate(dm, angles)[0]
    if np.max(np.abs(dm_new.imag)) < 1e-6:
        dm_new = dm_new.real
    print("S vector after rotation", stevens.get_mj(mol, dm_new, include_L=False))

    print("***** Test HF *****")
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("***** init energy *****", en0) 
    print("**** energy after rotation *****", en1)
    assert np.abs(en0 - en1) < 1e-6

    # test DFT
    print("***** Test LDA *****")
    gmf = dft.GKS(mol).density_fit()
    gmf.collinear = 'ncol'
    gmf.xc = 'lda'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("***** init energy *****", en0) 
    print("**** energy after rotation *****", en1)
    assert np.abs(en0 - en1) < 1e-6

    print("***** Test PBE *****")
    gmf = dft.GKS(mol).density_fit()
    gmf.collinear = 'mcol'
    gmf.xc = 'lda'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("***** init energy *****", en0) 
    print("**** energy after rotation *****", en1)
    assert np.abs(en0 - en1) < 1e-6
    
    return 
    # TODO not implemented yet in PySCF
    print("***** Test PBE0 *****")
    gmf = dft.GKS(mol).density_fit()
    gmf.collinear = 'mcol'
    gmf.xc = 'PBE0'
    en0 = gmf.energy_tot(dm=dm)
    en1 = gmf.energy_tot(dm=dm_new)
    print("***** init energy *****", en0) 
    print("**** energy after rotation *****", en1)
    assert np.abs(en0 - en1) < 1e-6
 
def test_rot_mol():
    from pyscf.lib import chkfile

    # User input
    geometry = 'NeH4_xz' # To start with, try 'H' 'C' 'Ho'
    angles = (np.pi/2, 0, np.pi/2)
    center = [0,0,0]

    # input a molecule
    if geometry == 'NeH4_xy':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0 -1  0
                    H  0  1  0
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            charge = 2, 
            verbose = 6)
    elif geometry == 'NeH4_xz':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0  0 -1
                    H  0  0  1
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            charge = 2, 
            verbose = 6)

    # input some MO
    if geometry[:4] == 'NeH4':
        mf = scf.UHF(mol).run() 
        # create a full p shell which is invariant under spatial rotation
        if True:
            mo_coeff = np.eye(mol.nao)
            mo_coeff[-4:, -4:] = np.array([[1,1,1,0],[1,1,-1,0],[1,-1,0,1],[1,-1,0,-1]])
            ao_ovlp = mf.get_ovlp()
            mo_coeff[:,:-3] = lo.orth.vec_lowdin(mo_coeff[:,:-3], ao_ovlp) 
            mo_coeff[:,-3:] = lo.orth.vec_lowdin(mo_coeff[:,-3:], ao_ovlp) 
            s = mo_coeff.conj().T @ ao_ovlp @ mo_coeff
            print("MO ovlp", s)
            mf.mo_coeff = np.array([mo_coeff, mo_coeff])
        print("mo", mo_coeff)
        # Convert the UHF solution to GHF 
        gmf = scf.addons.convert_to_ghf(mf)
        if mpirank == 0:
            gmf.analyze(with_meta_lowdin=True)
    dm = gmf.make_rdm1()
    en1 = gmf.energy_tot(dm=dm)
    print("***** init energy *****", en1) 
    mo_coeff_ori = np.asarray(gmf.mo_coeff.copy(), dtype=complex)
    mo_coeff = sph_mo_rotate(mol, mo_coeff_ori, angles) # new mo_coeff 
    
    # new mol of only the rotated atoms
    mol2 = mol.copy()
    nao = mol.nao
    natom = mol.natm
    atom = mol._atom
    atom_new = []
    for i in range(natom):
        R_old = atom[i][1]
        R_new = R_cartesian(R_old, center, angles) 
        atom_new.append(tuple((atom[i][0], R_new)))
    mol2.atom = atom_new
    mol2.unit = 'B'
    mol2.spin = mol.spin
    mol2.build()
    gmf.mol = mol2
    gmf.mo_coeff = mo_coeff
    dm_new = gmf.make_rdm1()
    en2 = gmf.energy_tot(dm=dm_new)
    print("**** energy after rotation *****", en2)
    assert np.abs(en1 - en2) < 1e-6

    # new mol of double size 
    mol = mol.copy()
    for i in range(natom):
        R_old = atom[i][1]
        R_new = R_cartesian(R_old, center, angles) 
        atom.append(tuple((atom[i][0], R_new)))
    mol.atom = atom
    mol.unit = 'B'
    mol.spin = mol.spin*2
    mol.build()
    gmf.mol = mol

    mo_coeff1 = np.zeros((nao * 4, nao*2), dtype=complex)
    mo_coeff1[:nao] = mo_coeff_ori[:nao]
    mo_coeff1[nao*2:nao*3] = mo_coeff_ori[nao:]
    mo_coeff2 = np.zeros((nao * 4, nao*2), dtype=complex)
    mo_coeff2[nao:nao*2] = mo_coeff[:nao]
    mo_coeff2[nao*3:] = mo_coeff[nao:]
    s = get_mo_ovlp(mo_coeff1[:,gmf.mo_occ>0], mo_coeff2[:,gmf.mo_occ>0], gmf)
    print(np.linalg.det(s))

def test_rot_cell():
    from pyscf.pbc import gto, scf, dft
    from pyscf.lib import chkfile

    # User input
    geometry = 'NeH4_xz' # To start with, try 'H' 'C' 'Ho'
    angles = (np.pi*2, 0, 0)
    center = [0,0,0]

    # input a molecule
    if geometry == 'NeH4_xy':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0 -1  0
                    H  0  1  0
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            a = np.diag([10,10,10]),
            charge = 2, 
            verbose = 6)
    elif geometry == 'NeH4_xz':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0  0 -1
                    H  0  0  1
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            a = np.diag([5,5,5]),
            charge = 2, 
            verbose = 6)

    # input some MO
    if geometry[:4] == 'NeH4':
        print("running SCF")
        mf = scf.UHF(mol).density_fit().run() 
        # create a full p shell which is invariant under spatial rotation
        if True:
            mo_coeff = np.eye(mol.nao)
            mo_coeff[-4:, -4:] = np.array([[1,1,1,0],[1,1,-1,0],[1,-1,0,1],[1,-1,0,-1]])
            ao_ovlp = mf.get_ovlp()
            mo_coeff[:,:-3] = lo.orth.vec_lowdin(mo_coeff[:,:-3], ao_ovlp) 
            mo_coeff[:,-3:] = lo.orth.vec_lowdin(mo_coeff[:,-3:], ao_ovlp) 
            print("mo", mo_coeff)
            s = mo_coeff.conj().T @ ao_ovlp @ mo_coeff
            print("MO ovlp", s)
            mf.mo_coeff = np.array([mo_coeff, mo_coeff])
        # Convert the UHF solution to GHF 
        gmf = scf.addons.convert_to_ghf(mf)
    dm = gmf.make_rdm1()
    en1 = gmf.energy_tot(dm=dm)
    print("***** init energy *****", en1) 
    mo_coeff = np.asarray(gmf.mo_coeff.copy(), dtype=complex)
    mo_coeff = sph_mo_rotate(mol, mo_coeff, angles) # new mo_coeff 
    
    # new mol of only the rotated atoms
    mol2 = mol.copy()
    nao = mol.nao
    natom = mol.natm
    atom = mol._atom
    atom_new = []
    for i in range(natom):
        R_old = atom[i][1]
        R_new = R_cartesian(R_old, center, angles) 
        atom_new.append(tuple((atom[i][0], R_new)))
    mol2.atom = atom_new
    mol2.unit = 'B'
    mol2.spin = mol.spin
    mol2.build()
    gmf.mol = mol
    dm_new = gmf.make_rdm1()
    en2 = gmf.energy_tot(dm=dm_new)
    print("**** energy after rotation *****", en2)
    assert np.abs(en1 - en2) < 1e-6

    # new mol of double size 
    mol = mol.copy()
    for i in range(natom):
        R_old = atom[i][1]
        R_new = R_cartesian(R_old, center, angles) 
        atom.append(tuple((atom[i][0], R_new)))
    mol.atom = atom
    mol.unit = 'B'
    mol.spin = mol.spin*2
    mol.build()
    gmf.mol = mol

    mo_coeff1 = np.zeros((nao * 4, nao*2), dtype=complex)
    mo_coeff1[:nao] = gmf.mo_coeff[:nao]
    mo_coeff1[nao*2:nao*3] = gmf.mo_coeff[nao:]
    mo_coeff2 = np.zeros((nao * 4, nao*2), dtype=complex)
    mo_coeff2[nao:nao*2] = mo_coeff[:nao]
    mo_coeff2[nao*3:] = mo_coeff[nao:]
    s = get_mo_ovlp(mo_coeff1[:,gmf.mo_occ>0], mo_coeff2[:,gmf.mo_occ>0], gmf)
    print(np.linalg.det(s))

def test_rot_mol_en():
    from pyscf.lib import chkfile

    # User input
    geometry = 'NeH4_xz' # To start with, try 'H' 'C' 'Ho'
    angles = np.array((0, np.pi/2, 0))
    center = [0,0,0]

    # input a molecule
    if geometry == 'NeH4_xy':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0 -1  0
                    H  0  1  0
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            charge = 2, 
            verbose = 6)
    elif geometry == 'NeH4_xz':
        mol = gto.M(
            atom = '''
                    Ne 0  0  0
                    H  0  0 -1
                    H  0  0  1
                    H -1  0  0
                    H  1  0  0''',
            symmetry = False,
            basis = 'sto6g', 
            spin = 0, 
            charge = 2, 
            verbose = 6)

    # input some MO
    if geometry[:4] == 'NeH4':
        mf = scf.UHF(mol).run() 
        print("mo", mf.mo_coeff)
        if False:
            mo_coeff = np.eye(mol.nao)
            mo_coeff[-4:, -4:] = np.array([[1,1,1,0],[1,1,-1,0],[1,-1,0,1],[1,-1,0,-1]])
            ao_ovlp = mf.get_ovlp()
            mo_coeff[:,:-3] = lo.orth.vec_lowdin(mo_coeff[:,:-3], ao_ovlp) 
            mo_coeff[:,-3:] = lo.orth.vec_lowdin(mo_coeff[:,-3:], ao_ovlp) 
            print("mo", mo_coeff)
            s = mo_coeff.conj().T @ ao_ovlp @ mo_coeff
            print("MO ovlp", s)
            mf.mo_coeff = np.array([mo_coeff, mo_coeff])
        # Convert the UHF solution to GHF 
        gmf = scf.addons.convert_to_ghf(mf)
        if mpirank == 0:
            gmf.analyze(with_meta_lowdin=True)
    dm = gmf.make_rdm1()
    gmf = gmf.x2c() 
    en1 = gmf.energy_tot(dm=dm)
    print("***** init energy *****", en1) 
    mo_coeff_ori = np.asarray(gmf.mo_coeff.copy(), dtype=complex)
    mo_coeff = sph_mo_rotate(mol, mo_coeff_ori, angles) # new mo_coeff 
    #print(mo_coeff_ori)
    #print("******")
    #print(mo_coeff)
    
    # new mol of only the rotated atoms
    mol2 = mol.copy()
    nao = mol.nao
    natom = mol.natm
    atom = mol._atom
    atom_new = []
    for i in range(natom):
        R_old = atom[i][1]
        R_new = R_cartesian(R_old, center, angles) 
        atom_new.append(tuple((atom[i][0], R_new)))
    mol2.atom = atom_new
    mol2.unit = 'B'
    mol2.spin = mol.spin
    mol2.build()
    hcore_old = gmf.get_hcore()
    veff_old = gmf.get_veff()
    mo_occ = gmf.mo_occ
    gmf.mol = mol2
    gmf.mo_coeff = mo_coeff
    gmf.analyze()
    dm_new = gmf.make_rdm1()
    print("new mo", gmf.mo_coeff)
    print("old mo", mo_coeff_ori)
    for i in range(mol.nao*2):
        cubegen.orbital(mol, '%s_%s_a%s.cube'%("old", 'Ne', i), mo_coeff_ori[:nao,i])
        cubegen.orbital(mol2, '%s_%s_a%s.cube'%("new", 'Ne', i), gmf.mo_coeff[:nao,i])
    exit()
    gmf = scf.GHF(mol).x2c()
    gmf.mo_coeff = mo_coeff
    gmf.mo_occ = mo_occ
    eri = mol2.intor('int2e', aosym='s1') 
    eri_old = mol.intor('int2e', aosym='s1') 
    print("new eri01\n", eri[5,5])
    print("old eri01\n", eri_old[5,5])
    exit()
    print("new h1", gmf.get_veff().real)
    print("old h1", veff_old.real)
    print("diff", np.max(np.abs(gmf.get_veff().real - veff_old.real)))
    en2 = gmf.energy_tot(dm=dm_new)
    print("**** energy after rotation *****", en2)
    assert np.abs(en1 - en2) < 1e-6

def sph_mo_rotate(mol, mo_coeff_ori, angles):
    from utils import Rz_spin, Ry_spin
    # Assume mo_coeff is from GHF/GKS
    angles = np.array(angles)
    if angles.ndim == 1:
        angles = angles[np.newaxis]
        single_angles = True
    mo_coeff = np.zeros_like(mo_coeff_ori)
    mo_coeff_lst = []

    nao = mol.nao
    natom = mol.natm
    ao_labels = mol.ao_labels()
    aoslices = mol.aoslice_by_atom()
    for alpha, beta, gamma in angles:
        orb_index = 0
        for spin in range(2):
            for i in range(natom):
                start, end = aoslices[i][:2] 
                for shell in range(end - start):
                    orb_symbol = ao_labels[orb_index%nao].split()[-1]
                    if 's' in orb_symbol:
                        L = 0
                    elif 'p' in orb_symbol:
                        L = 1
                    elif 'd' in orb_symbol:
                        L = 2
                    elif 'f' in orb_symbol:
                        L = 3
                    else:
                        raise NotImplementedError
                    wigner_D = pyscf.symm.Dmatrix.Dmatrix(L, alpha, beta, gamma, reorder_p=True)
                    norb_in_shell = len(wigner_D)
                    mo_coeff[orb_index : orb_index + norb_in_shell] = wigner_D @ \
                                    mo_coeff_ori[orb_index : orb_index + norb_in_shell]
                    orb_index += norb_in_shell
        ###mo_coeff = Rz_spin(gamma, nao) @ Ry_spin(beta, nao) @ Rz_spin(alpha, nao) @ mo_coeff
        mo_coeff_lst.append(mo_coeff)

    if single_angles:
        mo_coeff_lst = mo_coeff_lst[0]
    return mo_coeff_lst


if __name__ == "__main__":
    #test_rot_DFT()
    #test_rot_PBCDFT()
    test_rot_mol_en()

