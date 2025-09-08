import os, sys
import numpy as np
import scipy.linalg as la
import pyscf
from stevens import momentum

def find_close_atom(cell, coord, distance, element=None):
    coord_new = np.empty((0,3))
    for i in range(-1, 2): 
        for j in range(-1, 2):
            for k in range(-1, 2):
                coord_new = np.vstack((coord_new, coord + np.dot([i,j,k], cell.a)))
    coord = coord_new
    if element is None:
        idx = np.arange(cell.natm)
    else:
        idx = np.where([cell._atom[i][0] == element for i in range(cell.natm)])[0]
    target_coord = np.array(cell.atom_coords(unit='ANG')[idx])
    idx_new = []
    for i in range(len(idx)):
        dis =  min(la.norm(coord - target_coord[i], axis=1)) 
        if dis < distance and dis > 1e-6:
            idx_new.append(idx[i])
    return idx_new

def find_fragment(cell, distance):
    fragment = [[0]]
    atom_coords = cell.atom_coords(unit='ANG')
    for i in range(1, cell.natm):
        atom_idx = np.array(find_close_atom(cell, [atom_coords[i]], 2, None))
        if len(atom_idx) > 0 and (atom_idx < i).all():
            continue
        atom_idx = atom_idx[atom_idx > i]
        fragment_idx = [j for j in range(len(fragment)) if i in fragment[j]]
        fragment_element = [i] if len(fragment_idx) == 0 else np.array(fragment[fragment_idx[0]])
        for k in atom_idx:
            loc = np.where([k in fragment[j] for j in range(len(fragment))])[0]
            if len(loc) > 0:
                assert len(loc) == 1
                fragment_idx.append(loc[0])
                fragment_element = np.append(fragment_element, fragment[loc[0]])
            else:
                fragment_element = np.append(fragment_element, [k])
        if len(fragment_idx) == 0:
            fragment.append(fragment_element)
        else:
            fragment_idx = np.unique(fragment_idx)
            fragment[fragment_idx[0]] = np.unique(fragment_element)
            fragment_new = [fragment[j] for j in range(len(fragment)) if j not in fragment_idx[1:]]
            fragment = fragment_new
    atom = np.array([cell._atom[i][0] for i in range(cell.natm)])
    print("Fragments:")
    for i in range(len(fragment)):
        print(atom[fragment[i]])

    return fragment

def get_init_guess_fragment_UHF(cell_ori, fragment, frag_charge, frag_spin):
    has_pbc = getattr(cell_ori, 'dimension', 0) > 0
    if has_pbc:
        from pyscf.pbc.dft.uks import UKS 
    else:
        from pyscf.dft.uks import UKS
    dm_lst_alpha = []
    dm_lst_beta = []
    aoslice = cell_ori.aoslice_by_atom()[np.concatenate(fragment)]
    orb_lst = np.array([], dtype=int)
    for i in range(len(aoslice)):
        orb_lst = np.append(orb_lst, np.arange(aoslice[i][2], aoslice[i][3], dtype=int))
    nao_ori = cell_ori.nao
    assert len(orb_lst) == nao_ori

    for i in range(len(fragment)):
        print(i, fragment[i])
        cell = cell_ori.copy(deep=True)
        cell.atom = []
        cell.magmom = []
        cell._atom = [cell_ori._atom[j] for j in fragment[i]]
        cell.charge = frag_charge[i]
        cell.spin = frag_spin[i]
        cell.basis = cell_ori.basis
        cell.build()
        #orb_lst.append(cell.aoslice_by_atom()) ### TODO
        
        if has_pbc:
            mf = UKS(cell, [0,0,0]).density_fit()
        else:
            mf = UKS(cell)
        atom_charge = {'Ho': 3, 'Y': 3, 'C': 0, 'Cl': 3, 'H': 0, 'N': 1, 'O': -1}
        dm0 = mf.get_init_guess(key='atom', charge=atom_charge)
        mf.xc = 'lda'
        mf.max_cycle = 200
        mf.kernel()
        dm = mf.make_rdm1()
        dm_lst_alpha.append(dm[0])
        dm_lst_beta.append(dm[1])
        s = np.einsum('spq,qp->s', dm, mf.get_ovlp())
        print("fragment %s spin"%i, s[0] - s[1])
    dm_alpha = la.block_diag(*dm_lst_alpha)
    dm_beta = la.block_diag(*dm_lst_beta)
    dm_tmp = np.zeros_like(dm_alpha)
    dm_tmp[orb_lst] = dm_alpha
    dm_alpha[:, orb_lst] = dm_tmp
    dm_tmp[orb_lst] = dm_beta
    dm_beta[:, orb_lst] = dm_tmp
    return np.array([dm_alpha, dm_beta]) 

def get_init_guess_fragment_GHF(cell_ori, fragment, frag_charge, frag_spin):
    dm = get_init_guess_fragment_UHF(cell_ori, fragment, frag_charge, frag_spin)
    return la.block_diag(*dm)


def occ_config(l, ne):
    # generate all possible combinations of 3 occupied orbitals in a shell with orbital angular momentum l
    # return a list of tuples of indices of occupied orbitals 
    import itertools
    no = 2 * l + 1
    lst = []
    for it in itertools.combinations(np.arange(no), ne):
        lst.append(it)
    return lst

def Rz_spin(theta, nao):
    Rz_mat_aa = np.eye(nao) * np.exp(-1j * theta/2)
    Rz_mat_bb = np.eye(nao) * np.exp(1j * theta/2)
    Rz_mat = np.block([[Rz_mat_aa, np.zeros((nao, nao))],
                      [np.zeros((nao, nao)), Rz_mat_bb]])
    return Rz_mat

def Ry_spin(beta, nao):
    Ry_mat_aa = np.eye(nao) * np.cos(beta/2)
    Ry_mat_ab = - np.eye(nao) * np.sin(beta/2)
    Ry_mat_ba = np.eye(nao) * np.sin(beta/2)
    Ry_mat_bb = np.eye(nao) * np.cos(beta/2)
    Ry_mat = np.block([[Ry_mat_aa, Ry_mat_ab], [Ry_mat_ba, Ry_mat_bb]])
    return Ry_mat

def sph_Rz_spin(theta, l):
    nao = 2 * l + 1
    return Rz_spin(theta, nao)

def sph_Ry_spin(beta, l):
    nao = 2 * l + 1
    return Ry_spin(beta, nao)

def sph_Rz_orb(theta, l):
    nao = 2 * l + 1
    l_integral = np.diag(np.arange(-l, l+0.5))
    Rz_mat = la.expm(- 1j * theta * l_integral)
    u = pyscf.symm.sph.sph_pure2real(l)
    Rz_mat = u.conj().T @ Rz_mat @ u
    Rz_mat = np.block([[Rz_mat, np.zeros((nao, nao))], [np.zeros((nao,nao)), Rz_mat]])
    return Rz_mat

def sph_Ry_orb(beta, l):
    nao = 2 * l + 1
    l_integral = np.zeros((nao, nao), dtype=complex)
    for i in range(nao):
        for j in range(nao):
            l_integral[i,j] = ((i == j+1) - (i+1 == j)) / 2j * (l*(l+1) - (i-l)*(j-l))**0.5
    Ry_mat = la.expm(- 1j * beta * l_integral)
    u = pyscf.symm.sph.sph_pure2real(l)
    Ry_mat = u.conj().T @ Ry_mat @ u
    Ry_mat = np.block([[Ry_mat, np.zeros((nao, nao))], [np.zeros((nao,nao)), Ry_mat]])
    return Ry_mat

def sph_dm_rotate(dm, angles, l=None, include_L=True, include_S=True):
    '''
    Args:
        dm: a block of GHF density matrix in real AO basis of a specific shell
        angles: (a list of) Euler angles in z-y-z sequence or "inverse" Jx, Jy, Jz.
                Each set of Euler angles (alpha, beta, gamma) will rotate the dm first around
                z by alpha, then around y by beta, and finally around z by gamma.
    '''
    if l is None:
        l = int((dm.shape[-1]//2 - 1) // 2)
    if l > 5:
        raise ValueError
    # rotated the spin and space of a dm of an AO shell with some angles 
    # angles are either three Euler angles, or a list of multiple sets of three Euler angles.
    if (not isinstance(angles, list) and not isinstance(angles, np.ndarray)) or (len(np.array(angles).flatten()) == 3):
        angles = [angles]
    dm_lst = []
    for angle in angles:
        if isinstance(angle, str) and angle.lower()[:6] == "invers":
            if include_L and include_S:
                dm = dm.reshape(2, 2*l+1, 2, 2*l+1)[::-1, :, ::-1]
                dm[[0,1], :, [1,0]] = - dm[[0,1], :, [1,0]] 
                dm = dm.reshape(2 * (2*l+1), 2 * (2*l+1))
                dm = dm.conj()
            else:
                raise NotImplementedError
            dm_lst.append(dm)
        else:
            alpha, beta, gamma = angle
            if include_S:
                R = sph_Rz_spin(gamma, l) @ sph_Ry_spin(beta, l) @ sph_Rz_spin(alpha, l)
            else:
                R = np.eye(dm.shape[-1])
            if include_L:
                R = sph_Rz_orb(gamma, l) @ sph_Ry_orb(beta, l) @ sph_Rz_orb(alpha, l) @ R 
            dm_lst.append(R @ dm @ R.conj().T)
    return dm_lst

def spin_dm_rotate(dm, angles):
    nao = int(dm.shape[-1]//2)
    angles = np.array(angles)
    if angles.ndim == 1:
        angles = angles[np.newaxis]
    dm_lst = []
    for alpha, beta, gamma in angles:
        R = Rz_spin(gamma, nao) @ Ry_spin(beta, nao) @ Rz_spin(alpha, nao)
        dm_lst.append(R @ dm @ R.conj().T)
    return dm_lst

def dm_rotate(dm, angles, include_L=True, include_S=True):
    '''
    Args:
        dm: GHF density matrix in real AO basis
        angles: (a list of) Euler angles in z-y-z sequence or "inverse" Jx, Jy, Jz.
    '''
    nao = int(dm.shape[-1]//2) 
    if not isinstance(angles, list) and not isinstance(angles, np.ndarray):
        angles = [angles]
    dm_lst = []
    for angle in angles:
        if isinstance(angle, str) and angle.lower()[:6] == "invers":
            if include_L and include_S:
                dm = dm.reshape(2, nao, 2, nao)[::-1, :, ::-1]
                dm[[0,1], :, [1,0]] = - dm[[0,1], :, [1,0]] 
                dm = dm.reshape(2 * nao, 2 * nao)
                dm = dm.conj()
            else:
                raise NotImplementedError
            dm_lst.append(dm)
        else:
            raise NotImplementedError
    return dm_lst

def get_rotation_to_z(mol, dm):
    j = momentum.get_mj(mol, dm).real
    return get_rotation_to_z_from_vector(j)

def get_rotation_to_z_from_vector(j):
    assert np.max(np.abs(j.imag)) < 1e-6
    j = j.real
    j = j / la.norm(j)
    alpha = - np.arctan2(j[1], j[0])
    beta = - np.arccos(j[2])
    gamma = 0
    return (alpha, beta, gamma)


def change_dm_guess(mol, dm0, mode, atom_index=0, ao_shell='f', angles_from_z=None, direction=None, plane='xyz', occ_idx=None): 
    if mode is None:
        return dm0
    from stevens import cdft 
    ao_shell = ao_shell.lower()
    if not ao_shell == 'f':
        raise NotImplementedError
    if not (mode == 'rotate' or mode == 'positive_z'):
        raise ValueError
    nao = mol.nao
    if dm0.ndim == 3: # Unrestricted or restricted with spin index
        if mode == 'positive_z':
            HF_type = 'ghf' # i.e. keep uhf shape 
        else:
            spin = len(dm0)
            dm0 = la.block_diag(dm0[0], dm0[-1])
            HF_type = "uhf"
    elif len(dm0) == mol.nao:
        dm0 = la.block_diag(dm0, dm0)
        HF_type = 'rhf'
    else:
        HF_type = None
    aoslice = mol.aoslice_by_atom()[atom_index]
    ao_labels = mol.ao_labels()[aoslice[2]:aoslice[3]]
    idx = min(np.where([ao_shell in ao_labels[i] for i in range(len(ao_labels))])[0])
    idx += aoslice[2]
    if mode == 'rotate': # rotate 4f block of lanthanide with some angles
        idx = np.concatenate((np.arange(idx, idx+7), np.arange(idx+nao, idx+nao+7)))
        angles_init = get_rotation_to_z(mol, dm0) # first align with +z
        dm0[np.ix_(idx, idx)] = sph_dm_rotate(dm0[np.ix_(idx, idx)], angles_init)[0]
        if angles_from_z is None:
            if direction is None:
                angles_from_z = cdft.sample_rotation(plane, occ_idx, unit=20) 
            else:
                angles_from_z = - np.array(get_rotation_to_z_from_vector(direction))[::-1]
        dm0[np.ix_(idx, idx)] = sph_dm_rotate(dm0[np.ix_(idx, idx)], angles_from_z)[0]
        print("Rotated f-block dm by angles from +z", angles_from_z)
    elif mode == 'positive_z': # replace 4f block of lanthanide with J//+z guess
        if dm0.ndim == 2: # GHF
            dm0 = np.asarray(dm0, dtype=complex)
            dm0[idx:idx+7, idx:idx+7] = np.eye(7)  
            nbeta = {'Dy':2, 'Ho':3, 'Er':4}[mol._atom[atom_index][0]]
            beta_occ = np.zeros(7)
            beta_occ[-nbeta:] = 1
            u = pyscf.symm.sph.sph_pure2real(l=3) # rotate from pure to real spherical harmonics
            dm0[idx+nao:idx+7+nao, idx+nao:idx+7+nao] = u.conj().T @ np.diag(beta_occ) @ u
            dm0[idx:idx+7, idx+nao:idx+7+nao] = 0
            dm0[idx+nao:idx+7+nao, idx:idx+7] = 0
            print("Updated f-block dm with J~(0,0,J)")
        elif dm0.ndim == 3: # UHF: S > 0, L = 0
            dm0 = np.asarray(dm0)
            dm0[0:1, idx:idx+7, idx:idx+7] = np.eye(7)[np.newaxis]
            nbeta = {'Dy':2, 'Ho':3, 'Er':4}[mol._atom[atom_index][0]]
            beta_occ = nbeta / 7
            dm0[1:, idx:idx+7, idx:idx+7] = np.eye(7)[np.newaxis]*beta_occ
            print("Updated f-block dm with S~(0,0,S=%.1f)"%((7-nbeta)/2))
    if HF_type is not None:
        if HF_type == 'uhf':
            if spin == 2:
                dm0 = np.array([dm0[:mol.nao, :mol.nao], dm0[mol.nao:, mol.nao:]])
            elif spin == 1:
                dm0 = np.array([dm0[:mol.nao, :mol.nao]])
        elif HF_type == 'rhf':
            dm0 = dm0[:mol.nao, :mol.nao]
    return dm0

def project_dm_from_small_basis(mol, basis_small, chkfname_small, change_mode=None, atom_index=0, ao_shell='f', occ_idx=None):
    from pyscf import scf, gto
    from pyscf.lib import chkfile
    # read mo_coeff in a small basis
    has_pbc = getattr(mol, 'dimension', 0) > 0
    if has_pbc:
        mol_small = pyscf.pbc.gto.Cell()
        mol_small.rcut = mol.rcut
    else:
        mol_small = gto.Mole()
    mol_small.atom = mol.atom
    mol_small.basis = basis_small
    mol_small.spin = mol.spin
    mol_small.charge = mol.charge
    mol_small.max_memory = mol.max_memory
    mol_small.verbose = mol.verbose
    if hasattr(mol, "exp_to_discard"): 
        mol_small.exp_to_discard = mol.exp_to_discard
    mol_small.build()
    print('basis', mol.basis, mol.nao)
    print('basis', basis_small, mol_small.nao)
    mf_small = scf.GKS(mol_small).x2c()
    data = chkfile.load(chkfname_small, 'scf')
    mf_small.__dict__.update(data)
    if np.array(mf_small.mo_coeff).ndim == 3:
        mf_small = scf.UKS(mol_small).x2c()
        data = chkfile.load(chkfname_small, 'scf')
        mf_small.__dict__.update(data)
        unrestricted = True
    else:
        unrestricted = False
    dm = mf_small.make_rdm1()
    # change J by changing the 4f block in the small basis
    # This step has to be performed in the small basis instead of the large basis
    dm = change_dm_guess(mol_small, dm, change_mode, atom_index=atom_index, occ_idx=occ_idx)
    # project dm from the small basis to the large basis
    if has_pbc: 
        S_large = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=[[0,0,0]],
                            pbcopt=pyscf.lib.c_null_ptr()))[0] ### TODO
        S_overlap = np.array(pyscf.pbc.gto.cell.intor_cross('int1e_ovlp', mol, mol_small, hermi=0, kpts=[[0,0,0]],
                            pbcopt=pyscf.lib.c_null_ptr()))[0] ### TODO
    else:
        S_large = mol.intor_symmetric('int1e_ovlp') 
        S_overlap = gto.mole.intor_cross('int1e_ovlp', mol, mol_small)  
    p = la.inv(S_large) @ S_overlap
    if unrestricted:
        dm = np.einsum('ij,sjk,kl->sil', p, dm, p.conj().T, optimize=True)
    else:
        if len(dm) == p.shape[-1] * 2:
            p = la.block_diag(p, p)
        dm = p @ dm @ p.conj().T   
    return dm





