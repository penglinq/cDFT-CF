'''
Calculate S (spin), L (orbital), and J (total) angular momentum
'''
from pyscf import lib
import os, sys
import numpy as np
import scipy.linalg as la
log = lib.logger.Logger(sys.stdout, 4)

def get_dmj(ao_ovlp=None, l_integral=None, include_L=True, include_S=True, mol=None):
    """
    Calculate dmj, the derivative of J (L, or S) with respect to orbitals. 
    (J_x, J_y, J_z) = \Sum_{ij} \rho_{ij} x dmj_{ij} 

    Args
        ao_ovlp (Nao x Nao array): Atomic orbital overlap
        l_integral (Nao x Nao array): Angular momentum integral r x p. By default, treat (0,0,0) as the origin.
        include_L (bool) and inlcude_S (bool): 
            For J, include_L = True; include_S = True;
            for L, include_L = True; include_S = False;
            for S, include_L = False; include_S = True.
        mol (an instance of PySCF gto.Mole class, optional): The molecule. Supple ao_ovlp and l_integral. 

    Returns
        derivative of J (L, or S) with respect to orbitals (2*Nao x 2*Nao array).
    """
    if ao_ovlp is None or l_integral is None:
        assert mol is not None
        has_pbc = getattr(mol, 'dimension', 0) > 0
        if ao_ovlp is None:
            if has_pbc:
                kpts = [[0,0,0]] ### TODO
                ao_ovlp = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=kpts,
                               pbcopt=lib.c_null_ptr()))
                ao_ovlp = ao_ovlp[0]
            else:
                ao_ovlp = mol.intor_symmetric('int1e_ovlp')
        if l_integral is None:
            if has_pbc:
                kpts = [[0,0,0]] ### TODO
                l_integral = np.array(mol.pbc_intor('int1e_cg_irxp', kpts=kpts)) *(-1j) ### TODO double check 
                l_integral = l_integral[0] ### here assume only gamma point. TODO multiple k-point 
            else:
                l_integral = (mol.intor('int1e_cg_irxp') *(-1j))
    nao = ao_ovlp.shape[0]
    pauli = 0.5 * lib.PauliMatrices
    S = np.einsum('ij,xst->xsitj', ao_ovlp, pauli).reshape(3, nao*2, nao*2).conj()
    l_integral = l_integral.conj()
    print(l_integral.shape, nao)
    L = np.array([np.block([[l_integral[x], np.zeros((nao,nao))],[np.zeros((nao,nao)), l_integral[x]]]) for x in range(3)])
    dmj = S * include_S + L * include_L
    return (dmj + dmj.transpose(0,2,1).conj()) / 2 

def get_mj(mol, dm, ao_ovlp=None, l_integral=None, include_L=True, include_S=True, analyze=False, c_lo=None,
           f_only=False, kpts=None):
    """
    Calculate (J_x, J_y, J_z) (or the xyz components of L or S). 
    Math: (J_x, J_y, J_z) = \Sum_{ij} dm_{ij} x dmj_{ij} 

    Args
        mol (an instance of PySCF gto.Mole class): The molecule. To provide atomic orbital overlap and angular momentum
            integral. When analyze==False and ao_ovlp and l_integral are neither None, mol can be None.
        dm (2*Nao x 2*Nao array from GHF or 2 x Nao x Nao from UHF or nk x 2*Nao x 2*Nao from kGHF): density matrix in spin atomic orbital basis.
        ao_ovlp (Nao x Nao array for molecule or nk x Nao x Nao for PBC): Atomic orbital overlap
        l_integral (2 x Nao x Nao array): Angular momentum integral r x p. By default, treat (0,0,0) as the origin.
        include_L (bool) and inlcude_S (bool): 
            For J, include_L = True; include_S = True;
            for L, include_L = True; include_S = False;
            for S, include_L = False; include_S = True.
        analyze: If True, analyze the elemental decomposition of J_x, J_y, and J_z based on atomic orbitals.
        c_lo: Basis transformation from atomic orbitals to localized orbital (LO). If not None, analyze the elemental decomposition based on LO.

    Returns
        (J_x, J_y, J_z) (or the xyz components of L or S). 

    TODO:
        check if RHF solution should return J=0
    """
    nao = mol.nao 
    has_pbc = getattr(mol, 'dimension', 0) > 0
    # make all molecular dm of the form 2*Nao x 2*Nao
    # make all PBC dm of the form nk x 2*Nao x 2*Nao
    if has_pbc:
        if dm.ndim == 2:
            if dm.shape[0] == nao and dm.shape[1] == nao:
                dm = la.block_diag(dm, dm) # KRHF solution 
            elif dm.shape[0] != nao * 2 or dm.shape[1] != nao * 2:
                raise ValueError
            dm = dm[np.newaxis] # one k-point
        elif dm.ndim == 3:
            if dm.shape[1] == nao and dm.shape[2] == nao:
                if len(dm) == 1: # kRHF, otherwise kUHF
                    dm = la.block_diag(dm, dm) # KRHF with one k-point 
                else:
                    dm = la.block_diag(*dm)[np.newaxis] # kUHF with one k-point
            elif dm.shape[1] != nao * 2 or dm.shape[2] != nao * 2:
                raise ValueError
        elif dm.ndim == 4:
            # form spin x kpts x orb x orb
            if dm.shape[-2] == nao and dm.shape[-1] == nao:
                if dm.shape[0] == 1: # kRHF, otherwise kUHF
                    dm = np.array([la.block_diag(dm[0,k], dm[0,k]) for k in range(dm.shape[1])]) # KRHF with one k-point 
                else:
                    dm = np.array([la.block_diag(*dm[:,k]) for k in range(dm.shape[1])]) # KUHF with one k-point
            elif dm.shape[0] != nao * 2 or dm.shape[1] != nao * 2:
                raise ValueError 
    else:
        if dm.ndim == 3:
            if dm.shape[1] == nao and dm.shape[2] == nao:
                if len(dm) == 1: # RHF, otherwise UHF
                    return np.array([0,0,0]) # RHF solution has zero spin #TODO
                dm = np.block([[dm[0], np.zeros((nao, nao))],
                              [np.zeros((nao, nao)), dm[1]]])
            else:
                raise ValueError
        else:
            if dm.shape[0] == nao and dm.shape[1] == nao:
                return np.array([0,0,0]) # RHF solution has zero spin #TODO 
            elif dm.shape[0] != nao * 2 or dm.shape[1] != nao * 2:
                raise ValueError 
    if ao_ovlp is None:
        if has_pbc:
            if kpts is None:
                if len(dm) == 1:
                    kpts = [[0,0,0]] # gamma point by default
                else:
                    raise ValueError
            ao_ovlp = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=kpts,
                               pbcopt=lib.c_null_ptr()))
        else:
            ao_ovlp = mol.intor_symmetric('int1e_ovlp') 
    if l_integral is None:
        if has_pbc:
            l_integral = np.array(mol.pbc_intor('int1e_cg_irxp', kpts=kpts)) *(-1j) ### TODO double check 
        else:
            l_integral = (mol.intor('int1e_cg_irxp') *(-1j))
    if has_pbc and dm.shape[0] > 1:
        ### TODO update k-dependent dmj
        raise NotImplementedError
    elif has_pbc: ### Temporary solution
        dm = dm[0]
        if ao_ovlp.ndim == 3:
            ao_ovlp = ao_ovlp[0]
        if l_integral.ndim == 4:
            l_integral = l_integral[0]
    dmj = get_dmj(ao_ovlp=ao_ovlp, l_integral=l_integral, include_L=include_L, include_S=include_S)

    if analyze:
        if c_lo is not None:
            if isinstance(c_lo, str) and c_lo[-2:] == "h5":
                feri = h5py.File(c_lo, 'r')
                c_lo = np.array(feri['c_lo'])
                feri.close()
            elif isinstance(c_lo, str) and c_lo[-3:] == 'npy':
                c_lo = np.load(c_lo)
            ao_ovlp = la.block_diag(ao_ovlp, ao_ovlp)
            c_inv = c_lo.conj().T @  ao_ovlp
            dm = c_inv @ dm @ c_inv.conj().T
            dmj = (c_lo.conj().T @ dmj.conj() @ c_lo).conj()
        if len(dm) == 2*nao: # make sure the dm orbital index is the same as ao_labels index
            mulliken_J = np.einsum('ij,xij->xi', dm, dmj) ###TODO double check the definition 
            idx = np.where(np.abs(mulliken_J.real) > 0.00001)
            ao_labels = mol.ao_labels()
            ao_labels = np.append(ao_labels, ao_labels)
            axis_lst = ['x', 'y', 'z']
            atom_lst = np.unique(np.array([mol._atom[i][0] for i in range(mol.natm)])) 
            log.info("elemental J of %s" % atom_lst)
            J_atom = np.zeros((3, len(atom_lst)), dtype=complex)
            for i in range(len(idx[0])):
                if not f_only or 'f' in ao_labels[idx[1][i]]:
                    J_atom[idx[0][i], np.where(atom_lst == ao_labels[idx[1][i]].split(' ')[1])[0]] += mulliken_J[idx[0][i], idx[1][i]]
                    #print(axis_lst[idx[0][i]], np.where(atom_lst == ao_labels[idx[1][i]].split(' ')[1])[0], 
                    #        ao_labels[idx[1][i]], mulliken_J[idx[0][i], idx[1][i]])
                #print(axis_lst[idx[0][i]], ao_labels[idx[1][i]], mulliken_J[idx[0][i], idx[1][i]].real,
                #        mulliken_J[idx[0][i], idx[1][i]].imag)
            for i in range(len(atom_lst)):
                log.info("%4s "%atom_lst[i] + "{:.4f} ".format(J_atom[0, i]) + "{:.4f} ".format(J_atom[1, i]) 
                        + "{:.4f} ".format(J_atom[2, i]))
    return np.einsum('ij,xij->x', dm,  dmj)

def J_square(mol, dm, ao_ovlp=None, l_integral=None, include_L=True, include_S=True, lo_proj=None):
    """
    Calculate <J^2> (or <S^2> or <L^2>)
    Args
            lo_proj: c_lo @ c_lo.conj().T @ ao_ovlp 
    """
    symbol = ['S', 'L', 'J'][include_L*2 + include_S - 1]
    if ao_ovlp is None:
        ao_ovlp = mol.intor_symmetric('int1e_ovlp') 
    if l_integral is None:
        l_integral = (mol.intor('int1e_cg_irxp') *(-1j))
    dmj = get_dmj(ao_ovlp=ao_ovlp, l_integral=l_integral, include_L=include_L, include_S=include_S)

    J2 = 0
    dmj2 = get_dmj2(ao_ovlp=ao_ovlp, l_integral=l_integral, include_L=include_L, include_S=include_S)
    if lo_proj is not None:
        dmj = np.einsum('ki,xij,jl->xkl', lo_proj.T.conj(), dmj, lo_proj, optimize=True)
        dmj2 = np.einsum('ki,xij,jl->xkl', lo_proj.T.conj(), dmj2, lo_proj, optimize=True)
    for x in range(3):
        # same electron
        J2 += np.trace(dm @ dmj2[x].T)
        # two different electrons
        J2 += np.trace(dm @ dmj[x].T) * np.trace(dm @ dmj[x].T)
        J2 -= np.trace(dm @ dmj[x].T @ dm @ dmj[x].T)
    assert np.abs(J2.imag) < 1e-6
    J2 = J2.real
    log.info("%s quantum number %s"%(symbol, (J2 + 0.25) ** 0.5 - 0.5))
    return J2

def get_dmj2(ao_ovlp, l_integral, include_L=True, include_S=True):
    """
    Calculate dmj^2, the derivative of J^2 (L^2, or S^2) with respect to orbitals. 

    Args
        mol (an instance of PySCF gto.Mole class): The molecule. Need 
        overlap (Nao x Nao array): Atomic orbital overlap
        l_integral (Nao x Nao array): Angular momentum integral r x p. By default, treat (0,0,0) as the origin.
        include_L (bool) and inlcude_S (bool): 
            For J, include_L = True; include_S = True;
            for L, include_L = True; include_S = False;
            for S, include_L = False; include_S = True.

    Returns
        derivative of J^2 (L^2, or S^2) with respect to orbitals (2*Nao x 2*Nao array).
    """
    nao = ao_ovlp.shape[0]
    pauli = 0.5 * lib.PauliMatrices
    S = np.einsum('ij,xst->xsitj', ao_ovlp, pauli).reshape(3, nao*2, nao*2).conj()
    l_integral = (l_integral).conj()
    L = np.array([np.block([[l_integral[x], np.zeros((nao,nao))],[np.zeros((nao,nao)), l_integral[x]]]) for x in range(3)])
    dmj = S * include_S + L * include_L
    ovlp_inv = la.inv(ao_ovlp)
    ovlp_inv = np.block([[ovlp_inv, np.zeros((nao,nao))],[np.zeros((nao,nao)), ovlp_inv]])
    dmj2 = dmj @ ovlp_inv @ dmj
    return (dmj2 + dmj2.transpose(0,2,1).conj()) / 2 

