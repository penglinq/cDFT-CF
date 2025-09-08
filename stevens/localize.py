import pyscf
from pyscf import scf, lib, lo
from pyscf.tools import cubegen 
from pyscf.lib import chkfile
import os, sys
import numpy as np
import scipy.linalg as la
from mpi4py import MPI
from stevens import iao 
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
log = lib.logger.Logger(sys.stdout, 4)
MINAO = {'default': 'minao', 'Ho': 'sarc-dkh-minao', 'Er': 'sarc-dkh-minao', 'Dy': 'sarc-dkh-minao'}

def get_iao(mol, lo_chkfname, from_umf, ao_ovlp=None, verbose=False, diff_spin=False, kpts=[[0,0,0]], minao=MINAO): 
    # if diff_spin == False, use the solution of alpha spin to calculate IAO
    nao = mol.nao
    has_pbc = getattr(mol, 'dimension', 0) > 0
    if has_pbc and len(kpts) > 1:
        raise NotImplementedError # TODO implement multiple k-point
    if ao_ovlp is None:
        if has_pbc:
            ao_ovlp = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=kpts,
                                pbcopt=lib.c_null_ptr()))[0] ### TODO
        else:
            ao_ovlp = mol.intor_symmetric('int1e_ovlp') 
        if not from_umf:
            ao_ovlp = la.block_diag(ao_ovlp, ao_ovlp) 
    if lo_chkfname[-4:] == '.chk':
        lo_chkfname = lo_chkfname.replace(".chk", "")
    if from_umf: 
        log.info('localization from unrestricted solution')
        if has_pbc:
            mf_lo = pyscf.pbc.scf.UHF(mol, kpts[0]).x2c()
        else:
            mf_lo = pyscf.scf.UHF(mol).x2c()
        if os.path.isfile(lo_chkfname + '.chk'):
            data = chkfile.load(lo_chkfname + '.chk', 'scf')
            mf_lo.__dict__.update(data)
        else:
            log.info("Running UHF to obtain intrinsic atomic orbital")
            mf_lo.chkfile = lo_chkfname + '.chk'
            mf_lo.kernel()
        c_iao = []
        for s in range(diff_spin + 1):
            coeff = iao.iao(mol, mf_lo.mo_coeff[s][:, mf_lo.mo_occ[s]>1e-10], minao=minao,\
                        generalized=not from_umf, verbose=True if mpirank==0 else False)
            c_iao.append(lo.orth.vec_lowdin(coeff, ao_ovlp))
        if not diff_spin:
            c_iao.append(c_iao[0])
        niao = np.shape(c_iao)[-1] 

        # PAO for virtual, adapted from libdmet
        if nao == niao: # no virtual 
            c_iao = np.block([[c_iao[0], np.zeros((nao, niao))], [np.zeros((nao, niao)), c_iao[1]]])
            c_lo = c_iao
            nvirt = 0 
            if mpirank == 0:
                print("(spatial) nao, nval, nvirt, nlo_total\n%13d%6d%7d%8d"% \
                        (c_lo.shape[0] // 2, niao, nvirt, c_lo.shape[-1]))
            return c_lo
        else:
            c_virt = []
            for s in range(diff_spin + 1):
                coeff = iao.get_iao_virt(mol, c_iao[s], ao_ovlp, minao=minao, \
                        generalized=not from_umf, verbose=True if mpirank==0 else False)
                c_virt.append(lo.orth.vec_lowdin(coeff, ao_ovlp))
            if not diff_spin:
                c_virt.append(c_virt[0])
            nvirt = np.shape(c_virt)[-1]
            c_iao = np.block([[c_iao[0], np.zeros((nao, niao))], [np.zeros((nao, niao)), c_iao[1]]])
            c_virt = np.block([[c_virt[0], np.zeros((nao, nvirt))], [np.zeros((nao, nvirt)), c_virt[1]]])
    else:
        log.info('localization from generalized solution')
        if has_pbc:
            mf_lo = pyscf.pbc.scf.GKS(mol, kpts[0])
        else:
            mf_lo = pyscf.scf.GKS(mol)
        if os.path.isfile(lo_chkfname + '.chk'):
            data = chkfile.load(lo_chkfname + '.chk', 'scf')
            mf_lo.__dict__.update(data)
        else:
            log.info("Running GHF to obtain intrinsic atomic orbital")
            mf_lo.chkfile = lo_chkfname + '.chk'
            mf_lo.kernel()
        c_iao = iao.iao(mol, mf_lo.mo_coeff[:, mf_lo.mo_occ>1e-10], minao=minao, \
            generalized=not from_umf, verbose=True if mpirank==0 else False)
        c_iao = lo.orth.vec_lowdin(c_iao, ao_ovlp)

        # PAO for virtual, adapted from libdmet
        if c_iao.shape[1] == c_iao.shape[0]: # no virtual 
            c_lo = c_iao
            niao = int(c_iao.shape[-1]//2) 
            nvirt = 0 
            if mpirank == 0:
                print("(spatial) nao, nval, nvirt, nlo_total\n%13d%6d%7d%8d"% \
                        (c_lo.shape[0] // 2, niao, nvirt, c_lo.shape[-1]))
            return c_lo
        else:
            c_virt = iao.get_iao_virt(mol, c_iao, ao_ovlp, minao=minao,\
                    generalized=not from_umf, verbose=True if mpirank==0 else False)
            c_virt = lo.orth.vec_lowdin(c_virt, ao_ovlp)
            niao = int(c_iao.shape[-1]//2) 
            nvirt = int(c_virt.shape[-1]//2)
    # order LO based on the original order of AO (alpha first and then beta)
    pmol = iao.reference_mol(mol, minao)
    pmol_labels = pmol.ao_labels()
    mol_labels = mol.ao_labels()
    iao_idx = np.array([idx for idx, label in enumerate(mol_labels) 
                if (label in pmol_labels)])
    virt_idx = np.array([idx for idx, label in enumerate(mol_labels) 
                if (not label in pmol_labels)])
    c_lo = np.zeros((nao * 2, nao * 2), dtype=complex)
    c_lo[:, iao_idx] = c_iao[:, :niao]
    c_lo[:, virt_idx] = c_virt[:, :nvirt]
    c_lo[:, iao_idx + nao] = c_iao[:, niao:]
    c_lo[:, virt_idx + nao] = c_virt[:, nvirt:]
    #c_lo = lo.orth.vec_lowdin(c_lo, ao_ovlp)
    if len(ao_ovlp) == nao: 
        ao_ovlp = la.block_diag(ao_ovlp, ao_ovlp)
    on_error =  np.max(np.abs(c_lo.T.conj() @ ao_ovlp @ c_lo - np.eye(nao * 2)))
    log.info("LO error from I %s" % on_error)
    if on_error > 0.1:
        log.warn("LO error too large. Please check definition of MINAO.")
    return c_lo

def plot_lo(mol, c_lo, ao_shell, atom_index=0, loc_method='lo', alpha_only=False):
    # plot localized orbitals of a specific shell on the atom of interest
    aoslice = mol.aoslice_by_atom()[atom_index]
    ao_labels = mol.ao_labels()[aoslice[2]:aoslice[3]]
    nao = mol.nao
    idx = np.where([ao_shell in ao_labels[i] for i in range(len(ao_labels))])[0] + aoslice[2]
    atom = mol._atom[atom_index][0]
    if np.max(np.abs(c_lo.imag)) < 1e-6: # alpha and beta LO are separate, i.e. c_lo is block diagonal 
                                         # plot alpha and beta separately
        c_lo = c_lo.real
        nlo = c_lo.shape[-1] // 2
        for i in idx:
            cubegen.orbital(mol, '%s_%s_a%s.cube'%(loc_method, atom, i), c_lo[:nao,i])
            if not alpha_only:
                cubegen.orbital(mol, '%s_%s_b%s.cube'%(loc_method, atom, i), c_lo[nao:, i+nlo])
    else:
        nlo = c_lo.shape[-1] // 2
        for i in idx:
            cubegen.orbital(mol, '%s_%s_a_real%s.cube'%(loc_method, atom, i), c_lo.real[:nao,i])
            if not alpha_only:
                cubegen.orbital(mol, '%s_%s_b_real%s.cube'%(loc_method, atom, i), c_lo.real[nao:, i+nlo])
        for i in idx:
            cubegen.orbital(mol, '%s_%s_a_imag%s.cube'%(loc_method, atom, i), c_lo.imag[:nao,i])
            if not alpha_only:
                cubegen.orbital(mol, '%s_%s_b_imag%s.cube'%(loc_method, atom, i), c_lo.imag[nao:, i+nlo])
        return 
        for i in np.append(idx, np.array(idx+nlo)):
            dm = (c_lo[:,i:i+1] @ c_lo[:,i:i+1].conj().T).reshape(2, nao, 2, nao)
            dm = np.sum(dm, axis=(0,2))
            cubegen.density(mol, '%s_%s_density%s.cube'%(loc_method, atom, i), dm)

def print_AO_labels(mol, minao=MINAO):
    print("****************************")
    print("Complete AO labels of mol")
    print("****************************")
    ao_labels = mol.ao_labels()
    for i in range(len(ao_labels)):
        print(i, ao_labels[i])
    print("****************************")
    print("AO labels of reference mol")
    print("****************************")
    from stevens.iao import reference_mol
    ao_labels = reference_mol(mol, minao=minao).ao_labels()
    for i in range(len(ao_labels)):
        print(i, ao_labels[i])
