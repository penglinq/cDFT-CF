#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Paul J. Robinson <pjrobinson@ucla.edu>
#         Zhi-Hao Cui <zhcui0408@gmail.com>
#         Linqing Peng <linqingp@outlook.com>

'''
Intrinsic Atomic Orbitals
ref. JCTC, 9, 4834
'''

from functools import reduce
import numpy
import numpy as np
import scipy.linalg
from libdmet.utils import logger as log
from libdmet.utils.misc import mdot, kdot, format_idx
from pyscf.lib import logger
from pyscf import gto
from pyscf import scf
from pyscf import __config__
from pyscf.lo.orth import vec_lowdin

# Alternately, use ANO for minao
# orthogonalize iao with coefficients obtained by
#     vec_lowdin(iao_coeff, mol.intor('int1e_ovlp'))
MINAO = getattr(__config__, 'lo_iao_minao', 'minao')

def iao(mol, orbocc, minao=MINAO, kpts=None, generalized=False, verbose=False):
    '''Intrinsic Atomic Orbitals. [Ref. JCTC, 9, 4834]

    Args:
        mol : the molecule or cell object

        orbocc : 2D array
            occupied orbitals

    Returns:
        non-orthogonal IAO orbitals.  Orthogonalize them as C (C^T S C)^{-1/2},
        eg using :func:`orth.lowdin`

        >>> orbocc = mf.mo_coeff[:,mf.mo_occ>0]
        >>> c = iao(mol, orbocc)
        >>> numpy.dot(c, orth.lowdin(reduce(numpy.dot, (c.T,s,c))))
    '''
    if mol.has_ecp():
        logger.warn(mol, 'ECP/PP is used. MINAO is not a good reference AO basis in IAO.')

    pmol = reference_mol(mol, minao)
    if verbose:
        B2_labels = pmol.ao_labels()
        B1_labels = mol.ao_labels()
        iao_idx = [idx for idx, label in enumerate(B1_labels) 
                    if (label in B2_labels)]
        print("-" * 79)
        print("IAO core+valence:")
        print("AO index   label")
        for i in iao_idx:
            print("%7d   %s"%(i, B1_labels[i]))
        print("-" * 79)
    # For PBC, we must use the pbc code for evaluating the integrals lest the
    # pbc conditions be ignored.
    # DO NOT import pbcgto early and check whether mol is a cell object.
    # "from pyscf.pbc import gto as pbcgto and isinstance(mol, pbcgto.Cell)"
    # The code should work even pbc module is not availabe.
    if getattr(mol, 'pbc_intor', None):  # cell object has pbc_intor method
        from pyscf.pbc import gto as pbcgto
        s1 = numpy.asarray(mol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
        s2 = numpy.asarray(pmol.pbc_intor('int1e_ovlp', hermi=1, kpts=kpts))
        s12 = numpy.asarray(pbcgto.cell.intor_cross('int1e_ovlp', mol, pmol, kpts=kpts))
        if kpts is not None:
            raise NotImplementedError
        if generalized:
            nao1 = len(s1)
            assert len(orbocc) == nao1 * 2
            s1 = numpy.block([[s1, numpy.zeros((nao1, nao1))], [numpy.zeros((nao1, nao1)), s1]])
            nao2 = len(s2)
            s2 = numpy.block([[s2, numpy.zeros((nao2, nao2))], [numpy.zeros((nao2, nao2)), s2]])
            s12 = numpy.block([[s12, numpy.zeros((nao1, nao2))], [numpy.zeros((nao1, nao2)), s12]])
  
    else:
        #s1 is the one electron overlap integrals (coulomb integrals)
        s1 = mol.intor_symmetric('int1e_ovlp')
        if generalized:
            nao1 = len(s1)
            assert len(orbocc) == nao1 * 2
            s1 = numpy.block([[s1, numpy.zeros((nao1, nao1))], [numpy.zeros((nao1, nao1)), s1]])
        #s2 is the same as s1 except in minao
        s2 = pmol.intor_symmetric('int1e_ovlp')
        if generalized:
            nao2 = len(s2)
            s2 = numpy.block([[s2, numpy.zeros((nao2, nao2))], [numpy.zeros((nao2, nao2)), s2]])
        #overlap integrals of the two molecules
        s12 = gto.mole.intor_cross('int1e_ovlp', mol, pmol)
        if generalized:
            s12 = numpy.block([[s12, numpy.zeros((nao1, nao2))], [numpy.zeros((nao1, nao2)), s12]])

    if len(s1.shape) == 2:
        s21 = s12.conj().T
        s1cd = scipy.linalg.cho_factor(s1)
        s2cd = scipy.linalg.cho_factor(s2)
        p12 = scipy.linalg.cho_solve(s1cd, s12)
        ctild = scipy.linalg.cho_solve(s2cd, numpy.dot(s21, orbocc))
        ctild = scipy.linalg.cho_solve(s1cd, numpy.dot(s12, ctild))
        ctild = vec_lowdin(ctild, s1)
        ccs1 = reduce(numpy.dot, (orbocc, orbocc.conj().T, s1))
        ccs2 = reduce(numpy.dot, (ctild, ctild.conj().T, s1))
        #a is the set of IAOs in the original basis
        a = (p12 + reduce(numpy.dot, (ccs1, ccs2, p12)) * 2
             - numpy.dot(ccs1, p12) - numpy.dot(ccs2, p12))
    else: # k point sampling
        s21 = numpy.swapaxes(s12, -1, -2).conj()
        nkpts = len(kpts)
        a = numpy.zeros((nkpts, s1.shape[-1], s2.shape[-1]), dtype=numpy.complex128)
        for k in range(nkpts):
            # ZHC NOTE check the case, at some kpts, there is no occupied MO.
            s1cd_k = scipy.linalg.cho_factor(s1[k])
            s2cd_k = scipy.linalg.cho_factor(s2[k])
            p12_k = scipy.linalg.cho_solve(s1cd_k, s12[k])
            ctild_k = scipy.linalg.cho_solve(s2cd_k, numpy.dot(s21[k], orbocc[k]))
            ctild_k = scipy.linalg.cho_solve(s1cd_k, numpy.dot(s12[k], ctild_k))
            ctild_k = vec_lowdin(ctild_k, s1[k])
            ccs1_k = reduce(numpy.dot, (orbocc[k], orbocc[k].conj().T, s1[k]))
            ccs2_k = reduce(numpy.dot, (ctild_k, ctild_k.conj().T, s1[k]))
            #a is the set of IAOs in the original basis
            a[k] = (p12_k + reduce(numpy.dot, (ccs1_k, ccs2_k, p12_k)) * 2
                    - numpy.dot(ccs1_k, p12_k) - numpy.dot(ccs2_k, p12_k))
    return a

def reference_mol(mol, minao=MINAO):
    '''Create a molecule which uses reference minimal basis'''
    pmol = mol.copy()
    if getattr(pmol, 'rcut', None) is not None:
        pmol.rcut = None
    pmol.build(False, False, basis=minao)
    return pmol


def fast_iao_mullikan_pop(mol, dm, iaos, verbose=logger.DEBUG):
    '''
    Args:
        mol : the molecule or cell object

        iaos : 2D array
            (orthogonal or non-orthogonal) IAO orbitals

    Returns:
        mullikan population analysis in the basis IAO
    '''
    pmol = reference_mol(mol)
    if getattr(mol, 'pbc_intor', None):  # whether mol object is a cell
        ovlpS = mol.pbc_intor('int1e_ovlp')
    else:
        ovlpS = mol.intor_symmetric('int1e_ovlp')

# Transform DM in big basis to IAO basis
# |IAO> = |big> C
# DM_{IAO} = C^{-1} DM (C^{-1})^T = S_{IAO}^{-1} C^T S DM S C S_{IAO}^{-1}
    cs = numpy.dot(iaos.T.conj(), ovlpS)
    s_iao = numpy.dot(cs, iaos)
    iao_inv = numpy.linalg.solve(s_iao, cs)
    if isinstance(dm, numpy.ndarray) and dm.ndim == 2:
        dm = reduce(numpy.dot, (iao_inv, dm, iao_inv.conj().T))
        return scf.hf.mulliken_pop(pmol, dm, s_iao, verbose)
    else:
        dm = [reduce(numpy.dot, (iao_inv, dm[0], iao_inv.conj().T)),
              reduce(numpy.dot, (iao_inv, dm[1], iao_inv.conj().T))]
        return scf.uhf.mulliken_pop(pmol, dm, s_iao, verbose)

def get_iao_virt(cell, C_ao_iao, S, minao=MINAO, full_virt=False, pmol=None,
                 max_ovlp=False, verbose=True, B2=None, nomix=False, generalized=False): 
    """
    Get virtual orbitals from orthogonal IAO orbitals, C_ao_iao.
    Math: (1 - |IAO><IAO|) |B1> where B1 only choose the remaining virtual AO basis.
    """
    if max_ovlp:
        return get_iao_virt_max_ovlp(cell, C_ao_iao, S, minao=minao, 
                                     full_virt=full_virt, pmol=pmol)

    C_ao_iao = np.asarray(C_ao_iao)
    S = np.asarray(S)
    if pmol is None:
        pmol = reference_mol(cell, minao)

    B1_labels = cell.ao_labels()
    if full_virt:
        B2_labels = []
    else:
        B2_labels = pmol.ao_labels()

    virt_idx = [idx for idx, label in enumerate(B1_labels) 
                if (not label in B2_labels)]
    if verbose:
        print("-" * 79)
        print("IAO virtuals:")
        print("AO index   label")
        for i in virt_idx:
            print("%7d   %s"%(i, B1_labels[i]))
        print("-" * 79)
    nB1 = len(B1_labels)
    nB2 = len(B2_labels)
    nvirt = len(virt_idx)
    assert nB2 + nvirt == nB1
    
    if S.ndim == 3: # with kpts:
        nkpts = C_ao_iao.shape[-3]
        if C_ao_iao.ndim == 3:
            C_virt = np.zeros((nkpts, nB1, nvirt), dtype=np.complex128) 
            for k in range(nkpts):
                CCdS_k = mdot(C_ao_iao[k], C_ao_iao[k].conj().T, S[k])
                C_virt[k] = (np.eye(nB1) - CCdS_k)[:, virt_idx]
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nkpts, nB1, nvirt), dtype=np.complex128) 
            for s in range(spin):
                for k in range(nkpts):
                    CCdS_sk = mdot(C_ao_iao[s, k], C_ao_iao[s, k].conj().T, S[k])
                    C_virt[s, k] = (np.eye(nB1) - CCdS_sk)[:, virt_idx]
    else:
        if C_ao_iao.ndim == 2:
            CCdS = mdot(C_ao_iao, C_ao_iao.conj().T, S)
            if generalized: # combine alpha and beta spins
                    virt_idx = np.append(virt_idx, np.array(virt_idx) + nB1)
                    nB1 *= 2
            C_virt = (np.eye(nB1) - CCdS)[:, virt_idx]
        else:
            spin = C_ao_iao.shape[0]
            C_virt = np.zeros((spin, nB1, nvirt), dtype=C_ao_iao.dtype)
            for s in range(spin):
                CCdS_s = mdot(C_ao_iao[s], C_ao_iao[s].conj().T, S)
                C_virt[s] = (np.eye(nB1) - CCdS_s)[:, virt_idx]
    return C_virt

del(MINAO)
