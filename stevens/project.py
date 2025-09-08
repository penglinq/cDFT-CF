'''
Expand a Slater Determinant in eigenfunctions of J amd Jz
'''
import pyscf
from functools import reduce
from pyscf import scf, lib, lo
from pyscf.lib import chkfile
import os, sys
import h5py
import numpy as np
import scipy.linalg as la
import scipy
from stevens import localize 
from stevens import momentum
from stevens.momentum import get_mj
from stevens import iao
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
log = lib.logger.Logger(sys.stdout, 4)

ang2bohr = 1.8897259 
np.set_printoptions(precision=4, threshold=sys.maxsize, linewidth=380, suppress=True)
MINAO = {'default': 'minao', 'Ho': 'sarcdkhminao', 'Er': 'sarcdkhminao', 'Dy': 'sarcdkhminao'}

class Project():
    def __init__(self, mol=None, gmf=None, atom_index=None, ao_index=None, ao_shell=None, spin_separate=True, loc_method='iao', \
            lo_chkfname=None, trunc_R=True, nao=None, mo_coeff=None, mo_occ=None,  ao_ovlp=None, l_integral=None, verbose=False, \
            N_grid=None, occ_thres=1e-6, lo_from_umf=True, kpts=[[0,0,0]], minao=MINAO, small_mem=False):
        """
        Initialize an instance of Project.
        For PySCF users, passing in "mol" and "gmf" is enough.
        To inteface with other packages, pass in nao, mo_coeff, mo_occ, ao_ovlp, and l_integral.

        Args:
            mol (an instance of PySCF gto.Mole class): the molecule/unit cell.
            gmf (an instance of PySCF GHF or GKS class): a generalized mean-field calculation.
            nao (int): the number of spatial atomic orbital.
            mo_coeff ((2*nao x 2*nao) array): the rotation matrix from spin atomic orbitals to spin molecular orbitals.
            mo_occ ((2*nao) array): the occupation number of spin molecular orbitals, consisting of 1's and 0's.
            ao_ovlp ((2*nao x 2*nao) array): overlap integral between spin atomic orbitals.
            l_integral ((3 x nao x nao) array): Angular momentum integral r x p between spatial atomic orbitals. 
                                                By default, treat (0,0,0) as the origin.
            small_mem: True if memory is very limited. Discard attribute that probably won't be used again. Could cause
                       problem if using rot_DFT after JM projection.
        """
        if mol is None and gmf is None:
            assert not None in [nao, mo_coeff, mo_occ, ao_ovlp, l_integral] 
        elif mol is None:
            mol = gmf.mol
        if mol is not None:
            has_pbc = getattr(mol, 'dimension', 0) > 0
            self.mol = mol
            self.nelec = mol.nelectron
            nao = mol.nao
            self.spin = mol.spin
            if has_pbc:
                ao_ovlp = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=kpts,
                                   pbcopt=lib.c_null_ptr()))[0] ### TODO
                self.ao_ovlp = la.block_diag(ao_ovlp, ao_ovlp)
                l_integral = (np.array(mol.pbc_intor('int1e_cg_irxp', kpts=kpts)) *(-1j))[0] ### TODO
            else:
                ao_ovlp = mol.intor_symmetric('int1e_ovlp') 
                self.ao_ovlp = la.block_diag(ao_ovlp, ao_ovlp)
                l_integral = np.array(mol.intor('int1e_cg_irxp') *(-1j))
        self.ao_index = None  # the spatial AO included in rotation R
        self.L_ao_index = None  # the spatial AO included in L to generate R_L
        self.loc_method = loc_method
        self.lo_chkfname = lo_chkfname
        self.lo_from_umf = lo_from_umf
        self.verbose = verbose
        self.minao = minao
        self.small_mem = small_mem
        if gmf is not None:
            self._scf = gmf
            self.mo_coeff = np.array(gmf.mo_coeff[:, gmf.mo_occ>occ_thres])
            self.mo_occ = np.array(gmf.mo_occ[gmf.mo_occ>occ_thres])
            if nao is None: 
                nao = gmf.mol.nao
            if atom_index is not None:
                # localized AO by IAO and orthogonalize
                loc_method = loc_method.lower()
                c_lo = np.zeros((nao*2, nao*2))
                if mpirank == 0:
                    c_lo = self.c_lo
                comm.Barrier()
                c_lo = comm.bcast(c_lo, root=0)
                # self.c_lo_inv = c_lo.T.conj() @ self.ao_ovlp
                nlo = int(c_lo.shape[-1] // 2)
                if spin_separate:
                    assert np.max(np.abs(c_lo[nao:, :nlo])) < 1e-6 # True only if DM_ab and DM_ba are 0
                    assert np.max(np.abs(c_lo[:nao, nlo:])) < 1e-6 # True only if DM_ab and DM_ba are 0
                    c_lo_spat = c_lo[:nao, :nlo] ### TODO here assumed c_lo for two spins are the same
                if verbose >= 4 and mpirank == 0: # visualize localized orbitals in .cube files
                    localize.plot_lo(self, atom_index, ao_shell, loc_method)
                nao = nlo
        
                # rotate from AO to LO  
                self.mo_coeff = c_lo.T.conj() @ self.ao_ovlp @ self.mo_coeff 
                self.ao_ovlp = c_lo.T.conj() @ self.ao_ovlp @ c_lo
                log.info("LO diff from identity %s" % np.max(np.abs(self.ao_ovlp - np.eye(nao * 2))))
                if np.max(np.abs(self.ao_ovlp - np.eye(nao * 2))) > 1e-4:
                    log.warn("LO is not orthonormal!")
                if spin_separate:
                    l_integral = c_lo_spat.T.conj() @ l_integral @ c_lo_spat
                else:
                    tmp = np.empty((3, l_integral.shape[1]*2, l_integral.shape[2]*2), dtype=complex)
                    for i in range(3):
                        tmp[i] = la.block_diag(l_integral[i], l_integral[i])
                    l_integral = tmp
                    tmp = None
                    l_integral = c_lo.T.conj() @ l_integral
                    l_integral = l_integral @ c_lo

                # select LO to rotate
                _ = self.trunc_lo(c_lo, atom_index, ao_shell, ao_index, spin_separate) # only calculate self.ao_index
                if not trunc_R:
                    self.L_ao_index = self.ao_index
        else:
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
        self.nao = nao
        self.spin_separate = spin_separate
        np.set_printoptions(precision=4, threshold=sys.maxsize, linewidth=380, suppress=True)
        if spin_separate:
            self.l_integral = la.inv(self.ao_ovlp[:nao, :nao]) @ l_integral 
            l_integral = None # save memory
        else:
            if l_integral.shape[1] == nao:
                l_integral = [la.block_diag(l_integral[i], l_integral[i]) for i in range(3)]
            self.l_integral = la.inv(self.ao_ovlp) @ l_integral 
            l_integral = None # save memory
        self.rotated_mo_coeffs = None
        self.quad = None
        self.nocc = self.nelec
        self.J_max = -1
        self.double_domain = False
        self.N_grid = N_grid

    @property
    def c_lo(self, loc_method='iao'):
        # This method calculate c_lo when needed to avoid the memory cost of storing c_lo (N^2)
        mol = self.mol
        loc_method = self.loc_method
        lo_chkfname = self.lo_chkfname
        lo_from_umf = self.lo_from_umf
        verbose = self.verbose
        minao = self.minao

        if loc_method == 'iao': # IAO + PAO
            if os.path.isfile("c_lo.npy"):
                c_lo = np.load("c_lo.npy")
                log.info("Loaded localization c_lo.npy")
            else:
                c_lo = self.get_iao(lo_chkfname, from_umf=lo_from_umf, verbose=verbose) 
                np.save("c_lo.npy", c_lo)
        elif loc_method == 'lowdin': # lowdin
            c_lo = lo.orth.lowdin(self.ao_ovlp)
        elif loc_method == 'l_lowdin':
            if has_pbc:
                l_integral = (np.array(mol.pbc_intor('int1e_cg_irxp', kpts=kpts)) *(-1j))[0] ### TODO
            else:
                l_integral = np.array(mol.intor('int1e_cg_irxp') *(-1j))
            c_lo = iao.iao(mol, self.mo_coeff, \
                    minao=minao, generalized=True, verbose=True if mpirank==0 else False)
            c_lo = self.vec_L_lowdin(c_lo, la.expm(-1j * l_integral[2]))
            # PAO for virtual, adapted from libdmet
            c_virt = iao.get_iao_virt(mol, c_lo, self.ao_ovlp, minao=minao, generalized=True, \
                    verbose=True if mpirank==0 else False)
            c_virt = self.vec_L_lowdin(c_virt, la.expm(-1j * l_integral[2]))
            niao = int(c_lo.shape[-1]//2) 
            nvirt = int(c_virt.shape[-1]//2)
            # to preserve alpha and beta order (approximately) 
            c_lo = np.hstack((c_lo[:, :niao], c_virt[:, :nvirt], c_lo[:, niao:], c_virt[:, nvirt:]))
            if mpirank == 0:
                print("(spatial) nao, nval, nvirt, nlo_total\n%13d%6d%7d%7d"% \
                        (c_lo.shape[0] // 2, niao, nvirt, c_lo.shape[-1]))
        else:
            raise NotImplementedError
        return c_lo

    def get_iao(self, lo_chkfname, from_umf, verbose=False, diff_spin=False): 
        if lo_chkfname[-2:] == 'h5':
            # directly input c_lo
            feri = h5py.File(lo_chkfname, 'r')
            c_lo = np.array(feri['c_lo'])
            feri.close()
            return c_lo
        elif lo_chkfname[-3:] == 'npy':
            return np.load(lo_chkfname)
        mol = self.mol
        nao = mol.nao
        if from_umf:
            ao_ovlp = self.ao_ovlp[:nao, :nao]
        else:
            ao_ovlp = self.ao_ovlp
        c_lo = localize.get_iao(mol, lo_chkfname, from_umf, ao_ovlp=ao_ovlp, verbose=verbose, diff_spin=diff_spin)
        return c_lo 

    def trunc_lo(self, c_lo, atom_index, ao_shell, ao_index=None, spin_separate=False):
        mol = self.mol
        nao = mol.nao
        # select LO to rotate
        ao_labels = mol.ao_labels()
        if ao_index is None:
            assert ao_shell is not None
            ao_slice = mol.aoslice_by_atom()[atom_index] 
            ao_index = []
            for i in range(ao_slice[2], ao_slice[3]):
                if ao_shell in ao_labels[i]:
                    ao_index.append(i)
        if spin_separate:
            self.ao_index = ao_index
        else:
            self.ao_index = np.append(ao_index, np.array(ao_index) + nao)  
        if mpirank == 0:
            print("ao_index", self.ao_index)
            print("ao_labels", np.array(ao_labels)[ao_index])
        c_lo = c_lo[:, self.ao_index] # no longer unitary rotation
        return c_lo

    def plot_lo(self, atom_index, ao_shell, loc_method='lo', alpha_only=False):
        # plot localized orbitals
        mol = self.mol
        c_lo = self.c_lo
        localize.plot_lo(mol, c_lo, ao_shell, atom_index, loc_method, alpha_only=alpha_only)
        
    def lowdin(self, s):
        ''' new basis is |mu> c^{lowdin}_{mu i} '''
        e, v = scipy.linalg.eigh(s)
        idx = e > 1e-15
        return np.dot(v[:,idx]/np.sqrt(e[idx]), v[:,idx].conj().T)
    
    def vec_L_lowdin(self, c, l=1):
        ''' lowdin orth for the metric c.T*L*c and get x, then c*x'''
        if isinstance(l, np.ndarray) and len(c) == 2 * len(l):
            nao = len(l)
            l = np.block([[l, np.zeros((nao, nao))], [np.zeros((nao, nao)), l]])
        return np.dot(c, self.lowdin(reduce(np.dot, (c.conj().T,l,c))))

    def get_ao_ovlp(self):
        """
        Compute the AO overlap matrix S.
        """
        if self.ao_ovlp is None:
            self.ao_ovlp = self._scf.get_ovlp()
        return self.ao_ovlp
    
    def generate_quad(self, N_alpha=None, N_beta=None, N_gamma=None, verbose=False):
        """
        Generate quadrature points and weights. For integration over alpha and gamma, we use 
        a Trapezoid grid; for integration over beta, we use a Gauss-Legendre grid.
        Reminder: test numbers of alpha, beta, and gamma for convergence 
        """
        # beta.
        if N_beta is None:
            if J_max > -1:
                N_beta = int(np.ceil(self.J_max)) + 1  # Most likely good accuracy requires more than this 
            else:
                N_beta = sum(self.nelec) 
        
        # Quadrature points and weights.
        # x = cos(beta) in [-1, 1]
        xs, ws = np.polynomial.legendre.leggauss(N_beta)
        betas = np.arccos(xs)
        sorted_inds = np.argsort(betas)
        betas.sort()
        ws = ws[sorted_inds] # No need to / (- sin(beta)). The division cancels out with sin(beta)  
                             # in the integral factor over spherical coordinates
        
        # alpha.
        if N_alpha is None:
            if self.double_domain:
                N_alpha = 4 * N_beta
            else:
                N_alpha = 2 * N_beta
        
        if self.double_domain:
            alphas = np.linspace(4 * np.pi / N_alpha, 4 * np.pi, N_alpha)
        else:
            alphas = np.linspace(2 * np.pi / N_alpha, 2 * np.pi, N_alpha)

       
        # gamma.
        if N_gamma is None:
            N_gamma = 2 * N_beta
            
        gammas = np.linspace(2 * np.pi/ N_gamma, 2 * np.pi, N_gamma)
        
        if verbose:
            print(f'alphas: \n{alphas}\n')
            print(f'gammas: \n{gammas}\n')
            print(f'betas: \n{betas}\n')
            print(f'ws: \n{ws}\n')
                  
        self.quad = np.array([alphas, betas, gammas, ws], dtype=object)
        return alphas, betas, gammas, ws
        
    def get_wigner_d(self, J, m, k, beta):
        """
        Wigner small d-matrix.
        """
        return get_wigner_d(J, m, k, beta)

    def Ry_spin(self, mo_coeff, beta):
        """
        Rotates the spin components of the determinants in the coefficient matrix `mo_coeff` by angle `beta`
        about the y axis.
        math (when rotating all orbitals):
            Ry_mat = np.block([[np.cos(beta/2) * id_mat, -np.sin(beta/2) * id_mat],
                              [np.sin(beta/2) * id_mat, np.cos(beta/2) * id_mat]])

        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            beta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        nao = self.nao
        ao_index = self.ao_index[:len(self.ao_index)//2]
        Ry_mat_aa = np.eye(nao)
        Ry_mat_aa[(ao_index, ao_index)] = np.cos(beta/2)
        Ry_mat_ab = np.eye(nao) * 0
        Ry_mat_ab[(ao_index, ao_index)] = -np.sin(beta/2)
        Ry_mat_ba = np.eye(nao) * 0
        Ry_mat_ba[(ao_index, ao_index)] = np.sin(beta/2)
        Ry_mat_bb = np.eye(nao)
        Ry_mat_bb[(ao_index, ao_index)] = np.cos(beta/2)
        Ry_mat = np.block([[Ry_mat_aa, Ry_mat_ab], [Ry_mat_ba, Ry_mat_bb]])
        return lib.dot(Ry_mat, mo_coeff)
    
    def Rz_spin(self, mo_coeff, theta):
        """
        Rotates the spin components of the determinants in the coefficient matrix `mo_coeff` by angle `theta`
        about the z axis.
        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            theta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        nao = self.nao
        ao_index = self.ao_index[:len(self.ao_index)//2]
        Rz_mat_aa = np.eye(nao, dtype=complex)
        Rz_mat_aa[(ao_index, ao_index)] = np.exp(-1j * theta/2)
        Rz_mat_bb = np.eye(nao, dtype=complex)
        Rz_mat_bb[(ao_index, ao_index)] = np.exp(1j * theta/2)
        Rz_mat = np.block([[Rz_mat_aa, np.zeros((nao, nao))],
                          [np.zeros((nao, nao)), Rz_mat_bb]])
        return lib.dot(Rz_mat, mo_coeff)
                
    def Ry_orb(self, mo_coeff, beta, Ry_mat=None):
        """
        Rotates the orbital components of the determinants in the coefficient matrix `mo_coeff` by angle `beta`
        about the y axis.
        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            beta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        if Ry_mat is None:
            nao = self.nao
            if not self.spin_separate: 
                nao = 2 * nao
            l_integral_y = self.l_integral[1] 
            if self.L_ao_index is None:
                Ry_mat = la.expm(- 1j * beta * l_integral_y)
                if self.ao_index is not None:
                    Ry_mat_new = np.eye(nao, dtype=complex)
                    q, r = la.qr(Ry_mat[np.ix_(self.ao_index, self.ao_index)])
                    sign = np.sign(np.diag(r))
                    assert not (sign == 0).any()
                    Ry_mat = q @ np.diag(sign)
                    Ry_mat_new[np.ix_(self.ao_index, self.ao_index)] = Ry_mat
                    Ry_mat = Ry_mat_new
            else:
                Ry_mat = np.eye(len(l_integral_y), dtype=complex)
                Ry_mat[np.ix_(self.L_ao_index, self.L_ao_index)] = la.expm(- 1j * beta * \
                        l_integral_y[self.L_ao_index][:, self.L_ao_index])
            if self.spin_separate:
                Ry_mat = np.block([[Ry_mat, np.zeros((nao, nao))], [np.zeros((nao,nao)), Ry_mat]])
        return lib.dot(Ry_mat, mo_coeff)
    
    def Rz_orb(self, mo_coeff, theta, Rz_mat=None):
        """
        Rotates the orbital components of the determinants in the coefficient matrix `mo_coeff` by angle `theta`
        about the z axis.
        
        Args
            mo_coeff (ndarray): MO coefficient matrix from a GHF object.
            theta (float): Rotation angle.
            
        Returns
            The rotated coefficient matrix.
        """
        if Rz_mat is None:
            nao = self.nao
            if not self.spin_separate: 
                nao = 2 * nao
            l_integral_z = self.l_integral[2] 
            if self.L_ao_index is None:
                Rz_mat = la.expm(- 1j * theta * l_integral_z)
                if self.ao_index is not None:
                    Rz_mat_new = np.eye(nao, dtype=complex)
                    q, r = la.qr(Rz_mat[np.ix_(self.ao_index, self.ao_index)])
                    sign = np.sign(np.diag(r))
                    assert not (sign == 0).any()
                    Rz_mat = q @ np.diag(sign)
                    Rz_mat_new[np.ix_(self.ao_index, self.ao_index)] = Rz_mat
                    Rz_mat = Rz_mat_new
            else:
                Rz_mat = np.eye(len(l_integral_z), dtype=complex)
                Rz_mat[np.ix_(self.L_ao_index, self.L_ao_index)] = la.expm(- 1j * theta * \
                        l_integral_z[self.L_ao_index][:, self.L_ao_index])
            if self.spin_separate:
                Rz_mat = np.block([[Rz_mat, np.zeros((nao, nao))], [np.zeros((nao,nao)), Rz_mat]])
        return lib.dot(Rz_mat, mo_coeff)

    def get_rotated_mo_coeffs(self, mo_coeff, proj='full', N_alpha=None, N_beta=None, N_gamma=None, \
            verbose=False, normalize_MO=True):
        """
        Generates rotated coefficient matrices from `mo_coeff` at each quadrature point.
        """
        if not isinstance(self.quad, np.ndarray):
            self.generate_quad(N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, verbose=verbose)
        alphas, betas, gammas, ws = self.quad
         
        # generate all rotation matrices first to trade memory for time
        if proj in ['orb', 'full']:
            ''' 
            # serial version
            Rz_orb_lst = np.zeros((len(gammas), len(mo_coeff), len(mo_coeff)), dtype=complex)
            Ry_orb_lst = np.zeros((len(betas), len(mo_coeff), len(mo_coeff)), dtype=complex)
            for c, gamma_c in enumerate(gammas):
                Rz_orb_lst[c] = self.Rz_orb(np.eye(len(mo_coeff)), gamma_c)
            for b, beta_b in enumerate(betas):
                Ry_orb_lst[b] = self.Ry_orb(np.eye(len(mo_coeff)), beta_b)
            '''
            idx_lst = np.arange(mpirank, len(gammas) + len(betas), mpisize)
            print("rank = %s, idx_lst = %s"%(mpirank, idx_lst))
            for i in idx_lst:
                if i < len(gammas):
                    res = self.Rz_orb(np.eye(len(mo_coeff)), gammas[i])
                    np.save("Rz_%s.npy"%i, res)
                else:
                    res = self.Ry_orb(np.eye(len(mo_coeff)), betas[i-len(gammas)])
                    np.save("Ry_%s.npy"%(i - len(gammas)), res)
            res = None
            comm.Barrier()
            if not self.small_mem:
                Rz_orb_lst = np.empty((len(gammas), len(mo_coeff), len(mo_coeff)), dtype=complex)
                Ry_orb_lst = np.empty((len(betas), len(mo_coeff), len(mo_coeff)), dtype=complex)
                for i in range(len(gammas)):
                    Rz_orb_lst[i] = np.load('Rz_%s.npy'%i)
                for i in range(len(betas)):
                    Ry_orb_lst[i] = np.load('Ry_%s.npy'%i)
        if self.small_mem:
            self.l_integral = None # save memory
        # No need to rotate mo_coeff back to AO basis as long as LO are still spin-separated
        #if proj in ['spin', 'full'] and self.c_lo is not None:
        #    mo_coeff = self.c_lo @ mo_coeff

        # 1st index - alpha
        # 2nd index - beta
        # 3rd index - gamma
        # Angles in increasing order along all axes.
        # When mpisize > 1, only store the fraction of rotated mo_coeff to calculate det_ovlp
        njob = int(np.ceil(N_alpha * N_beta * N_gamma / mpisize))
        if mpirank == mpisize - 1: 
            nrot = N_alpha * N_beta * N_gamma - njob * mpirank
        else:
            nrot = njob
        rotated_mo_coeffs = np.empty((max(nrot, 0), mo_coeff.shape[0], mo_coeff.shape[1]), dtype=complex)
        count = 0
        for a in range(int(np.floor(njob * mpirank / (N_beta * N_gamma))), \
                int(min(N_alpha, np.ceil(njob * (mpirank + 1) / (N_beta * N_gamma))))):
            for b in range(int(max(0, np.floor((njob * mpirank - a * N_beta * N_gamma)/N_gamma))), \
                    int(min(N_beta, np.ceil((njob * (mpirank + 1) - a * N_beta * N_gamma)/N_gamma)))):
                for c in range(int(max(0, njob * mpirank - a * N_beta * N_gamma - b * N_gamma)), \
                    int(min(N_gamma, njob * (mpirank + 1) - a * N_beta * N_gamma - b * N_gamma))):
                    if proj == 'orb' or proj == 'full':
                        if self.small_mem:
                            Rz_gamma = np.load('Rz_%s.npy'%c)
                            Ry_beta = np.load('Ry_%s.npy'%b)
                            Rz_alpha = np.load('Rz_%s.npy'%a)
                        else:
                            Rz_gamma = Rz_orb_lst[c]
                            Ry_beta = Ry_orb_lst[b]
                            Rz_alpha = Rz_orb_lst[a]
                    if proj == 'orb':
                        mo_rot = self.Rz_orb(mo_coeff, gammas[c], Rz_mat=Rz_gamma)
                        mo_rot = self.Ry_orb(mo_rot, betas[b], Ry_mat=Ry_beta)
                        mo_rot = self.Rz_orb(mo_rot, alphas[a], Rz_mat=Rz_alpha)
                    elif proj == 'spin': 
                        mo_rot = self.Rz_spin(mo_coeff, gammas[c])
                        mo_rot = self.Ry_spin(mo_rot, betas[b])
                        mo_rot = self.Rz_spin(mo_rot, alphas[a])
                        # No need to rotate mo_coeff back to AO basis as long as LO are still spin-separated
                        #if self.c_lo is not None:
                        #    final = lib.dot(self.c_lo_inv, final) # rotate mo_coeff from AO basis back to LO basis
                    elif proj == 'full':
                        mo_rot = self.Rz_spin(mo_coeff, gammas[c])
                        mo_rot = self.Ry_spin(mo_rot, betas[b])
                        mo_rot = self.Rz_spin(mo_rot, alphas[a])
                        # No need to rotate mo_coeff back to AO basis as long as LO are still spin-separated
                        #if self.c_lo is not None:
                        #    zyz = lib.dot(self.c_lo_inv, zyz) # rotate mo_coeff from AO basis back to LO basis
                        mo_rot = self.Rz_orb(mo_rot, gammas[c], Rz_mat=Rz_gamma)
                        mo_rot = self.Ry_orb(mo_rot, betas[b], Ry_mat=Ry_beta)
                        mo_rot = self.Rz_orb(mo_rot, alphas[a], Rz_mat=Rz_alpha)
                    Rz_gamma, Ry_beta, Rz_alpha = None, None, None
                    rotated_mo_coeffs[count] = mo_rot
                    count += 1
        self.rotated_mo_coeffs = rotated_mo_coeffs
        return self.rotated_mo_coeffs
        
    def quad_coeffs(self, J, m, k, N_alpha=None, N_beta=None, N_gamma=None, verbose=False,):
        """
        Returns a 3D array of the coefficient at each quad point.
        """
        if not isinstance(self.quad, np.ndarray):
            alphas, betas, gammas, ws = self.generate_quad(N_alpha=N_alpha, N_beta=N_beta, 
                                                           N_gamma=N_gamma, verbose=verbose)
            
        alphas, betas, gammas, ws = self.quad
        N_alpha = len(alphas)
        N_beta = len(betas)
        N_gamma = len(gammas)
        
        prefactor = (2*J + 1) / (8 * np.pi**2) * (2 * np.pi) / N_alpha * (2 * np.pi) / N_gamma
        # The prefactor is the same if double_domain:
        # (2*J + 1) / (16 * np.pi**2) * (4 * np.pi) / N_alpha ...

        coeffs = []
        # When mpisize > 1, only store the fraction of rotated mo_coeff with which det_ovlp calculates
        njob = np.ceil(N_alpha * N_beta * N_gamma / mpisize)
        for a in range(int(np.floor(njob * mpirank / (N_beta * N_gamma))), \
                int(min(N_alpha, np.ceil(njob * (mpirank + 1) / (N_beta * N_gamma))))):
            for b in range(int(max(0, np.floor((njob * mpirank - a * N_beta * N_gamma)/N_gamma))), \
                    int(min(N_beta, np.ceil((njob * (mpirank + 1) - a * N_beta * N_gamma)/N_gamma)))):
                for c in range(int(max(0, njob * mpirank - a * N_beta * N_gamma - b * N_gamma)), \
                    int(min(N_gamma, njob * (mpirank + 1) - a * N_beta * N_gamma - b * N_gamma))):
                    coeffs.append(ws[b] * self.get_wigner_d(J, m, k, betas[b]) * \
                                        np.exp(1j * m * alphas[a]) * np.exp(1j * k * gammas[c]))
        
        coeffs = prefactor * np.array(coeffs)
        return coeffs

    def det_ovlp(self, mo_coeff1, mo_coeff2, mo_occ1, mo_occ2):
        """
        Calculates the overlap between two different determinants. It is the product
        of single values of molecular orbital overlap matrix.
        
        WARNING: Has some sign errors when computing overlaps between double excitations
                 < Psi_{ij}^{ab} | Psi_{kl}^{cd} >.
        Return: a complex scalar
        """
        if np.sum(mo_occ1) != np.sum(mo_occ2):
            raise RuntimeError('Electron numbers are not equal. Electronic coupling does not exist.')

        s = self.get_mo_ovlp(mo_coeff1[:,mo_occ1>0], mo_coeff2[:,mo_occ2>0])
        return np.linalg.det(s)

    def get_mo_ovlp(self, mo_coeff1, mo_coeff2):
        """
        Calculates the MO overlap matrix.
        """
        nao = mo_coeff1.shape[0] // 2
        ao_ovlp = self.get_ao_ovlp()
        s = reduce(lib.dot, (mo_coeff1.T.conj(), ao_ovlp, mo_coeff2))
        return s

    def get_proj_ovlp(self, J, m, k, mo_coeff=None, mo_occ=None, proj='full', N_alpha=None, N_beta=None, \
            N_gamma=None, verbose=False):
        """
        Compute overlap between the original DFT(/HF) determinant and JM-projected wavefunction <Psi|P|Psi>.
        To get P|Psi>, we calculate rotated_mo_coeff = P_operator |Psi> and quad_coeffs = w_quad * P_coeff.
        Math:
            P = (2J+1)/(8pi^2) \int_{\alpha, \beta, \gamma} sin(\alpha) (e^{-im\alpha} d^J_{mk}(\beta)
                 e^{-ik\gamma})^* e^{-i\alpha \hat{S}_z} e^{-i\beta \hat{S}_y} e^{-i\gamma \hat{S}_z}
            Note here the complex conjugate of the Wigner D-matrix. Wigner small d-matrix is real, so just to need to
            take * of alpha and beta term. 
        When using quadrature to integrate, 
            P = \sum_{\alpha, \beta, \gamma} w_quad (P_coeff)^* P_operator
        where P_coeff = (2J+1)/(8pi^2) 2*pi/N_\alpha 2*pi/N_\gamma e^{-im\alpha} d^J_{mk}(\beta) e^{-ik\gamma})^
        and P_operator = e^{-i\alpha \hat{S}_z} e^{-i\beta \hat{S}_y} e^{-i\gamma \hat{S}_z}. 

        Args
            J (int): Total spin J (L, or S) eigenvalue.
            m (int): Final Jz (Lz, or Sz) eigenvalue.
            k (int): Initial Jz (Lz or Sz) eigenvalue.
            
        Returns
            <Psi_{DFT}|P|Psi_{DFT}> 
        """ 
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if mo_occ is None:
            mo_occ = self.mo_occ
        if J > self.J_max:
            self.J_max = J
        if proj == 'full' or proj == 'spin': 
            spin = self.spin/2 
        elif proj == 'orb':
            spin = 0
        self.double_domain = self.double_domain or abs(round(spin - J) - (spin - J)) > 1e-4
        if verbose: 
            print("double domain", self.double_domain)
        nao, nmo = mo_coeff.shape
        if not isinstance(self.rotated_mo_coeffs, np.ndarray) or np.any(self.N_grid != (N_alpha, N_beta, N_gamma)):
            if self.small_mem:
                # release l_integral, load 3 rotation matrix
                print("current memory", lib.current_memory()[0], "R matrix",  
                    np.ceil(N_alpha*N_beta*N_gamma / mpisize + 1), nao, nmo )
                print("memory required %.0f MB" % (lib.current_memory()[0] + 
                    np.ceil(N_alpha*N_beta*N_gamma / mpisize + 1) * nao * nmo * 16 / 1024**2))
            # if J is a half-integer, double the integration domain by integrating alpha from 0 to 4pi
            self.get_rotated_mo_coeffs(mo_coeff, proj=proj, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma, \
                    verbose=verbose)
            if np.any(self.N_grid != (N_alpha, N_beta, N_gamma)):
                self.N_grid = (N_alpha, N_beta, N_gamma)
        rot_mo_coeffs = self.rotated_mo_coeffs
        coeffs = self.quad_coeffs(J, m, k, N_alpha=N_alpha, N_beta=N_beta, N_gamma=N_gamma)
        norm = 0.

        for i in range(len(coeffs)):
            rot_mo_coeff = rot_mo_coeffs[i]
            coeff = coeffs[i]
            _norm = self.det_ovlp(mo_coeff, rot_mo_coeff, mo_occ, mo_occ)
            if verbose >= 4:
                if mpisize == 1:
                    print(np.unravel_index(i, (N_alpha, N_beta, N_gamma)), _norm, coeff)
                else:
                    print(i, _norm, coeff)
            norm += coeff * _norm
        ### TODO: in the future, uncomment the following 
        ###norm = norm.conj() # add the complex conjugate on the D matrix. 
        comm.Barrier()
        norm = comm.reduce(norm, root=0)

        return norm 

    def rot_DFT(self, beta=None, alpha=None, gamma=None, mo_coeff=None):
        '''
        Rotate selected localized AOs in a localized AO to MO coefficient. 
        The rotated localized AO selection is determined by self.ao_index
        Note: all AOs here are orthogonal localized AOs. To obtain the original 
            AO to MO coefficient, use  self.c_lo @ mo_coeff
        '''
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        if alpha is not None:
            self.mo_coeff = self.Rz_orb(mo_coeff, alpha)
            self.mo_coeff = self.Rz_spin(self.mo_coeff, alpha)
        if beta is not None:
            self.mo_coeff = self.Ry_orb(mo_coeff, beta)
            self.mo_coeff = self.Ry_spin(self.mo_coeff, beta)
        if gamma is not None:
            self.mo_coeff = self.Rz_orb(mo_coeff, gamma)
            self.mo_coeff = self.Rz_spin(self.mo_coeff, gamma)
        return self.mo_coeff

def get_wigner_d(J, m, k, beta): 
    """
    Wigner small d-matrix.
    Note symmetry: by definition, d = d[::-1, ::-1].T
    """
    nmin = int(max(0, k-m))
    nmax = int(min(J+k, J-m))
    sum_arr = []

    for n in range(nmin, nmax+1):
        num = np.sqrt(
            scipy.special.factorial(J+k) * scipy.special.factorial(J-k) *
            scipy.special.factorial(J+m) * scipy.special.factorial(J-m))

        denom = (scipy.special.factorial(J+k-n) * scipy.special.factorial(J-n-m) * 
                 scipy.special.factorial(n-k+m) * scipy.special.factorial(n))
        
        sign = (-1.)**(n - k + m)
        cos = (np.cos(beta/2))**(2*J - 2*n + k - m)
        sin = (np.sin(beta/2))**(2*n - k + m)
        sum_arr.append(sign * num / denom * cos * sin)
    
    return sum(sum_arr)

if __name__ == "__main__":
    from pyscf import gto, scf, dft, lib
    from stevens.momentum import get_mj

    # User input
    N_grid = (20,10,20) 
    max_J = 8
    max_J_only = True # If True, calculate only J == max_J. Otherwise, calculate all J in [1, max_J]
    project_type = 'full' # "spin" (S), "orb" (L), or "full" (J)
    ao_shell = 'f'
    diag_only = False # If True, only calculate the diagonal element of C_{JM}* C_{JM'},
                      # i.e., C_{JM}* C_{JM}
    # load constrained DFT solution from running cdft.py
    chkfname = 'constrained_chkfile.chk'  

    mol = gto.Mole()
    mol.atom = '''
    Ho        0.000000    0.000000    0.000000'''
    mol.basis = 'sarc-dkh'
    mol.spin = 0
    mol.charge = 3 
    mol.verbose = 4
    mol.build()
    mol.set_common_origin(mol._atom[0][1])
    print("N_elec", mol.nelec)
    print("N_ao", mol.nao)

    gmf = scf.GKS(mol).x2c()
    gmf.xc = 'HF'
    if chkfname is not None and os.path.isfile(chkfname):
        data = chkfile.load(chkfname, 'scf')
        gmf.__dict__.update(data)
    else: 
        gmf.chkfile = chkfname 
        gmf.kernel()
    if mpirank == 0:
        gmf.analyze(with_meta_lowdin=True)
        dm = gmf.make_rdm1()
        print("S vector", get_mj(mol, dm, include_L=False))
        print("L vector", get_mj(mol, dm, include_S=False))
        print("J vector", get_mj(mol, dm))
    
    project = Project(mol, gmf, atom_index=0, ao_shell=ao_shell, spin_separate=False, \
            N_grid=N_grid, lo_chkfname='uhf_gs')

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
                    res = project.get_proj_ovlp(J=J, m=m, k=k, proj=project_type, N_alpha=N_grid[0], N_beta=N_grid[1], \
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



    
    
