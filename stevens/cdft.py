'''
constrained dft with given direction of angular momentum
Penalty method following VASP's I_CONSTRAINED_M=1 implementation
<vasp.at/wiki/index.php/I_CONSTRAINED_M>
'''
import pyscf
from pyscf import scf, dft, lib, lo, x2c
from pyscf.lib import chkfile
import os, sys
import h5py
import numpy as np
import scipy.linalg as la
from stevens import localize
from stevens import momentum 
np.set_printoptions(precision=12, threshold=sys.maxsize, linewidth=190, suppress=True)
MINAO = {'default': 'minao', 'Ho': 'sarcdkhminao', 'Er': 'sarcdkhminao', 'Dy': 'sarcdkhminao'}
from mpi4py import MPI
comm = MPI.COMM_WORLD
mpirank = comm.Get_rank()
mpisize = comm.Get_size()
log = lib.logger.Logger(sys.stdout, 4)

class CGKS(lib.StreamObject):
    def __init__(self, mol, direction, multiplier_guess=1, max_cycle=500, verbose=0, 
            lo_chkfname=None, atom_index=0, ao_shell='f', ao_index=None, 
            spin_separate=False, xc='HF', smearing=None, lo_from_umf=True, kpts=[[0,0,0]],
            gdf=None, gdf_fname='gdf_ints.h5', *args, **kwargs):
        self.mol = mol
        has_pbc = getattr(mol, 'dimension', 0) > 0
        if has_pbc:
            from pyscf.pbc import scf, dft, x2c, gto
            # TODO implement multiple k-point
            self.kpts = kpts
        else:
            from pyscf import scf, dft, x2c, gto
        if xc.upper() != 'HF':
            if has_pbc:
                gmf = dft.GKS(mol, kpts[0]).density_fit().x2c() ### TODO
                gmf.exxdiv = None
                gmf.with_df = gdf
                gmf.with_df._cderi = gdf_fname
                if xc.upper() != 'LDA':
                    raise NotImplementedError
                gmf.xc = xc
            else:
                gmf = dft.GKS(mol).x2c()
                gmf.xc = xc
            gmf.collinear = 'mcol'
            gmf.grids.level = 4
        else:
            if has_pbc:
                gmf = scf.GHF(mol, kpts[0]).density_fit().x2c() ### TODO
                gmf.exxdiv = None
                gmf.with_df = gdf
                gmf.with_df._cderi = gdf_fname
            else:
                ### TODO
                #gmf = scf.GHF(mol).x2c()
                gmf = dft.GKS(mol).x2c()
                gmf.xc = xc
                ###
        if smearing is not None:
            gmf = scf.addons.smearing_(gmf, sigma=smearing, method="fermi")
        gmf.chkfile = 'constrained_chkfile.chk'
        self._scf = gmf #x2c.x2c.x2c1e_ghf(gmf)
        self.veff_ks = self._scf.get_veff
        if has_pbc:
            self.get_veff = self.get_veff_cell
        else:
            self.get_veff = self.get_veff_mol
        if max_cycle is not None:
            self._scf.max_cycle = max_cycle


        # make direction a unit vector
        assert la.norm(direction) > 1e-6 # otherwise need to rescale input direction
        direction = np.array(direction) / la.norm(direction)
        self.direction = direction
        print("target direction:", self.direction)

        self.lagrange_multiplier = 1 if multiplier_guess is None else multiplier_guess
        if has_pbc:
            self.ao_ovlp = np.array(mol.pbc_intor('int1e_ovlp', hermi=0, kpts=kpts,
                               pbcopt=lib.c_null_ptr()))[0] ### TODO
            self.l_integral = (np.array(mol.pbc_intor('int1e_cg_irxp', kpts=kpts)) *(-1j))[0] ### TODO
        else:
            self.ao_ovlp = mol.intor_symmetric('int1e_ovlp') 
            self.l_integral = mol.intor('int1e_cg_irxp') *(-1j)
        self.dmj = self.get_dmj().transpose(0,2,1)

        # localize orbitals to define the lanthanide f-shells 
        if lo_chkfname is not None:
            # use a mf solution loaded from lo_chkfname to construct Intrinsic Atomic Orbital (IAO)
            c_lo = self.get_iao(lo_chkfname, from_umf=lo_from_umf, verbose=verbose) 
            c_lo = self.trunc_lo(c_lo, atom_index, ao_shell, ao_index, spin_separate)
            ao_ovlp = la.block_diag(self.ao_ovlp, self.ao_ovlp) 
            self.lo_proj = c_lo @ c_lo.conj().T @ ao_ovlp 
            self.dmj = np.einsum('ki,xij,jl->xkl', self.lo_proj.T.conj(), self.dmj, self.lo_proj, optimize=True)
        else:
            self.lo_proj = None 

    def get_iao(self, lo_chkfname, from_umf, verbose=False, diff_spin=False): 
        if lo_chkfname == 'c_lo.npy':
            c_lo = np.load(lo_chkfname)
            ao_ovlp = la.block_diag(self.ao_ovlp, self.ao_ovlp) 
            log.info("LO error from I %s" % np.max(np.abs(c_lo.T.conj() @ ao_ovlp @ c_lo - np.eye(c_lo.shape[-1]))))
            return c_lo 
        elif lo_chkfname[-2:] == 'h5':
            # directly input c_lo
            feri = h5py.File(lo_chkfname, 'r')
            c_lo = np.array(feri['c_lo'])
            feri.close()
            ao_ovlp = la.block_diag(self.ao_ovlp, self.ao_ovlp) 
            log.info("LO error from I %s" % np.max(np.abs(c_lo.T.conj() @ ao_ovlp @ c_lo - np.eye(c_lo.shape[-1]))))
            return c_lo
        mol = self.mol
        if from_umf:
            ao_ovlp = self.ao_ovlp
        else:
            ao_ovlp = la.block_diag(self.ao_ovlp, self.ao_ovlp) 
        c_lo = localize.get_iao(mol, lo_chkfname, from_umf, ao_ovlp=ao_ovlp, verbose=verbose, diff_spin=diff_spin)
        return c_lo 

    def trunc_lo(self, c_lo, atom_index, ao_shell, ao_index, spin_separate=False):
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

    def get_veff_mol(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1, **kwargs):
        if mol is None:
            mol = self.mol
        if dm is None:
            dm = self.make_rdm1()
        # veff is nonlinear wrt dm due to the new penalty term
        # Therefore, we compute the new vhf with the full dm instead of change of dm
        dm_last = 0
        vhf_last = 0
        veff = \
            self.veff_ks(mol=mol, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi, **kwargs)
        
        nao = mol.nao
        J = np.array([np.sum(self.dmj[x] * dm.T) for x in range(3)])
        assert np.max(np.abs(J.imag)) < 1e-10
        J = J.real
        log.info("J vector %s", J)
        dveff = self.lagrange_multiplier * np.einsum('x,xij->ij', J / la.norm(J) - self.direction, self.dmj)
        d_energy = self.lagrange_multiplier * (la.norm(J) - self.direction.dot(J))  
        log.info("energy penalty %s", d_energy)
        assert d_energy >= -1e-10 # otherwise self.direction may fail to be a unit vector
        if hasattr(veff, 'ecoul'):
            # KS
            veff = lib.tag_array(veff + dveff, ecoul=veff.ecoul, \
                    exc=veff.exc + d_energy , vj=veff.vj, vk=veff.vk)
        else:
            # HF
            veff += dveff
        return veff 

    def get_veff_cell(self, cell=None, dm=None, dm_last=0, vhf_last=0, hermi=1, kpt=None, kpts_band=None, **kwargs):
        if cell is None:
            cell = self.cell
        if dm is None:
            dm = self.make_rdm1()
        if kpt is None:
            kpt = self.kpts[0]
        veff = \
            self.veff_ks(cell=cell, dm=dm, dm_last=dm_last, vhf_last=vhf_last, hermi=hermi, kpt=kpt,
                         kpts_band=kpts_band, **kwargs)
        
        nao = cell.nao
        J = np.array([np.trace(self.dmj[x] @ dm) for x in range(3)])
        assert np.max(np.abs(J.imag)) < 1e-10
        J = J.real
        log.info("J vector %s", J)
        dveff = self.lagrange_multiplier * np.einsum('x,xij->ij', J / la.norm(J) - self.direction, self.dmj)
        d_energy = self.lagrange_multiplier * (la.norm(J) - self.direction.dot(J))  
        log.info("energy penalty %s", d_energy)
        assert d_energy >= -1e-10 # otherwise self.direction may fail to be a unit vector
        if hasattr(veff, 'ecoul'):
            # KS
            veff = lib.tag_array(veff + dveff, ecoul=veff.ecoul, \
                    exc=veff.exc + d_energy , vj=veff.vj, vk=veff.vk)
        else:
            # HF
            veff += dveff
        return veff 

        
    def get_mj(self, dm=None, ao_ovlp=None, l_integral=None):
        if dm is None:
            dm = self._scf.make_rdm1()
        if ao_ovlp is None:
            ao_ovlp = self.ao_ovlp
        if l_integral is None:
            l_integral = self.l_integral
        return momentum.get_mj(self.mol, dm, ao_ovlp=ao_ovlp, l_integral=l_integral)

    def get_dmj(self, ao_ovlp=None, l_integral=None):
        nao = self.mol.nao
        if ao_ovlp is None:
            ao_ovlp = self.ao_ovlp
        if l_integral is None:
            l_integral = self.l_integral
        return momentum.get_dmj(ao_ovlp=ao_ovlp, l_integral=l_integral, mol=self.mol)

    def get_dmj_numerical(self, dm, dx=1e-4):
        dm = np.array(dm)
        nao = self.mol.nao

        dmj = np.zeros((3,2*nao,2*nao), dtype=complex)
        for i in range(dm.shape[1]):
            for j in range(dm.shape[2]):
                dm_p = dm.copy()
                dm_p[i,j] += dx / 2
                mj_p = self.get_mj(dm_p)
                dm_m = dm.copy()
                dm_m[i,j] -= dx / 2
                mj_m = self.get_mj(dm_m)
                dmj[:,i,j] += (mj_p - mj_m) / dx
        return (dmj + dmj.transpose(0,2,1).conj()) / 2

    def kernel(self, **kwargs):

        def root_finding_function(lagrange_multiplier_input):
            self.lagrange_multiplier = lagrange_multiplier_input
            self._scf.get_veff = self.get_veff
            self._scf.kernel(**kwargs)
            dm = self._scf.make_rdm1()
            mj = self.get_mj(dm)
            self.lagrange_multiplier =  0
            energy = self._scf.energy_tot(dm=dm)
            log.info("**J_vector, dft energy** %s  %s", mj, energy)
            return mj, energy
        
        result = root_finding_function(self.lagrange_multiplier)
        return result

def get_dm_guess(mol, dm, shell='4f'):
    # revise a dm such that the 4f occupation satisfies the hund's rule for Ho3+
    ao_lst = []
    ao_labels = mol.ao_labels()
    for i in range(len(ao_labels)):
        if ao_labels[i].split()[2][:len(shell)] == shell:
            ao_lst.append(i)
    if shell[-1] == 'f':
        l = 3
    elif shell[-1] == 'd':
        l = 2
    elif shell[-1] == 'p':
        l = 1
    rot = pyscf.symm.sph.sph_pure2real(l, reorder_p=True) 
    l_integral = mol.intor('int1e_cg_irxp') *(-1j)
    dm = dm.reshape(2, nao, 2, nao).transpose(0,2,1,3)
    dmaa = dm[0,0]
    dmbb = dm[1,1]
    print('original dmaa in pure AO of shell',shell, rot @ dmaa[np.ix_(ao_lst, ao_lst)] @ rot.conj().T)
    print('original dmbb in pure AO of shell',shell, rot @ dmbb[np.ix_(ao_lst, ao_lst)] @ rot.conj().T)
    tmp = np.zeros((len(ao_lst), len(ao_lst)))
    tmp[[4,5,6],[4,5,6]] = 1
    dmbb[np.ix_(ao_lst, ao_lst)] = rot.conj().T @ tmp @ rot
    print("lz from guessed 4f bb pureAO", np.diag(rot @ l_integral[2][ao_lst][:, ao_lst] @ rot.conj().T @ tmp))
    dm[1,1] = dmbb
    tmp = np.eye(len(ao_lst))
    dmaa[np.ix_(ao_lst, ao_lst)] = rot.conj().T @ tmp @ rot
    dm[0,0] = dmaa
    print("lz from guessed 4f aa pure AO", np.diag(rot @ l_integral[2][ao_lst][:, ao_lst] @ rot.conj().T @ tmp))
    dm = dm.transpose(0,2,1,3).reshape(2*nao, 2*nao)
    return dm

def sample_direction(plane, idx, unit=20):
    if plane == 'xz':
        direction = [np.sin(np.pi/unit * idx), 0, np.cos(np.pi/unit * idx)]
    elif plane == 'yz':
        direction = [0, np.sin(np.pi/unit * idx), np.cos(np.pi/unit * idx)]
    elif plane == 'xy': 
        direction = [np.sin(np.pi/unit * idx), np.cos(np.pi/unit * idx), 0]
    elif plane == 'random':
        theta = np.random.rand(1)[0]*2-1
        theta = np.arcsin(theta) + np.pi/2 
        phi = np.random.rand(1)[0]*np.pi*2
        direction = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        log.info("Haar random target direction %s" % direction)
    return direction

def sample_rotation(plane, idx, unit=20, dir_init=(0,0,1), return_direction=False):
    # First we need to orient the initial direction to the +z direction
    dir_init = np.array(dir_init) / la.norm(dir_init)
    dir_init = dir_init.real
    angle_init = [np.arctan2(dir_init[1], dir_init[0]), -np.arccos(dir_init[2]), 0]

    # then calculate the [0, thete, phi] rotation angles needed to rotate +z to the desired direction
    if plane == 'xz':
        rotation = [0, np.pi / unit * idx, 0]
    elif plane == 'yz':
        rotation = [0, np.pi / unit * idx, np.pi / 2]
    elif plane == 'xy': 
        rotation = [0, np.pi / 2, np.pi / unit * idx]
    elif plane == 'xyz':
        idx = idx % unit 
        unit = int(unit//2)
        if idx < unit:
            rotation = [0, np.pi / unit * idx, 0]
        else:
            rotation = [0, np.pi / 2, np.pi / unit * (idx - unit + 1)]
    elif plane == 'random':
        theta = np.random.rand(1)[0]*2-1
        theta = np.arcsin(theta) + np.pi/2 
        phi = np.random.rand(1)[0]*np.pi*2
        rotation = [0, theta, phi]
    _, theta, phi = rotation 
    direction = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    log.info("Haar random target direction %s" % direction)

    # then combine two rotations
    rotation = np.array([angle_init[0], angle_init[1], 0]) + np.array(rotation)
    log.info("Haar random total rotation %s" % rotation)
    if return_direction:
        return rotation, direction
    else:
        return rotation


if __name__ == "__main__":
    from pyscf import gto, scf, dft, lib, x2c
    from stevens import utils
    
    # molecule setup
    ao_shell = 'f'
    mol = gto.Mole()
    mol.atom = '''
    Ho        0.000000    0.000000    0.000000'''
    mol.basis = 'sarc-dkh'
    mol.spin = 4
    mol.charge = 3 
    mol.verbose = 4
    mol.build()
    log.info("%s", mol.nelec)
    log.info("nao %s", mol.nao)

    # set up constrained HF
    multiplier = 0.01
    required_direction = np.random.rand(3) - 0.5
    required_direction = required_direction / la.norm(required_direction)
    mf = CGKS(mol, required_direction, multiplier_guess=multiplier, max_cycle=100, verbose=4,
            lo_chkfname='uhf_gs', atom_index=0, ao_shell=ao_shell)
    # generate a density matrix initial guess whose J vector is relatively close to required_direction
    dm0 = mf._scf.get_init_guess()
    dm0 = utils.change_dm_guess(mol, dm0, 'positive_z', atom_index=0, ao_shell=ao_shell) 
    dm0 = utils.change_dm_guess(mol, dm0, 'rotate', atom_index=0, ao_shell=ao_shell, direction=required_direction)
    log.info("init J vector %s", momentum.get_mj(mol, dm0))
    # run constrained HF 
    sol = mf.kernel(dm0=dm0)
    mj_opt = sol[0]
    # use "max" below to avoid division by 0 when mj_opt = [0,0,0]
    assert la.norm(mj_opt/max(0.0001, la.norm(mj_opt)) - required_direction) < 1e-3

