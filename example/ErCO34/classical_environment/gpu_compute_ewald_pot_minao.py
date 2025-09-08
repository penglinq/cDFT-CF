from pyscf import gto, lib, scf, dft
from pyscf.pbc import gto as pbcgto

import numpy as np
import cupy as cp

from cupyx.scipy import special

geometry = "Runde_789090"
basis = {'default': '631g', 'Er': 'sarc-dkh'} # select basis 
basis_final = {'default': 'minao', 'Er': 'cc-x2c-minao'} # select basis 
spin = 3
charge = -5
cell_spin = 0
cell_charge = 0

einsum = cp.einsum
cos = cp.cos
sin = cp.sin
norm = cp.linalg.norm
erf = special.erf
erfc = special.erfc

# Chemistry - A European Journal, (2009), 186-197, 15(1) (C, Cl, H, N, O)
# Shannon, “Revised Effective Ionic Radii and Systematic Studies of Interatomic Distances in Halides and Chalcogenides.” 
# (Ho3+ 8-coordinate, Er3+ 6-coordinate, Dy3+ 6-coordinate, Y3+)
ele2radius = {'Ho': 1.155, 'Er': 1.030, 'Dy': 1.052, 'Y': 1.159, 'C': 0.75, 'Cl': 0.99, 'H': 0.32, 'N': 0.71, 'O': 0.63, 'Cs': 2.02,
              'Na': 1.16} 
# assumed that Ho is VIII-coordinate, Dy is VI-coordinate, Cs is XII-coordinate, Na is VI-coordinate

atoms = open("%s_27cells.xyz"%geometry).readlines()[2:]
atoms = [[a.split()[0], np.array(a.split()[1:], dtype=float)] for a in atoms]

qm_indices = np.loadtxt("qm_index.dat", dtype=int)
mm_indices = [i for i in range(len(atoms)) if i not in qm_indices]

charges = np.loadtxt("mulliken_charges.dat", usecols=1)
charges = cp.asarray([charges] * 27).ravel()

qm_charges = charges[qm_indices]
mm_charges = charges[mm_indices]

qm_atoms = [atoms[i] for i in qm_indices]
mm_atoms = [atoms[i] for i in mm_indices]

coords = cp.asarray([a[1] for a in atoms]) / lib.param.BOHR
qm_coords = coords[qm_indices]
mm_coords = coords[mm_indices]

eta = lib.param.BOHR / cp.asarray([ele2radius[a[0]] for a in atoms])
qm_eta = eta[qm_indices]
mm_eta = eta[mm_indices]

mol = gto.Mole()
mol.spin = spin
mol.charge = charge
mol.basis = basis_final
mol.atom = qm_atoms
mol.build()
mol.tofile("qm.xyz")

cell = pbcgto.Cell()
cell.spin = cell_spin
cell.charge = cell_charge
cell.basis = basis
cell.atom = qm_atoms
cell.a = np.loadtxt("lattice.dat")
cell.a *= 3
cell.build()

mf = dft.RKS(mol)
mf.grids.build()

ew_eta, ew_cut = cell.get_ewald_params()
log_precision = np.log(cell.precision / (cp.abs(mm_charges).sum()*16*np.pi**2))
ke_cutoff = -2*ew_eta**2*log_precision.get()
mesh = cell.cutoff_to_mesh(ke_cutoff)
Lall = cp.asarray(cell.get_lattice_Ls(rcut=ew_cut))
Gv, Gvbase, Gweights = cell.get_Gv_weights(mesh)
Gv = cp.asarray(Gv)
absG2 = einsum('gx,gx->g', Gv, Gv)
absG2[absG2==0] = 1e200
coulG = 4 * np.pi / absG2
coulG *= Gweights
Gpref = cp.exp(-absG2/(4*ew_eta**2)) * coulG

def get_iGvR(Gv, coords, charges=None):
    GvR = einsum('gx,ix->ig', Gv, coords)
    cosGvR = cos(GvR)
    sinGvR = sin(GvR)
    
    if charges is not None:
        zcosGvR = einsum('i,ig->g', charges, cosGvR)
        zsinGvR = einsum('i,ig->g', charges, sinGvR)
    else:
        zcosGvR = zsinGvR = None
    return cosGvR, sinGvR, zcosGvR, zsinGvR

def rs_sum(coords1, coords2, charges2, f):
    rij = coords1[:,None,:] - coords2[None]
    rij = norm(rij, axis=-1)
    return einsum('ij,j->i', f(rij), charges2)
    
cosGvR, sinGvR, zcosGvR, zsinGvR = \
    get_iGvR(Gv, coords, charges)
v_ao = 0

prog = 0
for ao, mask, weight, grid_coords in mf._numint.block_loop(mol, mf.grids, blksize=1680):
    print(f"{prog}/{mf.grids.coords.shape[0]}")
    prog += grid_coords.shape[0]
    v = cp.zeros(len(grid_coords))
    grid_coords = cp.asarray(grid_coords)
    weight = cp.asarray(weight)

    # ewald real-space
    for T in Lall:
        # pot on grids from MM charges
        mm_coords_ = mm_coords + T
        v += rs_sum(grid_coords, mm_coords_, mm_charges,
                lambda r: (erf(mm_eta * r) - erf(ew_eta * r)) / r)
        # pot on grids from QM images
        if norm(T) < 1e-10:
            v -= rs_sum(grid_coords, qm_coords, qm_charges,
                    lambda r: erf(ew_eta * r) / r)
        else:
            qm_coords_ = qm_coords + T
            v += rs_sum(grid_coords, qm_coords_, qm_charges,
                    lambda r: (erf(qm_eta * r) - erf(ew_eta * r)) / r)

    # ewald g-space
    cosGvRg, sinGvRg, _, _ = get_iGvR(Gv, grid_coords)
    v += einsum('ig,g,g->i', cosGvRg, zcosGvR, Gpref)
    v += einsum('ig,g,g->i', sinGvRg, zsinGvR, Gpref)

    # represent v on AO basis
    v *= weight
    ao = cp.asarray(ao)
    aowv = einsum('g,gi->gi', v, ao)
    v_ao += einsum('gi,gj->ij', aowv, ao)

np.save("mm_hcore_minao.npy", -v_ao.get())

