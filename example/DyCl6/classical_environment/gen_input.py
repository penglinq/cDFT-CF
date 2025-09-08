import numpy as np
import os, sys
import scipy.linalg as la
import pyscf
from pyscf.pbc import scf
from pyscf.lib import chkfile
from libdmet.utils import read_poscar

geometry = "Cs2NaDyCl6" # The name of the .vasp file, not including the .vasp suffix
spin = 5 # number of unpaired electrons in a unit cell
charge = 0 # charge of the molecule or a unit cell
basis = {'default': '631g', 'Dy': 'sarc-dkh', 'Cs': 'cc-pvdz-x2c'} # select basis 
from_poscar = True
max_memory = 4000
exp_to_discard = 0.02 # minimal gaussian orbital exponent to keep 
vasp_dir = './k111/'
no_x2c = True

a2bohr = 1.889726124565

if from_poscar and os.path.isfile('./' + geometry + ".vasp"):
    print("Reading geometry from %s.vasp"%geometry)
    cell = read_poscar(fname='./' + geometry + ".vasp")
    cell.basis = basis
    cell.spin = spin
    cell.charge = charge
    cell.max_memory = max_memory
    cell.verbose = 4
    cell.exp_to_discard = exp_to_discard
    cell.build()
    cell.set_common_origin(cell._atom[0][1])
elif os.path.isfile('./' + geometry + ".xyz"):
    print("Reading geometry from %s.xyz"%geometry)
    cell = gto.M(
        atom = "./%s.xyz"%geometry,
        basis = basis, 
        verbose = 4,
        spin = spin,
        a = lattice,
        charge = charge,
        max_memory = max_memory,
        precision = 1e-14,
        )
    cell.set_common_origin(cell._atom[atom_index][1])
else:
    raise ValueError("No .xyz file available!")
print("N_elec %s, N_ao %s"%(cell.nelec, cell.nao))

with open("%s_27cells.xyz"%geometry, "w") as f:
    f.write("%d\n"%(cell.natm*27))
    f.write("\n")
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                for elem in cell._atom:
                    f.write("%s     %.10f  %.10f  %.10f\n"%(elem[0], *(np.array(elem[1])/a2bohr + i * cell.a[0] + j * cell.a[1] + k *
                                                                     cell.a[2])))

# For this simple molecule, we can manually select the DyCl6 cluster that will be treated quantum mechanically
qm_index = np.array([0,4,5,6,7,8,9], dtype=int) 
qm_index += 13 * cell.natm
np.savetxt("qm_index.dat", [qm_index], fmt="%d")
# More generally, we can input an .xyz file of all atoms belonging to the quantum cluster and automatically identify
# the indices of those atoms in the unit cell.
# qm_index = 0
# with open("%s_1mole.xyz"%geometry, "r") as f:
#     n_qm = int(f.readline())
#     qm_index = np.zeros((n_qm), dtype=int)
#     f.readline()
#     for i in range(n_qm):
#         tmp = np.asarray(f.readline().split()[1:], dtype=float)
#         index = np.argmin(la.norm(coord_lst - tmp, axis=1))
#         if la.norm(coord_lst - tmp, axis=1)[index] < 1e-4:
#             qm_index[i] = index
#         else:
#             print("Warning!!! Did not find %s-th atom at"%i, tmp)
# np.savetxt("qm_index.dat", [qm_index], fmt="%d")


# atomic charge from Bader/Mulliken charge analysis
os.system("grep 'ZVAL' %s/POTCAR | sed 's/.*ZVAL   =  //' | sed 's/mass.*//' > %s/ZVAL.dat"%(vasp_dir, vasp_dir)) 
chg = np.zeros((cell.natm))
with open(vasp_dir + "/ACF.dat", "r") as f:
    for i in range(2):
        f.readline()
    for i in range(cell.natm):
        chg[i] = f.readline().split()[4] 
zval = np.loadtxt(vasp_dir + 'ZVAL.dat').flatten() 
with open(vasp_dir + "/POSCAR", "r") as f:
    for i in range(6):
        f.readline()
    natm_per_elem = f.readline().split()
atom_charge = np.repeat(zval, natm_per_elem)
chg = atom_charge - chg
with open("mulliken_charges.dat", "w") as f:
    for i in range(len(chg)):
        f.write("%d  %.10f\n"%(i, chg[i]))

# lattice vector
np.savetxt("lattice.dat", cell.a)


