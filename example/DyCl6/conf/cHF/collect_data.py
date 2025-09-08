import numpy as np
import scipy.linalg as la
import scipy
import h5py
import os

# User input
c2_min = 0.90
J_gap = 200 # expected minimal gap between ground state J and 1st excited J manifold, in cm-1

# read JM projection
name_lst = []
data = []
for dirname in os.listdir("./"):
    if os.path.isfile(dirname + '/c2_Jmk.npy'):
        c2 = np.load(dirname + '/c2_Jmk.npy')
        c2[np.diag_indices(len(c2))] = c2[np.diag_indices(len(c2))].real 
        if np.trace(c2) > c2_min:
            data.append(c2)
            name_lst.append(dirname)
        else:
            print("exclude ", dirname, np.trace(c2))
data = np.asarray(data, dtype=complex)
print("c*c data shape", data.shape)
name_lst = np.asarray(name_lst, dtype=str)
np.set_printoptions(precision=4, suppress=True)
for i in range(data.shape[1]):
    print("i = ", -(data.shape[1]-1)/2+i, "has ndata", sum(data[:,i,i].real>0.05))

# read HF energy
np.set_printoptions(precision=8)
energy_dict = {}
with open('energy.out') as file:
    for line in file:
        tmp = line.split()
        energy_dict[tmp[0]] = tmp[1]
energy_lst = []
converged_lst = []
for idx, name in enumerate(name_lst):
    if name in energy_dict:
        energy_lst.append(energy_dict[name])
        converged_lst.append(idx)
data = data[converged_lst]
name_lst = name_lst[converged_lst]
energy_lst = np.asarray(energy_lst, dtype=float)
sort_energy_idx = np.argsort(energy_lst)
print("data idx sorted by increasing energy", sort_energy_idx)
print((np.sort(energy_lst)-min(energy_lst)) * 219474.6 )
inc_energy_lst = (energy_lst[sort_energy_idx[1:]] - energy_lst[sort_energy_idx[:-1]]) * 219474.6
cut_idx = np.where(inc_energy_lst > np.maximum(J_gap, energy_lst[sort_energy_idx[:-1]] - min(energy_lst)))[0]
if len(cut_idx) == 0:
    cut_idx = len(energy_lst)
else:
    cut_idx = cut_idx[0]+1
print("# of sample read", energy_lst.shape)
print("minimal energy", min(energy_lst))
print(name_lst[sort_energy_idx])

print("# of sample kept", cut_idx)
energy_lst = energy_lst[sort_energy_idx[:cut_idx]]
name_lst = name_lst[sort_energy_idx[:cut_idx]]
data = data[sort_energy_idx[:cut_idx]]
print("HF energy")
for i in range(cut_idx):
    print(energy_lst[i])
print("Use index")
for i in range(cut_idx):
    print(name_lst[i])

fn = 'cdft.h5' 
feri = h5py.File(fn, 'w')
feri['proj'] = data
feri['energy'] = energy_lst
feri.close()

