import numpy as np
import scipy.linalg as la
import scipy
import h5py
import os

# User input
c2_min = 0.90
J_gap = 200 # expected minimal gap between ground state J and 1st excited J manifold, in cm-1
functional_lst = ['PBE0', 'PBE', 'TPSS', 'B3LYP', 'B97D', 'R2SCAN', 'M06' ]

name_lst = []
data = []
for dirname in os.listdir("../"):
    if  os.path.isfile("../" + dirname + '/c2_Jmk.npy'):
        c2 = np.load("../" + dirname + '/c2_Jmk.npy')
        if np.trace(c2) > c2_min:
            data.append(c2)
            name_lst.append(dirname)
        else:
            print("exclude", np.trace(c2))
data = np.asarray(data, dtype=complex)
print(data.shape)
name_lst = np.asarray(name_lst, dtype=str)
print(name_lst)
np.set_printoptions(precision=4, suppress=True)

np.set_printoptions(precision=8)


energy_dict_HF = {}
with open('../energy.out') as file: # HF energy
    for line in file:
        tmp = line.split()
        energy_dict_HF[tmp[0]] = tmp[1]
energy_lst_HF = []
converged_lst = []
for idx, name in enumerate(name_lst):
    if name in energy_dict_HF:
        energy_lst_HF.append(energy_dict_HF[name])
        converged_lst.append(idx)
data = data[converged_lst]
name_lst = name_lst[converged_lst]
energy_lst_HF = np.asarray(energy_lst_HF, dtype=float)
sort_energy_idx = np.argsort(energy_lst_HF)
print(sort_energy_idx)
print((np.sort(energy_lst_HF)-min(energy_lst_HF)) * 219474.6 )
print(energy_lst_HF.shape)
print(min(energy_lst_HF))
print(name_lst[sort_energy_idx])
inc_energy_lst = (energy_lst_HF[sort_energy_idx[1:]] - energy_lst_HF[sort_energy_idx[:-1]]) * 219474.6
cut_idx = np.where(inc_energy_lst > np.maximum(J_gap, energy_lst_HF[sort_energy_idx[:-1]] - min(energy_lst_HF)))[0]
if len(cut_idx) == 0:
    cut_idx = len(energy_lst_HF)
else:
    cut_idx = cut_idx[0]+1
print("# of sample read", energy_lst_HF.shape)
print("minimal energy", min(energy_lst_HF))
print(name_lst[sort_energy_idx])

print("# of sample kept", cut_idx)
energy_lst_HF = energy_lst_HF[sort_energy_idx[:cut_idx]]
name_lst = name_lst[sort_energy_idx[:cut_idx]]
data = data[sort_energy_idx[:cut_idx]]
print("HF energy")
for i in range(cut_idx):
    print(energy_lst_HF[i])
print("Use index")
for i in range(cut_idx):
    print(name_lst[i])

for functional in functional_lst:
    energy_dict = {}
    with open('../energy_%s.out'%functional) as file:
        for line in file:
            tmp = line.split()
            energy_dict[tmp[0]] = tmp[1]
    energy_lst = []
    for name in name_lst:
        if name in energy_dict:
            energy_lst.append(energy_dict[name])
        else:
            print("NOOOO", name)
    energy_lst = np.asarray(energy_lst, dtype=float)
    print("DFT en in HF order", (energy_lst - min(energy_lst)) * 219474.6 )
    print(energy_lst.shape)
    print(min(energy_lst))
    print("%s energy"%functional)
    for i in range(len(energy_lst)):
        print(energy_lst[i])
    
    fn = 'cdft_%s.h5'%functional
    feri = h5py.File(fn, 'w')
    feri['proj'] = data
    feri['energy'] = energy_lst
    feri.close()
    
    
