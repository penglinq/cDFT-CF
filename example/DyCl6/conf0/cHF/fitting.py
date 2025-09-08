'''
Calculate extended Stevens Operators
Code adapted from EasySpin written for Matlab
<github.com/StollLab/EasySpin/blob/main/easyspin/stev.m>

References
  I.D.Ryabov, J.Magn.Reson. 140, 141-145 (1999)
    https://doi.org/10.1006/jmre.1999.1783
  C. Rudowicz, C.Y.Chung, J.Phys.:Condens.Matter 16, 1-23 (2004)

Consistent with
  Altshuler/Kozyrev, Electron Paramagnetic Resonance
  in Compounds of Transition Elements, Wiley, 2nd edn., (1974)
  Appendix V. p. 512
  Table 16, p. 863 in Abragam/Bleaney
  Electron Paramagnetic Resonance of Transition Ions, Dover (1986)
See also
 St. Stoll, PhD thesis, ETH Zurich, 2003
 C.Rudowicz, Magn.Reson.Rev. 13, 1-87 (1987)
'''
import pyscf
from pyscf import scf, lib
import os, sys
import numpy as np
import scipy.linalg as la
import scipy
import pycce
from pycce import sm
import h5py
ha2cm = 219474.6 
log = lib.logger.Logger(sys.stdout, 6)

if __name__ == '__main__':
    from stevens.cf_parameter import crystal_field_H, solve_Bkq, print_Bkq
    import h5py
    
    # User input
    algorithm = 'solve' # Obtain the least-square crystal field paramters by either 
                        # 'solve' (solving analytically) or 
                        # 'fit.' (scipy minimization). Recommend 'solve'  
    symmetry = 'Oh' # Currently support: None (i.e. C1), 'C2v', 'C<n>' (replace <n> with an integer),
                    # and 'Oh'
    j = 7.5 # J quantum number
    nidx = 'all' # an integer or 'all'. Calculate Bkq with only a randomly chosen portion of data
    nrun = 1 # When using only a randomly chosen portion of data, repeat the fitting nrun times, 
             # each time with a different random sampling of data

    fn = 'cdft.h5' 
    feri = h5py.File(fn, 'r')
    proj_ori = np.asarray(feri['proj'])
    energy_ori = np.asarray(feri['energy'])
    feri.close()
    idx_lst_ori = np.argsort(energy_ori) # change here if only using a fixed portion of data 

    # prepare data
    if nidx == 'all':
        nidx = len(idx_lst_ori)
        assert nrun == 1
    log.info("J = %s, Symmetry = %s, using %s data points." % (j, symmetry, nidx))
    E_lst = [] # spin-free energy
    H_lst = [] # crystal field Hamiltonian
    x_lst = [] # crystal field parameters
    ew_lst = [] # eigenvalues, i.e. relative energies
    ew_sum_lst = [] # eigenvectors, i.e. wavefunctions

    # calculate B_k^q
    for k in range(nrun):
        log.debug("######################## nrun %s ########################" % k)
        idx_lst = np.array(idx_lst_ori)[np.random.choice(len(idx_lst_ori), size=nidx, replace=False)]
        log.debug("%s selected idx %s" % (len(idx_lst), idx_lst))
        proj = proj_ori[idx_lst]
        log.debug2("c*c shape %s" % (proj.shape,))
        energy = energy_ori[idx_lst]
        log.debug2("energy shape %s" % (energy.shape,))
        if algorithm == 'fit':
            x = optimize_Bkq(energy, proj, j, symmetry)
        elif algorithm == 'solve':
            x = solve_Bkq(energy, proj, j, symmetry=symmetry, order=6)
        log.debug("Crystal field parameters %s" % x)
        
        H = crystal_field_H(x[:-1], j, symmetry=symmetry, order=6)
        log.debug2("hermitian? %s" % (np.max(np.abs(H - H.conj().T))<1e-6))
        ew, ev = la.eigh(H)
        # print eigenstate energies in cm^-1
        log.debug("relative energy in cm^-1")
        for e in ew:
            log.debug("%s," % e)
        # print eigenstate wavefunctions in |JM> basis
        for i in range(len(ew)):
            log.debug1("%sth ev c_JM**2"%i)
            c2 = ev[:, i] * ev[:, i].conj()
            for m in range(int(2*j+1)):
                if c2[m] > 1e-2:
                    log.debug1("mj = %s, |c|^2 = %s" % (m-j, c2[m].real))
        # calculate standard deviation 
        E_lst.append(x[-1])
        H_lst.append(H)
        x_lst.append(x[:-1])
        #ew_lst.append(ew)
        ew_lst.append(ew - min(ew))
        ew_sum_lst.append(np.sum(ew_lst))

        ### save data
        fn = 'H_HF_%s.h5'%k 
        feri = h5py.File(fn, 'w')
        feri['H'] = np.average(H_lst, axis=0)
        feri['E_spinfree'] = np.average(E_lst, axis=0)
        feri.close()
        np.save("x_HF_%s.npy"%k, x)

    log.info("relative energy in cm^-1")
    for e in np.average(ew_lst, axis=0):
        log.info("%s," % e)
   
    # save data
    fn = 'H_fit.h5' 
    feri = h5py.File(fn, 'w')
    feri['H'] = np.average(H_lst, axis=0)
    feri['E_spinfree'] = np.average(E_lst, axis=0)
    feri.close()
    
    print_Bkq(np.average(x_lst, axis=0), symmetry=symmetry, mode='phi')

    H_lst = np.array(H_lst).reshape(len(H_lst), int(2*j+1)**2)
    ew_lst = np.array(ew_lst)
    log.debug2("H std %s" % np.std(H_lst, axis=0))
    log.debug2("ew std %s" % np.std(ew_lst, axis=0))
    log.note("H std avg over terms %s" % np.average(np.std(H_lst, axis=0)))
    log.note("ew std avg over terms %s" % np.average(np.std(ew_lst, axis=0)))
    log.note("ew sum max magnitude %s" % max(np.abs(ew_sum_lst)))


