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
log = lib.logger.Logger(sys.stdout, 4)

if __name__ == '__main__':
    from stevens.cf_parameter import crystal_field_H, solve_Bkq, print_Bkq
    import h5py
    
    # User input
    data_source = 'solve' #'solve' # 'fit' 
    symmetry = 'Oh' # do not use "C2v" symmmetry here because the molecule and the crystal only have approximately C2v symemtry
    j = 7.5 
    nidx_lst = ['all'] #[30, 50, 70, 90, 110]  #'all' # an integer or 'all'. Calculate Bkq with only a randomly chosen portion of data
    nrun = 1
    functional_lst = ['PBE0', 'PBE', 'TPSS', 'M06', 'B3LYP', 'B97D', 'R2SCAN' ] 
    print_JM = False

    for functional in functional_lst:
        if functional == 'noname':
            suffix = ''
        else:
            suffix = '_' + functional
        print("******%s*******"%functional)
        fn = 'cdft%s.h5'%suffix 
        feri = h5py.File(fn, 'r')
        proj_ori = np.asarray(feri['proj'])
        energy_ori = np.asarray(feri['energy'])
        feri.close()
        idx_lst_ori = np.argsort(energy_ori) # change here if only using a fixed portion of data 
        for i in range(len(proj_ori)):
            proj_ori[i][np.diag_indices(int(2*j)+1)] = proj_ori[i][np.diag_indices(int(2*j)+1)].real
 
        idx_lst_lst = [] 
        de_max_lst = []
        for nidx in nidx_lst:
            # prepare data
            if nidx == 'all':
                nidx = len(idx_lst_ori)
                nrun == 1
            if np.abs((proj_ori.shape[-1] - 1) / 2 - j) > 1e-10:
                j = (proj_ori.shape[-1] - 1) / 2 
                log.warn("Wrong J input! J is corrected to %s" % j)
            log.info("J = %s, Symmetry = %s, using %s data points." % (j, symmetry, nidx))
            E_lst = [] # spin-free energy
            H_lst = [] # crystal field Hamiltonian
            ew_lst = [] # eigenvalues, i.e. relative energies
            ew_sum_lst = [] # eigenvectors, i.e. wavefunctions
            x_lst = []

            # calculate B_k^q
            for i in range(nrun):
                log.debug("######################## nrun %s ########################" % i)
                idx_lst = np.array(idx_lst_ori)[np.random.choice(len(idx_lst_ori), size=nidx, replace=False)]
                idx_lst_lst.append(idx_lst)
                log.debug("%s selected idx %s" % (len(idx_lst), idx_lst))
                proj = proj_ori[idx_lst]
                log.debug2("c*c shape %s" % (proj.shape,))
                energy = energy_ori[idx_lst]
                log.debug2("energy shape %s" % (energy.shape,))
                if data_source == 'fit':
                    x = optimize_Bkq(energy, proj, j, symmetry)
                elif data_source == 'solve':
                    x = solve_Bkq(energy, proj, j, symmetry=symmetry)
                log.debug("Crystal field parameters %s" % x)
                x_lst.append(x[:-1])
                
                H = crystal_field_H(x[:-1], j, symmetry=symmetry)
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
                ew_lst.append(ew - min(ew))
                ew_sum_lst.append(np.sum(ew_lst))
            # discard fittings that deviate the most
            dev_lst = la.norm(ew_lst - np.average(ew_lst, axis=0), axis=1)
            sort_idx = np.argsort(dev_lst)
            max_idx = np.argmax(dev_lst)
            log.info("relative energy (deviate most) in cm^-1")
            for i in range(1, 2):
                print("Deviation:", dev_lst[sort_idx[-i]])
                for e in ew_lst[sort_idx[-i]]:
                    log.info("%s," % e)
            if nrun > 1 :
                ndiscard = int(nrun//10)
                print("Discarding the %s fitting that deviate the most"%ndiscard)
                print("Discarded deviation", dev_lst[sort_idx[-ndiscard:]])
                H_lst = np.array(H_lst)[sort_idx[:-ndiscard]]
                x_lst = np.array(x_lst)[sort_idx[:-ndiscard]]
                E_lst = np.array(E_lst)[sort_idx[:-ndiscard]]
                ew_lst = np.array(ew_lst)[sort_idx[:-ndiscard]]
            # after discarding
            log.info("relative energy in cm^-1")
            for e in np.average(ew_lst, axis=0):
                log.info("%s," % e)
            H_avg = np.average(H_lst, axis=0)
            norm_dev = la.norm(np.array(H_avg) - np.array(H_lst), axis=(1,2))
            idx = np.argmin(norm_dev)
            x_final = x_lst[idx]
            H_final = H_lst[idx]
            E_final = E_lst[idx]
            ew_final, ev_final = la.eigh(H)
            print("CF parameters")

            # print CF parameter for Easyspin and Phi 
            x = x_final
            print_Bkq(x, symmetry=symmetry, mode='phi')
            print_Bkq(x, symmetry=symmetry, mode='EasySpin')
 
            # save data
            fn = 'H%s.h5'%suffix.upper()
            feri = h5py.File(fn, 'w')
            feri['H'] = np.array(H_final)
            feri['E_spinfree'] = E_final
            feri.close()
            np.save("ew_lst_%ssample%s.npy"%(nidx, suffix.upper()), ew_final) 
            np.save("ew_lst_%ssample%s_all.npy"%(nidx, suffix.upper()), ew_lst) 

            H_lst = np.array(H_lst).reshape(len(H_lst), int(2*j+1)**2)
            ew_lst = np.array(ew_lst)
            log.debug2("H std %s" % np.std(H_lst, axis=0))
            log.info("ew std %s" % np.std(ew_lst, axis=0))
            log.note("H std avg over terms %s" % np.average(np.std(H_lst, axis=0)))
            log.note("ew std avg over terms %s" % np.average(np.std(ew_lst, axis=0)))
            log.note("H std max over terms %s" % np.average(np.max(H_lst, axis=0)))
            log.note("ew std max over terms %s" % np.average(np.max(ew_lst, axis=0)))
            log.note("ew sum max magnitude %s" % max(np.abs(ew_sum_lst)))
            # print eigenstate wavefunctions in |JM> basis
            H =  H_final # same as crystal_field_H(x_final, j, symmetry=symmetry) # good for low excited states energy
            #H = crystal_field_H(np.average(x_lst, axis=0), j, symmetry=symmetry) # less accurate for small ew 
            ew, ev = la.eigh(H)
            log.info("relative energy in cm^-1")
            for e in ew:
                log.info("%s," % (e - min(ew)))
            for i in range(len(ew)):
                log.info("%sth ev c_JM**2"%i)
                c2 = ev[:, i] * ev[:, i].conj()
                for m in range(int(2*j+1)):
                    if c2[m] > 0.1:
                        log.info("mj = %s, |c|^2 = %s" % (m-j, c2[m].real))
 



