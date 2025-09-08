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

def get_O_lst(j, symmetry=None, order=6):
    '''
        Get a list of Steven's Operator Equivalents up to an order that satisfy the 
        crystal field symmetry. The operators are arranged in ascending order based on
        k and then q.

    Args:
        symmetry: point group symmetry of the crystal field
        order: max k of Steven's Operator Equivalents
    '''
    if symmetry is not None:
        symmetry = symmetry.upper()
    spin_matrix = sm.SpinMatrix(s=j)
    O_lst = []
    for k in range(2, order+1, 2):
        if symmetry is None:
            q_lst = range(-k, k+1)
        elif symmetry == 'C2V':
            q = int(symmetry[1])
            q_lst = np.arange(0, k+1, q)
        elif symmetry[0] == 'C':
            q = int(symmetry[1])
            q_lst = np.arange(-k, k+1)
            q_lst = q_lst[np.where(q_lst%q == 0)[0]]
        elif symmetry == 'OH':
            if k in [4, 6]:
                q_lst = [0, 4]
            else:
                q_lst = []
        for q in q_lst:
             O_lst.append(spin_matrix.stev(k, q))
    O_lst = np.array(O_lst)
    log.debug("Number of crystal field parameters %s" % len(O_lst))
    return O_lst

def crystal_field_H(parameters, j, symmetry=None, order=6):
    '''
        Calculate the crystal field Hamiltonian from crystal field parameters B_k^q
    '''
    O_lst = get_O_lst(j, symmetry=symmetry, order=order)
    H = np.einsum('i,ipq->pq', parameters, O_lst)
    return H

def error_en(parameters, energy, proj, j, symmetry=None, order=6):
    '''
        Calculate the norm error in cm^-1 in single determinant energies given crystal field parameters B_k^q
    '''

    H = crystal_field_H(parameters[:-1], j, symmetry=symmetry, order=order)
    c = parameters[-1]

    energy_new = (energy - c) * ha2cm
    scale = 1/np.einsum('kii->k', proj).real
    proj = np.einsum('kij, k->kij', proj, scale)
    en = [np.sum(proj[i] * H) for i in range(len(proj))]
    return la.norm(en - energy_new)

def optimize_Bkq(energy, proj, j, symmetry=None, p0=None, maxiter=10000):
    if p0 is None and symmetry is None:
        c = np.average(energy) 
        p0 = [1] * 5 + [0.01] * 9 + [0.0001] * 13 + [c]
    elif p0 is None:
        raise NotImplementedError
    res = scipy.optimize.minimize(error_en, p0, method='l-bfgs-b', 
            args=(energy, proj, j, symmetry), options={'maxiter':10000, 'maxfun':10000})
    print(res)
    return res.x

def solve_Bkq(energy, proj, j, symmetry=None, order=6):
    '''
        \Sum_i B_i G_ji = E_j where G_ji[:,:-1] = \Sum_pq CC_jpq O_ipq
         and G_ji[:,-1] = np.ones
    '''
    energy = ha2cm * energy
    scale = 1/np.einsum('kii->k', proj).real
    proj = np.einsum('kij, k->kij', proj, scale)
    O_lst = get_O_lst(j, symmetry, order=order)
    G = np.einsum('jpq,ipq->ji', proj, O_lst, optimize=True)
    G = np.hstack((G, np.ones((len(G), 1))))
    if False: 
        # deprecate
        x, residues, rank, sglval = la.lstsq(G, energy)
        log.debug1("least square fitting of Bkq")
        log.debug1("x %s" % x)
        log.debug1("residues %s" % residues)
    else:
        # if CC is hermitian, because O are also hermitian, G has to be real
        q, r = la.qr(G, mode='economic')
        log.debug1("r real max %s" % np.max(np.abs(r.real)))
        log.debug1("r real norm %s" % la.norm(r.real))
        log.debug1("r imag norm %s" % la.norm(r.imag))
        log.debug1("r imag max %s", np.max(np.abs(r.imag)))
        r = r.real
        q = q.real
        x = la.solve(r, q.T @ energy)
    return x

def solve_Bkq_fromH(H, j, symmetry=None, order=6):
    '''
        Solve \sum_{kq} B^k_q O^k_q = H for B^k_q. 
        O are reshaped into a list of O matrices, O_lst.
        B are the unknowns, also shaped as a list of Bkq parameters, x.
        We assume H is a crystal field Hamiltonian, which means it is hermitian and trace-zero.
    '''
    O_lst = get_O_lst(j, symmetry, order=order)
    O_lst = O_lst.transpose(1,2,0).reshape((int(2*j+1))**2, -1)
    q, r = la.qr(O_lst, mode='economic')
    log.debug2("Q imag %s" % np.max(np.abs(q.imag))) # Q is complex!
    log.debug2("R imag %s" % np.max(np.abs(r.imag))) # R is real or complex
    x = la.solve(r, q.T.conj() @ H.flatten())
    return x

def D_matrix(J, alpha, beta, gamma):
    '''
        Generate the wigner D-matrix in ascending m order. 
    '''
    from stevens.project import get_wigner_d
    m_lst = np.arange(-J, J+1)
    D = np.diag(np.exp(-1j*m_lst*gamma))
    if np.abs(beta) > 1e-6:
        d_small = np.zeros((int(2*J+1), int(2*J+1)))
        for m in m_lst:
            for k in m_lst:
                d_small[int(m+J),int(k+J)] = get_wigner_d(J, m, k, beta)
        D = d_small @ D
    D = np.diag(np.exp(-1j*m_lst*alpha)) @ D
    return D

def rot_proj(proj, rot_angle, J):
    '''
        rotate the JM basis of C_{jm'} x C^*_{jm}
        rot_angle is the rotation angle of the molecule, not the coordinates
    '''
    alpha, beta, gamma = rot_angle # rotation from old to new
    alpha, beta, gamma = -alpha, -beta, -gamma 
    D = D_matrix(J, alpha, beta, gamma)
    proj = np.einsum('nij,pi,jq->npq', proj, D.T, D.conj(), optimize=True)
    return proj

def rot_proj_new(proj, rot_angle, J):
    '''
        rotate the JM basis of C_{jm'} x C^*_{jm}
        rot_angle is the rotation angle of the molecule, not the coordinates
        ** rot_angle here is the rotation from the current molecule to target molecule **
        ** rot_angle here should be -rot_angle[::-1] of the rot_angle in rot_proj
    '''
    alpha, beta, gamma = rot_angle # rotation from new to old, i.e. from current to target
    D = D_matrix(J, alpha, beta, gamma)
    D = D[::-1, ::-1] # obtain D in descending m order
    proj = np.einsum('nij,pi,jq->npq', proj, D, D.conj().T, optimize=True)
    return proj

def print_Bkq(x, symmetry=None, order=6, mode='phi', hf_x=None):
    '''
        Print the Bkq parameters in various formats. 
        Args:
            x: a list of non-zero crystal field parameters under a certain symmetry. 
            symmetry: Currently support: None (i.e. C1), 'C<n>' (replace <n> with an integer), 'C2v', and 'Oh'
            order: highest k 
            mode: Current three formats are supported:
            1) mode = 'easyspin': print sys.B2, sys.B4, and sys.B6 for EasySpin simulation with ascending q order
            2) mode = 'phi': print the crystal field parameters in the format of 'phi'
            3) mode = 'latex': compile Bkq in a latex table. When hf_x is None, the table contains three columns: 
               k, q, and Bkq. When hf_x is not None, the table contains four columns: k, q, Bkq from hf_x, and 
               Bkq from x (usually Bkq from DFT). 
            hf_x: Only for mode == 'latex'. Bkq from hf to be printed with the Bkq from x. 
    '''
    mode = mode.lower()
    # print CF parameter for Easyspin and Phi 
    if symmetry is not None:
        symmetry = symmetry.upper()
        if symmetry[0] == 'C':
            q_symm = int(symmetry[1])
            count = 0
            for k in [2,4,6]:
                for q in range(k, -k-1, -1):
                    if q%q_symm > 0 or (symmetry == 'C2V' and q < 0):
                        x = np.insert(x, count, [0])
                    count += 1
        elif symmetry == 'OH':
            x_final = np.zeros(int((5+2*order+1)/2)*int(order//2))
            x_final[9] = x[0] # (4,0)
            x_final[13] = x[1] # (4,4)
            x_final[20] = x[2] # (6,0)
            x_final[24] = x[3] # (6,4)
            x = x_final
    if hf_x is not None:
        if symmetry is not None:
            symmetry = symmetry.upper()
            if symmetry[0] == 'C':
                q_symm = int(symmetry[1])
                count = 0
                for k in [2,4,6]:
                    for q in range(k, -k-1, -1):
                        if q%q_symm > 0 or (symmetry == 'C2V' and q < 0):
                            hf_x = np.insert(hf_x, count, [0])
                        count += 1
            elif symmetry == 'OH':
                hf_x_final = np.zeros(int((5+2*order+1)/2)*int(order//2))
                hf_x_final[9] = hf_x[0] # (4,0)
                hf_x_final[13] = hf_x[1] # (4,4)
                hf_x_final[20] = hf_x[2] # (6,0)
                hf_x_final[24] = hf_x[3] # (6,4)
                hf_x = hf_x_final
    if mode == 'easyspin':
        print("CF parameters in EasySpin format")
        print("B2", *x[:5][::-1])
        print("B4", *x[5:14][::-1])
        print("B6", *x[14:27][::-1])
    elif mode == 'phi':
        print("CF parameters in Phi format")
        for k in [2,4,6]:
            for q in range(k, -k-1, -1):
                print(1, k, q, x[[0,5,14][k//2-1] + q + k])
    elif mode == 'latex':
        print("----CF parameters in a Latex table----")
        print("\\toprule")
        if hf_x is None:
            print("$k$ & $q$ & $B_k^q$ \\\\")
        else:
            print("\\multirow{2}{*}{$k$} & \\multirow{2}{*}{$q$} & \\multicolumn{2}{c}{$B_k^q$($\mathrm{cm}^{-1}$)}\\\\")
            print("& & HF@HF & PBE0@HF \\\\")
        print("\\midrule")
        for k in [2,4,6]:
            for q in range(k, -k-1, -1):
                if np.abs(x[[0,5,14][k//2-1] + q + k].real)<1e-14:
                    if hf_x is None:
                        print("%s & %s & 0 \\\\"%(k, q))
                    else:
                        print("%s & %s & 0 & 0\\\\"%(k, q))
                else:
                    if hf_x is None:
                        print("%s & %s & %.7E \\\\"%(k, q, x[[0,5,14][k//2-1] + q + k].real))
                    else:
                        print("%s & %s & %.7E & %.7E \\\\"%(k, q, hf_x[[0,5,14][k//2-1] + q + k].real, x[[0,5,14][k//2-1] + q + k].real))
        print("\\bottomrule")
    return
          
def print_eigen(H, threshold=0.01):
        print("----Analysis of H_CF eigenstates----")
        log.debug2("hermitian? %s" % (np.max(np.abs(H - H.conj().T))<1e-6))
        assert np.max(np.abs(H - H.conj().T))<1e-6
        ew, ev = la.eigh(H)
        # print eigenstate energies in cm^-1
        print("relative energy in cm^-1")
        for e in ew:
            print("%s," % (e - min(ew)))
        # print eigenstate wavefunctions in |JM> basis
        for i in range(len(ew)):
            print("%sth eigenstate c_JM**2"%i)
            c2 = ev[:, i] * ev[:, i].conj()
            for m in range(int(2*j+1)):
                if c2[m] > threshold:
                    print("mj = %s, |c|^2 = %s" % (m-j, c2[m].real))
        return
 

if __name__ == '__main__':
    import h5py
    
    # User input
    symmetry = 'Oh' # Currently support: None (i.e. C1), 'C<n>' (replace <n> with an integer), 'C2v',
                    # and 'Oh'
    j = 7.5 # J quantum number
    filename = 'cdft.h5' 
    feri = h5py.File(filename, 'r')
    proj = np.asarray(feri['proj']) # an array of size N x (2j+1) x (2j+1)
                                    # C_{jm}' x C^*_{jm} of N Slater determinants
    energy = np.asarray(feri['energy']) # a list of size N
                                        # absolute energies (in Ha) of N Slater determinants 
    feri.close()

    # calculate B_k^q
    x = solve_Bkq(energy, proj, j, symmetry=symmetry) # analytically solve for the least-square Bkq (x[:-1])
                                                      # and constant energy shift (x[-1])
    H = crystal_field_H(x[:-1], j, symmetry=symmetry) # calculate the crystal field Hamiltonian from Bkq parameters
    print_eigen(H) # print relative energy and JM composition of each eigenstate
    print_Bkq(x, symmetry=symmetry, order=6, mode='latex', hf_x=None) # print a latex script to output a table of Bkq
