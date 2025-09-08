from pycce import sm
import numpy as np
import scipy
import scipy.linalg as la

def crystal_field_H_C4(parameters, j):
    spin_matrix = sm.SpinMatrix(s=j)
    b20, b4n4, b40, b44, b6n4, b60, b64 = parameters 
    H = b20 * spin_matrix.stev(2, 0) + b40 * spin_matrix.stev(4, 0) \
        + b4n4 * spin_matrix.stev(4, -4) + b44 * spin_matrix.stev(4, 4) \
        + b60 * spin_matrix.stev(6, 0) + b6n4 * spin_matrix.stev(6, -4) \
        + b64 * spin_matrix.stev(6, 4) 

    return H

def crystal_field_H_C2(parameters, j):
    spin_matrix = sm.SpinMatrix(s=j)
    b2n2, b20, b22, b4n4, b4n2, b40, b42, b44, b6n6, b6n4, b6n2, b60, b62, b64, b66= parameters 
    H = b20 * spin_matrix.stev(2, 0) + b2n2 * spin_matrix.stev(2, -2) \
        + b22 * spin_matrix.stev(2, 2) + b40 * spin_matrix.stev(4, 0) \
        + b44 * spin_matrix.stev(4, 4) + b4n4 * spin_matrix.stev(4, -4) \
        + b42 * spin_matrix.stev(4, 2) + b4n2 * spin_matrix.stev(4, -2) \
        + b60 * spin_matrix.stev(6, 0) + b6n6 * spin_matrix.stev(6, -6) \
        + b66 * spin_matrix.stev(6, 6) + b6n4 * spin_matrix.stev(6, -4) \
        + b64 * spin_matrix.stev(6, 4) + b6n2 * spin_matrix.stev(6, -2) \
        + b62 * spin_matrix.stev(6, 2) 
    return H

def test_stev_operator():
    spin_matrix = sm.SpinMatrix(s=2)
    spin_matrix_EasySpin = [[0, 0, 7.3485, 0, 0],
                            [0, 0, 0, -12.0000, 0],
                            [7.3485, 0, 0, 0, 7.3485],
                            [0, -12.0000, 0, 0, 0],
                            [0, 0, 7.3485, 0, 0]]
    print("max element-wise error", np.max(np.abs(spin_matrix_EasySpin - spin_matrix.stev(4, 2)))) 
    assert np.max(np.abs(spin_matrix_EasySpin - spin_matrix.stev(4, 2))) < 1e-4 # matches EasySpin result

def test_crystal_field_H():
    import cf_parameter  
    j = 7.5
    symmetry = 'C2'
    parameters = np.random.rand(15)
    H = cf_parameter.crystal_field_H(parameters, j, symmetry, order=6)
    H_ref = crystal_field_H_C2(parameters, j)
    assert np.max(np.abs(H - H_ref)) < 1e-10
    symmetry = 'C4'
    parameters = np.random.rand(7)
    H = cf_parameter.crystal_field_H(parameters, j, symmetry, order=6)
    H_ref = crystal_field_H_C4(parameters, j)
    assert np.max(np.abs(H - H_ref)) < 1e-10

def test_rot_proj():
    from cf_parameter import crystal_field_H, solve_Bkq, rot_proj
    import h5py
    
    # User input
    symmetry = 'Oh' 
    j = 7.5
    rot_axis = True
    rot_angle = [0, np.pi/6, np.pi/6]

    fn = 'cdft_rot_DyCl6_for_test.h5' # load 50 data points 
    feri = h5py.File(fn, 'r')
    proj = np.asarray(feri['proj'])
    energy = np.asarray(feri['energy'])
    feri.close()
    if rot_axis:
        proj = rot_proj(proj, rot_angle, j)

    # calculate B_k^q
    x = solve_Bkq(energy, proj, j, symmetry=symmetry)
    H = crystal_field_H(x[:-1], j, symmetry=symmetry)
    ew, ev = la.eigh(H)
    ew_ref = np.array([ 
    -1.480597100915998396e+02,
    -1.480597100915994986e+02,
    -1.229876941632931562e+02,
    -1.229876941632927583e+02,
    -1.214552312761638859e+02,
    -1.214552312761633175e+02,
    -3.436542292797949472e+01,
    -3.436542292797946629e+01,
    8.089578667247714350e+01,
    8.089578667247727140e+01,
    8.241923385960113535e+01,
    8.241923385960117798e+01,
    1.311713471942699414e+02,
    1.311713471942699698e+02,
    1.323816907326881847e+02,
    1.323816907326881847e+02])
    assert np.max(np.abs(ew - ew_ref)) < 1

test_rot_proj()
test_stev_operator()
test_crystal_field_H()
