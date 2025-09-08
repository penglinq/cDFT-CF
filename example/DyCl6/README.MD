## Procedures

1. Calculate atomic charges in the crystal using VASP, generate the molecular mechanics effective potential from the
   environmental charges, and store in mm_hcore.npy (sarc-basis for Ln and 6-31g for the rest) and mm_hcore_minao.npy (minimal basis). 
   See ./classical_environment. For detailed procedures, see ./classical_environment/procedure.txt
2. Copy the run file template ./conf to a run folder ./conf0, and copy the MM potential mm_hcore*.npy to ./conf0/
3. Copy the customized lanthanide minimal basis ./sarc-dkh2-minao.dat to your pyscf directory: 
   ```
   <pyscf_dir>/gto/basis/
   ```  
   and add this basis to the <pyscf_dir>/gto/basis/__init__.py by adding the line:   
   ```
   'sarcdkhminao' : 'sarc-dkh2-minao.dat',
   ```
4. First run a ground state GHF in the minimal basis in ./conf0/GS/ by  
   ```
   sbatch GEN_hpc.job
   ```
   Use the solution in the minimal basis to generate initial guess and run 20 ground state GHF starting from different J orientations 
   in ./conf0/GS/ by running run.sh. Pick the converged GHF solution with the lowest energy and copy its checkpoint file gs_ErCO34.chk 
   to ./conf0/. Copy its log file to ./conf0/GS/ for future reference
5. In the meantime, run UHF in ./conf0/UHF/ to generate localized atomic orbitals by running
   ./conf0/UHF/generate_state.py. The calculated localized orbitals will be stored in uhf_gs_ErCO34_lo.h5
6. After step 3 and step 4, run constrained HF (cHF) in ./conf0/cHF/ to sample low-energy determinants with different J
   orientaions. Specifically, assuming we want to sample 80 determinants:
    a) Generate the desired number of run folders from the template run folder by: 
       ```
       for id in `seq 0 79`; do  cp run "run_"$id -r; done
       ```
    b) Run constrained HF by:
       ```
       sbatch GEN_array_hpc.job
       ```
       This will run the "generate_state.py" file in each run folder, save the cHF solution as
       "constrained_chkfile.chk," and print the HF energy and PBE0 energy in the log file "slurm_generate_state.out"
    c) Analyze the JM components of each determinant by:
       ```
       sbatch JM_array_hpc.job
       ```
       This will run the "get_JM.py" file in each run folder and save c*_{JM}c_{JK} in "c2_Jmk.npy"
    d) (optional) If we want to analyze the DFT functional dependency of the fitted effective crystal field Hamiltonian, evaluate
       the energy with various functionals by:
       ```       
       sbatch EN_array_hpc.job
       ```
       This will run the "energy.py" file in each run folder and print the DFT energies in the log file
       "slurm_energy.out." This can be done simultaneously with step c)
    e) Collect results by:
       ```
       ./collect_data.sh
       python collect_data.py
       ```
7. Fit the effective crystal field Hamiltonian (H_CF). Specifically:
    a) Fit H_CF using the HF energies. Under the folder ./conf0/cHF/:
       ```
       python fitting.py > fitting.out
       ```
       The relative eigenstate energies from diagonalizing H_CF are printed under "the relative energy in cm^-1," and
       the main JM compositions |C_{JM}|^2 of each eigenstate are printed under "<x>th ev c_JM**2." The fitted H_CF is
       saved in "H_fit.h5"
    b) Fit H_CF using the PBE0 energies:
       ```
       cd dft_en
       python collect_data.py
       python fitting.py > fitting.out
       ```
       The eigenstate energies and JM compositions are similarly printed. The fitted H_CF is saved in "dft_en/H_fit.h5"
    c) (optional) Fit H_CF using energies from other DFT functionals:
       Under the same folder as step b), uncomment line 12 of collect_data.py. Then
       ```
       python collect_data.py
       python fitting_functional.py > fitting_functional.out
       ```
       The eigenstate energies and JM compositions are similarly printed. The fitted H_CF is saved in "dft_en/H_<functional>.h5"



