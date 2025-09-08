#cp energy_save100.out energy.out
#cp energy_pbe0_save100.out energy_pbe0.out
grep "dft energy\*\* " run*/slurm_generate_state*.out | sed 's/\/slurm.*\]//' > energy.out
#grep "energy without.* constraint " run*/slurm_generate_state*.out | sed 's/energy with.*constraint//' | sed 's/\/slurm.*out://' > energy.out
grep "energy .*functional PBE0 " run_*/slurm_generate_state*.out | sed 's/energy .*functional PBE0//' | sed 's/\/slurm.*out://' > energy_PBE0.out
grep "energy .*functional B3LYP " run_*/slurm_energy.out | sed 's/energy .*functional B3LYP//' | sed 's/\/slurm.*out://'> energy_B3LYP.out
grep "energy .*functional PBE " run_*/slurm_energy.out | sed 's/energy .*functional PBE//' | sed 's/\/slurm.*out://'> energy_PBE.out
grep "energy .*functional TPSS " run_*/slurm_energy.out | sed 's/energy .*functional TPSS//' | sed 's/\/slurm.*out://' > energy_TPSS.out
grep "energy .*functional M06 " run_*/slurm_energy.out | sed 's/energy .*functional M06//' | sed 's/\/slurm.*out://' > energy_M06.out
grep "energy .*functional B97-D " run_*/slurm_energy.out | sed 's/energy .*functional B97-D//' | sed 's/\/slurm.*out://'> energy_B97D.out
grep "energy .*functional R2SCAN " run_*/slurm_energy.out | sed 's/energy .*functional R2SCAN//' | sed 's/\/slurm.*out://'> energy_R2SCAN.out
#grep ' 7.5   ' run*/slurm_18918_J7.5.out | sed 's/\/slurm.*out://' > res.out
#grep 'J vector ' run?*/slurm_generate_state.out | sed 's/\/slurm.*out:J vector \[/ /' | sed 's/\]//' > J_vector.out
#cat energy.out | grep '15925\.77' | sed 's/runxyz\(.*\)_direct.*/\1/' | tr '\n' ',' >> JM_array.job 
#python collect_data.py
#python fitting.py > fitting.out
#cp energy_pbe0.out dft_en


