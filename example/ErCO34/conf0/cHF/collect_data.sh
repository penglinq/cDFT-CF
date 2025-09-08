grep "dft energy\*\* " run*/slurm_generate_state*.out | sed 's/\/slurm.*\]//' > energy.out
grep "energy .*functional PBE0 " run_*/slurm_generate_state*.out | sed 's/energy .*functional PBE0//' | sed 's/\/slurm.*out://' > energy_PBE0.out
grep "energy .*functional B3LYP " run_*/slurm_energy.out | sed 's/energy .*functional B3LYP//' | sed 's/\/slurm.*out://'> energy_B3LYP.out
grep "energy .*functional PBE " run_*/slurm_energy.out | sed 's/energy .*functional PBE//' | sed 's/\/slurm.*out://'> energy_PBE.out
grep "energy .*functional TPSS " run_*/slurm_energy.out | sed 's/energy .*functional TPSS//' | sed 's/\/slurm.*out://' > energy_TPSS.out
grep "energy .*functional M06 " run_*/slurm_energy.out | sed 's/energy .*functional M06//' | sed 's/\/slurm.*out://' > energy_M06.out
grep "energy .*functional B97-D " run_*/slurm_energy.out | sed 's/energy .*functional B97-D//' | sed 's/\/slurm.*out://'> energy_B97D.out
grep "energy .*functional R2SCAN " run_*/slurm_energy.out | sed 's/energy .*functional R2SCAN//' | sed 's/\/slurm.*out://'> energy_R2SCAN.out


