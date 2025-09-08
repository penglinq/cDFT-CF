for id in `seq 0 299`; do echo "runxyz"$id"_direct";  tail -n 3 "runxyz"$id"_direct"/slurm_gen*.out  | grep "cycle= "; done > tmp_energy.out
cat tmp_energy.out
echo "min"
grep 'cycle=' tmp_energy.out | awk -F ' ' '{print $4}' | awk 'BEGIN{a=1000}{if ($1<0+a) a=$1} END{print a}'
cat tmp_energy.out | grep -B 1 -- "-15924" | grep "runxyz" | tr "\n" " " ; echo
cat tmp_energy.out | grep -B 1 -- "-15925.[^7]" | grep "runxyz" | tr "\n" " " ; echo
cat tmp_energy.out | grep -B 1 -- "-15925.7[^7]" | grep "runxyz" | tr "\n" " " ; echo



