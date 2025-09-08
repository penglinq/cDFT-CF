#!/bin/bash
INITDIR=$PWD/run
PYFILE=$INITDIR/ground_state.py

for x in `seq 21 40`; do 
    dirname=$INITDIR"_"$x
    mkdir $dirname
    cp $INITDIR/* $dirname 
    cd $dirname
    sed "s/  occ_idx = .*/  occ_idx = $x/" $PYFILE > ground_state.py 
    sbatch -J "GS"$x GEN_hpc.job
    cd ..
done


    

