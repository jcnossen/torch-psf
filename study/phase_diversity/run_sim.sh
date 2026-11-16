#!/bin/bash
# simulation study (GPU ${GPU:-0}); output in ~/phasediv_out/sim
O=$HOME/phasediv_out/sim
PY="$HOME/miniconda3/envs/simcode-py311/bin/python -I"
S=$HOME/torch-spline-3d-pd/study/phase_diversity/sim_compare.py
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p $O
$PY $S $O/sim_dense.json --layout dense > $O/sim_dense.log 2>&1
$PY $S $O/sim_packed.json --layout packed > $O/sim_packed.log 2>&1
$PY $S $O/sim_dense_nominal.json --layout dense --na 1.49 --apod 0 0 0 --seeds 0 1 > $O/sim_dense_nominal.log 2>&1
# seed 2 of the dense layout again for the phase-diversity fits (the first run hit a NaN gradient, fixed since)
$PY $S $O/sim_dense_s2.json --layout dense --seeds 2 --methods pd pd_na_apod > $O/sim_dense_s2.log 2>&1
