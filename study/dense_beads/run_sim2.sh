#!/bin/bash
# packed layout (all neighbours within the patch, two isolated beads); runs after run_sim.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$HOME/torch-spline-3d-dense
PY=$HOME/miniconda3/envs/simcode-py311/bin/python
S=$HOME/torch-spline-3d-dense/study/dense_beads/sim_compare.py
O=$HOME/densebeads_out
while pgrep -f run_sim.sh > /dev/null; do sleep 30; done
$PY $S $O/sim_packed_tilt.json --layout packed --tilt 0.1
$PY $S $O/sim_packed_dim.json --layout packed --tilt 0.1 --photons 5000
