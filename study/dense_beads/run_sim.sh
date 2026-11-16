#!/bin/bash
# simulation scenarios (GPU 0); output in ~/densebeads_out
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONPATH=$HOME/torch-spline-3d-dense
PY=$HOME/miniconda3/envs/simcode-py311/bin/python
S=$HOME/torch-spline-3d-dense/study/dense_beads/sim_compare.py
O=$HOME/densebeads_out
$PY $S $O/sim_mixed_tilt.json --layout mixed --tilt 0.1
$PY $S $O/sim_dense_tilt.json --layout dense --tilt 0.1
$PY $S $O/sim_dense_notilt.json --layout dense --tilt 0
$PY $S $O/sim_dense_dim.json --layout dense --tilt 0.1 --photons 5000
