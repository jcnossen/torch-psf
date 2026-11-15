#!/bin/bash
# real-data runs (GPU 0): the comparison + regions, then the model variants; output in ~/densebeads_out/real
D=/mnt/data/multicolor/june/astig/2col/beads100nm_10nm_steps
O=$HOME/densebeads_out/real
PY="$HOME/miniconda3/envs/simcode-py311/bin/python -I"
S=$HOME/torch-spline-3d-dense/study/dense_beads
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p $O
[ "$1" == "variants" ] || $PY $S/real_fit.py $D $O --plane-step 2 --max-box 84 --max-members 10 > $O/real_fit.log 2>&1
$PY $S/real_variants.py $D $O/variants.json --plane-step 2 > $O/variants.log 2>&1
