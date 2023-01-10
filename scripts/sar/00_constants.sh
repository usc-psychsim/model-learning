#!/bin/bash

# ===============================================
# GENERAL OPTIONS ===============================
# ===============================================

ROOT_DIR="output/sar"

PROCESSES=-1 # num processes (usually = available cpus)
VERBOSITY=1
IMG_FORMAT="pdf"
CLEAR=true # whether to clear output/results directory
SEED=17

# ===============================================
# 1. GEN TRAJECTORIES OPTIONS ===================
# ===============================================

TEAM_CONFIG="model-learning/model_learning/res/sar/cond_known.json"
ENV_SIZE=3
NUM_VICTIMS=3
VICS_CLEARED_FEAT=false
DISCOUNT=0.7
HORIZON=2
AG_SELECTION="softmax" # agents' action selection criterion, to untie equal-valued actions
AG_RATIONALITY=20      # agents' rationality when selecting actions under a probabilistic criterion
PRUNE_THRESH=0.01
NUM_TRAJ=16
TRAJ_LEN=25

EXP_DIR="${TEAM_CONFIG##*/}"
EXP_DIR="${ROOT_DIR}/${EXP_DIR%.*}_s${ENV_SIZE}_v${NUM_VICTIMS}_h${HORIZON}_d${DISCOUNT}_r${AG_RATIONALITY}_t${NUM_TRAJ}_l${TRAJ_LEN}"
