#!/bin/bash

# ===============================================
# GENERAL OPTIONS ===============================
# ===============================================

# experiment params
PROFILES="model-learning/model_learning/res/sar/profiles.json"
GEN_TEAM_CONFIG="model-learning/model_learning/res/sar/cond_known.json"             # config to generate GT trajectories
INF_TEAM_CONFIG="model-learning/model_learning/res/sar/cond_unknown_gt_op_rnd.json" # config for model inference
ENV_SIZE=3
NUM_VICTIMS=3
VICS_CLEARED_FEAT=false
PRUNE_THRESH=0.01

ROOT_DIR="output/sar"
TMP1="${GEN_TEAM_CONFIG##*/}"
TMP2="${INF_TEAM_CONFIG##*/}"
ROOT_DIR="${ROOT_DIR}/gen_${TMP1%.*}_inf_${TMP2%.*}_s${ENV_SIZE}_v${NUM_VICTIMS}"

# common params
PROCESSES=20 #-1 # num processes (usually = available cpus)
VERBOSITY="info"
IMG_FORMAT="pdf"
CLEAR=true # whether to clear output/results directory
SEED=17

# ===============================================
# 1. GEN TRAJECTORIES OPTIONS ===================
# ===============================================
DISCOUNT=0.7
HORIZON=2
AG_SELECTION="random" #"softmax" # agents' action selection criterion, to untie equal-valued actions
AG_RATIONALITY=10     #20     # agents' rationality when selecting actions under a probabilistic criterion
NUM_TRAJ=16 #100 #16
TRAJ_LEN=25

ROOT_DIR="${ROOT_DIR}_t${NUM_TRAJ}_l${TRAJ_LEN}"
TRAJ_DIR="${ROOT_DIR}/1-trajectories"
TRAJ_FILE="${TRAJ_DIR}/trajectories.pkl.gz" # hardcoded, do not edit!

# ===============================================
# 2. FC ESTIMATION DIFF OPTIONS =================
# ===============================================
FC_EST_DIFF_DIR="${ROOT_DIR}/2-fc-est-diff"
NUM_EFC_TRAJS=100 #200 #100 # 16 32 # num Monte-Carlo trajectories to estimate feature counts

# ===============================================
# 3. MODEL INFERENCE OPTIONS ====================
# ===============================================
MODEL_INF_DIR="${ROOT_DIR}/3-model-inference"
INF_TRAJ_FILE="${MODEL_INF_DIR}/trajectories.pkl.gz" # hardcoded, do not edit!

# ===============================================
# 4. MIRL-TOM OPTIONS ===========================
# ===============================================
MIRL_DIR="${ROOT_DIR}/4-mirl-tom"
declare -a AGENTS=("Medic" "Explorer")
LEARNING_RATE=0.05 #0.01 #TEAM_LEARNING_RATE = [5e-2, 2e-1] # 0.05
DECREASE_RATE=true
NORM_THETA=true
MAX_EPOCHS=50
DIFF_THRESHOLD=0.005 # 0.005
EXACT=false
NUM_MC_TRAJECTORIES=100 #16        #128 #32 #16 # 10
RESULTS_FILE="results.pkl.gz" # hardcoded, do not edit!

# ===============================================
# 5. EVALUATION OPTIONS =========================
# ===============================================
EVAL_DIR="${ROOT_DIR}/5-eval"
