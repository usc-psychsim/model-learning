#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

for AGENT in "${AGENTS[@]}"; do
  AGENT_DIR="${MIRL_DIR}/${AGENT}"
  echo "========================================"
  echo "Performing MIRL-ToM using '${INF_TEAM_CONFIG}' for agent ${AGENT}, saving results in ${AGENT_DIR}..."
  python -m model_learning.bin.sar.mirl_tom \
    --profiles=$PROFILES \
    --team-config=$INF_TEAM_CONFIG \
    --traj-file=$INF_TRAJ_FILE \
    --agent=$AGENT \
    --output=$AGENT_DIR \
    --size=$ENV_SIZE \
    --victims=$NUM_VICTIMS \
    --vics-cleared-feature=$VICS_CLEARED_FEAT \
    --learning-rate=$LEARNING_RATE \
    --decrease-rate=$DECREASE_RATE \
    --normalize=$NORM_THETA \
    --epochs=$MAX_EPOCHS \
    --threshold=$DIFF_THRESHOLD \
    --exact=$EXACT \
    --monte-carlo=$NUM_MC_TRAJECTORIES \
    --discount=$DISCOUNT \
    --horizon=$HORIZON \
    --selection=$AG_SELECTION \
    --rationality=$AG_RATIONALITY \
    --prune=$PRUNE_THRESH \
    --img-format=$IMG_FORMAT \
    --processes=$PROCESSES \
    --seed=$SEED \
    --verbosity=$VERBOSITY \
    --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${AGENT_DIR}/packages.txt"
