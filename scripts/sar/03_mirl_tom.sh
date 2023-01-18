#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

echo "========================================"
echo "Performing MIRL-ToM using '${INF_TEAM_CONFIG}', saving results in ${MIRL_DIR}..."

python -m model_learning.bin.sar.mirl_tom \
  --team-config=$INF_TEAM_CONFIG \
  --traj-file=$INF_TRAJ_FILE \
  --agent="Medic" \
  --output=$MIRL_DIR \
  --size=$ENV_SIZE \
  --victims=$NUM_VICTIMS \
  --vics-cleared-feature=$VICS_CLEARED_FEAT \
  --learning-rate=0.05 \
  --decrease-rate=$DECREASE_RATE \
  --normalize=$NORM_THETA \
  --epochs=$MAX_EPOCHS \
  --threshold=$DIFF_THRESHOLD \
  --exact=$EXACT \
  --monte-carlo=$NUM_MC_TRAJECTORIES \
  --horizon=$HORIZON \
  --prune=$PRUNE_THRESH \
  --img-format=$IMG_FORMAT \
  --processes=$PROCESSES \
  --seed=$SEED \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${MIRL_DIR}/packages.txt"
