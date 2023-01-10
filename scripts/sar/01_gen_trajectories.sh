#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

echo "========================================"
echo "Processing '${TEAM_CONFIG}', saving results in ${EXP_DIR}..."

python -m model_learning.bin.sar.gen_trajectories \
  --team-config=$TEAM_CONFIG \
  --output=$EXP_DIR \
  --size=$ENV_SIZE \
  --victims=$NUM_VICTIMS \
  --vics-cleared-feature=$VICS_CLEARED_FEAT \
  --discount=$DISCOUNT \
  --horizon=$HORIZON \
  --selection=$AG_SELECTION \
  --rationality=$AG_RATIONALITY \
  --prune=$PRUNE_THRESH \
  --trajectories=$NUM_TRAJ \
  --length=$TRAJ_LEN \
  --img-format=$IMG_FORMAT \
  --processes=$PROCESSES \
  --seed=$SEED \
  --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${EXP_DIR}/packages.txt"
