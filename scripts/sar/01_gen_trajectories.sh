#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

echo "========================================"
echo "Processing '${GEN_TEAM_CONFIG}', saving results in ${TRAJ_DIR}..."

python -m model_learning.bin.sar.gen_trajectories \
  --profiles=$PROFILES \
  --team-config=$GEN_TEAM_CONFIG \
  --output=$TRAJ_DIR \
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
  --verbosity=$VERBOSITY \
  --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${TRAJ_DIR}/packages.txt"
