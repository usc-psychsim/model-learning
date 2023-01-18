#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

echo "========================================"
echo "Performing modl inference Using '${INF_TEAM_CONFIG}', saving results in ${MODEL_INF_DIR}..."

python -m model_learning.bin.sar.model_inference \
  --team-config=$INF_TEAM_CONFIG \
  --traj-file=$TRAJ_FILE \
  --output=$MODEL_INF_DIR \
  --size=$ENV_SIZE \
  --victims=$NUM_VICTIMS \
  --vics-cleared-feature=$VICS_CLEARED_FEAT \
  --prune=$PRUNE_THRESH \
  --img-format=$IMG_FORMAT \
  --processes=$PROCESSES \
  --seed=$SEED \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${MODEL_INF_DIR}/packages.txt"
