#!/bin/bash

# gets constants
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "$DIR/00_constants.sh"

# change to project root directory (in case invoked from other dir)
cd "$DIR/../../.." || exit
clear

RESULTS=""
for AGENT in "${AGENTS[@]}"; do
  RESULTS="${RESULTS} ${MIRL_DIR}/${AGENT}/${RESULTS_FILE}"
done

echo "========================================"
echo "Evaluating MIRL results in '${RESULTS}', saving results in ${EVAL_DIR}..."

python -m model_learning.bin.sar.evaluate \
  --traj-file=$INF_TRAJ_FILE \
  --results $RESULTS \
  --trajectories=$NUM_EFC_TRAJS \
  --profiles=$PROFILES \
  --team-config=$INF_TEAM_CONFIG \
  --output=$EVAL_DIR \
  --size=$ENV_SIZE \
  --victims=$NUM_VICTIMS \
  --vics-cleared-feature=$VICS_CLEARED_FEAT \
  --discount=$DISCOUNT \
  --horizon=$HORIZON \
  --selection=$AG_SELECTION \
  --rationality=$AG_RATIONALITY \
  --prune=$PRUNE_THRESH \
  --length=$TRAJ_LEN \
  --img-format=$IMG_FORMAT \
  --processes=$PROCESSES \
  --seed=$SEED \
  --verbosity=$VERBOSITY \
  --clear=$CLEAR

echo "========================================"
echo "Saving pip packages..."
pip freeze >"${EVAL_DIR}/packages.txt"
