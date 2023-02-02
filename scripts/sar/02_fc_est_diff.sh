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
echo "Comparing empirical vs. estimated feature counts, saving results in ${FC_EST_DIFF_DIR}..."

python -m model_learning.bin.sar.fc_est_diff \
  --traj-file=$TRAJ_FILE \
  --trajectories=$NUM_EFC_TRAJS \
  --profiles=$PROFILES \
  --team-config=$GEN_TEAM_CONFIG \
  --output=$FC_EST_DIFF_DIR \
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
pip freeze >"${FC_EST_DIFF_DIR}/packages.txt"
