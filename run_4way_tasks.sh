#!/usr/bin/env bash
set -euo pipefail

# W&B project (override: WANDB_PROJECT=foo ./run_4way_tasks.sh)
WANDB_PROJECT="${WANDB_PROJECT:-ss2r}"

# Run 4-way matrix for each task:
#   1) dr, stationary
#   2) dr, nonstationary
#   3) dr+spidr, stationary
#   4) dr+spidr, nonstationary
#
# Tasks:
#   - quadruped_run
#   - humanoid_walk
#   - cartpole_swingup
#   - go_to_goal

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

wandb_group_for_case() {
  local experiment="$1"
  local nonstationary="$2"
  local ns_label
  if [[ "${nonstationary}" == "true" ]]; then
    ns_label="nonstationary"
  else
    ns_label="stationary"
  fi
  echo "${experiment}__${ns_label}"
}

run_case() {
  local exp="$1"
  local nonstationary="$2"
  local wb_group
  wb_group="$(wandb_group_for_case "${exp}" "${nonstationary}")"
  echo
  echo "============================================================"
  echo "[START] wandb.project=${WANDB_PROJECT} wandb.group=${wb_group}"
  echo "        experiment=${exp} training.nonstationary=${nonstationary}"
  echo "        time=$(date -Iseconds)"
  echo "============================================================"

  python train_brax.py "+experiment=${exp}" "training.nonstationary=${nonstationary}" \
    "wandb.project=${WANDB_PROJECT}" "wandb.group=${wb_group}"

  echo "[DONE ] experiment=${exp} training.nonstationary=${nonstationary} time=$(date -Iseconds)"
}

# ---------------- quadruped_run ----------------
run_case "quadruped_run_dr" "false"
run_case "quadruped_run_dr" "true"
run_case "quadruped_run_dr_spidr" "false"
run_case "quadruped_run_dr_spidr" "true"

# ---------------- humanoid_walk ----------------
run_case "humanoid_walk_dr" "false"
run_case "humanoid_walk_dr" "true"
run_case "humanoid_walk_dr_spidr" "false"
run_case "humanoid_walk_dr_spidr" "true"

# --------------- cartpole_swingup --------------
run_case "cartpole_swingup_dr" "false"
run_case "cartpole_swingup_dr" "true"
run_case "cartpole_swingup_dr_spidr" "false"
run_case "cartpole_swingup_dr_spidr" "true"

# ------------------ go_to_goal -----------------
run_case "go_to_goal_dr" "false"
run_case "go_to_goal_dr" "true"
run_case "go_to_goal_dr_spidr" "false"
run_case "go_to_goal_dr_spidr" "true"

echo
echo "[ALL DONE] $(date -Iseconds)"
