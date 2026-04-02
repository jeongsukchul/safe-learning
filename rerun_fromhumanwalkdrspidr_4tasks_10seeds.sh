#!/usr/bin/env bash
# 4 tasks × 4-way matrix × 10 seeds (sequential runs).
#
# This script defaults to a *rerun* schedule: skip quadruped and humanoid DR,
# start at humanoid_walk_dr_spidr (stationary → nonstationary), then cartpole + go_to_goal.
#
# wandb.group = <experiment>__<stationary|nonstationary>
#
# Usage:
#   ./rerun_fromhumanwalkdrspidr_4tasks_10seeds.sh
#   FULL_SWEEP=1 ./rerun_fromhumanwalkdrspidr_4tasks_10seeds.sh   # same as run_4way_tasks_10seeds.sh (all tasks from start)
#   WANDB_PROJECT=my_proj ./rerun_fromhumanwalkdrspidr_4tasks_10seeds.sh
#   TASK=cartpole_swingup ./rerun_fromhumanwalkdrspidr_4tasks_10seeds.sh   # one task only (full 4-way for that task)
#
set -euo pipefail

# Set to 1 to run quadruped → humanoid (full) → cartpole → go_to_goal from the beginning.
FULL_SWEEP="${FULL_SWEEP:-0}"

# W&B project name (override on the command line: WANDB_PROJECT=foo ./run_4way_tasks_10seeds.sh)
WANDB_PROJECT="${WANDB_PROJECT:-ss2r}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

SEEDS="0,1,2,3,4,5,6,7,8,9"

# Same config => same group (all seeds); differs by experiment and nonstationary flag.
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

run_case_multiseed() {
  local experiment="$1"
  local nonstationary="$2"
  local wb_group
  wb_group="$(wandb_group_for_case "${experiment}" "${nonstationary}")"
  echo
  echo "============================================================"
  echo "[RUNx10] wandb.project=${WANDB_PROJECT} wandb.group=${wb_group}"
  echo "           +experiment=${experiment} training.nonstationary=${nonstationary}"
  echo "           seeds=${SEEDS}"
  echo "           time=$(date -Iseconds)"
  echo "============================================================"

  local seed
  for seed in 0 1 2 3 4 5 6 7 8 9; do
    echo "[SEED START] ${seed}  $(date -Iseconds)"
    python train_brax.py \
      "+experiment=${experiment}" \
      "training.nonstationary=${nonstationary}" \
      "training.seed=${seed}" \
      "wandb.project=${WANDB_PROJECT}" \
      "wandb.group=${wb_group}" \
      "wandb.resume=never"
    echo "[SEED DONE ] ${seed}  $(date -Iseconds)"
  done

  echo "[DONE] wandb.group=${wb_group} +experiment=${experiment} training.nonstationary=${nonstationary} time=$(date -Iseconds)"
}

run_task() {
  local exp_dr="$1"
  local exp_spidr="$2"

  run_case_multiseed "${exp_dr}" "false"
  run_case_multiseed "${exp_dr}" "true"
  run_case_multiseed "${exp_spidr}" "false"
  run_case_multiseed "${exp_spidr}" "true"
}

TASK="${TASK:-}"

if [[ -n "${TASK}" ]]; then
  case "${TASK}" in
    quadruped_run)    run_task quadruped_run_dr quadruped_run_dr_spidr ;;
    humanoid_walk)    run_task humanoid_walk_dr humanoid_walk_dr_spidr ;;
    cartpole_swingup) run_task cartpole_swingup_dr cartpole_swingup_dr_spidr ;;
    go_to_goal)       run_task go_to_goal_dr go_to_goal_dr_spidr ;;
    *)
      echo "Unknown TASK=${TASK}. Use: quadruped_run | humanoid_walk | cartpole_swingup | go_to_goal"
      exit 1
      ;;
  esac
else
  if [[ "${FULL_SWEEP}" == "1" ]]; then
    run_task quadruped_run_dr quadruped_run_dr_spidr
    run_task humanoid_walk_dr humanoid_walk_dr_spidr
    run_task cartpole_swingup_dr cartpole_swingup_dr_spidr
    run_task go_to_goal_dr go_to_goal_dr_spidr
  else
    # humanoid_walk_dr_spidr stationary, then nonstationary; then remaining tasks (full 4-way each).
    # run_case_multiseed "humanoid_walk_dr_spidr" "false"
    # run_case_multiseed "humanoid_walk_dr_spidr" "true"
    # run_task cartpole_swingup_dr cartpole_swingup_dr_spidr
    run_task humanoid_walk_dr humanoid_walk_dr_spidr
    run_task go_to_goal_dr go_to_goal_dr_spidr
  fi
fi

echo
echo "[ALL DONE] $(date -Iseconds)"
