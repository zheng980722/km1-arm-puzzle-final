#!/usr/bin/env bash
set -Eeo pipefail

readonly WORKSPACE="/home/wheeltec/WorkSpace/km1_arm_ws"
readonly LOCK_FILE="/tmp/km1_competition_task.lock"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "[Task2] Task1 or Task2 is already running; this request was not started." >&2
    exit 75
fi

source /opt/ros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash"

exec ros2 run km1_arm competition_task_runner --task 2
