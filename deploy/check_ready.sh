#!/usr/bin/env bash
set -uo pipefail

readonly WORKSPACE="/home/wheeltec/WorkSpace/km1_arm_ws"
failures=0

check_path() {
    local label="$1"
    local path="$2"
    if [[ -e "${path}" ]]; then
        echo "[OK] ${label}: ${path}"
    else
        echo "[FAIL] ${label}: ${path}" >&2
        failures=$((failures + 1))
    fi
}

check_path "camera" /dev/video0
check_path "serial" /dev/ttyCH341USB0
check_path "workspace" "${WORKSPACE}"
check_path "Task1 script" /home/wheeltec/111.sh
check_path "Task2 script" /home/wheeltec/222.sh
check_path "ROS install" "${WORKSPACE}/install/setup.bash"

if systemctl is-active --quiet km1-competition-buttons.service; then
    echo "[OK] button service: active"
else
    echo "[FAIL] button service is not active" >&2
    failures=$((failures + 1))
fi

if (( failures > 0 )); then
    echo "Readiness check failed: ${failures} item(s)." >&2
    exit 1
fi

echo "KM1 competition system is ready. No motion command was sent."
