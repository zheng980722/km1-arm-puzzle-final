#!/usr/bin/env bash
set -Eeo pipefail

readonly WORKSPACE="/home/wheeltec/WorkSpace/km1_arm_ws"

if [[ "$(pwd -P)" != "${WORKSPACE}" ]]; then
    echo "Run this script from ${WORKSPACE}." >&2
    exit 2
fi

source /opt/ros/humble/setup.bash
python3 -m pip install -r "${WORKSPACE}/puzzle_vision/requirements.txt"
colcon build --packages-select km1_arm --symlink-install

install -m 755 "${WORKSPACE}/deploy/111.sh" /home/wheeltec/111.sh
install -m 755 "${WORKSPACE}/deploy/222.sh" /home/wheeltec/222.sh
sudo install -m 644 \
    "${WORKSPACE}/deploy/km1-competition-buttons.service" \
    /etc/systemd/system/km1-competition-buttons.service
sudo systemctl daemon-reload
sudo systemctl enable --now km1-competition-buttons.service

echo "KM1 competition release installed."
echo "Task1: /home/wheeltec/111.sh"
echo "Task2: /home/wheeltec/222.sh"
