#!/usr/bin/env bash
# Reproducible FL-SLAM dev environment bootstrapper.
#
# Goals:
# - Ensure we are running the currently checked-out source (not a stale install)
# - Print an auditable provenance stamp (git SHA + import path)
# - Provide an optional per-package clean rebuild (avoids symlink-install collisions)
#
# Usage:
#   source tools/dev_env.sh
#
# Environment overrides:
#   FL_WS_DIR=fl_ws                 ROS 2 workspace directory (relative to repo root)
#   FL_ROS_SETUP=/opt/ros/jazzy/setup.bash
#   FL_PKG=fl_slam_poc              Package to build (default: fl_slam_poc)
#   FL_CLEAN=1                      If set, remove build/install/log for FL_PKG before build
#
# Notes:
# - Must be sourced (not executed) so the sourced setup.bash affects your shell.

set -euo pipefail

_fl_msg() { echo "[dev_env] $*"; }
_fl_err() { echo "[dev_env] ERROR: $*" >&2; }

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  _fl_err "This script must be sourced: 'source tools/dev_env.sh'"
  return 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FL_WS_DIR="${FL_WS_DIR:-fl_ws}"
FL_ROS_SETUP="${FL_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
FL_PKG="${FL_PKG:-fl_slam_poc}"

if [[ ! -d "${REPO_ROOT}/${FL_WS_DIR}" ]]; then
  _fl_err "Workspace not found: ${REPO_ROOT}/${FL_WS_DIR}"
  return 2
fi

if [[ ! -f "${FL_ROS_SETUP}" ]]; then
  _fl_err "ROS setup not found: ${FL_ROS_SETUP}"
  return 2
fi

cd "${REPO_ROOT}"

_fl_msg "Repo: ${REPO_ROOT}"
if command -v git >/dev/null 2>&1; then
  SHA="$(git rev-parse --short HEAD 2>/dev/null || true)"
  [[ -n "${SHA}" ]] && _fl_msg "Git SHA: ${SHA}"
else
  _fl_msg "Git: (not found)"
fi

_fl_msg "Sourcing ROS: ${FL_ROS_SETUP}"
set +u
source "${FL_ROS_SETUP}"
set -u

cd "${REPO_ROOT}/${FL_WS_DIR}"

if [[ "${FL_CLEAN:-0}" == "1" ]]; then
  _fl_msg "Cleaning package artifacts for ${FL_PKG} (build/install/log)"
  rm -rf "build/${FL_PKG}" "install/${FL_PKG}" log
fi

_fl_msg "Building: ${FL_PKG}"
colcon build --packages-select "${FL_PKG}" --symlink-install

_fl_msg "Sourcing workspace install: ${REPO_ROOT}/${FL_WS_DIR}/install/setup.bash"
set +u
source "${REPO_ROOT}/${FL_WS_DIR}/install/setup.bash"
set -u

_fl_msg "Import provenance:"
python3 - <<'PY'
import importlib
import inspect

mod = importlib.import_module("fl_slam_poc")
path = inspect.getfile(mod)
print(f"  fl_slam_poc.__file__ = {path}")
PY

_fl_msg "Ready."
