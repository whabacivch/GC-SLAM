#!/usr/bin/env bash
# Common venv Python detection and setup for FL-SLAM scripts
#
# Usage in scripts:
#   source "$(dirname "$0")/common_venv.sh"
#   # Then use $PYTHON for all Python invocations
#
# This ensures all scripts use the project's venv Python consistently.

# Detect project root (assuming this file is in tools/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect venv path - try multiple locations
if [ -n "${VENV_PATH:-}" ] && [ -d "${VENV_PATH}" ]; then
    # Use explicitly set VENV_PATH if it exists
    :
elif [ -d "${PROJECT_ROOT}/.venv" ]; then
    # Use project-local venv if it exists
    VENV_PATH="${PROJECT_ROOT}/.venv"
elif [ -d "${HOME}/.venv" ]; then
    # Fallback to home directory venv
    VENV_PATH="${HOME}/.venv"
else
    # Last resort: default to project venv
    VENV_PATH="${PROJECT_ROOT}/.venv"
fi

# Set PYTHON to venv's Python interpreter
if [ -d "$VENV_PATH" ] && [ -x "$VENV_PATH/bin/python" ]; then
    PYTHON="$VENV_PATH/bin/python"
    export VIRTUAL_ENV="$VENV_PATH"
    export PATH="$VENV_PATH/bin:$PATH"
    hash -r 2>/dev/null || true
else
    echo "ERROR: Python venv not found or invalid: $VENV_PATH" >&2
    echo "  Searched locations:" >&2
    echo "    - \$VENV_PATH: ${VENV_PATH:-<not set>}" >&2
    echo "    - ${PROJECT_ROOT}/.venv" >&2
    echo "    - ${HOME}/.venv" >&2
    echo "  Create venv with: python3 -m venv \"${PROJECT_ROOT}/.venv\"" >&2
    exit 1
fi

# Export for use in scripts
export PYTHON
export VENV_PATH
export PROJECT_ROOT
