#!/usr/bin/env bash
# Sets up a virtualenv for OmniVoice hosting/inference.
#
# Usage:
#   ./setup.sh
#
# Override the venv location with OMNIVOICE_VENV, e.g.:
#   OMNIVOICE_VENV=.venv2 ./setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

VENV_DIR="${OMNIVOICE_VENV:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn requests pydantic

echo
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
