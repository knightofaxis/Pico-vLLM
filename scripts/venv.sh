#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

create_venv() {
    if [ ! -d "${VENV_DIR}" ]; then
        "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    fi
}

activate_venv() {
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    cd "${PROJECT_ROOT}"
}

install_deps_if_requested() {
    if [ "${INSTALL_DEPS:-0}" = "1" ]; then
        python -m pip install --upgrade pip
        python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
    fi
}

create_venv
activate_venv
install_deps_if_requested

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$#" -gt 0 ]; then
        exec "$@"
    fi
    echo "Activated Pico-vLLM virtual environment: ${VIRTUAL_ENV}"
    echo "Run commands here, or use: source scripts/venv.sh"
    exec "${SHELL:-bash}"
fi
