#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip

TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if command -v nvidia-smi >/dev/null 2>&1; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi

pip install --extra-index-url "${TORCH_INDEX}" torch torchvision torchaudio
pip install transformers datasets accelerate evaluate sentencepiece
pip install lm_eval

echo "Environment setup complete. Activate with 'source ${VENV_DIR}/bin/activate'."
