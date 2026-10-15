#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
SETUP_PROFILE="${SETUP_PROFILE:-experiments}"
case "$SETUP_PROFILE" in
  experiments) extras='experiments,test' ;;
  smoke) extras='test' ;;
  *) echo 'SETUP_PROFILE must be experiments or smoke' >&2; exit 2 ;;
esac
"$PYTHON_BIN" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), "Use Python 3.11 or 3.12"'
if [[ ! -e "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
elif [[ ! -f "$VENV_DIR/pyvenv.cfg" ]]; then
  echo "$VENV_DIR exists and is not a virtual environment" >&2; exit 2
fi
venv_python="$VENV_DIR/bin/python"
"$venv_python" -m pip install --upgrade pip
if [[ "$(uname -s)" == Linux ]]; then
  # CPU audio wheels must come from the same index as CPU torch/vision.
  "$venv_python" -m pip install --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" \
    torch==2.14.1 torchvision==0.29.1 torchaudio==2.11.0
fi
"$venv_python" -m pip install -e ".[${extras}]"
"$venv_python" -m pip check
"$venv_python" -m mpsl.smoke --help >/dev/null
printf 'Environment ready. Activate with: source "%s/bin/activate"\n' "$VENV_DIR"
