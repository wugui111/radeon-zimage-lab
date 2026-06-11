#!/usr/bin/env bash
set -euo pipefail

FRPC_DIR="${FRPC_DIR:-/root/.cache/huggingface/gradio/frpc}"
FRPC_FILE="${FRPC_FILE:-frpc_linux_amd64_v0.3}"
FRPC_URL="${FRPC_URL:-https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64}"
FRPC_PATH="${FRPC_DIR}/${FRPC_FILE}"

mkdir -p "${FRPC_DIR}"

if [ -x "${FRPC_PATH}" ]; then
  echo "Gradio frpc already exists: ${FRPC_PATH}"
  ls -lh "${FRPC_PATH}"
  exit 0
fi

echo "Downloading Gradio frpc helper..."
echo "URL: ${FRPC_URL}"
echo "Target: ${FRPC_PATH}"

if command -v curl >/dev/null 2>&1; then
  curl -L --retry 3 --connect-timeout 30 -o "${FRPC_PATH}" "${FRPC_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${FRPC_PATH}" "${FRPC_URL}"
else
  echo "ERROR: neither curl nor wget is available."
  echo "Download this file manually and upload it to ${FRPC_PATH}:"
  echo "  ${FRPC_URL}"
  exit 1
fi

chmod +x "${FRPC_PATH}"
ls -lh "${FRPC_PATH}"

echo
echo "Gradio frpc is ready. Start the web app with:"
echo "  GRADIO_SHARE=1 python scripts/02_gradio_zimage_app.py"
