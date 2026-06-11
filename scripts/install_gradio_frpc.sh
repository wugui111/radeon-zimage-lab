#!/usr/bin/env bash
set -euo pipefail

FRPC_DIR="${FRPC_DIR:-/root/.cache/huggingface/gradio/frpc}"
FRPC_FILE="${FRPC_FILE:-frpc_linux_amd64_v0.3}"
FRPC_URL="${FRPC_URL:-https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64}"
FRPC_PATH="${FRPC_DIR}/${FRPC_FILE}"
LOCAL_FRPC="${LOCAL_FRPC:-}"

mkdir -p "${FRPC_DIR}"

if [ -x "${FRPC_PATH}" ]; then
  echo "Gradio frpc already exists: ${FRPC_PATH}"
  ls -lh "${FRPC_PATH}"
  exit 0
fi

copy_local_frpc() {
  local src="$1"
  if [ -n "${src}" ] && [ -f "${src}" ]; then
    echo "Using local frpc file: ${src}"
    cp "${src}" "${FRPC_PATH}"
    chmod +x "${FRPC_PATH}"
    ls -lh "${FRPC_PATH}"
    echo
    echo "Gradio frpc is ready. Start the web app with:"
    echo "  GRADIO_SHARE=1 python scripts/02_gradio_zimage_app.py"
    exit 0
  fi
}

copy_local_frpc "${LOCAL_FRPC}"
copy_local_frpc "./frpc_linux_amd64_v0.3"
copy_local_frpc "./frpc_linux_amd64"
copy_local_frpc "./scripts/frpc_linux_amd64_v0.3"
copy_local_frpc "./scripts/frpc_linux_amd64"

echo "Downloading Gradio frpc helper..."
echo "URL: ${FRPC_URL}"
echo "Target: ${FRPC_PATH}"

if command -v curl >/dev/null 2>&1; then
  if ! curl -L --retry 3 --connect-timeout 30 --max-time 300 -o "${FRPC_PATH}" "${FRPC_URL}"; then
    rm -f "${FRPC_PATH}"
    echo
    echo "ERROR: download timed out or failed."
    echo "Manual fallback:"
    echo "1. On your local computer, open this URL and download the file:"
    echo "   ${FRPC_URL}"
    echo "2. Upload it to the JupyterLab project directory: /workspace/radeon-zimage-lab"
    echo "3. Run this script again:"
    echo "   bash scripts/install_gradio_frpc.sh"
    exit 1
  fi
elif command -v wget >/dev/null 2>&1; then
  if ! wget -O "${FRPC_PATH}" "${FRPC_URL}"; then
    rm -f "${FRPC_PATH}"
    echo
    echo "ERROR: download timed out or failed."
    echo "Download this file manually and upload it to /workspace/radeon-zimage-lab:"
    echo "  ${FRPC_URL}"
    exit 1
  fi
else
  echo "ERROR: neither curl nor wget is available."
  echo "Download this file manually and upload it to /workspace/radeon-zimage-lab:"
  echo "  ${FRPC_URL}"
  exit 1
fi

chmod +x "${FRPC_PATH}"
ls -lh "${FRPC_PATH}"

echo
echo "Gradio frpc is ready. Start the web app with:"
echo "  GRADIO_SHARE=1 python scripts/02_gradio_zimage_app.py"
