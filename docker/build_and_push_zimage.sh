#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   REGISTRY_IMAGE=registry.example.com/your-namespace/radeon-zimage-turbo:rocm7.2 \
#     bash docker/build_and_push_zimage.sh
#
# Replace REGISTRY_IMAGE with the image address accepted by Radeon Cloud.

REGISTRY_IMAGE="${REGISTRY_IMAGE:-}"

if [[ -z "$REGISTRY_IMAGE" ]]; then
  echo "Please set REGISTRY_IMAGE first."
  echo "Example:"
  echo "  REGISTRY_IMAGE=registry.example.com/your-namespace/radeon-zimage-turbo:rocm7.2 bash docker/build_and_push_zimage.sh"
  exit 1
fi

docker build -f docker/Dockerfile.zimage -t "$REGISTRY_IMAGE" .
docker push "$REGISTRY_IMAGE"

echo "Pushed image: $REGISTRY_IMAGE"
echo "Use this image in Radeon Cloud Create Template -> Container Image."

