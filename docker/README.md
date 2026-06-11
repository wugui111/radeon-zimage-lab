# Lazy Image Deployment

This folder contains a reference Docker image for the teacher's pre-class preparation.

The image installs Z-Image-Turbo dependencies and pre-downloads the ModelScope checkpoint into `/opt/models`.
Students then start from this image in Radeon Cloud and avoid waiting for the model download during class.

Build and push:

```bash
REGISTRY_IMAGE=registry.example.com/your-namespace/radeon-zimage-turbo:rocm7.2 \
  bash docker/build_and_push_zimage.sh
```

Radeon Cloud must allow the pushed image in its `Container Image` selector. If the image does not appear, ask the platform administrator to add it or create the template for you.

