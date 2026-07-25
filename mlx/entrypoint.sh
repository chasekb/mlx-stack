#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
ERROR: container-native MLX inference is deprecated and unsupported.

mlx.core/mlx-lm must run in the Apple Silicon macOS host Python environment,
not inside a Linux Podman container. Use:

  uv venv
  source .venv/bin/activate
  uv pip install -e .[mlx-host]
  ai-dev pull-models
  ai-dev up

EOF
exit 2
