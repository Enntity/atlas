#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

# Fast GLM-5.3 development loop: sync once, build once on Spark 1, then send
# only the release executable directly to Spark 2 over the RoCE interface.
# Use a full image transfer when runtime libraries or FlashKDA itself change.

set -euo pipefail

SPARK1_HOST=${SPARK1_HOST:-ennspark01}
SPARK1_TREE=${SPARK1_TREE:-/tmp/sparkglm-atlas-glm53-20260831}
SPARK2_ROCE=${SPARK2_ROCE:-10.100.16.1}
PEER_KEY=${PEER_KEY:-/home/enntitysparkadmin/.ssh/id_ed25519_atlas_peer}
IMAGE=${IMAGE:-sparkglm-atlas:glm53-exl3-20260831}
DEV_BINARY=${DEV_BINARY:-/tmp/sparkglm53-dev/spark}
SSH_CONTROL_PATH=${SSH_CONTROL_PATH:-/tmp/atlas-glm53-ssh-%C}

ssh_opts=(
  -o ControlMaster=auto
  -o ControlPersist=600
  -o "ControlPath=$SSH_CONTROL_PATH"
)

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

rsync -az -e "ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=$SSH_CONTROL_PATH" \
  --exclude='/.git/' \
  --exclude='/target/' \
  --exclude='/build/' \
  --exclude='/.cache/' \
  --exclude='/.venv/' \
  --exclude='/vendor/cudarc/target/' \
  "$repo_root/" "$SPARK1_HOST:$SPARK1_TREE/"

ssh "${ssh_opts[@]}" "$SPARK1_HOST" \
  "docker build --build-arg ATLAS_TARGET_HW=gb10 --build-arg ATLAS_TARGET_MODEL=glm-5.3-flash --build-arg ATLAS_TARGET_QUANT=exl3 -f '$SPARK1_TREE/docker/gb10/Dockerfile' -t '$IMAGE' '$SPARK1_TREE'"

ssh "${ssh_opts[@]}" "$SPARK1_HOST" "
  set -e
  mkdir -p '$(dirname "$DEV_BINARY")'
  docker rm -f sparkglm53-binary-extract >/dev/null 2>&1 || true
  docker create --name sparkglm53-binary-extract '$IMAGE' --help >/dev/null
  docker cp sparkglm53-binary-extract:/usr/local/bin/spark '$DEV_BINARY.new'
  docker rm sparkglm53-binary-extract >/dev/null
  chmod 0755 '$DEV_BINARY.new'
  mv '$DEV_BINARY.new' '$DEV_BINARY'
  ssh -i '$PEER_KEY' -o BatchMode=yes -o StrictHostKeyChecking=accept-new \
    -o ControlMaster=auto -o ControlPersist=600 \
    -o ControlPath=/tmp/atlas-glm53-peer-%C '$SPARK2_ROCE' \
    'mkdir -p $(dirname "$DEV_BINARY")'
  local_sha=\$(sha256sum '$DEV_BINARY')
  local_sha=\${local_sha%% *}
  peer_sha=\$(ssh -i '$PEER_KEY' -o BatchMode=yes -o ControlMaster=auto \
    -o ControlPersist=600 -o ControlPath=/tmp/atlas-glm53-peer-%C '$SPARK2_ROCE' \
    'sha256sum "$DEV_BINARY" 2>/dev/null || true')
  peer_sha=\${peer_sha%% *}
  if [[ \$local_sha != \$peer_sha ]]; then
    scp -q -i '$PEER_KEY' -o BatchMode=yes -o ControlMaster=auto \
      -o ControlPersist=600 -o ControlPath=/tmp/atlas-glm53-peer-%C \
      '$DEV_BINARY' '$SPARK2_ROCE:$DEV_BINARY.new'
    ssh -i '$PEER_KEY' -o BatchMode=yes -o ControlMaster=auto \
      -o ControlPersist=600 -o ControlPath=/tmp/atlas-glm53-peer-%C '$SPARK2_ROCE' \
      'chmod 0755 "$DEV_BINARY.new" && mv "$DEV_BINARY.new" "$DEV_BINARY"'
    peer_sha=\$(ssh -i '$PEER_KEY' -o BatchMode=yes -o ControlMaster=auto \
      -o ControlPersist=600 -o ControlPath=/tmp/atlas-glm53-peer-%C '$SPARK2_ROCE' \
      'sha256sum "$DEV_BINARY"')
    peer_sha=\${peer_sha%% *}
  fi
  test \"\$local_sha\" = \"\$peer_sha\"
  echo \"deployed spark binary sha256=\$local_sha\"
"
