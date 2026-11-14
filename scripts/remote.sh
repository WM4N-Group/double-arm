#!/usr/bin/env bash
# Run a command on the training server; never put passwords in this script.
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=${SERVER_CONFIG:-"$PROJECT_ROOT/config/server.local.env"}
[[ -f "$CONFIG" ]] || { echo "Copy config/server.example.env to config/server.local.env first." >&2; exit 2; }
# shellcheck source=/dev/null
source "$CONFIG"
: "${SSH_HOST:?}" "${SSH_PORT:?}" "${SSH_USER:?}" "${REMOTE_ROOT:?}"
SSH_ARGS=(-p "$SSH_PORT" -o ConnectTimeout=20 -o ServerAliveInterval=20 -o ServerAliveCountMax=3)
if [[ -n ${SSH_CONTROL_PATH:-} && -S $SSH_CONTROL_PATH ]]; then
    SSH_ARGS+=(-S "$SSH_CONTROL_PATH")
fi
if (($#)); then
    printf -v COMMAND '%q ' "$@"
    exec ssh "${SSH_ARGS[@]}" "$SSH_USER@$SSH_HOST" "$COMMAND"
else
    exec ssh -t "${SSH_ARGS[@]}" "$SSH_USER@$SSH_HOST"
fi
