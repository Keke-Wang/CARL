#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG CHECKPOINT [CHECKPOINT ...]" >&2
  exit 2
fi

CONFIG="$1"
shift
for checkpoint in "$@"; do
  carl-evaluate --config "$CONFIG" --checkpoint "$checkpoint" --split test
done

