#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/mbart50_carl.json}"

carl-train --config "$CONFIG" --mode title_mle
carl-train --config "$CONFIG" --mode context_mle
carl-train --config "$CONFIG" --mode context_dropout_mle
carl-train --config "$CONFIG" --mode carl_rl
