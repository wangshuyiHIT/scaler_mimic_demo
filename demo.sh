#!/usr/bin/env bash
# One command: open the looping 43100 playback window.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$root/scripts/sim2sim.sh" --loop "$@"
