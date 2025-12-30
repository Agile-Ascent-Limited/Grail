#!/bin/bash
# restart-miners.sh - Clean restart script for GRAIL miners
# Equivalent to stop + start with full cleanup
#
# Usage: ./scripts/restart-miners.sh [ecosystem-config.js]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${1:-ecosystem.config.js}"

echo "=== Restarting GRAIL Miners ==="
echo ""

# Run stop script
"$SCRIPT_DIR/stop-miners.sh"

echo ""
echo "Waiting for GPU memory release..."
sleep 3

# Run start script
"$SCRIPT_DIR/start-miners.sh" "$CONFIG_FILE"
