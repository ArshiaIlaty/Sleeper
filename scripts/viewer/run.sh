#!/usr/bin/env bash
# Launch the PhysioNet 2026 biosignal viewer on pdmle.
#
# Run this as an account that can READ the dataset (a member of the `mlusers`
# group, e.g. arshia_ilaty_physio26) — NOT as ubuntu. The server itself reads
# the EDFs; there is no sudo per request.
#
#   bash run.sh              # serves on 127.0.0.1:8050
#   PORT=8060 bash run.sh    # custom port
#
# Then from your laptop, tunnel and open a browser:
#   ssh -L 8050:127.0.0.1:8050 <you>@AWOR-PDMLEAPP01
#   open http://127.0.0.1:8050
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8050}"
HOST="${HOST:-127.0.0.1}"

# Sanity: can this account read the dataset?
DATA="${PHYSIONET_DATA_ROOT:-/data-temp/shared-physionet26-dataset/extracted}"
if ! head -c1 "$DATA/demographics.csv" >/dev/null 2>&1; then
  echo "ERROR: cannot read $DATA/demographics.csv"
  echo "Run this as a user in the 'mlusers' group (e.g. arshia_ilaty_physio26), not ubuntu." >&2
  exit 1
fi

echo "Starting viewer on http://$HOST:$PORT  (data: $DATA)"
exec python3 "$HERE/app.py" --host "$HOST" --port "$PORT"
