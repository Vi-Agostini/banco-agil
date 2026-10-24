#!/bin/sh
set -eu

BANCO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "${BANCO_AGIL_PYTHON:-python3}" "$BANCO_ROOT/scripts/make_zip.py" "$@"
