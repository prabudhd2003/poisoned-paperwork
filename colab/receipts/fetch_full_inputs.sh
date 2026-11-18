#!/bin/bash
# Build csev_full_inputs.zip for the Colab receipt studies.
# 1. Copies the clean and saved-attack images listed in inputs/remote_files.txt
#    from the cluster where the CARC pipeline wrote them (read-only).
# 2. Recreates source.tar, the exact code snapshot the Colab runner checks (commit c5c5d52).
# 3. Verifies every file hash and writes csev_full_inputs.zip next to this script.
set -euo pipefail
cd "$(dirname "$0")"
: "${CARC_HOST:?Set CARC_HOST, for example CARC_HOST=user@discovery.usc.edu}"
mkdir -p staging
rsync -av --files-from=inputs/remote_files.txt "$CARC_HOST":/ staging/
git -C ../.. archive --format=tar c5c5d52 > inputs/source.tar
python3 pack_full.py
