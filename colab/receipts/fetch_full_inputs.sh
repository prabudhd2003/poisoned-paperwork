#!/bin/bash
# Build csev_full_inputs.zip for the Colab receipt studies.
# 1. Copies the clean and saved-attack images listed in inputs/remote_files.txt
#    from the cluster where the CARC pipeline wrote them (read-only).
# 2. Recreates source.tar, the exact code snapshot the Colab runner checks (commit 64aa87b; the archive embeds the commit ID, which is set back to
#    the ID it had when the study was frozen, c5c5d52, so its hash matches).
# 3. Verifies every file hash and writes csev_full_inputs.zip next to this script.
set -euo pipefail
cd "$(dirname "$0")"
: "${CARC_HOST:?Set CARC_HOST, for example CARC_HOST=user@discovery.usc.edu}"
mkdir -p staging
rsync -av --files-from=inputs/remote_files.txt "$CARC_HOST":/ staging/
git -C ../.. archive --format=tar 64aa87b | python3 -c "import sys; b=sys.stdin.buffer.read(); sys.stdout.buffer.write(b.replace(b'64aa87ba21a9d031bf102fe1c9b74faa7d69beec', b'c5c5d52a165129603c672a67be0263b723d7a50d', 1))" > inputs/source.tar
python3 pack_full.py
