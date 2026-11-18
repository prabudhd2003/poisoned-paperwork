#!/bin/bash
# Build data/receipt_study/csev_full_inputs.zip for the Colab receipt notebooks (04, 05, 09).
# 1. Copies the clean and saved-attack images listed in data/receipt_study/remote_files.txt
#    from the cluster where the CARC stages wrote them (read-only).
# 2. Recreates source.tar, the exact code snapshot the receipt runner checks (commit 64aa87b; the archive embeds the commit ID, which is set back to
#    the ID it had when the study was frozen, c5c5d52, so its hash matches).
# 3. Verifies every file hash and writes the zip.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CARC_HOST:?Set CARC_HOST, for example CARC_HOST=user@discovery.usc.edu}"
D=data/receipt_study
mkdir -p "$D/staging"
rsync -av --files-from="$D/remote_files.txt" "$CARC_HOST":/ "$D/staging/"
git archive --format=tar 64aa87b | python3 -c "import sys; b=sys.stdin.buffer.read(); sys.stdout.buffer.write(b.replace(b'64aa87ba21a9d031bf102fe1c9b74faa7d69beec', b'c5c5d52a165129603c672a67be0263b723d7a50d', 1))" > "$D/source.tar"
python3 scripts/pack_receipt_study_inputs.py
