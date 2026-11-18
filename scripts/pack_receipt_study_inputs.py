"""Verify the staged images and pack csev_full_inputs.zip for the Colab receipt studies."""
from pathlib import Path
import json, hashlib, zipfile

inputs = Path(__file__).resolve().parents[1] / 'data' / 'receipt_study'
staging = inputs / 'staging'
files = json.loads((inputs / 'file_hashes.json').read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


assert digest(inputs / 'source.tar') == '1859fd630629d16e9996dd9a4cf61df578a9ef87b27ae059c45f18329a50b261', \
    'source.tar must be built by the input-builder script'
for i, (remote, h) in enumerate(files.items()):
    p = staging / remote.lstrip('/')
    if not p.is_file() or digest(p) != h:
        raise SystemExit('Missing or changed: ' + str(p))
    if (i + 1) % 100 == 0:
        print('Verified', i + 1, 'files', flush=True)

archive, tmp = inputs / 'csev_full_inputs.zip', inputs / 'csev_full_inputs.partial.zip'
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
    for name in ['plan.json', 'csev_receipts_v1.json', 'source.tar', 'file_hashes.json', 'data_roles.json']:
        z.write(inputs / name, name)
    # Standard Tesseract TSV config, missing from the original bundle.
    td = json.loads((inputs / 'csev_receipts_v1.json').read_text())['legacy']['ocr']['tessdata_dir']
    z.writestr('staging/' + td.lstrip('/') + '/configs/tsv', 'tessedit_create_tsv 1\n')
    for remote in files:
        z.write(staging / remote.lstrip('/'), 'staging/' + remote.lstrip('/'))
tmp.replace(archive)
print('Ready:', archive, 'size GiB:', round(archive.stat().st_size / 2**30, 2))
