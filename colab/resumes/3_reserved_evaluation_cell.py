# Run below the completed saved-attack development notebook.
import json, hashlib, zipfile, subprocess, os, ast
from pathlib import Path
from datetime import datetime

RROOT = RESUME_ROOT / 'reserved_inputs_v1'
ROUT = RESUME_DRIVE / 'resume_reserved_results_v1'
RROOT.mkdir(exist_ok=True)
ROUT.mkdir(exist_ok=True)
BUNDLE = RESUME_ROOT / 'resume_complete_inputs_v1.zip'
assert BUNDLE.is_file(), 'Run the notebook setup and preparation cells first.'

def digest(b): return hashlib.sha256(b).hexdigest()
def replace_once(text, old, new):
    assert text.count(old) == 1, f'Unexpected source version: {old}'
    return text.replace(old, new, 1)

# Require the exact previously supplied scripts before adapting their data selection.
source = (RESUME_ROOT/'resume_saved_attack_development.py').read_text()
prepare = (RESUME_ROOT/'prepare_resume_saved_attacks.py').read_text()
assert digest(source.encode()) == 'b09b577474d73420f4f6891bd70ba4a486bbed1c40005d95439866be1e50b844'
assert digest(prepare.encode()) == '9bbf43db0621606faf4e160b7ac5a213926bc975458c17bc8b7fe6b001f63a03'

cfg = json.loads((RESUME_ROOT/'config.json').read_text())
assert len(cfg['reserved_ids']) == 60
assert not set(cfg['reserved_ids']) & set(cfg['development_ids'])
input_hashes = {}
def put(name, data):
    target = RROOT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    input_hashes[name] = digest(data)

for name in ('config.json','qwen_adapter.py'):
    put(name, (RESUME_ROOT/name).read_bytes())
with zipfile.ZipFile(BUNDLE) as z:
    hashes = json.loads(z.read('archive_hashes.json'))
    def checked(name):
        data = z.read(name)
        assert digest(data) == hashes[name], name
        return data
    inventory = json.loads(checked('resume_inventory.json'))
    selected = sorted((d for d in inventory['documents']
                       if d['record_id'] in cfg['reserved_ids']),
                      key=lambda d: d['record_id'])
    assert len(selected) == 60
    assert {d['record_id'] for d in selected} == set(cfg['reserved_ids'])
    documents = []
    for d in selected:
        images = []
        for name in d['images']:
            put(name, checked(name))
            images.append({'path':name, 'sha256':input_hashes[name]})
        documents.append({'record_id':d['record_id'],
                          'clean_target':d['clean_target'], 'images':images})
put('documents.json', json.dumps(documents, indent=2).encode())
(RROOT/'hashes.json').write_text(json.dumps(input_hashes, indent=2))

# One reserved resume has no saved attacks. Keep it in clean evaluation.
missing = {'resume_parsing_vision/validation/r00175'}
prepare = replace_once(prepare,
    "assert len(byid)==15 and set(byid)==set(c['development_ids'])",
    "assert len(byid)==60 and set(byid)==set(c['reserved_ids'])")
prepare = replace_once(prepare,
    "assert not set(byid)&set(c['reserved_ids'])",
    "assert not set(byid)&set(c['development_ids'])")
prepare = replace_once(prepare, 'assert len(keys)==len(set(keys))==60',
                       'assert len(keys)==len(set(keys))==236')
prepare = replace_once(prepare,
    'assert set(keys)=={(rid,e) for rid in byid for e in (2,4,8,16)}',
    'assert set(keys)=={(rid,e) for rid in set(byid)-'+repr(missing)+' for e in (2,4,8,16)}')
prepare = prepare.replace(
    'Verified 15 clean + 60 saved attacks', 'Verified 60 clean + 236 saved attacks'
).replace('Reserved documents excluded.', 'Development documents excluded.')
RP = RROOT/'prepare_reserved.py'
RP.write_text(prepare)
subprocess.run([str(RESUME_PYTHON), '-u', str(RP), str(RROOT), str(BUNDLE)], check=True)

source = replace_once(source,
    "assert len(docs)==75 and {d['record_id'] for d in docs}==set(c['development_ids'])",
    "assert len(docs)==296 and {d['record_id'] for d in docs}==set(c['reserved_ids'])")
source = replace_once(source, "'complete':len(results)==75", "'complete':len(results)==296")
source = replace_once(source,
    'Same 15 development documents. 60 standard saved attacks were not optimized against recovery. Repeated variants are not independent documents. Reserved resumes excluded.',
    '60 resumes reserved from current tuning; 236 saved standard attacks from 59 resumes. Parent validation data were used earlier in the project. Attacks did not target recovery; repeated variants are not independent documents.')
RR = RROOT/'run_reserved.py'
compile(source, str(RR), 'exec')
RR.write_text(source)
# Reuse the model cache to avoid downloading weights again.
if (RESUME_ROOT/'hf_cache').exists() and not (RROOT/'hf_cache').exists():
    (RROOT/'hf_cache').symlink_to(RESUME_ROOT/'hf_cache', target_is_directory=True)

log_path = ROUT/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
with log_path.open('w') as log:
    process = subprocess.Popen(
        [str(RESUME_PYTHON),'-u',str(RR),'--root',str(RROOT),'--out',str(ROUT)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, 'MPLBACKEND':'Agg'})
    try:
        for line in process.stdout:
            print(line,end=''); log.write(line); log.flush()
        status = process.wait()
    except BaseException:
        process.terminate()
        try: process.wait(timeout=15)
        except subprocess.TimeoutExpired: process.kill(); process.wait()
        raise
assert status == 0, f'Share this log if it failed: {log_path}'
archive = RESUME_DRIVE/'resume_reserved_results_v1.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(ROUT.rglob('*')):
        if f.is_file() and f.suffix in ('.json','.csv','.png','.log'):
            z.write(f, f.relative_to(ROUT))
print('Share:', archive)
from google.colab import files
files.download(str(archive))
