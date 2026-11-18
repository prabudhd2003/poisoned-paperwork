# ONE-CELL RUN: setup + part 2 of the frozen recovery-only direct attack (receipts 6-10).
# Connect to an A100, paste this as a new last cell of CSEV_Full_Colab_Study.ipynb, and run ONLY this cell.
# It does not rerun development, freeze, evaluation, the 120-run plan, or part 1.
import platform, shutil, subprocess, sys, json, os, tempfile, zipfile, tarfile, hashlib, threading
from pathlib import Path
from datetime import datetime
assert platform.system()=='Linux' and shutil.which('nvidia-smi'), 'Connect to a Colab GPU runtime first.'
subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv'],check=True)
from google.colab import drive
drive.mount('/content/drive')
DRIVE_FOLDER=Path('/content/drive/MyDrive/CSCI 566 Project/test/csev_full_colab')
BUNDLE=DRIVE_FOLDER/'csev_full_inputs.zip'
OUT=DRIVE_FOLDER/'full_study_results'/'receipt_bands_full_v1'
ROOT=Path('/content/csev_full_study')
assert BUNDLE.is_file(), f'Missing {BUNDLE}'
assert (OUT/'frozen.json').exists() and (OUT/'evaluation').is_dir(), f'Frozen study not found at {OUT}'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

# 1. Extract inputs to local disk (skipped if this runtime already has them).
if not (ROOT/'EXTRACTED.json').exists():
    assert not ROOT.exists(), 'Partial extraction exists at /content/csev_full_study. Use Runtime > Disconnect and delete runtime, then rerun.'
    with zipfile.ZipFile(BUNDLE) as z:
        required=sum(i.file_size for i in z.infolist())
    assert shutil.disk_usage('/content').free > BUNDLE.stat().st_size+required+2*1024**3, 'Insufficient local disk for extraction.'
    temp=Path(tempfile.mkdtemp(prefix='csev_extract_',dir='/content'))
    print('Copying bundle from Drive (4 GB, a few minutes)...',flush=True)
    local=temp/'bundle.zip';shutil.copy2(BUNDLE,local)
    print('Extracting...',flush=True)
    with zipfile.ZipFile(local) as z:
        for item in z.infolist():
            assert (temp/item.filename).resolve().is_relative_to(temp)
        z.extractall(temp)
    local.unlink()
    source=temp/'source';source.mkdir()
    with tarfile.open(temp/'source.tar') as t:
        for m in t.getmembers():
            assert (source/m.name).resolve().is_relative_to(source)
            assert m.isfile() or m.isdir()
        t.extractall(source,filter='data')
    (temp/'EXTRACTED.json').write_text(json.dumps({'bundle_name':BUNDLE.name}))
    temp.rename(ROOT)
config=json.loads((ROOT/'csev_receipts_v1.json').read_text())
assert file_sha(ROOT/'source.tar')=='1859fd630629d16e9996dd9a4cf61df578a9ef87b27ae059c45f18329a50b261', 'Source archive differs from the tested code snapshot.'

# 2. Restore the exact frozen runner from Drive (hash must match the study contract).
RUNNER_SHA='3e617b6ef285e856d5ad33615b25abe25bf36cb18962e12e1e1199abace1a5a9'
RUNNER=ROOT/'full_study.py'
if not (RUNNER.exists() and file_sha(RUNNER)==RUNNER_SHA):
    for candidate in (OUT/'runner_used.py',DRIVE_FOLDER/'notebook_tools'/'full_study.py'):
        if candidate.is_file() and file_sha(candidate)==RUNNER_SHA:
            shutil.copy2(candidate,RUNNER);break
assert RUNNER.exists() and file_sha(RUNNER)==RUNNER_SHA, 'Frozen runner not found on Drive. Run the notebook cell that writes full_study.py (section 2, second cell), then rerun this cell.'
print('Frozen runner restored:',RUNNER)

# 3. Recorded Python environment (skipped parts are reused if already installed).
def checked(args):
    print(' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),check=True)
checked([sys.executable,'-m','pip','install','-q','uv'])
PYTHON=ROOT/'venv/bin/python'
if not PYTHON.exists():checked(['uv','venv','--python',config['runtime_versions']['python'],'--seed',ROOT/'venv'])
v=config['runtime_versions']
constraints=ROOT/'constraints.txt'
constraints.write_text(''.join(k+'=='+ver+'\n' for k,ver in v.items() if k!='python'))
checked(['uv','pip','install','--python',PYTHON,'--index-url','https://download.pytorch.org/whl/cu124','torch=='+v['torch'],'torchvision=='+v['torchvision']])
packages=[k+'=='+ver for k,ver in v.items() if k not in ('python','torch','torchvision')]
checked(['uv','pip','install','--python',PYTHON,'--constraint',constraints,*packages,'pyarrow','sentencepiece','accelerate'])
checked(['apt-get','update','-qq'])
checked(['apt-get','install','-y','-qq','tesseract-ocr'])
(OUT/('packages_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.txt')).write_text(subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True))

# 4. Preflight with the frozen runner (same check the notebook runs before any stage).
env=os.environ.copy();env['HF_HOME']=str(ROOT/'hf_cache')
pre=subprocess.run([str(PYTHON),'-u',str(RUNNER),'preflight','--root',str(ROOT),'--out',str(OUT)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
(OUT/('preflight_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')).write_text(pre.stdout)
print(pre.stdout[-3000:])
assert pre.returncode==0, 'Preflight failed; nothing was attacked. Send me the output above.'

# 5. Part 2 of the recovery-only direct attack.
os.environ['MPLBACKEND']='Agg'
assert ROOT.is_dir() and (OUT/'frozen.json').exists() and (OUT/'evaluation').is_dir()
before={p.parent.name for p in (OUT/'adaptive').glob('*/result.json')}
assert len(before)>=25, f'Expected the 25 finished runs in {OUT}/adaptive, found {len(before)}'
if 'process' in globals():
    assert process.poll() is not None, 'A previous child process is still running.'
LAUNCHER_SOURCE=r'''"""Scheduling-only continuation, part 2, of the frozen Qwen study; no method changes."""
import argparse, hashlib, importlib.util, inspect, json, textwrap
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out)
spec=importlib.util.spec_from_file_location('frozen_study',root/'full_study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.Study(root,out)  # Verifies the original runner, source, input and policy contract.
f=s.frozen()
first=m.read(out/'recovery_budget_schedule.json')
assert first['original_contract']==s.sig and first['document_ids']==f['adaptive_ids'][:5]
schedule={'original_contract':s.sig,'method':'verified_recovery','document_ids':f['adaptive_ids'][5:10],'epsilons':[4,8],'restarts':[0,1],'max_runs':20,'wall_seconds_per_session':14400,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'IDs six to ten in the original frozen order, completing the frozen ten-receipt list; not selected on outcomes. Only scheduling changes; original gradient, query, replay and checkpoint code retained.'}
manifest=out/'recovery_budget_schedule_part2.json'
if manifest.exists():assert m.read(manifest)==schedule,'Existing part-2 schedule differs.'
else:m.write(manifest,schedule)
source=textwrap.dedent(inspect.getsource(m.Study.adaptive))
needle="for rid in ids:\n"
assert source.count(needle)==1
source=source.replace(needle,"for rid in ids[5:10]:\n")
needle="for method in POLICY['adaptive_methods']:\n"
assert source.count(needle)==1
source=source.replace(needle,needle+"   if method != 'verified_recovery':continue\n")
namespace={};exec(compile(source,'<recovery_schedule_part2>','exec'),m.__dict__,namespace)
print('Recovery-only schedule, part 2:',json.dumps(schedule,indent=2),flush=True)
try:
 namespace['adaptive'](s,pilot=False)
finally:
 if s.wrapper:s.wrapper.unload()
'''
launcher=ROOT/'continue_recovery_part2.py'
launcher.write_text(LAUNCHER_SOURCE)
log_path=OUT/('recovery_budget_part2_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
env=os.environ.copy();env['HF_HOME']=str(ROOT/'hf_cache')
expired=threading.Event()
with log_path.open('w') as log:
    process=subprocess.Popen([str(PYTHON),'-u',str(launcher),'--root',str(ROOT),'--out',str(OUT)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    def stop_at_budget():
        if process.poll() is None:
            expired.set();process.terminate()
    timer=threading.Timer(4*3600,stop_at_budget);timer.start()
    try:
        for line in process.stdout:
            print(line,end='');log.write(line);log.flush()
        code=process.wait()
    except BaseException:
        process.terminate()
        try:process.wait(timeout=15)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        raise
    finally:
        timer.cancel()
if expired.is_set():
    print('Four-hour budget reached. Finished runs and the latest checkpoint are kept; rerun this cell to continue.')
elif code:
    raise RuntimeError(f'Part-2 continuation failed: {code}. See {log_path}')
else:
    print('Part-2 schedule completed.')

# Recompute the recovery-only table from individual records (the 120-run adaptive_summary.json stays incomplete by design).
frozen_ids=json.loads((OUT/'frozen.json').read_text())['adaptive_ids']
clean={}
for p in (OUT/'evaluation').glob('*.json'):
    r=json.loads(p.read_text())
    if r['family']=='clean' and r['record_id'] in frozen_ids:
        v=r['methods']['verified_recovery'];clean[r['record_id']]=bool(v['accepted'] and v['correct'])
runs=[json.loads(p.read_text()) for p in (OUT/'adaptive').glob('*/result.json')]
runs=[r for r in runs if r['method_attacked']=='verified_recovery']
print('\nRecovery-only runs on disk:',len(runs),'of 40 planned (10 receipts x 2 budgets x 2 restarts)')
for eps in (4,8):
    done={};hit={}
    for r in runs:
        if r['epsilon']!=eps:continue
        v=r['methods']['verified_recovery'];rid=r['record_id']
        done.setdefault(rid,set()).add(r['restart'])
        hit[rid]=hit.get(rid,False) or bool(v['accepted'] and v['targeted'])
    complete=[i for i in frozen_ids if done.get(i)=={0,1}]
    eligible=[i for i in complete if clean.get(i)]
    print(f'eps {eps}: receipts with both restarts {len(complete)}/10 | success, all: {sum(hit[i] for i in complete)}/{len(complete)}'
          f' | success, clean-correct-and-accepted: {sum(hit[i] for i in eligible)}/{len(eligible)}')
    for i in frozen_ids:
        print('   ',i.split('/')[-1],'eligible' if clean.get(i) else 'clean-rejected','restarts done:',sorted(done.get(i,[])),'SUCCESS' if hit.get(i) else '')
