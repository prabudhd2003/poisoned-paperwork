# Paste this whole cell into the EXISTING Qwen notebook; do not restart its runtime.
import os, subprocess, threading, json
from datetime import datetime
os.environ['MPLBACKEND']='Agg'
assert ROOT.is_dir() and (OUT/'frozen.json').exists()
if 'process' in globals():
    assert process.poll() is not None, 'The previous child process is still running.'
LAUNCHER_SOURCE='"""Scheduling-only continuation of the frozen Qwen study; no method changes."""\nimport argparse, hashlib, importlib.util, inspect, json, sys, textwrap\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument(\'--root\',required=True);p.add_argument(\'--out\',required=True);a=p.parse_args()\nroot=Path(a.root);out=Path(a.out)\nspec=importlib.util.spec_from_file_location(\'frozen_study\',root/\'full_study.py\')\nm=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\ns=m.Study(root,out)  # Verifies the original runner, source, input and policy contract.\nf=s.frozen()\nschedule={\'original_contract\':s.sig,\'method\':\'verified_recovery\',\'document_ids\':f[\'adaptive_ids\'][:5], \'epsilons\':[4,8],\'restarts\':[0,1],\'max_runs\':20,\'wall_seconds_per_session\':14400,\'launcher_sha256\':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\'selection\':\'First five IDs in the original frozen order; not selected on outcomes. Only scheduling changes; original gradient, query, replay and checkpoint code retained.\'}\nmanifest=out/\'recovery_budget_schedule.json\'\nif manifest.exists():assert m.read(manifest)==schedule,\'Existing schedule differs.\'\nelse:m.write(manifest,schedule)\nsource=textwrap.dedent(inspect.getsource(m.Study.adaptive))\nneedle="for rid in ids:\\n"\nassert source.count(needle)==1\nsource=source.replace(needle,"for rid in ids[:5]:\\n")\nneedle="for method in POLICY[\'adaptive_methods\']:\\n"\nassert source.count(needle)==1\nsource=source.replace(needle,needle+"   if method != \'verified_recovery\':continue\\n")\n# Match indentation of the original method (two spaces within each nested loop).\ncompile(source,\'<recovery_schedule>\',\'exec\')\nnamespace={};exec(compile(source,\'<recovery_schedule>\',\'exec\'),m.__dict__,namespace)\nprint(\'Recovery-only schedule:\',json.dumps(schedule,indent=2),flush=True)\ntry:\n namespace[\'adaptive\'](s,pilot=False)\nfinally:\n if s.wrapper:s.wrapper.unload()\n'
launcher=ROOT/'continue_recovery.py'
launcher.write_text(LAUNCHER_SOURCE)
log_path=OUT/('recovery_budget_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.log')
env=os.environ.copy();env['HF_HOME']=str(ROOT/'hf_cache')
expired=threading.Event()
with log_path.open('w') as log:
    process=subprocess.Popen([str(PYTHON),'-u',str(launcher),'--root',str(ROOT),'--out',str(OUT)],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
    def stop_at_budget():
        if process.poll() is None:
            expired.set()
            process.terminate()
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
    print('Four-hour budget reached. Completed results and the latest saved checkpoint are preserved.')
elif code:
    raise RuntimeError(f'Recovery continuation failed: {code}. See {log_path}')
else:
    print('Recovery schedule completed.')
schedule=json.loads((OUT/'recovery_budget_schedule.json').read_text())
results=[json.loads(p.read_text()) for p in (OUT/'adaptive').glob('*/result.json')]
results=[r for r in results if r['method_attacked']=='verified_recovery' and r['record_id'] in schedule['document_ids']]
print('Completed recovery runs:',len(results),'/ 20 planned; interrupted runs are not counted.')
print('Next: rerun analysis, display, and export. The original 120-run summary remains intentionally incomplete.')
