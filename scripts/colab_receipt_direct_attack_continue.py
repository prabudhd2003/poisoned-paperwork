"""Scheduling-only continuation of the frozen Qwen study; no method changes."""
import argparse, hashlib, importlib.util, inspect, json, sys, textwrap
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out)
spec=importlib.util.spec_from_file_location('frozen_study',root/'full_study.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=m.Study(root,out)  # Verifies the original runner, source, input and policy contract.
f=s.frozen()
schedule={'original_contract':s.sig,'method':'verified_recovery','document_ids':f['adaptive_ids'][:5], 'epsilons':[4,8],'restarts':[0,1],'max_runs':20,'wall_seconds_per_session':14400,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':'First five IDs in the original frozen order; not selected on outcomes. Only scheduling changes; original gradient, query, replay and checkpoint code retained.'}
manifest=out/'recovery_budget_schedule.json'
if manifest.exists():assert m.read(manifest)==schedule,'Existing schedule differs.'
else:m.write(manifest,schedule)
source=textwrap.dedent(inspect.getsource(m.Study.adaptive))
needle="for rid in ids:\n"
assert source.count(needle)==1
source=source.replace(needle,"for rid in ids[:5]:\n")
needle="for method in POLICY['adaptive_methods']:\n"
assert source.count(needle)==1
source=source.replace(needle,needle+"   if method != 'verified_recovery':continue\n")
# Match indentation of the original method (two spaces within each nested loop).
compile(source,'<recovery_schedule>','exec')
namespace={};exec(compile(source,'<recovery_schedule>','exec'),m.__dict__,namespace)
print('Recovery-only schedule:',json.dumps(schedule,indent=2),flush=True)
try:
 namespace['adaptive'](s,pilot=False)
finally:
 if s.wrapper:s.wrapper.unload()
