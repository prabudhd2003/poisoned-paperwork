"""Run on CARC: export 15 deterministic clean development resumes, no jobs or edits."""
import argparse, hashlib, importlib.metadata, io, json, platform, subprocess, sys, zipfile
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
 root=Path(a.root).resolve();out=Path(a.out).resolve();assert not out.exists(),f'Already exists: {out}; preserve the original.'
 sys.path.insert(0,str(root/'src'))
 import pandas as pd
 from data import load_processed_rows,row_images
 manifest=root/'data/processed/master_manifest.parquet';frame=pd.read_parquet(manifest)
 frame=frame.loc[frame.dataset_name.eq('resume_parsing_vision') & frame.experiment_split.eq('validation') & frame.usable.eq(True)]
 assert len(frame)==75 and not frame.record_id.duplicated().any()
 labels=['secondary','certificate_or_diploma','associate','bachelor','postgraduate','master','doctorate']
 assert set(frame.clean_target)<=set(labels)
 ids=sorted(frame.record_id.tolist(),key=lambda x:sha(('566:resume_development_v1:'+x).encode()))
 dev=ids[:15];reserved=ids[15:];rows=load_processed_rows(root,'validation',dev)
 cfgpath=root/'outputs/baseline/baseline_validation_v1/config.json'
 assert cfgpath.exists(),'Original baseline config missing; do not substitute guessed settings.'
 cfg=json.loads(cfgpath.read_text());model=cfg['models']['qwen'];prompt=cfg['prompts']['resume_highest_degree']['text']
 versions={k:importlib.metadata.version(k) for k in ['torch','torchvision','transformers','Pillow','numpy','accelerate','tokenizers','huggingface_hub','safetensors']}
 config={'model':model,'question':prompt,'labels':labels,'python':platform.python_version(),'versions':versions,
  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
  'baseline_config_sha256':sha(cfgpath.read_bytes()),'manifest_sha256':sha(manifest.read_bytes()),
  'selection':'First 15 of 75 usable validation resumes ranked by SHA256(566:resume_development_v1: + record_id); no correctness filtering.',
  'development_ids':dev,'reserved_ids':reserved,'note':'Reserved IDs are excluded from this diagnostic; parent validation data were used in earlier project experiments.'}
 out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.partial');hashes={};docs=[]
 try:
  with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
   def put(name,b):z.writestr(name,b);hashes[name]=sha(b)
   put('qwen_adapter.py',(root/'src/models/qwen.py').read_bytes())
   put('config.json',json.dumps(config,indent=2).encode())
   for i,rid in enumerate(dev):
    row=frame.loc[frame.record_id.eq(rid)].iloc[0];images=[]
    for j,im in enumerate(row_images(rows[rid])):
     buf=io.BytesIO();im.convert('RGB').save(buf,format='PNG');name=f'images/document_{i:02d}/page_{j:02d}.png';put(name,buf.getvalue());images.append({'path':name,'sha256':hashes[name]})
    docs.append({'record_id':rid,'clean_target':str(row.clean_target),'images':images})
    print(f'Exported {i+1}/15: {rid}; {len(images)} pages',flush=True)
   put('documents.json',json.dumps(docs,indent=2).encode())
   z.writestr('hashes.json',json.dumps(hashes,indent=2))
  tmp.replace(out)
 except BaseException:
  if tmp.exists():tmp.unlink()
  raise
 print(f'BUNDLE: {out}\nSIZE: {out.stat().st_size/1024**2:.1f} MiB\nSHA256: {sha(out.read_bytes())}')
if __name__=='__main__':main()
