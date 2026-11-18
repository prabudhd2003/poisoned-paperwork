"""Archive resume inputs for later studies; does not run models or change experiments."""
import argparse,hashlib,json,subprocess,sys,zipfile,io
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root).resolve();out=Path(a.out).resolve()
assert not out.exists(),f'Preserve existing archive: {out}'
dev=out.parent/'resume_development_inputs_v1.zip'
if not dev.exists():
 subprocess.run([sys.executable,str(Path(__file__).with_name('export_resume_development.py')),'--root',str(root),'--out',str(dev)],check=True)
with zipfile.ZipFile(dev) as z:
 cfg=json.loads(z.read('config.json'))
 for name,h in json.loads(z.read('hashes.json')).items():assert hashlib.sha256(z.read(name)).hexdigest()==h,name
sys.path.insert(0,str(root/'src'))
import pandas as pd
from data import load_processed_rows,row_images
frame=pd.read_parquet(root/'data/processed/master_manifest.parquet')
frame=frame.loc[frame.dataset_name.eq('resume_parsing_vision') & frame.usable.eq(True) & frame.experiment_split.isin(['validation','test'])]
assert not frame.record_id.duplicated().any()
assert set(cfg['development_ids']+cfg['reserved_ids'])==set(frame.loc[frame.experiment_split.eq('validation'),'record_id'])
attack=root/'outputs/qwen_attack/qwen_resume_validation_v1'
assert (attack/'merged/attacks.parquet').is_file(),'Missing merged standard resume attacks; stop rather than export an incomplete archive.'
out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix('.partial');hashes={};docs=[]
try:
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  def put(name,b):z.writestr(name,b);hashes[name]=hashlib.sha256(b).hexdigest()
  def add(path,name):
   h=hashlib.sha256()
   with path.open('rb') as src,z.open(name,'w',force_zip64=True) as dest:
    for block in iter(lambda:src.read(8*1024*1024),b''):h.update(block);dest.write(block)
   hashes[name]=h.hexdigest()
  add(dev,dev.name)
  for split in ['validation','test']:
   part=frame.loc[frame.experiment_split.eq(split)].sort_values('record_id')
   rows=load_processed_rows(root,split,part.record_id.tolist())
   for i,(_,r) in enumerate(part.iterrows()):
    paths=[]
    for j,im in enumerate(row_images(rows[r.record_id])):
     name=f'clean/{split}/document_{i:03d}/page_{j:02d}.png';b=io.BytesIO();im.convert('RGB').save(b,format='PNG');put(name,b.getvalue());paths.append(name)
    role='development' if r.record_id in cfg['development_ids'] else 'reserved_from_current_tuning'
    docs.append({'record_id':r.record_id,'clean_target':str(r.clean_target),'split':split,'role':role,'images':paths})
    print(f'{split}: {i+1}/{len(part)} exported',flush=True)
  files=[f for f in attack.rglob('*') if f.is_file() and f.suffix.lower() in ('.png','.json','.parquet','.csv')]
  assert any(f.suffix.lower()=='.png' for f in files),'No saved attack PNGs found.'
  for i,f in enumerate(files):
   add(f,str(f.relative_to(root)))
   if i%100==0:print(f'Attack artifacts: {i+1}/{len(files)}',flush=True)
  for rel in ['data/processed/master_manifest.parquet','outputs/baseline/baseline_validation_v1/config.json','outputs/baseline/baseline_validation_v1/merged/predictions.parquet']:
   f=root/rel
   if not f.is_file():raise FileNotFoundError(f)
   add(f,rel)
  put('resume_inventory.json',json.dumps({'documents':docs,'development_ids':cfg['development_ids'],'note':'Only the embedded 15-document bundle is for development. Original attack tables retain CARC paths; future notebooks must remap them to archived paths and verify per-page hashes. Includes existing standard attacks, not future defense-aware attacks. Test data provenance must be checked before describing it as unseen.'},indent=2).encode())
  z.writestr('archive_hashes.json',json.dumps(hashes,indent=2))
 tmp.replace(out)
except BaseException:
 if tmp.exists():tmp.unlink()
 raise
print(f'Complete: {out}; {out.stat().st_size/1024**3:.2f} GiB; {len(docs)} clean documents; {len(files)} attack artifacts')
