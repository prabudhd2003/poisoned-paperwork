import zipfile,json,hashlib,sys
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(sys.argv[1]);bundle=Path(sys.argv[2])
read=lambda p:json.loads(p.read_text())
sha=lambda b:hashlib.sha256(b).hexdigest()
c=read(root/'config.json');clean=read(root/'documents.json');byid={d['record_id']:d for d in clean}
assert len(byid)==15 and set(byid)==set(c['development_ids'])
assert not set(byid)&set(c['reserved_ids'])
cases=[dict(d,case_id='clean_'+str(i),family='clean',epsilon_255=0,attack_target=None) for i,d in enumerate(clean)]
with zipfile.ZipFile(bundle) as z:
 manifest=json.loads(z.read('archive_hashes.json'))
 def checked(name):
  b=z.read(name);assert sha(b)==manifest[name],name;return b
 selected=[]
 for name in sorted(z.namelist()):
  if '/records/' not in name or not name.endswith('.json'):continue
  r=json.loads(checked(name))
  if r['record_id'] in byid:selected.append((name,r))
 keys=[(r['record_id'],r['epsilon_255']) for _,r in selected]
 assert len(keys)==len(set(keys))==60
 assert set(keys)=={(rid,e) for rid in byid for e in (2,4,8,16)}
 for index,(name,r) in enumerate(sorted(selected,key=lambda x:(x[1]['record_id'],x[1]['epsilon_255']))):
  d=byid[r['record_id']];assert r['status']=='complete'
  assert r['clean_target']==d['clean_target']
  assert r['attack_target'] in c['labels'] and r['attack_target']!=d['clean_target']
  assert r['model_id']==c['model']['model_id'] and r['model_revision']==c['model']['revision']
  assert r['question']==c['question']
  for key in ('precision','min_pixels','max_pixels','max_new_tokens'):
   assert r['processor_settings'][key]==c['model'][key],key
  assert len(r['images'])==len(d['images'])==r['document_page_count']
  assert r['attacked_image_sha256']==[im['sha256'] for im in r['images']]
  images=[]
  for j,im in enumerate(r['images']):
   assert im['page']==j
   stored=im['path'];prefix='outputs/qwen_attack/qwen_resume_validation_v1/'
   assert stored.count(prefix)==1
   archive_path=prefix+stored.split(prefix,1)[1]
   b=checked(archive_path);assert sha(b)==im['sha256']
   target=root/'saved_attacks'/f'{index:03d}'/f'page_{j:02d}.png';target.parent.mkdir(parents=True,exist_ok=True)
   target.write_bytes(b)
   with Image.open(target) as attacked,Image.open(root/d['images'][j]['path']) as original:
    aa=np.asarray(attacked.convert('RGB'),dtype=np.int16);cc=np.asarray(original.convert('RGB'),dtype=np.int16)
    assert aa.shape==cc.shape,(r['record_id'],j)
    assert int(np.abs(aa-cc).max())<=r['epsilon_255'],(r['record_id'],j,'pixel budget exceeded')
   images.append({'path':str(target.relative_to(root)),'sha256':im['sha256']})
  cases.append({'record_id':r['record_id'],'clean_target':d['clean_target'],'images':images,
    'case_id':r['experiment_key'],'family':'standard','epsilon_255':r['epsilon_255'],'attack_target':r['attack_target'],
    'source_record':name,'source_record_sha256':manifest[name],'historical_targeted_success':r['targeted_success_reloaded']})
(root/'attack_cases.json').write_text(json.dumps(cases,indent=2))
print('Verified 15 clean + 60 saved attacks, all page hashes, page order, model settings, and pixel bounds. Reserved documents excluded.')
