"""Isolated Colab receipt study. Do not use as a production CARC launcher."""
import argparse,copy,hashlib,json,math,os,platform,sys,time,collections,csv,io
from pathlib import Path
METHODS=['none','jpeg','resize_vote','selective_multiview','semantic_consistency','ocr_only','evidence','fixed_band','band_answer','band_agreement','band_csev','verified_recovery']
POLICY={'version':1,'seed':566,'band_min_height':.12,'band_margin':.03,'band_max_height':.35,'expanded_margin':.03,'expanded_max_height':.45,'fixed_band_vertical':[.25,.75],'methods':METHODS,'adaptive_methods':['band_agreement','band_csev','verified_recovery'],'adaptive_epsilons':[4,8],'adaptive_restarts':2,'adaptive_steps':100,'adaptive_queries':100,'refresh_interval':10,'adaptive_documents':10,'primary_method':'verified_recovery'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False));tmp.replace(p)
def band(box,size):
 if box is None:return None
 W,H=size;_,t,_,b=box;hh=max(round(POLICY['band_min_height']*H),b-t+2*round(POLICY['band_margin']*H))
 if hh>POLICY['band_max_height']*H:return None
 top=max(0,min(H-hh,round((t+b-hh)/2)));return (0,top,W,top+hh)
def expand(box,size):
 if box is None:return None
 W,H=size;_,t,_,b=box;pad=round(H*POLICY['expanded_margin']);top=max(0,t-pad);bottom=min(H,b+pad)
 return (0,top,W,bottom) if (bottom-top)/H<=POLICY['expanded_max_height'] else None

def choose(o,tau):
 full=o.get('full');crop=o.get('crop');wide=o.get('wide');gap=o.get('gap');agree=full is not None and full==crop
 return {'none':full,'band_answer':crop,'band_agreement':full if agree else None,'band_csev':full if agree and gap is not None and gap>=tau else None,'verified_recovery':crop if crop is not None and crop==wide else None}
def record(pred,d):return {'prediction':pred,'accepted':pred is not None,'correct':pred is not None and pred==d['clean_target'],'targeted':pred is not None and pred==d['frozen_attack_target']}

class Study:
 def __init__(self,root,out):
  self.root=Path(root);self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
  sys.path.insert(0,str(self.root/'source/src'))
  self.c=read(self.root/'csev_receipts_v1.json');self.plan=read(self.root/'plan.json');self.roles=read(self.root/'data_roles.json')
  self.docs={d['record_id']:d for split in ['development','evaluation'] for d in self.plan[split]}
  self.contract={'source_archive':sha(self.root/'source.tar'),'runner':sha(__file__),'inputs':sha(self.root/'file_hashes.json'),'plan':sha(self.root/'plan.json'),'config':sha(self.root/'csev_receipts_v1.json'),'roles':sha(self.root/'data_roles.json'),'policy':POLICY}
  path=self.out/'contract.json'
  if path.exists():assert read(path)==self.contract,'Code or inputs changed. Use a NEW STUDY_ID.'
  else:
   write(path,self.contract)
   (self.out/'runner_used.py').write_text(Path(__file__).read_text())
  self.sig=digest(self.contract);self.wrapper=None
 def local(self,path):return self.root/'staging'/str(path).lstrip('/')
 def ids(self,phase):
  if phase=='development':return self.roles['public_development']+self.roles['collected_development']
  if phase=='evaluation':return self.roles['collected_reserved_evaluation']
  raise ValueError(phase)
 def tasks(self,phase):
  tasks=[]
  for rid in self.ids(phase):
   d=self.docs[rid]
   for a in [{'family':'clean','epsilon':0,'images':d['images']}]+d['attacks']:
    key=digest([rid,a['family'],a['epsilon']]);tasks.append(dict(key=key,document=d,**a))
  return tasks
 def preflight(self):
  from PIL import Image
  from csev.study import bounds
  hashes=read(self.root/'file_hashes.json')
  for i,(path,h) in enumerate(hashes.items()):
   assert sha(self.local(path))==h,path
   if (i+1)%100==0:print('Verified input hashes:',i+1,flush=True)
  dev=set(self.ids('development'));ev=set(self.ids('evaluation'))
  assert not dev&ev and len(dev)==75 and len(ev)==45
  assert dev|ev==set(self.docs)
  for rid,d in self.docs.items():
   assert isinstance(d['clean_target'],str) and isinstance(d['frozen_attack_target'],str)
   assert d['clean_target']!=d['frozen_attack_target']
   assert len(d['images'])==1
   clean=Image.open(self.local(d['images'][0]['path'])).convert('RGB')
   seen=set()
   for a in d['attacks']:
    assert (a['family'],a['epsilon']) not in seen;seen.add((a['family'],a['epsilon']))
    assert a['family'] in ['standard','uniform_eot','hardview'] and a['epsilon'] in [4,8,16] and len(a['images'])==1
    attacked=Image.open(self.local(a['images'][0]['path'])).convert('RGB');bounds([clean],[attacked],a['epsilon'])
  self.environment()
  # Validate OCR output format on one image before reserving model memory.
  import subprocess
  td=self.local(self.c['legacy']['ocr']['tessdata_dir']);(td/'configs').mkdir(exist_ok=True)
  (td/'configs/tsv').write_text('tessedit_create_tsv 1\n')
  image=self.local(next(iter(self.docs.values()))['images'][0]['path'])
  for lang in ['eng','eng+ind']:
   r=subprocess.run(['/usr/bin/tesseract',str(image),'stdout','--tessdata-dir',str(td),'-l',lang,'--oem','1','--psm','3','tsv'],capture_output=True,text=True,timeout=120,check=True)
   header=r.stdout.splitlines()[0].split('\t') if r.stdout else []
   assert {'level','text','left','top','width','height','conf'}.issubset(header),(lang,r.stderr,header)
  result={'contract':self.sig,'documents':len(self.docs),'saved_attack_records':sum(len(d['attacks']) for d in self.docs.values()),'development_cases':len(self.tasks('development')),'evaluation_cases':len(self.tasks('evaluation'))}
  write(self.out/'preflight.json',result);print(json.dumps(result,indent=2),flush=True)
 def environment(self):
  import importlib.metadata,torch,subprocess
  expected=self.c['runtime_versions'];versions={k:platform.python_version() if k=='python' else importlib.metadata.version(k) for k in expected}
  assert versions==expected,{'expected':expected,'actual':versions}
  assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA with BF16 required.'
  env={'versions':versions,'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'ocr_binary_sha256':sha('/usr/bin/tesseract'),'ocr_version':subprocess.check_output(['/usr/bin/tesseract','--version'],text=True).splitlines()[0]}
  p=self.out/'environment.json'
  if p.exists():assert read(p)==env,'Runtime differs from prior work. Use a new study ID; do not mix environments.'
  else:write(p,env)
  return env
 def model(self):
  if self.wrapper:return
  assert read(self.out/'preflight.json')['contract']==self.sig,'Run preflight first.'
  self.environment()
  import torch
  from models.qwen import QwenVL
  from csev.defense import Verifier
  os.environ['HF_HOME']=str(self.root/'hf_cache');torch.manual_seed(566)
  self.wrapper=QwenVL(device=torch.device('cuda'),**self.c['legacy']['model']);self.v=Verifier(self.wrapper,self.c['settings'],self.c['legacy']['prompts']['receipt']);self.prompt=self.v.prompt
 def observe(self,image,ds,scoring=True):
  from csev.defense import parse_box,control_box,mask
  from normalization import normalize_receipt_total
  norm=lambda s:normalize_receipt_total(s,ds)
  full_raw=self.v.predict([image],self.prompt);o={'full':norm(full_raw),'full_raw':full_raw,'crop':None,'wide':None,'gap':None,'box':None,'expanded_box':None,'control':None,'reason':'invalid_region'}
  # Localization runs even if full prediction is invalid: recovery may still work.
  raw=self.v.predict([image],self.c['settings']['localization_prompt']);grid=self.wrapper.last_input_info['image_grid_thw'];patch=self.wrapper.processor.image_processor.patch_size
  assert len(grid)==1 and isinstance(patch,int)
  _,gh,gw=grid[0];rs=(gw*patch,gh*patch);o.update(localization_raw=raw,processor_size=rs)
  try:
   s=raw.strip()
   if s.startswith('```'):s=s.split('\n',1)[-1].rsplit('```',1)[0].strip()
   val=json.loads(s)['box'];assert isinstance(val,list) and len(val)==4
   assert all(not isinstance(x,bool) and isinstance(x,(int,float)) and math.isfinite(x) for x in val)
   tight=parse_box(json.dumps({'box':[x*1000/d for x,d in zip(val,(*rs,*rs))]}),image.size,self.c['settings'])
  except (AssertionError,ValueError,KeyError,TypeError):tight=None
  box=band(tight,image.size);wide=expand(box,image.size);o.update(tight_box=tight,box=box,expanded_box=wide)
  if box is None:return o
  raw=self.v.predict([image.crop(box)],self.prompt);o.update(crop=norm(raw),crop_raw=raw,reason='crop_disagreement')
  if wide:
   raw=self.v.predict([image.crop(wide)],self.prompt);o.update(wide=norm(raw),wide_raw=raw)
  o['control']=control_box(box,image.size)
  if scoring and o['full'] is not None and o['crop']==o['full'] and o['control']:
   sc=self.v.score(mask(image,o['control'],128),self.prompt,o['full']);se=self.v.score(mask(image,box,128),self.prompt,o['full'])
   assert math.isfinite(sc) and math.isfinite(se)
   o.update(gap=sc-se,score_control=sc,score_evidence=se,reason='scored')
  return o
 def run(self,phase,limit=None):
  from PIL import Image
  from types import SimpleNamespace
  from stages.final_protocol import evaluate
  from normalization import normalize_receipt_total
  tau=0 if phase=='development' else self.frozen()['threshold']
  tasks=self.tasks(phase);self.model()
  for i,t in enumerate(tasks):
   if limit is not None and i>=limit:break
   path=self.out/phase/(t['key']+'.json')
   if path.exists():assert read(path)['contract']==self.sig;print('RESUME',i+1,flush=True);continue
   d=t['document'];item=t['images'][0];local=self.local(item['path']);assert sha(local)==item['sha256']
   image=Image.open(local).convert('RGB');start=time.monotonic();o=self.observe(image,d['dataset_name']);preds=choose(o,tau);costs={}
   # Baselines use the same model and prompts; OCR executable differs from CARC.
   legacy=copy.deepcopy(self.c['legacy']);legacy['ocr']['executable']='/usr/bin/tesseract';legacy['ocr']['tessdata_dir']=str(self.local(legacy['ocr']['tessdata_dir']))
   if d['dataset_name']=='collected_receipts':legacy['ocr']['language']='eng'
   for name in ['jpeg','resize_vote','selective_multiview','semantic_consistency','ocr_only','evidence']:
    result=evaluate([image],d,name,SimpleNamespace(predict=self.v.predict),legacy);preds[name]=result['prediction'];costs[name]=result['runtime_seconds']
   W,H=image.size;fixed=(0,round(H*.25),W,round(H*.75));preds['fixed_band']=normalize_receipt_total(self.v.predict([image.crop(fixed)],self.prompt),d['dataset_name'])
   row={'contract':self.sig,'record_id':d['record_id'],'dataset':d['dataset_name'],'family':t['family'],'epsilon':t['epsilon'],'input_sha256':item['sha256'],'clean_target':d['clean_target'],'attack_target':d['frozen_attack_target'],'observation':o,'methods':{k:record(preds[k],d) for k in METHODS},'elapsed_seconds':time.monotonic()-start,'baseline_seconds':costs,'timing_note':'Shared calls: elapsed time is for the whole comparison, not any individual proposed method.'}
   write(path,row);print(f'{phase} {i+1}/{len(tasks)} {d["record_id"]} {t["family"]} eps={t["epsilon"]} recovery={preds["verified_recovery"]}',flush=True)
  self.report(phase,tau)
 def records(self,phase,complete=False):
  tasks=self.tasks(phase);rs=[]
  for t in tasks:
   p=self.out/phase/(t['key']+'.json')
   if not p.exists():
    if complete:raise ValueError(f'Incomplete {phase}: missing {t["key"]}')
    continue
   r=read(p);assert r['contract']==self.sig and r['record_id']==t['document']['record_id'] and r['input_sha256']==t['images'][0]['sha256'];rs.append(r)
  return rs
 def report(self,phase,tau):
  rows=self.records(phase);groups=collections.defaultdict(list)
  for row in rows:
   d=self.docs[row['record_id']]
   for k,v in choose(row['observation'],tau).items():row['methods'][k]=record(v,d)
   groups[(row['dataset'],row['family'],row['epsilon'])].append(row)
  summary={'completed':len(rows),'expected':len(self.tasks(phase)),'threshold':tau,'groups':[]}
  flat=[]
  for (ds,f,e),rs in sorted(groups.items()):
   for name in METHODS:
    counts={k:sum(r['methods'][name][k] for r in rs) for k in ['accepted','correct','targeted']}
    val=dict(dataset=ds,family=f,epsilon=e,method=name,n=len(rs),**counts)
    val['accepted_wrong']=counts['accepted']-counts['correct'];val['coverage']=counts['accepted']/len(rs);val['correct_rate']=counts['correct']/len(rs);val['target_rate']=counts['targeted']/len(rs)
    flat.append(val)
  summary['groups']=flat;write(self.out/f'{phase}_summary.json',summary)
  if flat:
   with (self.out/f'{phase}_summary.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
  print(f'Summary saved: {phase}, {len(rows)}/{summary["expected"]} cases',flush=True)
  if phase=='development':
   clean=[r for r in rows if r['family']=='clean'];gaps=[r['observation']['gap'] for r in clean if r['observation']['gap'] is not None]
   candidates=sorted(set([0.]+[g for g in gaps if g>=0]));table=[]
   for threshold in candidates:
    by={}
    for ds in sorted({r['dataset'] for r in clean}):
     rr=[r for r in clean if r['dataset']==ds];pred=[choose(r['observation'],threshold)['band_csev'] for r in rr]
     by[ds]={'n':len(rr),'accepted':sum(x is not None for x in pred),'correct':sum(x is not None and x==r['clean_target'] for x,r in zip(pred,rr))}
    table.append({'threshold':threshold,'clean_by_dataset':by})
   write(self.out/'development_thresholds.json',table)
 def freeze(self,tau):
  assert math.isfinite(tau) and tau>=0
  rows=self.records('development',True)
  # Fail closed if evaluation started with an unfrozen protocol.
  ids=sorted(self.ids('evaluation'),key=lambda x:digest([566,'adaptive',x]))[:10]
  snapshot=digest([sha(self.out/'development'/(t['key']+'.json')) for t in self.tasks('development')])
  frozen={'contract':self.sig,'threshold':tau,'development_hash':snapshot,'adaptive_ids':ids,'primary_method':'verified_recovery','note':'Recovery checks band/expanded-band agreement only, not a guarantee. Threshold applies to masking ablation only.'}
  p=self.out/'frozen.json'
  if p.exists():assert read(p)==frozen,'Already frozen; cannot silently change settings.'
  else:
   assert not list((self.out/'evaluation').glob('*.json'))
   write(p,frozen)
  self.report('development',tau);print('FROZEN. Evaluation and adaptive testing can now run.',flush=True)
 def frozen(self):
  f=read(self.out/'frozen.json');assert f['contract']==self.sig
  self.records('development',True)
  assert f['development_hash']==digest([sha(self.out/'development'/(t['key']+'.json')) for t in self.tasks('development')])
  return f
 def adaptive(self,pilot=False):
  from PIL import Image
  import numpy as np
  from csev.adaptive import search,gradient_step
  from csev.defense import answer_objective
  from models.donut import pil_to_rgb_tensor,rgb_tensor_to_pil
  import torch
  frozen=None if pilot else self.frozen();tau=0 if pilot else frozen['threshold'];self.model()
  ids=sorted(self.roles['collected_development'])[:1] if pilot else frozen['adaptive_ids']
  base=copy.deepcopy(self.c['settings']);base['adaptive'].update(steps=2 if pilot else 100,queries=5 if pilot else 100,refresh_interval=1 if pilot else 10)
  outer=self
  class Adapter:
   def __init__(self,method):self.method=method;self.prompt=outer.prompt
   def score(self,*args):return outer.v.score(*args)
   def __call__(self,im,ds):
    o=outer.observe(im,ds,scoring=self.method=='band_csev');pred=choose(o,tau)[self.method]
    return dict(full_prediction=pred,crop_agrees=pred is not None,gap=0. if pred is not None else None,box=o['box'],control=o['control'],raw_observation=o)
  for rid in ids:
   d=self.docs[rid];clean=Image.open(self.local(d['images'][0]['path'])).convert('RGB');array=np.asarray(clean)
   for method in POLICY['adaptive_methods']:
    for epsilon in ([4] if pilot else POLICY['adaptive_epsilons']):
     for restart in ([0] if pilot else range(2)):
      key=digest([rid,method,epsilon,restart]);directory=self.out/('adaptive_pilot' if pilot else 'adaptive')/key
      contract=digest([self.sig,method,rid,epsilon,restart,tau,base])
      adapter=Adapter(method)
      def step(image,obs):
       if method!='verified_recovery':
        settings=copy.deepcopy(base)
        if method=='band_agreement':settings['adaptive']['score_weight']=0
        return gradient_step(self.wrapper,image,array,d['frozen_attack_target'],self.prompt,obs,tau if method=='band_csev' else -1e6,epsilon,settings)
       # Target both views used by recovery; hold coordinates fixed during each step.
       # Actual verifier queries re-localize on current perturbed pixels.
       x=pil_to_rgb_tensor(image,self.wrapper.device).detach().requires_grad_(True)
       o=obs['raw_observation'];boxes=[b for b in [o['box'],o['expanded_box']] if b]
       views=boxes if boxes else [None];grad=torch.zeros_like(x);loss_sum=0.
       for box in views:
        if box:l,t,r,b=box;view=x[:,t:b,l:r]
        else:view=x
        obj=answer_objective(self.wrapper,rgb_tensor_to_pil(view),self.prompt,d['frozen_attack_target'],base['processor_tolerance']);obj.parity([view]);loss,_=obj.loss([view]);(g,)=torch.autograd.grad(loss,x)
        assert torch.isfinite(loss) and torch.isfinite(g).all();grad+=g.detach();loss_sum+=float(loss.detach());del obj,loss,g,view
       assert bool(grad.abs().max()>0),'Zero gradient'
       with torch.no_grad():updated=x-(epsilon/255)*.25*grad.sign()
       from csev.adaptive import project
       return project(np.asarray(rgb_tensor_to_pil(updated)),array,epsilon),loss_sum
      print('Adaptive',rid,method,epsilon,restart,flush=True)
      state=search(clean,d['dataset_name'],d['frozen_attack_target'],epsilon,restart,base,0.,adapter,step,directory/'state.pkl',contract)
      final=Image.open(directory/'selected.png').convert('RGB');o=self.observe(final,d['dataset_name']);preds=choose(o,tau)
      write(directory/'result.json',{'contract':contract,'record_id':rid,'method_attacked':method,'epsilon':epsilon,'restart':restart,'queries':state['queries'],'steps':state['step'],'runtime_seconds':state['runtime_seconds'],'linf_255':state['linf_255'],'image_sha256':sha(directory/'selected.png'),'methods':{m:record(preds[m],d) for m in POLICY['adaptive_methods']},'observation':o,'attack_note':'Bounded surrogate PGD plus verifier query search. Not an exhaustive adaptive attack or a robustness certificate.'})
      print('Adaptive saved:',directory,flush=True)
 def analysis(self):
  import numpy as np
  frozen=self.frozen();rows=self.records('evaluation',True);self.report('evaluation',frozen['threshold'])
  # Cluster bootstrap by document, keeping all its attack variants together.
  intervals=[]
  for family in ['clean','standard','uniform_eot','hardview']:
   rr=[r for r in rows if r['family']==family];ids=sorted({r['record_id'] for r in rr})
   if not ids:continue
   for method in METHODS:
    for metric in ['correct','targeted','accepted']:
     totals=np.array([[sum(r['methods'][method][metric] for r in rr if r['record_id']==rid),sum(r['record_id']==rid for r in rr)] for rid in ids],dtype=float)
     rng=np.random.default_rng(566);boot=totals[rng.integers(0,len(ids),size=(2000,len(ids)))].sum(axis=1);values=boot[:,0]/boot[:,1]
     intervals.append({'family':family,'method':method,'metric':metric,'documents':len(ids),'cases':len(rr),'estimate':totals[:,0].sum()/totals[:,1].sum(),'lower':float(np.quantile(values,.025)),'upper':float(np.quantile(values,.975))})
  write(self.out/'evaluation_cluster_intervals.json',intervals)
  adaptive=[];missing=[]
  for rid in frozen['adaptive_ids']:
   for method in POLICY['adaptive_methods']:
    for e in [4,8]:
     results=[]
     for restart in range(2):
      key=digest([rid,method,e,restart]);p=self.out/'adaptive'/key/'result.json'
      if p.exists():results.append(read(p))
      else:missing.append(key)
     if len(results)==2:adaptive.append({'record_id':rid,'method_attacked':method,'epsilon':e,'target_success_any_restart':any(r['methods'][method]['targeted'] for r in results)})
  write(self.out/'adaptive_summary.json',{'expected_runs':120,'completed_runs':120-len(missing),'missing_keys':missing,'per_document_budget':adaptive})
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  methods=['none','jpeg','band_answer','band_agreement','band_csev','verified_recovery'];fig,axes=plt.subplots(1,2,figsize=(12,4))
  for ax,fam,metric,title in [(axes[0],'clean','correct','Clean correct-answer rate'),(axes[1],None,'targeted','Saved-attack target acceptance')]:
   rs=[r for r in rows if (r['family']=='clean')==(fam=='clean')];vals=[100*sum(r['methods'][m][metric] for r in rs)/len(rs) for m in methods]
   ax.bar(methods,vals);ax.set_ylim(0,100);ax.set_ylabel('% of cases');ax.set_title(title);ax.tick_params(axis='x',rotation=45)
  fig.tight_layout();fig.savefig(self.out/'main_comparison.png',dpi=180);plt.close(fig)
  write(self.out/'compute.json',{'gpu':read(self.out/'environment.json')['gpu'],'recorded_comparison_seconds':sum(r['elapsed_seconds'] for phase in ['development','evaluation'] for r in self.records(phase)),'adaptive_seconds':sum(read(p)['runtime_seconds'] for p in (self.out/'adaptive').glob('*/result.json')),'note':'Recorded stage time excludes setup/model load and may undercount interrupted work; not whole-project GPU accounting.'})
  print('Analysis saved. Adaptive completeness:',120-len(missing),'/120',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','development','freeze','evaluation','adaptive-pilot','adaptive','analysis']);p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--threshold',type=float,default=0);p.add_argument('--limit',type=int);a=p.parse_args();s=Study(a.root,a.out)
 try:
  if a.command=='preflight':s.preflight()
  elif a.command in ['development','evaluation']:s.run(a.command,a.limit)
  elif a.command=='freeze':s.freeze(a.threshold)
  elif a.command in ['adaptive','adaptive-pilot']:s.adaptive(a.command=='adaptive-pilot')
  else:s.analysis()
 finally:
  if s.wrapper:s.wrapper.unload()
if __name__=='__main__':main()
