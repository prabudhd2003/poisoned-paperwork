"""Clean development only: multi-page contextual education views."""
import argparse,hashlib,json,math,sys,time,os,platform,importlib.metadata,csv
from pathlib import Path
LOC='Find the education and qualifications sections on this resume page. Locate whole sections, including their headings, degree or diploma names, institutions, dates and any continuation lines. Include stated qualifications even if the page does not explicitly say completed or awarded. Include education entries outside a heading too. Do not infer qualifications from job titles, work experience or skills. Group nearby lines in the same section into one box rather than returning a box per word or line. Return only JSON in this exact format: {"boxes": [[left, top, right, bottom]]}. Use pixel coordinates on the displayed image of width {w} and height {h}. Return {"boxes": []} only if no education or qualification entries are visible. Return at most four section boxes. Do not include explanations or transcribed text.'
POLICY={'version':'resume_development_retry_v4','localization_attempts':2,'minimum_height':.25,'margin':.05,'expansion':.05,'max_height':.70,'expanded_max_height':.85,'max_regions_per_page':4,'max_raw_boxes':16,'max_crops_per_document':8,'localization_tokens':768,'fixed_band':[.2,.8]}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def parse(raw,processed,original):
 import re
 s=raw.strip()
 # Accept one fenced JSON response, even when preceded by explanation.
 fences=re.findall(r"```(?:json)?\s*(.*?)```",s,flags=re.S|re.I)
 if fences:
  if len(fences)!=1:raise ValueError('Ambiguous multiple JSON blocks')
  s=fences[0].strip()
 empty_phrases={
  'there are none.',
  'there are no regions indicating completed education qualifications or awarded degrees on this resume page.',
  'there are no regions stating completed education qualifications or awarded degrees on this resume page.'}
 if s.lower() in empty_phrases:return []
 # Only repair trailing commas outside strings; do not extract arbitrary numbers.
 cleaned=[];quoted=False;escaped=False
 for i,ch in enumerate(s):
  if quoted:
   cleaned.append(ch)
   if escaped:escaped=False
   elif ch=='\\':escaped=True
   elif ch=='"':quoted=False
  else:
   if ch=='"':quoted=True
   if ch==',' and s[i+1:].lstrip().startswith((']', '}')):continue
   cleaned.append(ch)
 obj=json.loads(''.join(cleaned))
 def unpack(x):
  if isinstance(x,dict):
   schemas=[k for k in ('boxes','bbox_2d') if k in x]
   named=all(k in x for k in ('left','top','right','bottom'))
   if len(schemas)+int(named)!=1:raise ValueError('Unknown or ambiguous box schema')
   return unpack(x[schemas[0]]) if schemas else [[x[k] for k in ('left','top','right','bottom')]]
  if isinstance(x,list):
   if len(x)==4 and all(isinstance(v,(int,float)) and not isinstance(v,bool) for v in x):return [x]
   out=[]
   for item in x:out.extend(unpack(item))
   return out
  raise ValueError('Invalid box representation')
 vals=unpack(obj)
 # Bound raw output, but merge line boxes before limiting final regions.
 if len(vals)>POLICY['max_raw_boxes']:raise ValueError('Too many raw boxes')
 pw,ph=processed;w,h=original;out=[]
 for v in vals:
  if len(v)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in v):raise ValueError('Invalid coordinates')
  l,t,r,b=v
  if l>=r or t>=b:raise ValueError('Box endpoints inverted or zero area')
  if not (0<=l and r<=pw and 0<=t and b<=ph):raise ValueError('Box outside processed image')
  top,bottom=round(t*h/ph),round(b*h/ph)
  if top>=bottom:raise ValueError('Box collapses when mapped to original image')
  out.append((0,top,w,bottom))
 return out

def views(boxes,size):
 w,h=size;intervals=[]
 for _,t,_,b in boxes:
  height=max(round(.25*h),b-t+2*round(.05*h))
  if height>.70*h:return None
  top=max(0,min(h-height,round((t+b-height)/2)));intervals.append((top,top+height))
 merged=[]
 for t,b in sorted(intervals):
  if merged and t<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
  else:merged.append((t,b))
 if len(merged)>POLICY['max_regions_per_page']:return None
 pairs=[]
 for t,b in merged:
  t2=max(0,t-round(.05*h));b2=min(h,b+round(.05*h))
  if b-t>.70*h or b2-t2>.85*h:return None
  pairs.append(((0,t,w,b),(0,t2,w,b2)))
 return pairs

def localize(model,im,rs,grid):
 prompt=LOC.replace('{w}',str(rs[0])).replace('{h}',str(rs[1]))
 attempts=[];previous_tokens=model.max_new_tokens
 try:
  model.max_new_tokens=POLICY['localization_tokens']
  for attempt in range(POLICY['localization_attempts']):
   query=prompt
   if attempt:
    query+=' Your previous localization output was invalid. Inspect the page again and return fresh section boxes. Every box must satisfy 0 <= left < right <= '+str(rs[0])+' and 0 <= top < bottom <= '+str(rs[1])+'. Use positive-area boxes, group nearby education lines, and output only the requested JSON. Do not guess any qualification.'
   raw=model.predict([im],query)
   assert model.last_input_info['image_grid_thw']==grid
   try:
    boxes=parse(raw,rs,im.size);pairs=views(boxes,im.size)
    if pairs is None:raise ValueError('Merged regions exceed size or count limits')
    attempts.append({'raw':raw,'status':'valid'})
    return {'raw':raw,'processed_size':rs,'boxes':boxes,'pairs':pairs,'status':'ok' if pairs else 'no_education','attempts':attempts}
   except (ValueError,KeyError,TypeError) as exc:
    attempts.append({'raw':raw,'status':'invalid','error':str(exc)})
  return {'raw':attempts[-1]['raw'],'processed_size':rs,'status':'invalid','error':attempts[-1]['error'],'attempts':attempts}
 finally:model.max_new_tokens=previous_tokens

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.out)
 hashes=read(root/'hashes.json')
 for name,h in hashes.items():assert sha(root/name)==h,name
 c=read(root/'config.json');docs=read(root/'attack_cases.json')
 assert len(docs)==75 and {d['record_id'] for d in docs}==set(c['development_ids'])
 assert not set(c['development_ids'])&set(c['reserved_ids'])
 for k,v in c['versions'].items():assert importlib.metadata.version(k)==v,(k,v)
 assert platform.python_version()==c['python']
 import torch
 from PIL import Image
 sys.path.insert(0,str(root));from qwen_adapter import QwenVL
 assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
 contract={'inputs':sha(root/'hashes.json'),'attack_cases':sha(root/'attack_cases.json'),'frozen_v4_source_sha256':'19f651e1c296e8de53795b3a8a35e189541b246c4022a7a3fcc9121e1a401ca2','script':sha(__file__),'policy':POLICY,'prompt':LOC,'gpu':torch.cuda.get_device_name(),'versions':c['versions'],'cuda':torch.version.cuda}
 out.mkdir(parents=True,exist_ok=True)
 if (out/'contract.json').exists():assert read(out/'contract.json')==contract,'Changed inputs/environment/code: use a new output directory.'
 else:write(out/'contract.json',contract)
 os.environ['HF_HOME']=str(root/'hf_cache');torch.manual_seed(566)
 model=QwenVL(device=torch.device('cuda'),**c['model']);norm=lambda x:x.strip().lower() if isinstance(x,str) and x.strip().lower() in c['labels'] else None
 results=[]
 try:
  for i,d in enumerate(docs):
   dest=out/'cases'/f'{i:02d}.json'
   if dest.exists():results.append(read(dest));print(f'RESUME {i+1}/{len(docs)}',flush=True);continue
   for im in d['images']:assert sha(root/im['path'])==im['sha256'],im['path']
   pages=[Image.open(root/x['path']).convert('RGB') for x in d['images']];start=time.monotonic()
   fullraw=model.predict(pages,c['question']);full=norm(fullraw)
   fixedraw=model.predict([im.crop((0,round(.2*im.height),im.width,round(.8*im.height))) for im in pages],c['question'])
   regions=[];bands=[];expanded=[];valid=True
   for j,im in enumerate(pages):
    # Obtain the exact processor grid, then ask explicitly for that coordinate system.
    proc=model.processor.image_processor(images=[im],return_tensors='pt');grid=proc['image_grid_thw'].tolist();assert len(grid)==1
    _,gh,gw=grid[0];patch=model.processor.image_processor.patch_size;rs=(gw*patch,gh*patch);del proc
    region=localize(model,im,rs,grid);region['page']=j;regions.append(region)
    if region['status']=='invalid':
     valid=False;continue
    for k,(b,e) in enumerate(region['pairs']):
     bands.append(im.crop(b));expanded.append(im.crop(e))
     target=out/'crops'/f'{i:02d}_{j:02d}_{k:02d}.png';target.parent.mkdir(exist_ok=True);im.crop(b).save(target)
     im.crop(e).save(target.with_name(target.stem+'_expanded.png'))
   raw1=raw2=None
   if valid and 0<len(bands)<=POLICY['max_crops_per_document']:
    raw1=model.predict(bands,c['question']);raw2=model.predict(expanded,c['question'])
   crop,wide=norm(raw1),norm(raw2)
   preds={'none':full,'fixed_middle':norm(fixedraw),'localized_only':crop,'full_region_agreement':full if full is not None and full==crop else None,'contextual_recovery':crop if crop is not None and crop==wide else None}
   row={'case_id':d['case_id'],'family':d['family'],'epsilon_255':d['epsilon_255'],'attack_target':d['attack_target'],'record_id':d['record_id'],'clean_target':d['clean_target'],'page_count':len(pages),'regions':regions,'crop_count':len(bands),'localization_valid':valid,
        'raw_answers':{'full':fullraw,'fixed':fixedraw,'localized':raw1,'expanded':raw2},'methods':{k:{'prediction':v,'correct':v is not None and v==d['clean_target'],'accepted':v is not None,'targeted':d['attack_target'] is not None and v==d['attack_target']} for k,v in preds.items()},'elapsed_seconds':time.monotonic()-start}
   write(dest,row);results.append(row);print(f'{i+1}/{len(docs)} {d["case_id"]} pages={len(pages)} regions={len(bands)} '+str(preds),flush=True)
  summary=[]
  for family,epsilon in [('clean',0)]+[('standard',e) for e in (2,4,8,16)]+[('standard','all')]:
   group=[r for r in results if r['family']==family and (epsilon=='all' or r['epsilon_255']==epsilon)]
   for method in results[0]['methods']:
    correct=sum(r['methods'][method]['correct'] for r in group);accepted=sum(r['methods'][method]['accepted'] for r in group)
    targeted=sum(r['methods'][method]['targeted'] for r in group);n=len(group)
    summary.append({'family':family,'epsilon_255':epsilon,'method':method,'n':n,'correct':correct,'wrong':accepted-correct,'rejected':n-accepted,'targeted':targeted,'correct_rate':correct/n,'coverage':accepted/n,'accepted_error_rate':(accepted-correct)/n,'target_rate':targeted/n})
  write(out/'summary.json',{'complete':len(results)==75,'cases':len(results),'method_records':len(results)*5,'groups':summary,'note':'Same 15 development documents. 60 standard saved attacks were not optimized against recovery. Repeated variants are not independent documents. Reserved resumes excluded.'})
  with (out/'summary.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
  print(json.dumps(summary,indent=2))
 finally:model.unload()
if __name__=='__main__':main()
