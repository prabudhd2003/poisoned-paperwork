"""Bounded direct attack on the frozen v4 resume contextual-recovery rule. New study; the defense is unchanged."""
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


# ---------------------------------------------------------------------------------------------
# Everything above this line is copied verbatim from the frozen v4 runner (see FROZEN_SOURCE_SHA256).
# Everything below is new: a bounded direct attack on the frozen contextual-recovery rule.
# ---------------------------------------------------------------------------------------------
FROZEN_SOURCE_SHA256='b09b577474d73420f4f6891bd70ba4a486bbed1c40005d95439866be1e50b844'
ATTACK={'version':'resume_direct_recovery_v1','documents':10,'epsilons':[8,16],'restarts':[0,1],
        'steps':100,'refresh_interval':10,'queries':40,'step_fraction':0.25,'seed':566,
        'wall_seconds':14400,'run_margin_seconds':1500,'early_stop_on_accepted_target':True,
        'processor_tolerance':{'max':1e-6,'mean':1e-8,'surrogate_mean':0.05},
        'objective':'cross-entropy of the target label tokens on the localized views and on the expanded views; boxes held fixed between verifier queries',
        'order':'all restart-0 runs first, then restart-1; inside a restart by frozen document order, then epsilon'}

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def project(array,clean,eps):
    import numpy as np
    return np.clip(np.rint(array),np.maximum(0,clean.astype(float)-eps),np.minimum(255,clean.astype(float)+eps)).astype('uint8')

def rank_of(obs,target):
    return (int(obs['prediction']==target),int(obs['crop']==target)+int(obs['wide']==target))

def run_attack(clean_pages,target,eps,restart,seed_key,observe_fn,step_fn,settings):
    """clean_pages: uint8 HxWx3 arrays. observe_fn(arrays)->observation. step_fn(arrays,observation)->(arrays,loss)."""
    import numpy as np
    rng=np.random.default_rng(int(seed_key[:8],16)+restart)
    refreshes=(settings['steps']+settings['refresh_interval']-1)//settings['refresh_interval']
    assert settings['queries']>=refreshes+2,'query budget cannot cover refreshes and final verification'
    current=[project(c.astype(float)+rng.integers(-eps,eps+1,c.shape),c,eps) for c in clean_pages]
    state={'queries':0,'step':0,'trace':[],'best':None,'best_rank':None}
    def query(arrays,phase,loss=None,strict=False):
        obs=observe_fn(arrays);r=rank_of(obs,target);state['queries']+=1
        state['trace'].append({'phase':phase,'step':state['step'],'query':state['queries'],'prediction':obs['prediction'],
            'crop':obs['crop'],'wide':obs['wide'],'localization_valid':obs['valid'],'crops':len(obs['boxes']),'surrogate_loss':loss})
        better=state['best_rank'] is None or (r>state['best_rank'] if strict else r>=state['best_rank'])
        if better:state['best']=[a.copy() for a in arrays];state['best_rank']=r
        return obs
    usable=query(current,'initial')
    stop=lambda:settings['early_stop_on_accepted_target'] and state['best_rank'][0]==1
    while state['step']<settings['steps'] and not stop():
        current,loss=step_fn(current,usable)
        current=[project(a,c,eps) for a,c in zip(current,clean_pages)]
        state['step']+=1
        if state['step']%settings['refresh_interval']==0 or state['step']==settings['steps']:
            obs=query(current,'gradient',loss)
            if obs['boxes']:usable=obs  # keep the last usable regions when localization fails
    # Reserve the last verifier query for the saved, reloaded pages.
    while state['queries']<settings['queries']-1 and not stop():
        cand=[a.copy() for a in state['best']];j=int(rng.integers(0,len(cand)));h,w=cand[j].shape[:2]
        size=max(1,round(min(h,w)*(0.15 if state['queries']<settings['queries']//2 else 0.05)))
        y=int(rng.integers(0,h-size+1));x=int(rng.integers(0,w-size+1));signs=rng.choice([-1,1],size=(1,1,3))
        patch=clean_pages[j][y:y+size,x:x+size]
        cand[j][y:y+size,x:x+size]=project(patch.astype(float)+eps*signs,patch,eps)
        query(cand,'query_search',strict=True)
    return state

def observe(model,pages,question,norm):
    """Frozen v4 recovery rule on the current pages (same order of calls as the frozen runner)."""
    regions=[];boxes=[];valid=True
    for j,im in enumerate(pages):
        proc=model.processor.image_processor(images=[im],return_tensors='pt');grid=proc['image_grid_thw'].tolist();assert len(grid)==1
        _,gh,gw=grid[0];patch=model.processor.image_processor.patch_size;rs=(gw*patch,gh*patch);del proc
        region=localize(model,im,rs,grid);region['page']=j;regions.append(region)
        if region['status']=='invalid':
            valid=False;continue
        for b,e in region['pairs']:boxes.append((j,tuple(b),tuple(e)))
    raw1=raw2=None
    if valid and 0<len(boxes)<=POLICY['max_crops_per_document']:
        raw1=model.predict([pages[j].crop(b) for j,b,_ in boxes],question)
        raw2=model.predict([pages[j].crop(e) for j,_,e in boxes],question)
    crop,wide=norm(raw1),norm(raw2)
    return {'regions':regions,'boxes':boxes if valid else [],'valid':valid,'raw_localized':raw1,'raw_expanded':raw2,
            'crop':crop,'wide':wide,'prediction':crop if crop is not None and crop==wide else None}

def summarize(out,plan):
    results=[read(p) for p in sorted((out/'runs').glob('*/result.json'))]
    docs=plan['documents'];rows=[]
    for eps in ATTACK['epsilons']:
        done={};hit={}
        for r in results:
            if r['epsilon_255']!=eps:continue
            done.setdefault(r['record_id'],set()).add(r['restart'])
            m=r['methods']['contextual_recovery'];hit[r['record_id']]=hit.get(r['record_id'],False) or bool(m['accepted'] and m['targeted'])
        attempted=[d for d in docs if done.get(d['record_id'])]
        both=[d for d in docs if done.get(d['record_id'])==set(ATTACK['restarts'])]
        eligible=[d for d in attempted if d['clean_recovery']=='correct']
        runs=[r for r in results if r['epsilon_255']==eps]
        rows.append({'epsilon_255':eps,'documents_planned':len(docs),'documents_with_any_restart':len(attempted),'documents_with_all_restarts':len(both),
            'success_documents_all':sum(hit[d['record_id']] for d in attempted),'eligible_documents':len(eligible),
            'success_documents_eligible':sum(hit[d['record_id']] for d in eligible),'runs_completed':len(runs),
            'runs_succeeded':sum(bool(r['methods']['contextual_recovery']['accepted'] and r['methods']['contextual_recovery']['targeted']) for r in runs),
            'runs_rejected':sum(not r['methods']['contextual_recovery']['accepted'] for r in runs),
            'runs_correct':sum(bool(r['methods']['contextual_recovery']['correct']) for r in runs),
            'saved_standard_attack_undefended_targeted':sum(bool(d['saved_standard'][str(eps)]['undefended_targeted']) for d in docs),
            'success_ids':[d['record_id'] for d in attempted if hit[d['record_id']]]})
    planned=len(docs)*len(ATTACK['epsilons'])*len(ATTACK['restarts'])
    summary={'complete':len(results)==planned,'runs_completed':len(results),'runs_planned':planned,'groups':rows,
        'note':'Bounded direct attack on frozen v4 resume contextual recovery. First ten reserved resumes with saved attacks in the frozen order, not selected on outcomes. Missing runs are not attack failures. A document counts as a success if any completed restart yields an accepted target on the saved, reloaded pages. Not a robustness guarantee.'}
    write(out/'summary.json',summary)
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='success_ids']);w.writeheader()
        for row in rows:w.writerow({k:v for k,v in row.items() if k!='success_ids'})
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.out)
    hashes=read(root/'hashes.json')
    for name,h in hashes.items():assert sha(root/name)==h,name
    c=read(root/'config.json');plan=read(root/'attack_plan.json');docs=plan['documents']
    assert len(docs)==ATTACK['documents'] and len({d['record_id'] for d in docs})==len(docs)
    assert {d['record_id'] for d in docs}<=set(c['reserved_ids']) and not {d['record_id'] for d in docs}&set(c['development_ids'])
    for d in docs:assert d['attack_target'] in c['labels'] and d['attack_target']!=d['clean_target']
    for k,v in c['versions'].items():assert importlib.metadata.version(k)==v,(k,v)
    assert platform.python_version()==c['python']
    os.environ['HF_HOME']=str(root/'hf_cache')
    import numpy as np,torch
    from PIL import Image
    sys.path.insert(0,str(root/'src'))
    from attacks.qwen import Objective,clear_rope
    from models.donut import rgb_tensor_to_pil
    from models.qwen import QwenVL
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    contract={'inputs':sha(root/'hashes.json'),'plan':sha(root/'attack_plan.json'),'frozen_rule_source_sha256':FROZEN_SOURCE_SHA256,'script':sha(__file__),
              'policy':POLICY,'prompt':LOC,'attack':ATTACK,'versions':c['versions'],'model':c['model'],'question':c['question']}
    out.mkdir(parents=True,exist_ok=True)
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract,'Changed inputs/environment/code: use a new output directory.'
    else:write(out/'contract.json',contract)
    sig=digest(contract)
    sessions=read(out/'sessions.json') if (out/'sessions.json').exists() else []
    sessions.append({'started':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'gpu':torch.cuda.get_device_name(),'cuda':torch.version.cuda});write(out/'sessions.json',sessions)
    torch.manual_seed(ATTACK['seed'])
    model=QwenVL(device=torch.device('cuda'),**c['model'])
    raw_predict=model.predict
    def predict(images,prompt):
        clear_rope(model.model)  # gradient forwards leave cached rotary offsets
        return raw_predict(images,prompt)
    model.predict=predict
    norm=lambda x:x.strip().lower() if isinstance(x,str) and x.strip().lower() in c['labels'] else None
    question=c['question'];tokenizer=model.processor.tokenizer
    observe_fn=lambda arrays:observe(model,[Image.fromarray(x) for x in arrays],question,norm)
    def make_step(target,eps,diagnostics):
        cache={}
        def objective(views):
            obj=Objective(model,[rgb_tensor_to_pil(v) for v in views],question,target,ATTACK['processor_tolerance'])
            ids=tokenizer.encode(target,add_special_tokens=False);assert ids
            tail=torch.tensor([ids],device=model.device,dtype=torch.long);prefix=obj.prefix['input_ids']
            obj.ids=torch.cat([prefix,tail],dim=1)
            obj.positions=torch.arange(prefix.shape[1]-1,obj.ids.shape[1]-1,device=model.device);obj.target_tensor=tail.flatten()
            with torch.no_grad():stats=obj.parity(views)
            diagnostics['surrogate_mean_max']=max(diagnostics['surrogate_mean_max'],stats['processor_surrogate_mean_abs_difference'])
            return obj
        def step(arrays,obs):
            xs=[torch.from_numpy(x.astype('float32')/255).permute(2,0,1).contiguous().to(model.device).requires_grad_(True) for x in arrays]
            boxes=obs['boxes']
            groups=[[(j,b) for j,b,_ in boxes],[(j,e) for j,_,e in boxes]] if boxes else [[(j,None) for j in range(len(arrays))]]
            diagnostics['steps_on_regions' if boxes else 'steps_on_full_pages']+=1
            grads=[torch.zeros_like(x) for x in xs];total=0.
            for group in groups:
                views=[xs[j] if box is None else xs[j][:,box[1]:box[3],box[0]:box[2]] for j,box in group]
                key=tuple((j,box,tuple(v.shape)) for (j,box),v in zip(group,views))
                if key not in cache:
                    if len(cache)>=4:cache.clear()
                    cache[key]=objective(views)
                loss,_=cache[key].loss(views)
                used=sorted({j for j,_ in group})
                gs=torch.autograd.grad(loss,[xs[j] for j in used])
                assert torch.isfinite(loss)
                for j,g in zip(used,gs):
                    assert torch.isfinite(g).all();grads[j]+=g.detach()
                total+=float(loss.detach());del loss,gs,views
            assert any(bool(g.abs().max()>0) for g in grads),'Zero gradient'
            with torch.no_grad():
                updated=[np.asarray(rgb_tensor_to_pil(x-(eps/255)*ATTACK['step_fraction']*g.sign())) for x,g in zip(xs,grads)]
            del xs,grads
            return updated,total
        return step
    schedule=[(restart,d,eps) for restart in ATTACK['restarts'] for d in docs for eps in ATTACK['epsilons']]
    started=time.monotonic()
    try:
        for index,(restart,d,eps) in enumerate(schedule):
            key=digest([d['record_id'],eps,restart]);directory=out/'runs'/key;dest=directory/'result.json'
            if dest.exists():
                print(f'DONE {index+1}/{len(schedule)} {d["record_id"]} eps={eps} restart={restart}',flush=True);continue
            if time.monotonic()-started>ATTACK['wall_seconds']-ATTACK['run_margin_seconds']:
                print('Session budget reached before starting the next run. Rerun the notebook to continue.',flush=True);break
            for im in d['images']:assert sha(root/im['path'])==im['sha256'],im['path']
            clean=[np.asarray(Image.open(root/x['path']).convert('RGB')) for x in d['images']]
            print(f'ATTACK {index+1}/{len(schedule)} {d["record_id"]} eps={eps} restart={restart} target={d["attack_target"]}',flush=True)
            t0=time.monotonic();diagnostics={'surrogate_mean_max':0.,'steps_on_regions':0,'steps_on_full_pages':0}
            state=run_attack(clean,d['attack_target'],eps,restart,digest([sig,d['record_id'],eps,restart]),observe_fn,make_step(d['attack_target'],eps,diagnostics),ATTACK)
            directory.mkdir(parents=True,exist_ok=True);images=[];reloaded=[];linf=0
            for j,(best,original) in enumerate(zip(state['best'],clean)):
                path=directory/f'page_{j:02d}.png';Image.fromarray(best).save(path)
                with Image.open(path) as im:arr=np.asarray(im.convert('RGB')).copy()
                assert arr.shape==original.shape
                linf=max(linf,int(np.abs(arr.astype(np.int16)-original.astype(np.int16)).max()))
                reloaded.append(arr);images.append({'path':str(path.relative_to(out)),'sha256':sha(path)})
            assert linf<=eps,'pixel budget exceeded'
            final=observe_fn(reloaded);queries=state['queries']+1
            fullraw=model.predict([Image.fromarray(x) for x in reloaded],question);full=norm(fullraw)
            preds={'none':full,'localized_only':final['crop'],'full_region_agreement':full if full is not None and full==final['crop'] else None,'contextual_recovery':final['prediction']}
            result={'contract':sig,'record_id':d['record_id'],'clean_target':d['clean_target'],'attack_target':d['attack_target'],'method_attacked':'contextual_recovery',
                'epsilon_255':eps,'restart':restart,'steps':state['step'],'queries':queries,'runtime_seconds':time.monotonic()-t0,'linf_255':linf,'images':images,
                'candidate_rank':list(state['best_rank']),'saved_png_matches_candidate':int(final['prediction']==d['attack_target'])==state['best_rank'][0],
                'methods':{k:{'prediction':v,'correct':v is not None and v==d['clean_target'],'accepted':v is not None,'targeted':v is not None and v==d['attack_target']} for k,v in preds.items()},
                'raw_answers':{'full':fullraw,'localized':final['raw_localized'],'expanded':final['raw_expanded']},'regions':final['regions'],'crop_count':len(final['boxes']),
                'localization_valid':final['valid'],'diagnostics':diagnostics,'trace':state['trace'],'clean_recovery':d['clean_recovery'],
                'attack_note':'Bounded surrogate PGD on localized and expanded views plus verifier query search. Boxes fixed between verifier queries. Not an exhaustive adaptive attack or a robustness certificate.'}
            write(dest,result)
            print(f'SAVED {d["record_id"]} eps={eps} restart={restart} recovery={preds["contextual_recovery"]} full={full} steps={state["step"]} queries={queries} seconds={result["runtime_seconds"]:.0f}',flush=True)
            torch.cuda.empty_cache()
        summarize(out,plan)
    finally:model.unload()
if __name__=='__main__':main()
