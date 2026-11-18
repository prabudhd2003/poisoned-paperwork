"""Development-only Donut geometry comparison; never selects a winning rule."""
import argparse, collections, csv, hashlib, json, os, sys, time
from pathlib import Path

VARIANTS = {
    'smaller_expansion': (.12, .03, .01),
    'taller_18': (.18, .03, .03),
    'taller_24': (.24, .04, .03),
}

def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False));t.replace(p)
def boxes(tight,size,params):
    if tight is None:return None,None
    w,h=size;_,top,_,bottom=tight;minimum,margin,extra=params
    height=max(round(minimum*h),bottom-top+2*round(margin*h))
    if height<=0 or height>.35*h:return None,None
    top=max(0,min(h-height,round((top+bottom-height)/2)))
    first=(0,top,w,top+height);pad=round(extra*h)
    second=(0,max(0,top-pad),w,min(h,top+height+pad))
    if second[3]-second[1]>.45*h:return first,None
    return first,second

def summarize(rows,out):
    groups=collections.defaultdict(list)
    for r in rows:
        for ds in [r['dataset'],'ALL']:
            family='clean' if r['family']=='clean' else 'saved_transfer'
            for name,v in r['methods'].items():groups[(ds,family,name)].append(v)
    flat=[]
    for (ds,fam,name),vv in sorted(groups.items()):
        n=len(vv);c=sum(v['correct'] for v in vv);a=sum(v['accepted'] for v in vv)
        flat.append(dict(dataset=ds,family=fam,method=name,n=n,correct=c,wrong=a-c,rejected=n-a,
                         correct_rate=c/n,accepted_error_rate=(a-c)/n,coverage=a/n,
                         targeted=sum(v['targeted'] for v in vv)))
    with (out/'geometry_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    write(out/'summary.json',{'completed_cases':len(rows),'expected_cases':210,'complete':len(rows)==210,
        'note':'Development only. Saved attacks target Qwen, not Donut. No method automatically selected.', 'groups':flat})
    for r in flat:
        if r['dataset']=='ALL':print(r,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--baseline',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root,baseline,out=map(Path,[a.root,a.baseline,a.out])
    assert out.resolve()!=baseline.resolve() and not out.resolve().is_relative_to(baseline.resolve()),'Use separate output folder.'
    assert not (baseline/'frozen.json').exists(),'Baseline is frozen. Review before further development.'
    assert not list((baseline/'evaluation').glob('*.json')),'Reserved evaluation already started; do not silently tune.'
    roles=read(root/'data_roles.json');ids=set(roles['public_development']+roles['collected_development'])
    assert len(ids)==75 and not ids & set(roles['collected_reserved_evaluation'])
    plan=read(root/'plan.json');docs={d['record_id']:d for phase in ['development','evaluation'] for d in plan[phase] if d['record_id'] in ids}
    tasks={}
    for rid in sorted(ids):
        d=docs[rid]
        for item in [{'family':'clean','epsilon':0,'images':d['images']}]+d['attacks']:
            tasks[(rid,item['family'],item['epsilon'])]=item
    files=sorted((baseline/'development').glob('*.json'));old=[read(f) for f in files]
    assert len(old)==len(tasks)==210
    assert {(r['record_id'],r['family'],r['epsilon']) for r in old}==set(tasks)
    bc=read(baseline/'contract.json');sig=hashlib.sha256(json.dumps(bc,sort_keys=True).encode()).hexdigest()
    for r in old:assert r['contract']==sig
    assert sha(root/'source.tar')==bc['source_archive'] and sha(root/'plan.json')==bc['plan']
    assert sha(root/'csev_receipts_v1.json')==bc['config'] and sha(root/'data_roles.json')==bc['roles']
    contract={'script':sha(__file__),'variants':VARIANTS,'baseline_contract':sig,'baseline_records':{f.name:sha(f) for f in files},
              'scope':'210 development cases only; saved per-image localization reused; no reserved evaluation'}
    # JSON round-trip normalizes tuples for restart comparisons.
    contract=json.loads(json.dumps(contract));out.mkdir(parents=True,exist_ok=True)
    if (out/'contract.json').exists():assert read(out/'contract.json')==contract,'Diagnostic changed; use a new output folder.'
    else:write(out/'contract.json',contract)
    sys.path.insert(0,str(root/'source/src'))
    import torch, platform, importlib.metadata
    from PIL import Image
    from models.donut import DonutDocVQA
    from normalization import normalize_receipt_total
    env=read(baseline/'environment.json')
    for k,v in env['versions'].items():assert (platform.python_version() if k=='python' else importlib.metadata.version(k))==v,(k,v)
    assert torch.cuda.is_available() and torch.cuda.get_device_name()==env['gpu']
    write(out/'environment.json',env)
    os.environ['HF_HOME']=str(root/'hf_cache');torch.manual_seed(566)
    model=DonutDocVQA(device=torch.device('cuda'),**bc['policy']['reader'])
    prompt=read(root/'csev_receipts_v1.json')['legacy']['prompts']['receipt']
    rows=[]
    try:
        for i,(f,r) in enumerate(zip(files,old),1):
            dest=out/'cases'/f.name
            if dest.exists():rows.append(read(dest));print(f'RESUME {i}/210',flush=True);continue
            item=tasks[(r['record_id'],r['family'],r['epsilon'])]['images'][0]
            imagepath=root/'staging'/str(item['path']).lstrip('/')
            assert sha(imagepath)==item['sha256']==r['input_sha256']
            image=Image.open(imagepath).convert('RGB');o=r['observation'];cache={}
            for key,answer in [('box','crop'),('expanded_box','wide')]:
                if o.get(key):cache[tuple(o[key])]={'prediction':o.get(answer),'raw':o.get(answer+'_raw'),'reused':True}
            start=time.monotonic();methods=dict(r['methods']);detail={}
            for name,params in VARIANTS.items():
                pair=boxes(o.get('tight_box'),image.size,params);answers=[]
                for box in pair:
                    if box is None:answers.append(None);continue
                    if box not in cache:
                        raw=model.predict([image.crop(box)],prompt)
                        cache[box]={'prediction':normalize_receipt_total(raw,r['dataset']),'raw':raw,'reused':False}
                    answers.append(cache[box]['prediction'])
                pred=answers[0] if answers[0] is not None and answers[0]==answers[1] else None
                methods[name]={'prediction':pred,'accepted':pred is not None,'correct':pred is not None and pred==r['clean_target'],
                               'targeted':pred is not None and pred==r['attack_target']}
                detail[name]={'boxes':pair,'answers':answers}
            row={k:r[k] for k in ['record_id','dataset','family','epsilon','clean_target','attack_target','input_sha256']}
            row.update(methods=methods,variants=detail,crops=[dict(box=k,**v) for k,v in cache.items()],elapsed_seconds=time.monotonic()-start)
            write(dest,row);rows.append(row)
            print(f'{i}/210 {r["record_id"]} {r["family"]} eps={r["epsilon"]} '+str({k:methods[k]['prediction'] for k in VARIANTS}),flush=True)
            if i%10==0:summarize(rows,out)
        summarize(rows,out)
    finally:model.unload()
if __name__=='__main__':main()
