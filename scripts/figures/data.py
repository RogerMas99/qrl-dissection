import re, csv, gzip, collections, os, pathlib, numpy as np
# Input locations (the only change with respect to the thesis copy of this script). Override with the environment variables
# FIG_TABLES (dossier v3 + per_seed_metrics.csv + curves_*.csv; default: <repo>/handoff_memoria/) and FIG_PAPER_CURVES.
REPO=pathlib.Path(__file__).resolve().parents[2]
H=os.environ.get('FIG_TABLES', str(REPO/'handoff_memoria'))+'/'
PAPER_CURVES=os.environ.get('FIG_PAPER_CURVES', str(REPO/'data'/'paper_ppo_curves.csv.gz'))
def parse_dossier():
    t={}
    txt=open(H+'01_DOSSIER_RESULTADOS_v3.md').read()
    sec=txt[txt.index('## 3. Per-experiment tables'):txt.index('## 4. Pre-specified')]
    for line in sec.splitlines():
        m=re.match(r"\| `([^`]+)` \| (on|off) \| ([^|]+) \| (\d+) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \| ([\d.]+) \(([\d.]+)-([\d.]+)\) \|",line)
        if not m: continue
        arm,fix,par,n=m.group(1),m.group(2),m.group(3).strip(),int(m.group(4))
        k=(arm,fix)
        if k in t: continue
        t[k]=dict(params=par,n=n,final=float(m.group(5)),final_ci=(float(m.group(6)),float(m.group(7))),
                  greedy=float(m.group(8)),greedy_ci=(float(m.group(9)),float(m.group(10))))
    return t
def per_seed():
    d=collections.defaultdict(dict)
    for r in csv.DictReader(open(H+'per_seed_metrics.csv')):
        if r['exp']=='exp01_60k': continue
        key=(r['arm']+('__'+r['tag'] if r['tag'] else ''),r['fix'])
        d[key][int(r['seed'])]=(float(r['final']),float(r['greedy']))
    return d
def curves(fname,col):
    d=collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(H+fname)):
        if r['exp']=='exp01_60k': continue
        key=(r['arm']+('__'+r['tag'] if r['tag'] else ''),r['fix'])
        d[key][int(r['seed'])].append((int(r['global_step']),float(r[col])))
    return d
def iqm(x):
    x=np.sort(np.asarray(x,float)); n=len(x)
    lo=int(np.floor(0.25*n)); hi=int(np.ceil(0.75*n))
    return x[lo:hi].mean()
def boot(x,f=iqm,B=2000,seed=0):
    rng=np.random.default_rng(seed); x=np.asarray(x,float)
    s=[f(rng.choice(x,len(x))) for _ in range(B)]
    return np.percentile(s,2.5),np.percentile(s,97.5)
def paper_curves(configs, grid):
    rows=collections.defaultdict(list)
    with gzip.open(PAPER_CURVES,'rt') as f:
        for r in csv.DictReader(f):
            if r['config'] in configs and r['block']=='DR':
                rows[(r['config'],int(r['seed']))].append((int(r['step']),float(r['ep_return'])))
    out=collections.defaultdict(dict)
    for (c,s),v in rows.items():
        v.sort(); st=np.array([a for a,_ in v]); ret=np.array([b for _,b in v])
        ma=np.array([ret[max(0,i-49):i+1].mean() for i in range(len(ret))])
        vals=[]
        for g in grid:
            idx=np.searchsorted(st,g,side='right')-1
            vals.append(ma[idx] if idx>=0 else np.nan)
        out[c][s]=np.array(vals)
    return out
