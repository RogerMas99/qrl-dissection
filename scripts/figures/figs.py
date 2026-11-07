import sys, pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent)); REPO=pathlib.Path(__file__).resolve().parents[2]
import numpy as np, matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import data
plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True,'font.size':8.5,
  'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'lines.linewidth':1.2})
OUT=sys.argv[1]
C=dict(ent='#010073', noent='#5C8AA8', matched='#C0672E', rep='#B8860B', ref='#9A9A9A', ceil='#2E7D4F', big='#555555')
T=data.parse_dossier(); PS=data.per_seed()
W=6.6
def tt(s): return r'\texttt{'+s.replace('_',r'\_')+'}'

# ---------- Fig 4.1: capacity ladder ----------
arms=[('paper_linear','reference\n(26)',C['ref']),('matched_classical','matched\n(135)',C['matched']),
      ('hybrid_fig4','hybrid\n(126)',C['ent']),('oversized_mlp','large network\n(10,934)',C['big'])]
fig,axs=plt.subplots(1,2,figsize=(W,2.35))
rng=np.random.default_rng(1)
for j,(m,lab) in enumerate([('final','training return'),('greedy','greedy return')]):
    ax=axs[j]
    for i,(a,name,col) in enumerate(arms):
        v=[x[j] for x in PS[(a,'on')].values()]
        ax.scatter(i+rng.uniform(-0.12,0.12,len(v)),v,s=11,color=col,alpha=0.55,lw=0)
        r=T[(a,'on')]; c=r[m+'_ci']; y=r[m]
        ax.errorbar(i+0.27,y,yerr=[[y-c[0]],[c[1]-y]],fmt='o',color=col,ms=4,capsize=2,lw=1)
    ax.axhline(9.5,color='k',ls=':',lw=0.7); ax.axhline(22,color='k',ls='--',lw=0.6)
    ax.set_yscale('log'); ax.set_ylim(7,500); ax.set_xticks(range(4)); ax.set_xticklabels([a[1] for a in arms])
    ax.set_yticks([10,22,50,100,200,500]); ax.set_yticklabels(['10','22','50','100','200','500'])
    ax.set_title(lab); ax.set_xlim(-0.5,3.6)
axs[0].set_ylabel('return (log scale)')
axs[1].text(3.55,9.5,'constant',fontsize=6.5,va='bottom',ha='right'); axs[1].text(3.55,22,'random',fontsize=6.5,va='bottom',ha='right')
fig.tight_layout(); fig.savefig(OUT+'/ch4_capacity.pdf'); plt.close(fig)

# ---------- Fig 4.2: output reuse ----------
R=[4,8,16,32]
# exp11: matched control with the same replication as the hybrid agent (R=4 is identical to classical_or_matched_r4)
import csv
for row in csv.DictReader(open(REPO/'handoff_exp11'/'summary_exp11.csv')):
    if row['arm'].startswith('classical_or_matched_rep_r'):
        T[(row['arm'],'on')]={'final':float(row['final_perf_iqm']),'final_ci':(float(row['final_ci_lo']),float(row['final_ci_hi'])),
                              'greedy':float(row['greedy_final_iqm']),'greedy_ci':(float(row['greedy_ci_lo']),float(row['greedy_ci_hi']))}
T[('classical_or_matched_rep_r4','on')]=T[('classical_or_matched_r4','on')]
for row in csv.DictReader(open(REPO/'handoff_exp12'/'summary_exp12.csv')):
    if row['arm'].startswith('classical_twin'):
        T[(row['arm'],'on')]={'final':float(row['final_perf_iqm']),'final_ci':(float(row['final_ci_lo']),float(row['final_ci_hi'])),
                              'greedy':float(row['greedy_final_iqm']),'greedy_ci':(float(row['greedy_ci_lo']),float(row['greedy_ci_hi']))}
series=[('hybrid_OR{}', 'hybrid, entangled',C['ent'],'o',-0.12),
        ('hybrid_or_r{}_noent','hybrid, unentangled',C['noent'],'s',-0.06),
        ('classical_or_matched_r{}','matched control',C['matched'],'^',0.0),
        ('classical_twin_r{}','classical twin',C['rep'],'D',0.06),
        ('classical_OR{}','reference control',C['ref'],'v',0.12)]
base=[('hybrid_fig4',C['ent'],'o',-0.12),('hybrid_fig4_noent',C['noent'],'s',-0.06),('classical_twin_r1',C['rep'],'D',0.06)]
fig,axs=plt.subplots(1,2,figsize=(W,2.85))
for j,(m,lab) in enumerate([('final','training return'),('greedy','greedy return')]):
    ax=axs[j]
    for pat,name,col,mk,dx in series:
        xs=np.array([1,2,3,4])+dx; ys=[T[(pat.format(r),'on')][m] for r in R]
        lo=[T[(pat.format(r),'on')][m+'_ci'][0] for r in R]; hi=[T[(pat.format(r),'on')][m+'_ci'][1] for r in R]
        ax.errorbar(xs,ys,yerr=[np.array(ys)-lo,np.array(hi)-ys],fmt=mk+'-',color=col,ms=3.5,capsize=1.5,lw=1,label=name)
    for a,col,mk,dx in base:
        r=T[(a,'on')]; y=r[m]; c=r[m+'_ci']
        ax.errorbar(0+dx,y,yerr=[[y-c[0]],[c[1]-y]],fmt=mk,color=col,ms=3.5,capsize=1.5,lw=1,mfc='white')
    ax.axvline(0.5,color='k',lw=0.4,alpha=0.4)
    ax.set_xticks([0,1,2,3,4]); ax.set_xticklabels(['none','4','8','16','32']); ax.set_xlabel('reuse factor $R$')
    ax.set_title(lab); ax.set_ylim(0,320 if j==0 else 440)
axs[0].set_ylabel('return (IQM, 95% CI)')
h,l=axs[0].get_legend_handles_labels()
fig.legend(h,l,loc='lower center',frameon=False,ncol=3,handlelength=1.8,columnspacing=1.2)
fig.tight_layout(rect=[0,0.13,1,1]); fig.savefig(OUT+'/ch4_or.pdf'); plt.close(fig)

# ---------- Fig 4.3: depth, DQN vs PPO ----------
TR=data.curves('curves_train_ma50.csv','train_ma50')
grid=np.arange(1000,100001,1000)
P=data.paper_curves({'Skolik_8Q_L1','Skolik_8Q_L2','Skolik_8Q_L5'},grid)
cols={1:'#9DB8D9',2:'#4F79B0',5:C['ent']}
fig,axs=plt.subplots(1,2,figsize=(W,2.45),sharey=True)
def band(ax,M,col,lab,ls='-'):
    M=np.array(M); mid=np.array([data.iqm(M[:,k][~np.isnan(M[:,k])]) for k in range(M.shape[1])])
    ci=np.array([data.boot(M[:,k][~np.isnan(M[:,k])],B=400) for k in range(M.shape[1])])
    ax.plot(grid/1000,mid,color=col,label=lab,ls=ls); ax.fill_between(grid/1000,ci[:,0],ci[:,1],color=col,alpha=0.18,lw=0)
for L in (1,2,5):
    runs=TR[('hybrid_DR%d'%L,'on')]
    M=[]
    for s,v in runs.items():
        d=dict(v); M.append([d.get(g,np.nan) for g in grid])
    band(axs[0],M,cols[L],'$L=%d$'%L)
    band(axs[1],list(P['Skolik_8Q_L%d'%L].values()),cols[L],'$L=%d$'%L)
for ax,ttl in zip(axs,['(a) DQN, this work','(b) PPO, reference study']):
    ax.set_title(ttl); ax.set_xlabel('environment steps ($\\times 10^3$)'); ax.axhline(22,color='k',ls='--',lw=0.6)
    ax.set_xlim(0,100)
axs[0].set_ylabel('training return, 50-episode mean')
axs[0].legend(frameon=False,loc='upper left')
fig.tight_layout(); fig.savefig(OUT+'/ch4_depth.pdf'); plt.close(fig)

# ---------- Fig 4.4: FrozenLake ----------
EV=data.curves('curves_eval_greedy.csv','greedy_return')
fig,axs=plt.subplots(1,2,figsize=(W,2.45))
ax=axs[0]
gsteps=np.arange(10000,100001,10000)
for a,lab,col,mk in [('frozen_onehot_mlp','one-hot network (5,508)',C['big'],'D'),
                     ('frozen_binary_4q_L5','binary, entangled (60)',C['ent'],'o'),
                     ('frozen_binary_4q_noent_L5','binary, unentangled (60)',C['noent'],'s'),
                     ('frozen_binary_4q_fourier_ceiling','linear ceiling (20)',C['ceil'],'^')]:
    runs=EV[(a,'on')]
    M=np.array([[dict(v).get(g,np.nan) for g in gsteps] for v in runs.values()])
    frac=(M>0.5).mean(axis=0)
    ax.plot(gsteps/1000,frac,marker=mk,ms=3,color=col,label=lab)
ax.set_ylim(-0.03,1.05); ax.set_xlabel('environment steps ($\\times 10^3$)'); ax.set_ylabel('fraction of seeds solving (greedy)')
ax.set_title('(a) Binary circuit, $L=5$'); ax.legend(frameon=False,loc='center left',bbox_to_anchor=(0.0,0.6),fontsize=6.8)
ax=axs[1]
Ls=[1,5,10,15]
for pat,lab,col,mk,dx in [('frozen_scalar_1q_L{}','scalar circuit',C['noent'],'s',-0.08),
                          ('frozen_scalar_1q_fourier_ceiling_L{}__flkw','Fourier ceiling',C['ceil'],'^',0.08)]:
    ys=[T[(pat.format(L),'on')]['final'] for L in Ls]
    lo=[T[(pat.format(L),'on')]['final_ci'][0] for L in Ls]; hi=[T[(pat.format(L),'on')]['final_ci'][1] for L in Ls]
    ax.errorbar(np.arange(4)+dx,ys,yerr=[np.array(ys)-lo,np.array(hi)-ys],fmt=mk+'-',color=col,ms=3.5,capsize=1.5,label=lab)
ax.axhline(T[('frozen_onehot_mlp','on')]['final'],color=C['big'],ls='--',lw=0.7)
ax.text(3.3,0.9,'one-hot network',fontsize=6.5,ha='right',va='top',color=C['big'])
ax.set_xticks(range(4)); ax.set_xticklabels([str(L) for L in Ls]); ax.set_xlabel('depth $L$')
ax.set_ylabel('success rate, training'); ax.set_ylim(-0.03,1.05)
ax.set_title('(b) Scalar circuit against its ceilings'); ax.legend(frameon=False,loc='center left',fontsize=6.8)
fig.tight_layout(); fig.savefig(OUT+'/ch4_frozenlake.pdf'); plt.close(fig)
print('done')

# ---------- Fig: output reuse dynamics ----------
fig,axs=plt.subplots(1,2,figsize=(W,2.45),sharey=True)
shades=['#C9D6EA','#8FA9D2','#4F79B0',C['ent']]
oshades=['#F2D3BF','#E3A57E','#D07E4C',C['matched']]
for M_,(a,lab) in zip(shades,[('hybrid_fig4','no reuse'),('hybrid_OR4','$R=4$'),('hybrid_OR16','$R=16$'),('hybrid_OR32','$R=32$')]):
    runs=TR[(a,'on')]; M=[[dict(v).get(g,np.nan) for g in grid] for v in runs.values()]
    band(axs[0],M,M_,lab)
for M_,(a,lab) in zip(oshades,[('classical_or_matched_r4','$R=4$'),('classical_or_matched_r8','$R=8$'),('classical_or_matched_r16','$R=16$'),('classical_or_matched_r32','$R=32$')]):
    runs=TR[(a,'on')]; M=[[dict(v).get(g,np.nan) for g in grid] for v in runs.values()]
    band(axs[1],M,M_,lab)
for ax,ttl in zip(axs,['(a) hybrid agent','(b) matched classical control']):
    ax.set_title(ttl); ax.set_xlabel('environment steps ($\\times 10^3$)'); ax.axhline(22,color='k',ls='--',lw=0.6); ax.set_xlim(0,100)
    ax.legend(frameon=False,loc='upper left')
axs[0].set_ylabel('training return, 50-episode mean')
fig.tight_layout(); fig.savefig(OUT+'/ch4_or_curves.pdf'); plt.close(fig)
print('done2')
