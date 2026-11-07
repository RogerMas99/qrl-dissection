import sys, csv, numpy as np, matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams.update({'text.usetex':False,'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm','axes.unicode_minus':False,'axes.formatter.use_mathtext':True,'font.size':8.5,
  'axes.labelsize':8.5,'axes.titlesize':9,'legend.fontsize':7.2,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
  'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':0.6,'lines.linewidth':1.2})
OUT=sys.argv[1]; D=sys.argv[2]
C=dict(ent='#010073', noent='#5C8AA8', matched='#C0672E', rep='#B8860B', ref='#9A9A9A', ceil='#2E7D4F', big='#555555')
S={}
for r in csv.DictReader(open(D+'/summary_exp13.csv')):
    d={}
    for k,src in [('final_perf_iqm','final_perf_iqm'),('final_perf_ci_lo','final_ci_lo'),('final_perf_ci_hi','final_ci_hi'),('best_ma50_iqm','best_ma50_iqm'),('best_ma50_ci_lo','best_ma50_ci_lo'),('best_ma50_ci_hi','best_ma50_ci_hi')]:
        if r.get(src) not in (None,'','nan'): d[k]=float(r[src])
    S[r['arm']]=d
def pt(a,m):
    s=S[a]; return s[m+'_iqm'],s[m+'_ci_lo'],s[m+'_ci_hi']
fig,axs=plt.subplots(1,2,figsize=(6.6,2.75))
ax=axs[0]; m='final_perf'
def plot(ax,xs,arms,m,fmt,col,lab,mfc=None,dx=1.0):
    y=[pt(a,m) for a in arms]; v=np.array([t[0] for t in y]); lo=np.array([t[1] for t in y]); hi=np.array([t[2] for t in y])
    ax.errorbar(np.array(xs)*dx,v,yerr=[v-lo,hi-v],fmt=fmt,color=col,ms=3.5,capsize=1.5,lw=1,label=lab,mfc=mfc or col)
ks=[1,2,4,8,16,32,64,128]
plot(ax,ks,['classical_twin_reuse_r1']+['classical_twin_headlr_x%d'%k for k in ks[1:]],m,'o-',C['ent'],'learning rate of the head $\\times k$')
plot(ax,[1,4,8,16,32],['classical_twin_reuse_r%d'%r for r in [1,4,8,16,32]],m,'D',C['rep'],'output reuse $R$',dx=1.12)
plot(ax,[4,8,16,32],['classical_twin_collapsed_r%d'%r for r in [4,8,16,32]],m,'D',C['rep'],'learning rate $\\times R$, collapsed start',mfc='white',dx=0.89)
plot(ax,[8,32],['classical_twin_gradx8','classical_twin_gradx32'],m,'s',C['ref'],'gradient $\\times R$')
ax.set_xscale('log',base=2); ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.minorticks_off()
ax.set_xlabel('$R$ or $k$'); ax.set_ylabel('training return (IQM, 95% CI)'); ax.set_ylim(0,300)
ax.set_title('(a) DQN, classical twin')
ax.legend(loc='upper left',frameon=False,handlelength=1.4)
ax=axs[1]; m='best_ma50'; Rs=[1,4,8,16,32]
plot(ax,Rs,['ppo_reuse_clip_r%d'%r for r in Rs],m,'D-',C['rep'],'output reuse $R$',dx=1.0)
plot(ax,Rs[1:],['ppo_lrx_clip_r%d'%r for r in Rs[1:]],m,'o--',C['ent'],'learning rate $\\times R$',mfc='white',dx=1.1)
ref={4:(263.94,18.38),8:(320.0,36.29),16:(381.7,37.32),32:(330.2,63.44)}
xr=np.array(list(ref))*0.9; ax.errorbar(xr,[v[0] for v in ref.values()],yerr=[v[1] for v in ref.values()],fmt='s',color=C['big'],ms=3.5,capsize=1.5,lw=1,label='reference study (mean $\\pm$ sd)')
ax.set_xscale('log',base=2); ax.set_xticks(Rs); ax.set_xticklabels([str(r) for r in Rs]); ax.minorticks_off()
ax.set_xlabel('$R$'); ax.set_ylabel('best 50-episode moving average'); ax.set_ylim(0,450)
ax.set_title('(b) PPO, classical control of the reference study')
ax.legend(loc='lower right',frameon=False,handlelength=1.4)
fig.tight_layout(); fig.savefig(OUT+'/ch4_lr.pdf'); plt.close(fig)
