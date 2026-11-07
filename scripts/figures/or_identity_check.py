# Numerical check: Output Reuse (R copies of the features before a linear layer, trained with Adam)
# is equivalent to the same layer without reuse whose learning rate is multiplied by R.
import numpy as np
rng=np.random.default_rng(0)
n,f,o,R,lr,steps=256,8,4,16,2.5e-4,3000
X=np.tanh(rng.normal(size=(n,f))); Y=rng.normal(size=(n,o))*30   # bounded features, large targets
def adam(p,g,s,lr,b1=.9,b2=.999,eps=1e-8):
    s['t']+=1; s['m']=b1*s['m']+(1-b1)*g; s['v']=b2*s['v']+(1-b2)*g*g
    mh=s['m']/(1-b1**s['t']); vh=s['v']/(1-b2**s['t']); return p-lr*mh/(np.sqrt(vh)+eps)
# (a) Output Reuse: W has shape (o, f*R), input = X repeated R times
W=rng.uniform(-1,1,(o,f*R))/np.sqrt(f*R); b=np.zeros(o)
Weff0=W.reshape(o,R,f).sum(1)            # effective weight on each feature
# (b) no reuse, same effective initial weights, lr multiplied by R on W (bias lr unchanged)
V=Weff0.copy(); c=np.zeros(o)
sW=dict(t=0,m=0*W,v=0*W); sb=dict(t=0,m=0*b,v=0*b); sV=dict(t=0,m=0*V,v=0*V); sc=dict(t=0,m=0*c,v=0*c)
Xr=np.tile(X,R)
for t in range(steps):
    Pa=Xr@W.T+b; ga=2*(Pa-Y)/n; W=adam(W,ga.T@Xr,sW,lr); b=adam(b,ga.sum(0),sb,lr)
    Pb=X@V.T+c;  gb=2*(Pb-Y)/n; V=adam(V,gb.T@X,sV,lr*R); c=adam(c,gb.sum(0),sc,lr)
print('max |difference| between the two models after',steps,'Adam steps:',np.abs((Xr@W.T+b)-(X@V.T+c)).max())
print('loss with reuse:',round(float(((Xr@W.T+b-Y)**2).mean()),3),' loss without reuse, lr x R:',round(float(((X@V.T+c-Y)**2).mean()),3))
# (c) for reference: no reuse, ordinary lr
U=Weff0.copy(); d=np.zeros(o); sU=dict(t=0,m=0*U,v=0*U); sd=dict(t=0,m=0*d,v=0*d)
for t in range(steps):
    Pc=X@U.T+d; gc=2*(Pc-Y)/n; U=adam(U,gc.T@X,sU,lr); d=adam(d,gc.sum(0),sd,lr)
print('loss without reuse, ordinary lr:',round(float(((X@U.T+d-Y)**2).mean()),3))
print('initial std of effective weights, R=16 vs R=1 default init:',round(float(Weff0.std()),3), round(float((rng.uniform(-1,1,(4000,))/np.sqrt(f)).std()),3))
