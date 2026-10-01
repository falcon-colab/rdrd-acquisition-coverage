"""Mock with the REAL RDRD layout: data/<Class>/<session>/<n>.csv"""
import numpy as np, os, shutil
rng=np.random.default_rng(11); H,W=11,61
def run(n,peak,sr,sd,seed):
    r=np.random.default_rng(seed); out=[]
    r0,d0=r.uniform(3,8),r.uniform(15,45)
    for _ in range(n):
        r0+=r.normal(0,0.04); d0+=r.normal(0,0.12)
        rr,dd=np.meshgrid(np.arange(H),np.arange(W),indexing="ij")
        m=peak*np.exp(-(((rr-r0)/sr)**2+((dd-d0)/sd)**2))
        m=20*np.log10(np.maximum(m+r.lognormal(-2,0.4,(H,W)),1e-6))
        out.append(np.maximum(m,-120.0))
    return out
root="/tmp/mock3"; shutil.rmtree(root,ignore_errors=True)
cfg={"Cars":(40.0,2.6,3.0,["13-44","14-02","15-10"]),
     "Drones":(1.2,0.7,1.2,["12-34","12-58","16-05"]),
     "People":(10.0,1.6,7.0,["12-50f","13-20","17-41"])}
for cls,(pk,sr,sd,sess) in cfg.items():
    for si,s in enumerate(sess):
        d=os.path.join(root,"data",cls,s); os.makedirs(d)
        mats=run(120,pk,sr,sd,seed=hash(cls+s)%10000)
        n=1
        for m in mats:
            if cls=="Drones" and rng.random()<0.12:   # dropped detections
                n+=1; continue
            np.savetxt(os.path.join(d,"%d.csv"%n),m,delimiter=",",fmt="%.4f"); n+=1
print("wrote",root)
