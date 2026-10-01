"""Mock reproducing BOTH pathologies: duplicated tree + duplicate sessions."""
import numpy as np, os, shutil
H,W=11,61
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

root="/tmp/mock4"; shutil.rmtree(root,ignore_errors=True)
cfg={"Cars":(40.0,2.6,3.0,["13-13","13-23","13-44","17-09"]),
     "Drones":(1.5,0.7,1.2,["12-34","12-41","15-21","16-09"]),
     "People":(10.0,1.6,7.0,["11-00f","11-00i","12-50f","15-58"])}
for cls,(pk,sr,sd,sess) in cfg.items():
    for s in sess:
        d=os.path.join(root,"data",cls,s); os.makedirs(d)
        for i,m in enumerate(run(100,pk,sr,sd,seed=abs(hash(cls+s))%10000),1):
            np.savetxt(os.path.join(d,"%d.csv"%i),m,delimiter=",",fmt="%.4f")
    # duplicate SESSION: 17-09 -> 17-09p (identical content)
    if cls=="Cars":
        shutil.copytree(os.path.join(root,"data",cls,"17-09"),
                        os.path.join(root,"data",cls,"17-09p"))
    if cls=="People":
        shutil.copytree(os.path.join(root,"data",cls,"15-58"),
                        os.path.join(root,"data",cls,"15-58i"))
# duplicate the WHOLE TREE at the archive root
for cls in cfg:
    shutil.copytree(os.path.join(root,"data",cls), os.path.join(root,cls))
n=sum(len(f) for _,_,f in os.walk(root))
print("wrote",root,"total files",n)
