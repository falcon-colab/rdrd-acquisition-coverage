"""Two harder mocks: shuffled filenames, and session-encoded filenames."""
import numpy as np, os, shutil
rng = np.random.default_rng(7)
H, W, FLOOR = 11, 61, -120.0

def make_run(n, peak, sr, sd):
    out=[]; r0,d0 = rng.uniform(3,8), rng.uniform(20,40)
    for _ in range(n):
        r0 += rng.normal(0,0.05); d0 += rng.normal(0,0.15)
        rr,dd = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        m = peak*np.exp(-(((rr-r0)/sr)**2 + ((dd-d0)/sd)**2))
        m = 20*np.log10(np.maximum(m + rng.lognormal(-2,0.4,(H,W)), 1e-6))
        out.append(np.maximum(m, FLOOR))
    return out

# drone: WEAK but COMPACT (this is the case the naive metric got wrong)
cfg = {"Drones": dict(peak=1.2, sr=0.7, sd=1.2),
       "Cars":   dict(peak=40.0, sr=2.6, sd=3.0),
       "People": dict(peak=10.0, sr=1.6, sd=7.0)}

for tag, mode in [("shuffled","shuffle"), ("sessioned","session")]:
    root = "/tmp/mock2_%s" % tag
    shutil.rmtree(root, ignore_errors=True)
    for cls,c in cfg.items():
        d=os.path.join(root,cls); os.makedirs(d)
        allm=[]; sess=[]
        for s in range(6):
            for m in make_run(60, c["peak"], c["sr"], c["sd"]):
                allm.append(m); sess.append(s+1)
        if mode=="shuffle":
            idx=rng.permutation(len(allm))
            for newi,i in enumerate(idx,1):
                np.savetxt(os.path.join(d,"%s_%04d.csv"%(cls[:-1],newi)),
                           allm[i],delimiter=",",fmt="%.4f")
        else:
            cnt={}
            for m,s in zip(allm,sess):
                cnt[s]=cnt.get(s,0)+1
                np.savetxt(os.path.join(d,"%s_s%02d_f%04d.csv"%(cls[:-1],s,cnt[s])),
                           m,delimiter=",",fmt="%.4f")
    print("wrote",root)
