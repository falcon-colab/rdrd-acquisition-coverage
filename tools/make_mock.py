"""Build a mock RDRD-like tree to exercise the inventory script."""
import numpy as np, os, shutil
rng = np.random.default_rng(1)
root = "/tmp/mock_rdrd"
shutil.rmtree(root, ignore_errors=True)

H, W = 11, 61
FLOOR = -120.0

def make_run(n, peak, spread_r, spread_d, drift):
    """One continuous pass: target slides slowly across the window."""
    out = []
    r0, d0 = rng.uniform(3, 8), rng.uniform(20, 40)
    for t in range(n):
        r0 += drift * rng.normal(0, 0.05)
        d0 += drift * rng.normal(0, 0.15)
        rr, dd = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        m = peak * np.exp(-(((rr-r0)/spread_r)**2 + ((dd-d0)/spread_d)**2))
        m = 20*np.log10(np.maximum(m + rng.lognormal(-2, 0.4, (H, W)), 1e-6))
        out.append(np.maximum(m, FLOOR))
    return out

cfg = {
    "Drones":  dict(n_runs=6, per_run=60, peak=3.0,  sr=0.9, sd=2.0, drift=1.0),
    "Cars":    dict(n_runs=6, per_run=60, peak=40.0, sr=2.6, sd=3.0, drift=1.0),
    "People":  dict(n_runs=6, per_run=60, peak=10.0, sr=1.6, sd=7.0, drift=1.0),
}
for cls, c in cfg.items():
    d = os.path.join(root, cls); os.makedirs(d)
    idx = 1
    for _ in range(c["n_runs"]):
        for m in make_run(c["per_run"], c["peak"], c["sr"], c["sd"], c["drift"]):
            np.savetxt(os.path.join(d, "%s_%04d.csv" % (cls[:-1], idx)),
                       m, delimiter=",", fmt="%.4f")
            idx += 1
print("mock written to", root)
