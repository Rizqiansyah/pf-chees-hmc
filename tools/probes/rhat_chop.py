import numpy as np, arviz as az, xarray as xr, sys
idata = az.from_netcdf(sys.argv[1])
post = idata.posterior
drop = [v for v in post.data_vars if v not in ("mu", "tau", "theta")]  # deterministics aside
post = post[[v for v in post.data_vars if v in ("mu","tau","theta","theta_t","theta_tilde")]] if any(v in post for v in ("mu",)) else post
print("vars:", list(post.data_vars), dict(post.sizes))
ess = az.ess(post, method="bulk"); n_tot = post.sizes["chain"] * post.sizes["draw"]
tau_by = {v: float(n_tot / np.min(ess[v].values)) for v in post.data_vars}
print("full run: max R-hat %.4f ; IAT (N/ESS_bulk) worst per var:" % float(max(float(az.rhat(post)[v].max()) for v in post.data_vars)), {k: round(t,2) for k,t in tau_by.items()})
def chop(seg):
    C, D = post.sizes["chain"], post.sizes["draw"]; k = D // seg
    out = {}
    for v in post.data_vars:
        a = post[v].values[:, :k*seg]                      # (C, k*seg, ...)
        a = a.reshape((C, k, seg) + a.shape[2:]).reshape((C*k, seg) + a.shape[2:])
        out[v] = a
    return az.from_dict(posterior=out).posterior
rng = np.random.default_rng(0)
for seg, label in [(39, "doc: 1024x39 -> R-hat 1.0404"), (20, "doc: 512x20 -> R-hat 1.08-1.11"), (78, "doc: 512x78 -> 1.0229"), (313, "doc: 128x313 -> 1.0060")]:
    p = chop(seg)
    # subsample chains to the doc's chain count so the estimator sees the same shape
    nc = {39:1024, 20:512, 78:512, 313:128}[seg]
    idx = rng.choice(p.sizes["chain"], size=min(nc, p.sizes["chain"]), replace=False)
    p = p.isel(chain=idx)
    rh = az.rhat(p); mx = max(float(rh[v].max()) for v in p.data_vars)
    print(f"stationary chop into {p.sizes['chain']} x {seg}: max R-hat {mx:.4f}   [{label}]")
# iid control: same shape, independent normal draws (tau = 1)
z = rng.normal(size=(1024, 39, 10)); print("iid control 1024x39 d=10: max R-hat %.4f" % float(az.rhat(az.from_dict(posterior={'x': z}).posterior)['x'].max()))
z = rng.normal(size=(512, 20, 10)); print("iid control 512x20 d=10: max R-hat %.4f" % float(az.rhat(az.from_dict(posterior={'x': z}).posterior)['x'].max()))
