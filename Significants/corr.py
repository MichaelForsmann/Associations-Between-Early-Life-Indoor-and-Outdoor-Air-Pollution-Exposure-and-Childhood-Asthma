
def fit_corr(
    X,
    num_warmup=500,
    num_samples=500,
    num_chains=4,
    seed=0,
    transform="auto",        # "auto" | "log" | "rank" | "none"
):
    """
    transform:
      "log"  -> np.log1p(X), then standardize  (good for positive skewed data)
      "rank" -> rank-transform, then standardize (Spearman-style; most robust)
      "none" -> just standardize
      "auto" -> "log" if all columns are non-negative, else "rank"
    """
    # 1) DataFrame -> ndarray
    if isinstance(X, pd.DataFrame):
        col_names = X.columns.tolist()
        X = X.to_numpy(dtype=np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)
        col_names = [f"x{i}" for i in range(X.shape[1])]

    # 2) Drop NaN rows
    mask = ~np.isnan(X).any(axis=1)
    if mask.sum() < len(X):
        print(f"Dropping {len(X) - mask.sum()} rows with NaN")
    X = X[mask]

    # 3) Choose transform
    if transform == "auto":
        transform = "log" if (X >= 0).all() else "rank"

    if transform == "log":
        if (X < 0).any():
            raise ValueError("log transform requires non-negative data.")
        X = np.log1p(X)                             # handles zeros safely
    elif transform == "rank":
        X = pd.DataFrame(X).rank().to_numpy(dtype=np.float64)
    elif transform != "none":
        raise ValueError(f"Unknown transform: {transform}")

    # 4) Guard constant columns
    sd = X.std(0)
    bad = np.where(sd == 0)[0]
    if bad.size:
        raise ValueError(
            f"Constant columns after transform: "
            f"{[col_names[i] for i in bad]} — drop them."
        )

    # 5) Standardize
    X = (X - X.mean(0)) / sd

    # 6) Final sanity check — fail loudly if still pathological
    mx = float(np.abs(X).max())
    if mx > 15:
        print(f"WARNING: max |X| = {mx:.1f} after transform+standardize. "
              "Consider transform='rank' for heavier-tailed data.")

    # 7) Fit
    kernel = NUTS(
        corr_model,
        target_accept_prob=0.95,
        init_strategy=init_to_median(num_samples=30),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(jax.random.PRNGKey(seed), X=jnp.asarray(X))
    return az.from_numpyro(mcmc)
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
import numpyro.distributions as dist
import numpyro.distributions as dist
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value,Predictive
from numpyro.infer.util import log_density
import jax
import jax.numpy as jnp
import numpy as np
import arviz as az
from arviz_base import from_dict     # <-- ArviZ 1.x location
import numpyro.distributions as dist
from numpyro.infer import MCMC, Predictive
import pandas as pd
import numpyro
def corr_model(X):
    """
    Multivariate-normal model on standardized data.
    Returns posterior over the full DxD correlation matrix.

    X : (N, D) array, already standardized (mean 0, sd 1) per column
        Use ranks (then standardize) for a Spearman-style version.
    """
    N, D = X.shape

    # Means (should be ~0 after standardization, but let the model see)
    mu = numpyro.sample("mu", dist.Normal(0.0, 1.0).expand([D]))

    # Per-variable scale (should be ~1 after standardization)
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0).expand([D]))

    # LKJ prior on correlation matrix via its Cholesky factor.
    # concentration=1 -> uniform over correlation matrices.
    # >1 favours identity (weaker correlations); <1 favours stronger.
    L_corr = numpyro.sample(
        "L_corr", dist.LKJCholesky(D, concentration=1.0)
    )

    # Reconstruct full correlation & covariance matrices as deterministics
    corr = numpyro.deterministic("corr", L_corr @ L_corr.T)
    scale_tril = sigma[:, None] * L_corr
    numpyro.deterministic("cov", scale_tril @ scale_tril.T)

    numpyro.sample(
        "obs",
        dist.MultivariateNormal(loc=mu, scale_tril=scale_tril),
        obs=X,
    )
def corr(Data,varibles,transform="rank"):
    corr_data=Data.loc[:,varibles]
    idata_b = fit_corr(corr_data,transform=transform)
    corr_data=idata_b["posterior"]["corr"]
    R_mean = corr_data.mean(axis=(0,1))
    R_lo   = np.quantile(corr_data, 0.05, axis=(0,1))
    R_hi   = np.quantile(corr_data, 0.95, axis=(0,1)) 
    df_corr=pd.DataFrame(R_mean.values,index=varibles,columns=varibles)
    hdi=pd.DataFrame(((np.sign(R_hi)-np.sign(R_lo))==0),index=varibles,columns=varibles)
    return idata_b,R_mean,df_corr,hdi