"""
Repeated stratified CV for a Bayesian logistic regression in NumPyro.

Per fold we fit the model with NUTS, then summarise:
  - AUROC and Brier as posterior distributions (one value per draw)
  - posterior-mean probabilities for point-estimate calibration
  - pooled out-of-fold probabilities for a single calibration curve
    over the whole dataset (the thing you want in a paper).
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive, log_likelihood
import arviz as az
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.calibration import calibration_curve
from scipy.stats import beta as beta_dist


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def logistic_regression_model(features, target=None):
    """Logistic regression with a half-normal global scale on weights."""
    D = features.shape[1]
    tau = numpyro.sample("tau", dist.HalfNormal(2))
    weight_raw = numpyro.sample(
        "weight_raw",
        dist.Normal(jnp.zeros(D), jnp.ones(D)).to_event(1),
    )
    weight = numpyro.deterministic("weight", tau * weight_raw)
    bias = numpyro.sample("bias", dist.Normal(0.0, 1.5))
    logits = features @ weight + bias
    with numpyro.plate("data", features.shape[0]):
        numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=target)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_inference(features, target, rng_key,
                  num_samples=1000, num_warmup=1000,
                  num_chains=4, chain_method="parallel"):
    kernel = NUTS(logistic_regression_model, target_accept_prob=0.95)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        # Multi-chain parallel + progress bar can deadlock; only show it
        # when we're actually running serially.
        progress_bar=(num_chains == 1 or chain_method == "sequential"),
    )
    # extra_fields=("diverging",) is REQUIRED for get_extra_fields() below.
    mcmc.run(rng_key, features, target, extra_fields=("diverging",))
    return mcmc


def train_model(X_train, y_train, fold_i, repeat_i, name,
                num_samples=1000, num_warmup=1000, seed=0,
                save_idata=True):
    """Fit one fold. Returns (mcmc, posterior_samples)."""
    rng_key = random.PRNGKey(seed)
    k_run, k_post, k_prior = random.split(rng_key, 3)

    mcmc = run_inference(
        X_train, y_train, k_run,
        num_samples=num_samples, num_warmup=num_warmup,
    )
    posterior_samples = mcmc.get_samples()  # <-- this was the missing piece

    posterior_predictive = Predictive(
        logistic_regression_model, posterior_samples
    )(k_post, features=X_train)

    prior_predictive = Predictive(
        logistic_regression_model, num_samples=num_samples
    )(k_prior, features=X_train)

    ll = log_likelihood(
        logistic_regression_model, posterior_samples,
        features=X_train, target=y_train,
    )

    idata = az.from_numpyro(
        mcmc,
        prior=prior_predictive,
        posterior_predictive=posterior_predictive,
        log_likelihood=ll,
    )
    if save_idata:
        idata.to_netcdf(f"{name}_repeat{repeat_i}_fold{fold_i}.nc")

    return mcmc, posterior_samples


# ---------------------------------------------------------------------------
# Predictions and per-draw metrics
# ---------------------------------------------------------------------------
def posterior_probs(posterior_samples, X):
    """Per-draw probabilities of shape (N, S)."""
    W = posterior_samples["weight"]    # (S, D)
    b = posterior_samples["bias"]      # (S,)
    logits = X @ W.T + b[None, :]      # (N, S)
    return np.asarray(jax.nn.sigmoid(logits))


def _per_draw_auc_brier(y_true, probs_samples, max_draws=500, rng=None):
    """AUROC and Brier for every posterior draw. Optionally subsample draws."""
    S = probs_samples.shape[1]
    if max_draws is not None and S > max_draws:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(S, size=max_draws, replace=False)
        probs_samples = probs_samples[:, idx]
        S = max_draws
    aucs = np.empty(S)
    briers = np.empty(S)
    ici = np.empty(S)

    for s in range(S):
        aucs[s] = roc_auc_score(y_true, probs_samples[:, s])
        briers[s] = brier_score_loss(y_true, probs_samples[:, s])
        ici[s] =calibration_metrics(y_true, probs_samples[:, s])
    return aucs, briers,ici


# ---------------------------------------------------------------------------
# Repeated stratified CV
# ---------------------------------------------------------------------------
def nested_cross_bayesian_logistic(
    X, y, Number_CV, number_split, name,
    num_samples=1000, warmup_steps=1000,
    n_neighbors_imputer=20,
    metric_subsample=500,   # e.g. 500 to speed up per-draw metrics
    save_idata=True,
    verbose=True,
):
    n_folds = Number_CV * number_split

    test_cols = [
        "roc_test", "brier_test",
        "ici_test","bss_test"]
    train_cols = [c.replace("test", "train") for c in test_cols]
    test_arr  = {c: np.zeros((n_folds,metric_subsample)) for c in test_cols}
    train_arr = {c: np.zeros((n_folds,metric_subsample)) for c in train_cols}

    calib_train_per_fold = []
    calib_test_per_fold  = []
    pooled_oof_roca_train=[]
    pooled_oof_brier_train=[]
    pooled_oof_bss_train=[]
    pooled_oof_roca_test=[]
    pooled_oof_brier_test=[]
    pooled_oof_bss_test=[]
    pooled_oof_ici_train=[]
    pooled_oof_ici_test=[]
    # Pooled out-of-fold predictions: one row per repeat. Within a repeat
    # every sample is in exactly one test fold, so each row fills entirely.
    oof_probs = np.full((Number_CV, len(y)), np.nan)
    fold_diagnostics = []

    prevalence     = float(np.mean(y))
    baseline_brier = prevalence * (1.0 - prevalence)

    j = 0
    mcmc = None  # last-fold handle for return value

    for repeat_i in range(Number_CV):
        skf = StratifiedKFold(
            n_splits=number_split, shuffle=True, random_state=repeat_i,
        )

        for fold_i, (train_ix, test_ix) in enumerate(skf.split(X, y)):
            X_train_raw = X.iloc[train_ix]
            X_test_raw  = X.iloc[test_ix]
            y_train_raw = y.iloc[train_ix]
            y_test_raw  = y.iloc[test_ix]

            # Fit scaler/imputer on TRAIN only, transform both.
            scaler  = StandardScaler()
            imputer = KNNImputer(n_neighbors=n_neighbors_imputer)
            X_train_imp = imputer.fit_transform(scaler.fit_transform(X_train_raw))
            X_test_imp  = imputer.transform(scaler.transform(X_test_raw))

            X_train_j = jnp.asarray(X_train_imp, dtype=jnp.float32)
            X_test_j  = jnp.asarray(X_test_imp,  dtype=jnp.float32)
            y_train_j = jnp.asarray(y_train_raw.values, dtype=jnp.float32)
            y_test_j = jnp.asarray(y_test_raw.values, dtype=jnp.float32)

            mcmc, posterior_samples = train_model(
                X_train_j, y_train_j, fold_i, repeat_i, name,
                num_samples=num_samples,
                num_warmup=warmup_steps,
                seed=repeat_i * number_split + fold_i,
                save_idata=save_idata,
            )

            divergences = int(mcmc.get_extra_fields()["diverging"].sum())
            if divergences > 0 and verbose:
                print(
                    f"WARNING repeat {repeat_i} fold {fold_i}: "
                    f"{divergences} divergent transitions"
                )
            fold_diagnostics.append({
                "repeat": repeat_i,
                "fold": fold_i,
                "divergences": divergences,
            })

            train_probs_samples = posterior_probs(posterior_samples, X_train_j)
            test_probs_samples  = posterior_probs(posterior_samples, X_test_j)

            # Posterior-mean point estimates
            train_probs = train_probs_samples.mean(axis=1)
            test_probs  = test_probs_samples.mean(axis=1)

            # Posterior distribution over the metric itself.
            # NB: the original code passed test_probs_samples to the train
            # metric calls — fixed here.
            
            train_arr["roc_train"][j], train_arr["brier_train"][j], train_arr["ici_train"][j] = _per_draw_auc_brier(y_train_raw.values, train_probs_samples,max_draws=metric_subsample)
            test_arr["roc_test"][j],   test_arr["brier_test"][j],   test_arr["ici_test"][j]   =  _per_draw_auc_brier(y_test_raw.values,  test_probs_samples,max_draws=metric_subsample)

            train_arr["bss_train"][j] = 1.0 - train_arr["brier_train"][j] / baseline_brier
            test_arr["bss_test"][j]   = 1.0 - test_arr["brier_test"][j]   / baseline_brier
            pooled_oof_roca_train.append(roc_auc_score(y_train_raw.values, train_probs))
            pooled_oof_brier_train.append(brier_score_loss(y_train_raw.values, train_probs))
            pooled_oof_bss_train.append(1 - brier_score_loss(y_train_raw.values, train_probs) / baseline_brier)
            pooled_oof_roca_test.append(roc_auc_score(y_test_raw.values, test_probs))
            pooled_oof_brier_test.append(brier_score_loss(y_test_raw.values, test_probs))
            pooled_oof_bss_test.append(1 - brier_score_loss(y_test_raw.values, test_probs) / baseline_brier)
            pooled_oof_ici_train.append(calibration_metrics(y_train_raw.values, train_probs))
            pooled_oof_ici_test.append(calibration_metrics(y_test_raw.values, test_probs))

            # Per-fold quantile-binned calibration kept for diagnostics.
            calib_train_per_fold.append(
                calibration_curve(y_train_raw.values, train_probs,
                                  n_bins=10, strategy="quantile"))
            calib_test_per_fold.append(
                calibration_curve(y_test_raw.values, test_probs,
                                  n_bins=10, strategy="quantile"))
            
         
            oof_probs[repeat_i, test_ix] = test_probs

            j += 1

    # Pooled OOF: average probability per sample across repeats, then
    # one calibration curve over the whole dataset. This is the figure
    # for the paper.
    oof_mean_probs = np.nanmean(oof_probs, axis=0)
    assert not np.isnan(oof_mean_probs).any(), "Some samples never appeared in test"

    pooled_calibration_test = calibration_curve(
        y.values, oof_mean_probs, n_bins=10, strategy="quantile",
    )

    summary = {
    "prevalence":          prevalence,
    "baseline_brier":      baseline_brier,
    "pooled_oof_roc_train":      np.array(pooled_oof_roca_train),
    "pooled_oof_brier_train":    np.array(pooled_oof_brier_train),
    "pooled_oof_bss_train":      np.array(pooled_oof_bss_train),
    "pooled_oof_roc_test":      np.array(pooled_oof_roca_test),
    "pooled_oof_brier_test":    np.array(pooled_oof_brier_test),
    "pooled_oof_bss_test":      np.array(pooled_oof_bss_test),
    "pooled_oof_ici_train":   np.array(pooled_oof_ici_train), 
    "pooled_oof_ici_test": np.array(pooled_oof_ici_test),
    "mean_per_fold_roc":   float(np.nanmean(test_arr["roc_test"])),
    "mean_per_fold_brier": float(np.nanmean(test_arr["brier_test"])),
    "mean_per_fold_ici":   float(np.nanmean(test_arr["ici_test"])),
    "mean_per_fold_bss":   float(np.nanmean(test_arr["bss_test"])),
    "total_divergences":   int(sum(d["divergences"] for d in fold_diagnostics)),
}

    return {
        "performance_train":      train_arr,
        "performance_test":       test_arr,
        "summary":                summary,
        "model":                  logistic_regression_model,
        "mcmc_last_fold":         mcmc,
        "calibration_train":      calib_train_per_fold,
        "calibration_test":       calib_test_per_fold,
        "calibration_pooled_oof": pooled_calibration_test,
        "oof_probs":              oof_mean_probs,
        "fold_diagnostics":       pd.DataFrame(fold_diagnostics),
    }


# ---------------------------------------------------------------------------
# Bayesian calibration curve with Beta(prior_a, prior_b) prior per bin.
# Jeffreys (0.5, 0.5) is the standard non-informative choice.
# ---------------------------------------------------------------------------
def bayesian_calibration_curve(y_true, probs, n_bins=10, strategy="quantile",
                               prior_a=0.5, prior_b=0.5, ci=0.95):
    y_true = np.asarray(y_true)
    probs  = np.asarray(probs)

    if strategy == "quantile":
        edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    edges[0]  -= 1e-9
    edges[-1] += 1e-9

    bin_idx = np.clip(np.digitize(probs, edges) - 1, 0, n_bins - 1)
    alpha = (1 - ci) / 2

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        k = int(y_true[mask].sum())
        a_post = prior_a + k
        b_post = prior_b + n - k
        rows.append((
            probs[mask].mean(),                            # mean predicted
            a_post / (a_post + b_post),                    # posterior mean
            beta_dist.ppf(alpha,     a_post, b_post),      # lower
            beta_dist.ppf(1 - alpha, a_post, b_post),      # upper
            n,
        ))

    if not rows:
        empty_f = np.empty(0)
        return empty_f, empty_f, empty_f, empty_f, empty_f.astype(int)

    mean_pred, post_mean, lo, hi, n_per_bin = map(np.array, zip(*rows))
    return mean_pred, post_mean, lo, hi, n_per_bin


# ---------------------------------------------------------------------------
# Stratified bootstrap of an operating point at quantile threshold q.
# ---------------------------------------------------------------------------
def bootstrap_op_point(y_arr, probs, q, n_boot=10000, seed=0):
    y_arr = np.asarray(y_arr)
    probs = np.asarray(probs)
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y_arr == 1)[0]
    neg_idx = np.where(y_arr == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Need both classes present for stratified bootstrap.")

    ps = np.empty(n_boot)
    rs = np.empty(n_boot)
    keep = 0
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        thr = np.quantile(probs[idx], q)
        preds = (probs[idx] >= thr).astype(int)
        if preds.sum() == 0:
            continue
        ps[keep] = precision_score(y_arr[idx], preds, zero_division=0)
        rs[keep] = recall_score(y_arr[idx], preds, zero_division=0)
        keep += 1
    return ps[:keep], rs[:keep]
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def calibration_metrics(y_true, y_prob, frac=0.75):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if np.std(y_prob) < 1e-8:
        return float(np.mean(np.abs(y_prob - y_true.mean())))

    order = np.argsort(y_prob)
    p_sorted = y_prob[order]
    y_sorted = y_true[order]

    smoothed = lowess(y_sorted, p_sorted, frac=frac, it=0, return_sorted=False)
    smoothed = np.clip(smoothed, 0.0, 1.0)

    if not np.all(np.isfinite(smoothed)):
        return np.nan

    return float(np.mean(np.abs(p_sorted - smoothed)))



