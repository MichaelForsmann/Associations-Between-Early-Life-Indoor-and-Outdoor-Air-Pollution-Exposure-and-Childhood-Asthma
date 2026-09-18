"""
Nested cross-validation for a calibrated random forest, with out-of-fold SHAP
main effects and per-fold SHAP interaction statistics.

Designed to support three claims in a manuscript:

  1. Discrimination and calibration of the model, fold-wise, on held-out data
     (`fold_metrics`, `performance_table`). Apparent (in-fold) performance is
     reported alongside, clearly labelled, so the optimism gap is visible.
  2. Which features drive the prediction (`shap_importance`, `shap_explanation`).
  3. Whether any feature *pair* departs from additivity
     (`interaction_stats`, `permutation_null_threshold`, `apply_criterion`),
     with a permutation-calibrated threshold so that a null result is
     interpretable rather than an absence of evidence.

Design decisions worth stating in Methods
-----------------------------------------
* Preprocessing (z-scaling, KNN imputation) is fitted on the training rows of
  each outer fold and applied to the held-out rows. Nothing is fitted on the
  full dataset.
* Hyperparameters are selected by RandomizedSearchCV *inside* each outer
  training fold, so the outer test folds are never used for tuning.
* Probabilities come from CalibratedClassifierCV(..., ensemble=False). With
  ensemble=False scikit-learn fits the calibrator by cross_val_predict on the
  training fold and then refits the forest on the whole training fold, so a
  single forest -- trained on all training rows and never on the test rows --
  underlies both the probabilities and the SHAP values.
* SHAP values are computed on outer-test rows only, using that forest, with
  `feature_perturbation="tree_path_dependent"`. For a scikit-learn forest the
  raw output is the class probability, so SHAP values are in probability units
  and base + sum(values) reproduces the *uncalibrated* forest probability.
  Sigmoid calibration is monotone, so it rescales but does not reorder.
* Interaction statistics are computed within each fold and then pooled, so the
  reported spread is a genuine fold-level spread and R_ij is a median of
  ratios rather than a ratio of means.
* The Brier skill score baseline is the training-fold prevalence forecast
  scored on the evaluation rows ("applied"), which is the honest reference.
  `bss_baseline="train_variance"` reproduces the older p(1-p) convention.

Requires: numpy, pandas, scipy, scikit-learn >= 1.0, shap, statsmodels,
matplotlib.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import beta as _beta_dist
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess



# ===================================================================== #
# 1. metrics
# ===================================================================== #

def calibration_metrics(y_true, y_prob, frac=0.75):
    """Integrated calibration index: mean |p - E[y | p]|, E estimated by loess.

    Austin & Steyerberg (2019). Lower is better; 0 is perfect calibration.
    Returned on the probability scale, so an ICI of 0.02 means the predicted
    risk is off by 2 percentage points on average.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    ok = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true, y_prob = y_true[ok], y_prob[ok]
    if y_true.size == 0:
        return float("nan")

    # Degenerate forecast: loess is undefined, fall back on the constant-model
    # calibration error.
    if np.std(y_prob) < 1e-8:
        return float(np.mean(np.abs(y_prob - y_true.mean())))

    order = np.argsort(y_prob, kind="mergesort")
    p_sorted, y_sorted = y_prob[order], y_true[order]

    smoothed = lowess(y_sorted, p_sorted, frac=frac, it=0, return_sorted=False)
    smoothed = np.clip(smoothed, 0.0, 1.0)
    if not np.all(np.isfinite(smoothed)):
        return float("nan")
    return float(np.mean(np.abs(p_sorted - smoothed)))


def _safe_auc(y_true, p):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, p))


def _metrics(y_true, p, baseline_brier):
    """AUROC, Brier, Brier skill score and ICI for one set of predictions."""
    y_true = np.asarray(y_true)
    brier = float(brier_score_loss(y_true, p))
    return {
        "auc": _safe_auc(y_true, p),
        "brier": brier,
        "bss": 1.0 - brier / baseline_brier if baseline_brier > 0 else float("nan"),
        "ici": float(calibration_metrics(y_true, p)),
    }


# ===================================================================== #
# 2. preprocessing and cross-validation helpers
# ===================================================================== #

def _rows(A, idx):
    """Positional row selection that works for DataFrame or ndarray."""
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]


def _make_imputer(n_neighbors):
    try:  # sklearn >= 1.2 keeps all-missing columns instead of dropping them
        return KNNImputer(n_neighbors=n_neighbors, keep_empty_features=True)
    except TypeError:
        return KNNImputer(n_neighbors=n_neighbors)


def _fit_preprocess(X_fit_raw, n_neighbors):
    """Fit scaler + KNN imputer on the training rows only.

    Imputation is done in standardised space so that the neighbour distance is
    not dominated by whichever variable happens to have the largest units, then
    mapped back to the original scale so SHAP dependence plots stay readable.

    Returns (imputed_fit_array, transform_fn).
    """
    A = np.asarray(X_fit_raw, dtype=float)
    if A.shape[0] == 0:
        raise ValueError("empty training fold")
    all_nan = np.isnan(A).all(axis=0)
    if all_nan.any():
        raise ValueError(
            f"columns {np.flatnonzero(all_nan).tolist()} are entirely missing in "
            "a training fold; drop them before running the pipeline"
        )

    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(A)          # StandardScaler ignores NaN
    imputer = _make_imputer(n_neighbors)
    X_fit_imp = scaler.inverse_transform(imputer.fit_transform(X_fit_s))

    def transform(X_new_raw):
        B = np.asarray(X_new_raw, dtype=float)
        return scaler.inverse_transform(imputer.transform(scaler.transform(B)))

    return X_fit_imp, transform


def _split_list(X, y, groups, n_splits, seed):
    """Stratified splits, group-aware when `groups` is supplied."""
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(X, y))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups=groups))


def _sanitize_param_dist(pdist):
    """Wrap bare scalars in a list so RandomizedSearchCV accepts them."""
    if pdist is None:
        return None
    out = {}
    for key, val in pdist.items():
        if hasattr(val, "rvs"):                                   # scipy dist
            out[key] = val
        elif isinstance(val, (list, tuple, np.ndarray)):
            out[key] = list(val)
        else:
            warnings.warn(
                f"param_distributions['{key}'] is a bare scalar ({val!r}); "
                "wrapping it in a list. If you meant a distribution, check that "
                "you imported randint from scipy.stats."
            )
            out[key] = [val]
    return out


def _make_calibrated(rf_params, method, cv):
    """CalibratedClassifierCV(ensemble=False) across sklearn versions."""
    rf = RandomForestClassifier(**rf_params)
    for kwargs in (
        dict(estimator=rf, method=method, cv=cv, ensemble=False),
        dict(base_estimator=rf, method=method, cv=cv, ensemble=False),
        dict(base_estimator=rf, method=method, cv=cv),
    ):
        try:
            return CalibratedClassifierCV(**kwargs)
        except TypeError:
            continue
    raise RuntimeError("could not construct CalibratedClassifierCV")


def _base_estimator_of(calibrated_clf):
    """Recover the forest the calibrator wraps (only valid for ensemble=False).

    With ensemble=False that forest was fitted on the entire training fold, so
    it is the right object to explain. Returns None when it cannot be
    recovered, in which case the caller refits an identical forest.
    """
    ccs = getattr(calibrated_clf, "calibrated_classifiers_", [])
    if len(ccs) != 1:
        return None
    for attr in ("estimator", "base_estimator"):
        est = getattr(ccs[0], attr, None)
        if isinstance(est, RandomForestClassifier):
            return est
    return None


# --- SHAP output-shape normalisation ---------------------------------- #

def _shap_pos(values, n_features, class_index=1):
    """(n, p) positive-class SHAP matrix, whatever layout shap returned."""
    if isinstance(values, list):
        v = np.asarray(values[class_index] if len(values) > 1 else values[0])
    else:
        v = np.asarray(values)
    if v.ndim == 2:
        return v
    if v.ndim == 3 and v.shape[1] == n_features:      # (n, p, n_classes)
        return v[:, :, class_index] if v.shape[2] > 1 else v[:, :, 0]
    raise ValueError(f"unexpected SHAP value shape {v.shape}")


def _shap_inter_pos(values, n_features, class_index=1):
    """(n, p, p) positive-class SHAP interaction tensor."""
    if isinstance(values, list):
        v = np.asarray(values[class_index] if len(values) > 1 else values[0])
    else:
        v = np.asarray(values)
    if v.ndim == 3 and v.shape[1] == v.shape[2] == n_features:
        return v
    if v.ndim == 4:                                   # (n, p, p, n_classes)
        return v[..., class_index] if v.shape[3] > 1 else v[..., 0]
    raise ValueError(f"unexpected SHAP interaction shape {v.shape}")


# ===================================================================== #
# 3. main pipeline
# ===================================================================== #

def nested_cv_calibrated_rf(
    X,
    y,
    *,
    groups=None,
    n_repeats=5,
    n_splits=5,
    n_neighbors_imputer=25,
    param_distributions=None,
    fixed_rf_params=None,
    n_iter_search=20,
    search_cv_folds=5,
    search_scoring="roc_auc",
    calibration_method="sigmoid",
    calibration_cv_folds=5,
    base_rf_kwargs=None,
    bss_baseline="applied",
    compute_train=True,
    compute_shap=True,
    compute_interactions=True,
    interaction_features=None,
    interaction_max_rows=None,
    n_jobs=-1,
    random_state=42,
    verbose=True,
):
    """Repeated nested CV for a calibrated random forest.

    Parameters
    ----------
    X : DataFrame or ndarray, shape (n, p). May contain NaN.
    y : binary target, shape (n,).
    groups : array-like or None
        Family / subject identifiers. Splitting, hyperparameter search and
        calibration all become group-aware when supplied. Repeats are obtained
        by varying the shuffle seed, since StratifiedGroupKFold has no repeated
        variant.
    param_distributions : dict or None
        Search space for RandomizedSearchCV. If None, no search is run and
        `fixed_rf_params` is used directly (this is what the permutation null
        does -- tuning against a permuted outcome is meaningless and dominates
        the runtime).
    fixed_rf_params : dict or None
        Forest hyperparameters used when `param_distributions` is None.
    bss_baseline : {"applied", "train_variance"}
        Reference forecast for the Brier skill score.
    compute_train : bool
        Also score the calibrated model on its own training rows. These are
        apparent, in-sample numbers -- a random forest largely memorises its
        training data, so they are reported only to show the optimism gap and
        must never be quoted as performance.
    interaction_features : list of names/indices or None
        Restrict the reported interaction matrix to these features. Note that
        TreeExplainer computes the full p x p tensor regardless; this controls
        what is stored and screened, not the cost.
    interaction_max_rows : int or None
        Explain at most this many test rows per fold for the interaction
        tensor (rows are sampled without replacement). SHAP interaction values
        cost roughly p times a normal SHAP call, so this is the knob to turn
        when the run is too slow. Main effects always use all test rows.

    Returns
    -------
    dict. `fold_metrics` and `summary` are the performance results;
    `interaction_stats` and `fold_interactions` feed the interaction figures.
    """
    param_distributions = _sanitize_param_dist(param_distributions)
    base_rf_kwargs = dict(base_rf_kwargs or {})
    base_rf_kwargs.setdefault("n_jobs", n_jobs)
    base_rf_kwargs.pop("random_state", None)          # set per fold instead
    fixed_rf_params = dict(fixed_rf_params or {})
    fixed_rf_params.pop("random_state", None)

    if param_distributions is None and not fixed_rf_params:
        warnings.warn(
            "no param_distributions and no fixed_rf_params: falling back to "
            "scikit-learn's RandomForestClassifier defaults"
        )
    if bss_baseline not in ("applied", "train_variance"):
        raise ValueError("bss_baseline must be 'applied' or 'train_variance'")

    feature_names = (
        list(X.columns) if hasattr(X, "columns")
        else [f"x{i}" for i in range(X.shape[1])]
    )
    y_arr = np.asarray(y).ravel()
    if np.unique(y_arr).size != 2:
        raise ValueError("this pipeline is for binary outcomes")
    n_samples, n_features = X.shape
    prevalence = float(np.mean(y_arr))

    # ---- interaction bookkeeping ----
    if interaction_features is None:
        inter_idx = np.arange(n_features)
    else:
        inter_idx = np.array([
            f if isinstance(f, (int, np.integer)) else feature_names.index(f)
            for f in interaction_features
        ])
    inter_names = [feature_names[i] for i in inter_idx]

    # ---- outer splits ----
    if groups is None:
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
        outer_splits = list(splitter.split(X, y_arr))
        repeat_id = np.repeat(np.arange(n_repeats), n_splits)
    else:
        groups = np.asarray(groups)
        outer_splits, repeat_id = [], []
        for r in range(n_repeats):
            spl = _split_list(X, y_arr, groups, n_splits, random_state + r)
            outer_splits.extend(spl)
            repeat_id.extend([r] * len(spl))
        repeat_id = np.asarray(repeat_id)
    n_folds = len(outer_splits)

    # ---- storage ----
    rows_metrics = []
    best_params_per_fold = []

    oof_sum = np.zeros(n_samples)
    oof_count = np.zeros(n_samples)

    shap_sum = np.zeros((n_samples, n_features))
    shap_count = np.zeros(n_samples)
    base_sum = np.zeros(n_samples)
    disp_sum = np.zeros((n_samples, n_features))
    fold_interactions = []            # list of (row_idx, (m, k, k) float32)

    def _baseline_brier(p_ref, y_eval):
        if bss_baseline == "train_variance":
            return float(p_ref * (1.0 - p_ref))
        return float(np.mean((p_ref - np.asarray(y_eval, dtype=float)) ** 2))

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    for j, (train_idx, test_idx) in enumerate(outer_splits):

        X_train_raw, X_test_raw = _rows(X, train_idx), _rows(X, test_idx)
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]
        g_train = None if groups is None else groups[train_idx]

        fold_prev = float(np.mean(y_train))
        base_test = _baseline_brier(fold_prev, y_test)
        base_train = _baseline_brier(fold_prev, y_train)

        # ---- preprocessing, fitted on the training rows only ----
        X_train_imp, tf = _fit_preprocess(X_train_raw, n_neighbors_imputer)
        X_test_imp = tf(X_test_raw)

        # ---- step 1: hyperparameter search inside the training fold ----
        if param_distributions is None:
            best_params = dict(fixed_rf_params)
        else:
            if groups is None:
                search_cv = StratifiedKFold(
                    n_splits=search_cv_folds, shuffle=True,
                    random_state=random_state + j,
                )
                fit_kw = {}
            else:
                search_cv = StratifiedGroupKFold(
                    n_splits=search_cv_folds, shuffle=True,
                    random_state=random_state + j,
                )
                fit_kw = {"groups": g_train}

            search = RandomizedSearchCV(
                estimator=RandomForestClassifier(
                    **base_rf_kwargs, random_state=random_state + j
                ),
                param_distributions=param_distributions,
                n_iter=n_iter_search,
                scoring=search_scoring,
                cv=search_cv,
                n_jobs=n_jobs,
                random_state=random_state + j,
                refit=False,          # the refit happens inside calibration
            )
            search.fit(X_train_imp, y_train, **fit_kw)
            best_params = dict(search.best_params_)

        best_params_per_fold.append(dict(best_params))
        rf_params = {**base_rf_kwargs, **best_params,
                     "random_state": random_state + j}

        # ---- step 2: calibrated model on the full training fold ----
        cal_cv = (
            calibration_cv_folds if groups is None
            else _split_list(X_train_imp, y_train, g_train,
                             calibration_cv_folds, random_state + j)
        )
        calibrated_clf = _make_calibrated(rf_params, calibration_method, cal_cv)
        calibrated_clf.fit(X_train_imp, y_train)

        test_probs = calibrated_clf.predict_proba(X_test_imp)[:, 1]
        rec = {
            "fold": j,
            "repeat": int(repeat_id[j]),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "prev_train": fold_prev,
            "prev_test": float(np.mean(y_test)),
            "baseline_brier_test": base_test,
        }
        rec.update({f"{k}_test": v
                    for k, v in _metrics(y_test, test_probs, base_test).items()})

        if compute_train:
            train_probs = calibrated_clf.predict_proba(X_train_imp)[:, 1]
            rec.update({f"{k}_train_apparent": v for k, v in
                        _metrics(y_train, train_probs, base_train).items()})

        rows_metrics.append(rec)
        oof_sum[test_idx] += test_probs
        oof_count[test_idx] += 1

        # ---- step 3: SHAP on the held-out rows ----
        if compute_shap:
            interp_rf = _base_estimator_of(calibrated_clf)
            if interp_rf is None:                       # older sklearn fallback
                interp_rf = RandomForestClassifier(**rf_params)
                interp_rf.fit(X_train_imp, y_train)

            explainer = shap.TreeExplainer(
                interp_rf, feature_perturbation="tree_path_dependent"
            )
            sv = _shap_pos(
                explainer.shap_values(X_test_imp, check_additivity=False),
                n_features,
            )
            ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
            base_val = float(ev[1]) if ev.size > 1 else float(ev[0])

            shap_sum[test_idx] += sv
            shap_count[test_idx] += 1
            base_sum[test_idx] += base_val
            disp_sum[test_idx] += X_test_imp

            if compute_interactions:
                sel = np.arange(len(test_idx))
                if interaction_max_rows and len(sel) > interaction_max_rows:
                    sel = np.sort(rng.choice(len(sel), interaction_max_rows,
                                             replace=False))
                iv = _shap_inter_pos(
                    explainer.shap_interaction_values(X_test_imp[sel]),
                    n_features,
                )
                iv = iv[:, inter_idx][:, :, inter_idx].astype(np.float32)
                fold_interactions.append((np.asarray(test_idx)[sel], iv))
                del iv

        if verbose:
            msg = (f"fold {j + 1:>3}/{n_folds}  "
                   f"test AUC={rec['auc_test']:.3f}  "
                   f"BSS={rec['bss_test']:+.3f}  ICI={rec['ici_test']:.3f}")
            if compute_train:
                msg += f"   | apparent AUC={rec['auc_train_apparent']:.3f}"
            print(msg, flush=True)
    # ------------------------------------------------------------------ #

    fold_metrics = pd.DataFrame(rows_metrics)

    if not (oof_count > 0).all():
        raise RuntimeError("some rows never appeared in a test fold")
    oof_probs = oof_sum / oof_count
    pooled_brier = float(brier_score_loss(y_arr, oof_probs))
    mean_baseline = float(fold_metrics["baseline_brier_test"].mean())

    summary = {
        "model": "calibrated_rf",
        "calibration": calibration_method,
        "calibration_ensemble": False,
        "grouped_cv": groups is not None,
        "n_repeats": n_repeats,
        "n_splits": n_splits,
        "n_folds": n_folds,
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "prevalence": prevalence,
        "bss_baseline": bss_baseline,
        "mean_fold_baseline_brier": mean_baseline,
        "test_auc_mean": float(fold_metrics["auc_test"].mean()),
        "test_auc_sd": float(fold_metrics["auc_test"].std(ddof=1)),
        "test_auc_q05": float(fold_metrics["auc_test"].quantile(0.05)),
        "test_auc_q95": float(fold_metrics["auc_test"].quantile(0.95)),
        "test_brier_mean": float(fold_metrics["brier_test"].mean()),
        "test_bss_mean": float(fold_metrics["bss_test"].mean()),
        "test_bss_sd": float(fold_metrics["bss_test"].std(ddof=1)),
        "test_ici_mean": float(fold_metrics["ici_test"].mean()),
        "pooled_oof_auc": _safe_auc(y_arr, oof_probs),
        "pooled_oof_brier": pooled_brier,
        "pooled_oof_bss": 1.0 - pooled_brier / mean_baseline,
        "pooled_oof_ici": float(calibration_metrics(y_arr, oof_probs)),
        "pooled_note": (
            "pooled_oof_* averages each row's prediction over the repeats and is "
            "therefore an ensemble; report the fold-wise test_* as primary"
        ),
    }
    if compute_train:
        summary["train_apparent_auc_mean"] = float(
            fold_metrics["auc_train_apparent"].mean())
        summary["train_apparent_note"] = (
            "in-sample fit of a random forest; shown only as an optimism check"
        )

    # ---- SHAP aggregation (main effects; interactions stay per fold) ----
    shap_values_oof = shap_importance = base_values_oof = None
    X_display = shap_explanation = interaction_stats = None

    if compute_shap and shap_count.sum() > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            shap_values_oof = shap_sum / shap_count[:, None]
            base_values_oof = base_sum / shap_count
            X_display = disp_sum / shap_count[:, None]
        shap_values_oof = np.nan_to_num(shap_values_oof, nan=0.0)
        base_values_oof = np.nan_to_num(base_values_oof, nan=prevalence)
        X_display = np.nan_to_num(X_display, nan=0.0)

        shap_importance = (
            pd.DataFrame({
                "feature": feature_names,
                "mean_abs_shap": np.abs(shap_values_oof).mean(axis=0),
            })
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_explanation = shap.Explanation(
            values=shap_values_oof,
            base_values=base_values_oof,
            data=X_display,
            feature_names=feature_names,
        )
        if compute_interactions and fold_interactions:
            interaction_stats = interaction_stats_by_fold(
                fold_interactions, inter_names
            )

    return {
        "summary": summary,
        "fold_metrics": fold_metrics,
        "oof_probs": oof_probs,
        "oof_count": oof_count,
        "oof_y": y_arr,
        "best_params_per_fold": best_params_per_fold,
        "feature_names": feature_names,
        "shap_values_oof": shap_values_oof,
        "shap_count": shap_count,
        "shap_importance": shap_importance,
        "base_values_oof": base_values_oof,
        "X_display": X_display,
        "shap_explanation": shap_explanation,
        "fold_interactions": fold_interactions,
        "interaction_features": inter_names,
        "interaction_stats": interaction_stats,
    }


def performance_table(res, decimals=3):
    """Publication-ready performance summary: mean (SD) [5th, 95th percentile].

    One row per metric; test performance first, apparent performance last.
    """
    fm = res["fold_metrics"]
    order = ["auc", "brier", "bss", "ici"]
    pretty = {"auc": "AUROC", "brier": "Brier score",
              "bss": "Brier skill score", "ici": "ICI"}

    rows = []
    for split, suffix in (("Held-out test", "_test"),
                          ("Training (apparent)", "_train_apparent")):
        for m in order:
            col = m + suffix
            if col not in fm:
                continue
            v = fm[col].to_numpy(dtype=float)
            rows.append({
                "split": split,
                "metric": pretty[m],
                "mean": round(float(np.nanmean(v)), decimals),
                "sd": round(float(np.nanstd(v, ddof=1)), decimals),
                "q05": round(float(np.nanquantile(v, 0.05)), decimals),
                "q95": round(float(np.nanquantile(v, 0.95)), decimals),
                "n_folds": int(np.sum(np.isfinite(v))),
            })
    out = pd.DataFrame(rows)
    out["reported"] = [
        f"{r['mean']:.{decimals}f} ({r['sd']:.{decimals}f}) "
        f"[{r['q05']:.{decimals}f}, {r['q95']:.{decimals}f}]"
        for _, r in out.iterrows()
    ]
    return out


# ===================================================================== #
# 4. interaction statistics and the permutation null
# ===================================================================== #

def interaction_stats_by_fold(fold_interactions, inter_names, eps=1e-12):
    """R_ij and sign consistency computed within each fold, then pooled.

        R_ij = |2 Phi_ij| / (0.5 * (|Phi_ii| + |Phi_jj|))

    Evaluated fold-wise, so the median of ratios is reported rather than a
    ratio of means, and a fold-level spread exists for the stability criterion.

    Returns one row per unordered pair with both the normalised R_ij (is the
    model additive for this pair?) and the unnormalised |2 Phi_ij| in model
    output units (does the pair matter in absolute terms?).
    """
    k = len(inter_names)
    n_folds = len(fold_interactions)
    if n_folds == 0:
        return pd.DataFrame()

    R = np.full((n_folds, k, k), np.nan)
    A = np.full((n_folds, k, k), np.nan)
    S = np.full((n_folds, k, k), np.nan)

    for f, (_, iv) in enumerate(fold_interactions):
        iv = np.asarray(iv, dtype=np.float64)
        abs_off = np.abs(iv).mean(axis=0)
        signed = iv.mean(axis=0)
        main = np.abs(np.einsum("nii->ni", iv)).mean(axis=0)
        denom = 0.5 * (main[:, None] + main[None, :]) + eps
        R[f] = (2.0 * abs_off) / denom
        A[f] = 2.0 * abs_off
        S[f] = 2.0 * signed

    rows = []
    for a, b in zip(*np.triu_indices(k, k=1)):
        r = R[:, a, b]
        sgn = np.sign(S[:, a, b])
        n_pos, n_neg = np.nansum(sgn > 0), np.nansum(sgn < 0)
        dominant = 1.0 if n_pos >= n_neg else -1.0
        rows.append({
            "feature_i": inter_names[a],
            "feature_j": inter_names[b],
            "R_median": np.nanmedian(r),
            "R_q05": np.nanquantile(r, 0.05),
            "R_q95": np.nanquantile(r, 0.95),
            "R_mean": np.nanmean(r),
            "abs_2phi_median": np.nanmedian(A[:, a, b]),
            "signed_2phi_median": np.nanmedian(S[:, a, b]),
            "sign_consistency": float(np.nanmean(sgn == dominant)),
            "dominant_sign": dominant,
            "n_folds": int(np.sum(np.isfinite(r))),
        })

    return (pd.DataFrame(rows)
            .sort_values("R_median", ascending=False)
            .reset_index(drop=True))


def apply_criterion(stats, R_threshold=0.1, sign_frac=0.80, abs_threshold=None):
    """Flag pairs meeting the pre-specified non-null criterion.

    Pre-specify R_threshold and sign_frac in the protocol. The permutation null
    can then only tighten the rule, never loosen it -- report both numbers.
    """
    out = stats.copy()
    keep = (out["R_median"] > R_threshold) & (out["sign_consistency"] >= sign_frac)
    if abs_threshold is not None:
        keep &= out["abs_2phi_median"] > abs_threshold
    out["non_null"] = keep
    return out


def _permute(y, groups, rng):
    """Break the X -> y link while preserving the (y, group) joint structure.

    Permuting the (outcome, group-label) pairs together, rather than the
    outcome alone, keeps the number and size of clusters and the within-cluster
    outcome pattern intact, so the null run faces the same clustered design as
    the observed run.
    """
    y = np.asarray(y).ravel()
    perm = rng.permutation(y.size)
    if groups is None:
        return y[perm], None
    return y[perm], np.asarray(groups)[perm]


def permutation_null_threshold(
    X, y, *, groups=None, n_perm=200, quantile=0.95, fixed_rf_params=None,
    seed=0, verbose=True, **fit_kwargs
):
    """Calibrate the R_ij threshold against outcome-permuted data.

    Mean absolute interaction is non-negative and upward-biased, and k(k-1)/2
    pairs are screened, so the relevant null is the distribution of the LARGEST
    R_ij per permuted run. The returned quantile therefore controls the
    family-wise false-positive rate across pairs.

    Pass the modal best_params_ from the observed run as `fixed_rf_params`; no
    hyperparameter search is run, because tuning against a permuted outcome is
    meaningless and would dominate the runtime. Keep `n_splits` the same as the
    observed run so the fold-level medians are comparable; `n_repeats` may be
    reduced for speed.
    """
    rng = np.random.default_rng(seed)
    fit_kwargs.setdefault("n_repeats", 1)
    maxima = np.empty(n_perm)

    for b in range(n_perm):
        y_perm, g_perm = _permute(y, groups, rng)
        res = nested_cv_calibrated_rf(
            X, y_perm,
            groups=g_perm,
            param_distributions=None,
            fixed_rf_params=fixed_rf_params,
            compute_train=False,
            compute_shap=True,
            compute_interactions=True,
            verbose=False,
            random_state=10_000 + b,
            **fit_kwargs,
        )
        maxima[b] = float(res["interaction_stats"]["R_median"].max())
        if verbose and (b + 1) % 10 == 0:
            running = np.quantile(maxima[:b + 1], quantile)
            print(f"  permutation {b + 1}/{n_perm}  "
                  f"running q{int(quantile * 100)}={running:.3f}", flush=True)

    return {
        "threshold": float(np.quantile(maxima, quantile)),
        "null_maxima": maxima,
        "quantile": quantile,
        "n_perm": n_perm,
    }


# ===================================================================== #
# 5. reading the interaction tensors back out
# ===================================================================== #

def pooled_interaction_matrix(res, signed=False):
    """Pooled interaction matrices as labelled DataFrames.

    Returns (M, R) where M holds the total pairwise effect 2*Phi_ij off the
    diagonal and the main effect Phi_ii on it, and R is the normalised ratio
    from the same pooled matrix. Fold-wise statistics from
    `interaction_stats_by_fold` are the ones to report; this is for heatmaps.
    """
    names = res["interaction_features"]
    k = len(names)
    total = np.zeros((k, k))
    n_rows = 0
    for _, iv in res["fold_interactions"]:
        iv = np.asarray(iv, dtype=np.float64)
        total += (iv if signed else np.abs(iv)).sum(axis=0)
        n_rows += iv.shape[0]
    if n_rows == 0:
        raise ValueError("no interaction tensors were stored")

    cm = total / n_rows
    M = cm * 2.0
    np.fill_diagonal(M, np.diag(cm))
    d = np.abs(np.diag(M))
    R = np.abs(M) / (0.5 * np.add.outer(d, d) + 1e-12)
    frame = lambda a: pd.DataFrame(a, index=names, columns=names)
    return frame(M), frame(R)


def interaction_pair_values(res, feat_i, feat_j):
    """Per-row total interaction 2*Phi_ij, aligned with the rows of X.

    NaN for rows that were never included in a stored interaction tensor
    (possible when `interaction_max_rows` was used).
    """
    names = res["interaction_features"]
    a, b = names.index(feat_i), names.index(feat_j)
    n = len(res["oof_y"])
    s, c = np.zeros(n), np.zeros(n)
    for idx, iv in res["fold_interactions"]:
        s[idx] += 2.0 * np.asarray(iv[:, a, b], dtype=np.float64)
        c[idx] += 1
    out = np.full(n, np.nan)
    seen = c > 0
    out[seen] = s[seen] / c[seen]
    return out


def prevalence_grid(X, y, exposure, modifier, n_bins=3):
    """Model-free companion: observed prevalence by bin x bin, exact 95% CIs.

    Tests the same hypothesis with no SHAP machinery, which is the version an
    epidemiology reviewer will trust.
    """
    df = pd.DataFrame({
        "e": np.asarray(X[exposure], dtype=float),
        "g": np.asarray(X[modifier], dtype=float),
        "y": np.asarray(y, dtype=float),
    }).dropna()
    df["y"] = df["y"].astype(int)
    df["e_bin"] = pd.qcut(df.e, n_bins, labels=False, duplicates="drop")
    df["g_bin"] = pd.qcut(df.g, n_bins, labels=False, duplicates="drop")

    rows = []
    for (g, e), sub in df.groupby(["g_bin", "e_bin"], observed=True):
        n, k = len(sub), int(sub.y.sum())
        lo = float(_beta_dist.ppf(0.025, k, n - k + 1)) if k > 0 else 0.0
        hi = float(_beta_dist.ppf(0.975, k + 1, n - k)) if k < n else 1.0
        rows.append({
            f"{modifier}_bin": int(g) + 1,
            f"{exposure}_bin": int(e) + 1,
            "n": n, "cases": k, "prevalence": k / n,
            "ci_low": lo, "ci_high": hi,
        })
    return pd.DataFrame(rows)


# ===================================================================== #
# 6. SHAP display helpers
# ===================================================================== #

def _force_payload(expl, idx=None):
    """JSON-safe (scalar base, values, feature frame) for legacy force plots."""
    idx = np.arange(expl.values.shape[0]) if idx is None else np.asarray(idx)
    vals = np.asarray(expl.values, dtype=np.float64)[idx]
    base = np.asarray(expl.base_values, dtype=np.float64)[idx]
    data = np.asarray(expl.data, dtype=np.float64)[idx]
    names = [str(f) for f in expl.feature_names]

    b0 = float(base.mean())
    drift = float(np.abs(base - b0).max())
    if drift > 1e-3:
        print(f"[warn] base value varies by up to {drift:.4g} across rows; the "
              f"plot uses the mean, so additivity is approximate.")
    return b0, vals, pd.DataFrame(data, columns=names)


def shap_overview(res, top_k=15, show=True):
    """Beeswarm, bar and a dependence scatter for the top feature."""
    expl = res["shap_explanation"]
    try:
        shap.plots.beeswarm(expl, max_display=top_k, show=show)
        shap.plots.bar(expl, max_display=top_k, show=show)
        top_feat = res["shap_importance"].iloc[0]["feature"]
        shap.plots.scatter(expl[:, top_feat], color=expl, show=show)
    except Exception:                                    # older shap
        shap.summary_plot(
            res["shap_values_oof"], res["X_display"],
            feature_names=res["feature_names"], max_display=top_k,
        )


def interactive_force_html(res, out_html="shap_oof_force.html",
                           max_samples=500, order_by="prediction",
                           random_state=0):
    """Stacked force plot (JS/D3). Rendering thousands of rows in a browser is
    slow, so the rows are subsampled across the prediction range."""
    expl = res["shap_explanation"]
    n = expl.values.shape[0]
    if n > max_samples:
        if order_by == "prediction":
            pred = np.asarray(expl.base_values) + np.asarray(expl.values).sum(axis=1)
            idx = np.argsort(pred)[np.linspace(0, n - 1, max_samples).astype(int)]
        else:
            idx = np.random.default_rng(random_state).choice(
                n, max_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    b0, vals, feats = _force_payload(expl, idx)
    plot = shap.force_plot(b0, vals, feats)
    shap.save_html(out_html, plot)
    print(f"[saved] {out_html}  ({len(idx)} rows)")
    return plot


def interactive_single_subject(res, i, out_html=None):
    """Force plot plus waterfall for one subject (i indexes the rows of X)."""
    expl = res["shap_explanation"]
    b0, vals, feats = _force_payload(expl, [i])
    plot = shap.force_plot(b0, vals[0], feats.iloc[0])
    if out_html:
        shap.save_html(out_html, plot)
        print(f"[saved] {out_html}")
    try:
        shap.plots.waterfall(expl[i], max_display=12, show=True)
    except Exception:
        shap.waterfall_plot(expl[i], max_display=12, show=True)
    return plot


# ===================================================================== #
# 7. figures
# ===================================================================== #

import matplotlib as mpl              # noqa: E402  (kept with the figure code)
import matplotlib.pyplot as plt       # noqa: E402
from matplotlib.lines import Line2D   # noqa: E402

MM = 1 / 25.4
SINGLE, DOUBLE = 88 * MM, 180 * MM     # Nature-family column widths

JOURNAL_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.9,
    "savefig.dpi": 600,
    "figure.dpi": 150,
    "pdf.fonttype": 42,       # editable text, required by most journals
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def use_journal_style():
    """Apply the figure defaults. Call once before building figures."""
    mpl.rcParams.update(JOURNAL_RC)


NAVY, RED, GREY = "#1f3b73", "#b2182b", "#8c8c8c"

PRETTY = {
    "NO2": r"NO$_2$", "NOx": r"NO$_x$", "PM2.5": r"PM$_{2.5}$",
    "Street NO2": r"Street NO$_2$", "Street NOx": r"Street NO$_x$",
    "Street PM2.5": r"Street PM$_{2.5}$",
    "Black Carbon": "Black carbon", "Formaldehyde": "Formald.",
    "Acetaldehyde": "Acetald.", "Gestational age": "Gest. age",
    "Mother education": "Mat. educ.",
}


def lbl(name):
    return PRETTY.get(name, str(name))


def plot_interaction_caterpillar(ax, stats, threshold=None, top_n=12,
                                 colour=NAVY, show_ylabels=True, title=None,
                                 xmax=None, threshold_label=None):
    """Top pairs by R_ij with their fold spread, against the null threshold.

    This is the primary interaction panel. A heatmap of faint cells asks the
    reader to trust that nothing is there; this shows the comparison.
    """
    d = stats.head(top_n).iloc[::-1]
    ypos = np.arange(len(d))

    if threshold is not None:
        ax.axvspan(threshold, xmax if xmax else float(d.R_q95.max()) * 1.2,
                   color=RED, alpha=0.06, lw=0)
        ax.axvline(threshold, color=RED, ls=(0, (3, 2)), lw=0.9, zorder=3,
                   label=threshold_label)

    ax.hlines(ypos, d.R_q05, d.R_q95, color=GREY, lw=1.1, zorder=1)
    ax.scatter(d.R_median, ypos, s=11, color=colour, zorder=2,
               edgecolor="white", linewidth=0.3)

    ax.set_yticks(ypos)
    ax.set_yticklabels(
        [f"{lbl(a)} \u00d7 {lbl(b)}" for a, b in zip(d.feature_i, d.feature_j)]
        if show_ylabels else []
    )
    ax.set_ylim(-0.8, len(d) - 0.2)
    ax.set_xlim(0, xmax)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color="0.9", lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=3)
    if threshold_label:
        ax.legend(frameon=False, loc="lower right", handlelength=1.4)
    return ax


R_AXIS_LABEL = (r"$R_{ij}=|2\Phi_{ij}|\,/\,"
                r"\frac{1}{2}(|\Phi_{ii}|+|\Phi_{jj}|)$")


def plot_permutation_null(ax, null_maxima, observed_max, threshold,
                          quantile=0.95):
    """Null distribution of the largest R_ij, with the observed maximum."""
    ax.hist(null_maxima, bins=24, color="0.82", edgecolor="white", lw=0.3)
    ax.axvline(threshold, color=RED, ls=(0, (3, 2)), lw=0.9)
    ax.axvline(observed_max, color=NAVY, lw=1.2)
    ax.set_xlabel(r"max $R_{ij}$ per permuted run")
    ax.set_ylabel("Permutations")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Line2D([], [], color=NAVY, lw=1.2, label="observed"),
        Line2D([], [], color=RED, ls=(0, (3, 2)), lw=0.9,
               label=f"null q{int(quantile * 100)}"),
    ], frameon=False, loc="upper right", handlelength=1.4, borderpad=0.2)
    return ax


def plot_shap_dependence(ax, shap_values, X_display, feature_names,
                         exposure, modifier, cax=None, s=4):
    """SHAP dependence for one pre-specified pair, coloured by the modifier.

    A pre-specified negative deserves its own panel: the dependence plot shows
    whether there is crossover shape that a single summary index averages away.
    """
    xi = feature_names.index(exposure)
    mi = feature_names.index(modifier)
    sc = ax.scatter(X_display[:, xi], shap_values[:, xi], c=X_display[:, mi],
                    cmap="coolwarm", s=s, linewidths=0, alpha=0.85,
                    rasterized=True)
    ax.axhline(0, color="0.75", lw=0.6, zorder=0)
    ax.set_xlabel(f"{lbl(exposure)} (SD)")
    ax.set_ylabel(f"SHAP value, {lbl(exposure)}")
    ax.spines[["top", "right"]].set_visible(False)
    if cax is not None:
        cb = plt.colorbar(sc, cax=cax)
        cb.set_label(f"{lbl(modifier)} (SD)", labelpad=2)
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(width=0.4, length=1.8)
    return ax


def plot_prevalence_grid(ax, grid, modifier, exposure):
    """Observed prevalence by exposure bin, within each modifier bin."""
    gcol, ecol = f"{modifier}_bin", f"{exposure}_bin"
    levels = sorted(grid[gcol].unique())
    offsets = np.linspace(-0.18, 0.18, len(levels))
    shades = plt.cm.Blues(np.linspace(0.45, 0.95, len(levels)))

    for gi, g in enumerate(levels):
        sub = grid[grid[gcol] == g].sort_values(ecol)
        x = sub[ecol].to_numpy(float) + offsets[gi]
        ax.vlines(x, sub.ci_low, sub.ci_high, color=shades[gi], lw=1.0)
        ax.plot(x, sub.prevalence, "o", ms=3.2, color=shades[gi],
                mec="white", mew=0.3, label=f"T{int(g)}")

    ticks = sorted(grid[ecol].unique())
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"T{int(t)}" for t in ticks])
    ax.set_xlabel(f"{lbl(exposure)} bin")
    ax.set_ylabel("Outcome prevalence")
    ax.set_ylim(0, float(grid.ci_high.max()) * 1.28)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(title=lbl(modifier), frameon=False, ncol=len(levels),
                    handlelength=0.8, columnspacing=0.8, loc="upper left",
                    borderpad=0.1)
    leg.get_title().set_fontsize(6)
    return ax


def plot_reliability(ax, y_true, y_prob, n_bins=10, frac=0.75):
    """Reliability diagram: loess curve plus quantile bins, for the ICI panel."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(y_prob, kind="mergesort")
    p, yv = y_prob[order], y_true[order]

    sm = np.clip(lowess(yv, p, frac=frac, it=0, return_sorted=False), 0, 1)
    ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, ls=(0, (3, 2)))
    ax.plot(p, sm, color=NAVY, lw=1.1, label="loess")

    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bx = np.array([p[which == b].mean() for b in range(n_bins)])
    by = np.array([yv[which == b].mean() for b in range(n_bins)])
    ax.plot(bx, by, "o", ms=3.0, color=RED, mec="white", mew=0.3,
            label="decile")

    ax.set_xlabel("Predicted risk")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, max(p.max(), by.max()) * 1.05)
    ax.set_ylim(0, max(p.max(), by.max()) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=1.2)
    return ax


def _panel_letter(ax, letter, dx=0.0, dy=1.16):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=8,
            fontweight="bold", va="top", ha="left")


def build_figure(stats_by_endpoint, null_by_endpoint, shap_values, X_display,
                 feature_names, prev_grid, exposure, modifier, top_n=12,
                 endpoint_colours=None):
    """Six-panel double-column interaction figure.

      a-c  R_ij caterpillars, one per endpoint, shared x-axis, with the
           permutation-calibrated null threshold.
      d    Null distribution of max R_ij with the observed maximum overlaid.
      e    SHAP dependence for the pre-specified pair.
      f    Model-free prevalence grid with exact 95% CIs.

    `stats_by_endpoint` and `null_by_endpoint` are dicts keyed identically
    (e.g. by age); up to three endpoints are drawn on the top row.
    """
    use_journal_style()
    keys = sorted(stats_by_endpoint)[:3]
    if endpoint_colours is None:
        palette = ["#3b6ea5", "#d1802b", "#4a8f5b"]
        endpoint_colours = {k: palette[i % 3] for i, k in enumerate(keys)}

    fig = plt.figure(figsize=(DOUBLE, 4.4))
    gs = fig.add_gridspec(
        2, 4, height_ratios=[1.35, 1.0], width_ratios=[1.30, 1.0, 1.0, 0.045],
        hspace=0.50, wspace=0.34,
        left=0.205, right=0.935, top=0.905, bottom=0.105,
    )

    xmax = max(
        max(stats_by_endpoint[k].head(top_n).R_q95.max() for k in keys),
        max(null_by_endpoint[k]["threshold"] for k in keys),
    ) * 1.12

    axes_top = []
    for i, k in enumerate(keys):
        ax = fig.add_subplot(gs[0, i])
        plot_interaction_caterpillar(
            ax, stats_by_endpoint[k], null_by_endpoint[k]["threshold"],
            top_n=top_n, colour=endpoint_colours[k], show_ylabels=(i == 0),
            title=str(k), xmax=xmax,
        )
        _panel_letter(ax, "abc"[i])
        axes_top.append(ax)
    axes_top[len(axes_top) // 2].set_xlabel(R_AXIS_LABEL, labelpad=2)

    ref = keys[len(keys) // 2]
    ax_d = fig.add_subplot(gs[1, 0])
    plot_permutation_null(
        ax_d, null_by_endpoint[ref]["null_maxima"],
        float(stats_by_endpoint[ref].R_median.max()),
        null_by_endpoint[ref]["threshold"],
        quantile=null_by_endpoint[ref].get("quantile", 0.95),
    )
    _panel_letter(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 3])
    plot_shap_dependence(ax_e, shap_values, X_display, feature_names,
                         exposure, modifier, cax=cax)
    _panel_letter(ax_e, "e")

    ax_f = fig.add_subplot(gs[1, 2])
    plot_prevalence_grid(ax_f, prev_grid, modifier, exposure)
    _panel_letter(ax_f, "f")

    return fig


# ===================================================================== #
# 8. reproducibility
# ===================================================================== #

def session_info():
    """Package versions, for the reproducibility statement."""
    import platform
    import sklearn
    import scipy
    import statsmodels
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "shap": shap.__version__,
        "statsmodels": statsmodels.__version__,
        "matplotlib": mpl.__version__,
    }


# ===================================================================== #
if __name__ == "__main__":
    HOWTO = """
    from scipy.stats import randint, uniform

    param_dist = {
        "n_estimators":      randint(300, 1200),
        "max_depth":         randint(2, 12),
        "min_samples_leaf":  randint(5, 60),
        "max_features":      uniform(0.1, 0.7),
        "class_weight":      ["balanced", None],
    }

    res = nested_cv_calibrated_rf(
        X, y, groups=family_id,
        n_repeats=5, n_splits=5,
        param_distributions=param_dist,
        interaction_features=["PRS", "NO2", "PM2.5", "BMI"],
        interaction_max_rows=400,
    )

    print(performance_table(res).to_string(index=False))

    # interactions: pre-specified rule first, then the permutation calibration
    modal = pd.DataFrame(res["best_params_per_fold"]).mode().iloc[0].to_dict()
    null = permutation_null_threshold(
        X, y, groups=family_id, n_perm=200, fixed_rf_params=modal,
        n_splits=5, n_repeats=1,
        interaction_features=["PRS", "NO2", "PM2.5", "BMI"],
        interaction_max_rows=400,
    )
    stats = apply_criterion(res["interaction_stats"],
                            R_threshold=max(0.1, null["threshold"]))
    stats.to_csv("supp_interactions.csv", index=False)

    fig = build_figure({"12 years": stats}, {"12 years": null},
                       res["shap_values_oof"], res["X_display"],
                       res["feature_names"],
                       prevalence_grid(X, y, "NO2", "PRS"),
                       exposure="NO2", modifier="PRS")
    fig.savefig("figure_interactions.pdf", bbox_inches="tight")
    """
    print(HOWTO)
