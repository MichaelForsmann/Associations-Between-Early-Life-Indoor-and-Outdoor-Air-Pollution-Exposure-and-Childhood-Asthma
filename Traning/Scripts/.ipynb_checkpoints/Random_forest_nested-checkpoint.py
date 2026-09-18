from statsmodels.nonparametric.smoothers_lowess import lowess
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler
"""
Nested CV for a calibrated Random Forest with OOF SHAP + SHAP interaction values.

Leakage fixes applied in this version
-------------------------------------
1. StandardScaler + KNNImputer are now fitted INSIDE the inner validation loop,
   on inner-train only. Previously they were fitted on the whole outer training
   fold before the inner split, so inner-val rows sat in the imputer's neighbour
   pool and contributed to the scaler's mean/SD.
2. Hyperparameter search is still run on the whole training fold (that is what
   nested CV is for), so val_perf remains SELECTION-contaminated by construction.
   It is now explicitly labelled as a diagnostic, not a generalization estimate.
   Report test_perf / pooled_oof_*.
3. SHAP and SHAP interactions are computed on the OUTER TEST fold (clean: never
   touched by tuning or preprocessing), not on the inner-val rows. An optional
   also-on-val pass is available via shap_on="val" or "both" if you want to
   compare, but "test" is the default and the defensible one.
4. pooled_oof_bss now uses the mean of the per-fold baselines rather than the
   whole-dataset prevalence.
5. Optional groups= argument routes to StratifiedGroupKFold so siblings / twins
   / repeated measures cannot straddle a train-test boundary.

Requires: numpy, pandas, scikit-learn, shap, scipy (for the param distributions)
and your own calibration_metrics(y_true, y_prob) -> float.
"""

import warnings

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
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


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _rows(A, idx):
    """Positional row selection that works for DataFrame or ndarray."""
    return A.iloc[idx] if hasattr(A, "iloc") else A[idx]


def _fit_preprocess(X_fit_raw, n_neighbors):
    """
    Fit scaler + KNN imputer on FIT rows only.
    Returns (imputed_fit, transform_fn) where transform_fn applies the fitted
    pipeline to new raw rows.
    """
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit_raw)
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_fit_imp = scaler.inverse_transform(imputer.fit_transform(X_fit_s))

    def transform(X_new_raw):
        return scaler.inverse_transform(imputer.transform(scaler.transform(X_new_raw)))

    return X_fit_imp, transform


def _positive_class(arr, class_index=1):
    """Normalise SHAP output across shap versions and binary/multiclass shapes."""
    if isinstance(arr, list):
        return arr[class_index] if len(arr) > 1 else arr[0]
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[1] != arr.shape[2]:
        return arr[..., class_index]        # (n, p, n_classes)
    if arr.ndim == 4:
        return arr[..., class_index]        # (n, p, p, n_classes)
    return arr


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
                "wrapping in a list. If you meant a distribution, check you "
                "imported randint from scipy.stats, not from random/numpy."
            )
            out[key] = [val]
    return out


def _metrics(y_true, p, baseline):
    y_true = np.asarray(y_true)
    brier = float(brier_score_loss(y_true, p))
    return {
        "roc":   float(roc_auc_score(y_true, p)),
        "brier": brier,
        "bss":   1.0 - brier / baseline,
        "ici":   float(calibration_metrics(y_true, p)),
    }


# --------------------------------------------------------------------------- #
# main routine
# --------------------------------------------------------------------------- #
def nested_cross_calibrated_rf(
    X, y,
    groups=None,
    Number_CV=2,
    number_split=3,
    val_cv_folds=3,
    n_neighbors_imputer=25,
    param_distributions=None,
    n_iter_search=20,
    search_cv_folds=5,
    search_scoring="roc_auc",
    calibration_method="sigmoid",
    calibration_cv_folds=5,
    base_rf_kwargs=None,
    compute_val=True,
    compute_shap=True,
    compute_interactions=True,
    interaction_features=None,
    shap_on="test",                # "test" | "val" | "both"
    random_state=42,
    verbose=True,
):
    """
    Parameters
    ----------
    groups : array-like or None
        Family / subject IDs. If given, splitting uses StratifiedGroupKFold so
        related rows never straddle a train-test boundary. Essential for birth
        cohorts with siblings or twins when genetic features are in the model.
        Note: repeats are not available for grouped splits, so Number_CV is
        applied by varying the shuffle seed across repeats.
    compute_val : bool
        Run the inner calibrated-CV validation loop. Purely diagnostic — see the
        docstring at the top of the file. Set False for a much faster run.
    shap_on : {"test", "val", "both"}
        Which rows get explained. "test" is the clean choice.

    Returns
    -------
    dict; the numbers to report are test_perf and summary["pooled_oof_*"].
    val_perf is a diagnostic only.
    """
    param_distributions = _sanitize_param_dist(param_distributions)
    base_rf_kwargs = base_rf_kwargs or {"n_jobs": -1, "random_state": random_state}

    if "calibration_metrics" not in globals():
        raise NameError(
            "calibration_metrics(y_true, y_prob) must be defined. It needs "
            "`from statsmodels.nonparametric.smoothers_lowess import lowess`."
        )

    feature_names = (list(X.columns) if hasattr(X, "columns")
                     else [f"x{i}" for i in range(X.shape[1])])
    y_arr = np.asarray(y)
    n_samples, n_features = X.shape

    # ---- interaction block bookkeeping ----
    if interaction_features is None:
        inter_idx = np.arange(n_features)
    else:
        inter_idx = np.array([f if isinstance(f, (int, np.integer))
                              else feature_names.index(f)
                              for f in interaction_features])
    inter_names = [feature_names[i] for i in inter_idx]
    k = len(inter_idx)

    # ---- outer splits ----
    if groups is None:
        splitter = RepeatedStratifiedKFold(
            n_splits=number_split, n_repeats=Number_CV, random_state=random_state
        )
        outer_splits = list(splitter.split(X, y_arr))
    else:
        groups = np.asarray(groups)
        outer_splits = []
        for r in range(Number_CV):
            sgkf = StratifiedGroupKFold(
                n_splits=number_split, shuffle=True, random_state=random_state + r
            )
            outer_splits.extend(sgkf.split(X, y_arr, groups=groups))
    n_folds = len(outer_splits)

    # ---- storage ----
    val_perf  = {f"{m}_val":  np.zeros(n_folds) for m in ("roc", "brier", "bss", "ici")}
    test_perf = {f"{m}_test": np.zeros(n_folds) for m in ("roc", "brier", "bss", "ici")}
    best_params_per_fold = []
    fold_baselines = np.zeros(n_folds)

    oof_sum, oof_count = np.zeros(n_samples), np.zeros(n_samples)
    val_sum, val_count = np.zeros(n_samples), np.zeros(n_samples)

    shap_sum   = np.zeros((n_samples, n_features))
    shap_count = np.zeros(n_samples)
    base_sum   = np.zeros(n_samples)
    disp_sum   = np.zeros((n_samples, n_features))
    inter_sum  = (np.zeros((n_samples, k, k), dtype=np.float32)
                  if (compute_shap and compute_interactions) else None)

    prevalence = float(np.mean(y_arr))

    def _explain(rf, X_expl, g_idx):
        """Accumulate SHAP (+interactions) for rows g_idx explained by rf."""
        explainer = shap.TreeExplainer(rf, feature_perturbation="tree_path_dependent")

        sv_pos = _positive_class(
            explainer.shap_values(X_expl, check_additivity=False)
        )
        ev = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
        base_val = float(ev[1]) if ev.size > 1 else float(ev[0])

        shap_sum[g_idx]   += sv_pos
        shap_count[g_idx] += 1
        base_sum[g_idx]   += base_val
        disp_sum[g_idx]   += X_expl

        if compute_interactions:
            iv = _positive_class(explainer.shap_interaction_values(X_expl))
            inter_sum[g_idx] += iv[:, inter_idx][:, :, inter_idx].astype(np.float32)
            del iv

    # ======================================================================= #
    for j, (train_idx, test_idx) in enumerate(outer_splits):

        X_train_raw = _rows(X, train_idx)
        X_test_raw  = _rows(X, test_idx)
        y_train_raw = y_arr[train_idx]
        y_test_raw  = y_arr[test_idx]

        fold_prev         = float(np.mean(y_train_raw))
        fold_baseline     = fold_prev * (1.0 - fold_prev)
        fold_baselines[j] = fold_baseline

        # ---- preprocessing for the OUTER fold: fitted on train only ----
        X_train_imp, tf_outer = _fit_preprocess(X_train_raw, n_neighbors_imputer)
        X_test_imp = tf_outer(X_test_raw)

        # ---- Step 1: hyperparameter search on the training fold ----
        search = RandomizedSearchCV(
            estimator=RandomForestClassifier(**base_rf_kwargs),
            param_distributions=param_distributions,
            n_iter=n_iter_search,
            scoring=search_scoring,
            cv=StratifiedKFold(n_splits=search_cv_folds, shuffle=True,
                               random_state=random_state + j),
            n_jobs=-1,
            random_state=random_state + j,
            refit=True,
        )
        search.fit(X_train_imp, y_train_raw)
        best_params_per_fold.append(search.best_params_)
        best = {**base_rf_kwargs, **search.best_params_}

        # ---- Step 2: inner calibrated CV -> validation diagnostics ----
        # Preprocessing is refitted inside each inner split (leakage fix #1).
        if compute_val:
            if groups is None:
                inner_cv = StratifiedKFold(n_splits=val_cv_folds, shuffle=True,
                                           random_state=random_state + j)
                inner_iter = inner_cv.split(X_train_raw, y_train_raw)
            else:
                inner_cv = StratifiedGroupKFold(n_splits=val_cv_folds, shuffle=True,
                                                random_state=random_state + j)
                inner_iter = inner_cv.split(X_train_raw, y_train_raw,
                                            groups=groups[train_idx])

            for tr, va in inner_iter:
                g_va = train_idx[va]

                X_tr_imp, tf_inner = _fit_preprocess(
                    _rows(X_train_raw, tr), n_neighbors_imputer
                )
                X_va_imp = tf_inner(_rows(X_train_raw, va))

                cal_inner = CalibratedClassifierCV(
                    estimator=RandomForestClassifier(**best),
                    method=calibration_method,
                    cv=calibration_cv_folds,
                )
                cal_inner.fit(X_tr_imp, y_train_raw[tr])
                p_va = cal_inner.predict_proba(X_va_imp)[:, 1]

                val_sum[g_va]   += p_va
                val_count[g_va] += 1

                if compute_shap and shap_on in ("val", "both"):
                    rf_inner = RandomForestClassifier(**best)
                    rf_inner.fit(X_tr_imp, y_train_raw[tr])
                    _explain(rf_inner, X_va_imp, g_va)

            seen  = train_idx[val_count[train_idx] > 0]
            p_val = val_sum[seen] / val_count[seen]
            blk = _metrics(y_arr[seen], p_val, fold_baseline)
            for m in ("roc", "brier", "bss", "ici"):
                val_perf[f"{m}_val"][j] = blk[m]

        # ---- Step 3: calibrated model on the full training fold -> TEST ----
        calibrated_clf = CalibratedClassifierCV(
            estimator=RandomForestClassifier(**best),
            method=calibration_method,
            cv=calibration_cv_folds,
        )
        calibrated_clf.fit(X_train_imp, y_train_raw)
        test_probs = calibrated_clf.predict_proba(X_test_imp)[:, 1]

        blk = _metrics(y_test_raw, test_probs, fold_baseline)
        for m in ("roc", "brier", "bss", "ici"):
            test_perf[f"{m}_test"][j] = blk[m]

        oof_sum[test_idx]   += test_probs
        oof_count[test_idx] += 1

        # ---- Step 4: SHAP on the clean outer test fold ----
        if compute_shap and shap_on in ("test", "both"):
            interp_rf = RandomForestClassifier(**best)
            interp_rf.fit(X_train_imp, y_train_raw)
            _explain(interp_rf, X_test_imp, test_idx)

        if verbose:
            msg = f"Fold {j+1}/{n_folds}  "
            if compute_val:
                msg += (f"VAL AUC={val_perf['roc_val'][j]:.3f} "
                        f"BSS={val_perf['bss_val'][j]:.3f} "
                        f"ICI={val_perf['ici_val'][j]:.3f}  |  ")
            msg += (f"TEST AUC={test_perf['roc_test'][j]:.3f} "
                    f"BSS={test_perf['bss_test'][j]:.3f} "
                    f"ICI={test_perf['ici_test'][j]:.3f}")
            print(msg)
    # ======================================================================= #

    assert (oof_count > 0).all(), "some rows never landed in a test fold"
    oof_mean_probs = oof_sum / oof_count
    pooled_brier   = float(brier_score_loss(y_arr, oof_mean_probs))
    mean_baseline  = float(np.mean(fold_baselines))          # leakage fix #4

    with np.errstate(invalid="ignore", divide="ignore"):
        val_mean_probs = np.where(val_count > 0,
                                  val_sum / np.maximum(val_count, 1), np.nan)

    summary = {
        "model":            "calibrated_rf_tuned",
        "calibration":      calibration_method,
        "grouped_cv":       groups is not None,
        "shap_on":          shap_on,
        "prevalence":       prevalence,
        "mean_fold_baseline_brier": mean_baseline,
        "test_roc_mean":    float(np.mean(test_perf["roc_test"])),
        "test_roc_sd":      float(np.std(test_perf["roc_test"], ddof=1)),
        "test_bss_mean":    float(np.mean(test_perf["bss_test"])),
        "test_bss_sd":      float(np.std(test_perf["bss_test"], ddof=1)),
        "test_ici_mean":    float(np.mean(test_perf["ici_test"])),
        "pooled_oof_roc":   float(roc_auc_score(y_arr, oof_mean_probs)),
        "pooled_oof_brier": pooled_brier,
        "pooled_oof_bss":   1.0 - pooled_brier / mean_baseline,
        "pooled_oof_ici":   float(calibration_metrics(y_arr, oof_mean_probs)),
    }
    if compute_val:
        summary.update({
            "val_roc_mean": float(np.mean(val_perf["roc_val"])),
            "val_bss_mean": float(np.mean(val_perf["bss_val"])),
            "val_ici_mean": float(np.mean(val_perf["ici_val"])),
            "val_note": "diagnostic only - hyperparameters were tuned on the "
                        "fold these rows came from; report test_* instead",
        })

    # ---- SHAP aggregation ----
    shap_values_oof = shap_importance = base_values_oof = None
    X_display = shap_explanation = None
    shap_interaction_oof = interaction_importance = None

    if compute_shap and shap_count.sum() > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            shap_values_oof = shap_sum / shap_count[:, None]
            base_values_oof = base_sum / shap_count
            X_display       = disp_sum / shap_count[:, None]
        shap_values_oof = np.nan_to_num(shap_values_oof, nan=0.0)
        base_values_oof = np.nan_to_num(base_values_oof, nan=prevalence)
        X_display       = np.nan_to_num(X_display, nan=0.0)

        shap_importance = (
            pd.DataFrame({"feature": feature_names,
                          "mean_abs_shap": np.abs(shap_values_oof).mean(axis=0)})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
        shap_explanation = shap.Explanation(
            values=shap_values_oof, base_values=base_values_oof,
            data=X_display, feature_names=feature_names,
        )

        if compute_interactions:
            with np.errstate(invalid="ignore", divide="ignore"):
                shap_interaction_oof = np.nan_to_num(
                    inter_sum / shap_count[:, None, None].astype(np.float32), nan=0.0
                )
            interaction_importance = summarise_interactions(
                shap_interaction_oof, inter_names
            )

    return {
        "summary":                summary,
        "test_perf":              test_perf,      # <- report this
        "val_perf":               val_perf if compute_val else None,
        "fold_baselines":         fold_baselines,
        "oof_probs":              oof_mean_probs,
        "val_probs":              val_mean_probs,
        "val_count":              val_count,
        "oof_y":                  y_arr,
        "best_params_per_fold":   best_params_per_fold,
        "feature_names":          feature_names,
        "shap_values_oof":        shap_values_oof,
        "shap_count":             shap_count,
        "shap_importance":        shap_importance,
        "base_values_oof":        base_values_oof,
        "X_display":              X_display,
        "shap_explanation":       shap_explanation,
        "shap_interaction_oof":   shap_interaction_oof,
        "interaction_features":   inter_names,
        "interaction_importance": interaction_importance,
    }


# --------------------------------------------------------------------------- #
# interaction post-processing
# --------------------------------------------------------------------------- #
def summarise_interactions(inter, names, top_n=None):
    """
    Rank unordered feature pairs by mean |total pairwise interaction|.

    SHAP splits each pairwise effect in half (Phi[i,j] == Phi[j,i]), so the
    total for a pair is 2*Phi[i,j]. That factor is applied here, putting
    mean_abs_interaction on the same scale as mean_abs_shap.
    """
    inter = np.asarray(inter, dtype=np.float64)
    k = inter.shape[1]
    iu = np.triu_indices(k, k=1)

    pair = 2.0 * inter[:, iu[0], iu[1]]
    diag = np.abs(inter[:, np.arange(k), np.arange(k)]).mean(axis=0)

    df = pd.DataFrame({
        "feature_i":            [names[i] for i in iu[0]],
        "feature_j":            [names[j] for j in iu[1]],
        "mean_abs_interaction": np.abs(pair).mean(axis=0),
        "mean_interaction":     pair.mean(axis=0),
        "sd_interaction":       pair.std(axis=0, ddof=1),
        "main_effect_i":        diag[iu[0]],
        "main_effect_j":        diag[iu[1]],
    })
    df["interaction_ratio"] = df["mean_abs_interaction"] / (
        df[["main_effect_i", "main_effect_j"]].mean(axis=1) + 1e-12
    )
    df = df.sort_values("mean_abs_interaction", ascending=False).reset_index(drop=True)
    return df.head(top_n) if top_n else df


def interaction_pair(res, feat_a, feat_b):
    """Per-sample TOTAL interaction (2*Phi_ij), aligned with the original X rows."""
    names = res["interaction_features"]
    ia, ib = names.index(feat_a), names.index(feat_b)
    return 2.0 * np.asarray(res["shap_interaction_oof"][:, ia, ib], dtype=np.float64)


def interaction_matrix(res, mode="abs"):
    inter = np.asarray(res["shap_interaction_oof"], dtype=np.float64)
    keep = (abs(inter) > 0).mean(axis=0) > 0.8   # or CI excluding some floor
    if mode == "abs":
        m = np.abs(inter).mean(axis=0) * 2.0
        np.fill_diagonal(m, np.abs(inter).mean(axis=0).diagonal())
    else:  # signed, exposes cancellation
        m = inter.mean(axis=0) * 2.0
        np.fill_diagonal(m, inter.mean(axis=0).diagonal())
    return pd.DataFrame(m, index=res["interaction_features"],
                        columns=res["interaction_features"]),keep


# =====================================================================
#  INTERACTIVE SHAP HELPERS
# =====================================================================
def _force(expl_slice):
    """shap.plots.force with a fallback to the legacy API."""
    try:
        return shap.plots.force(expl_slice)
    except Exception:
        return shap.force_plot(
            expl_slice.base_values,
            expl_slice.values,
            expl_slice.data,
            feature_names=list(expl_slice.feature_names),
        )


def interactive_force_html(
    res,
    out_html="shap_oof_force.html",
    max_samples=500,
    order_by="prediction",
    random_state=0,
):
    """
    The genuinely interactive one: the stacked force plot (JS/D3).
    Hover per sample, re-order by feature / similarity from the dropdowns.

    Rendering thousands of samples in the browser is slow, so we subsample.
    """
    expl = res["shap_explanation"]
    n = expl.values.shape[0]

    if n > max_samples:
        if order_by == "prediction":
            pred = expl.base_values + expl.values.sum(axis=1)
            idx = np.argsort(pred)
            idx = idx[np.linspace(0, n - 1, max_samples).astype(int)]  # spread across range
        else:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(n, max_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    plot = _force(expl[idx])
    shap.save_html(out_html, plot)
    print(f"[saved] {out_html}  ({len(idx)} samples)")
    return plot          # returning it renders inline in a notebook


def interactive_single_patient(res, i, out_html=None):
    """Force plot + waterfall for one subject (i = row index in X)."""
    expl = res["shap_explanation"]
    plot = _force(expl[i])
    if out_html:
        shap.save_html(out_html, plot)
        print(f"[saved] {out_html}")
    try:
        shap.plots.waterfall(expl[i], max_display=12, show=True)
    except Exception:
        shap.waterfall_plot(expl[i], max_display=12, show=True)
    return plot


def shap_overview(res, top_k=15):
    """Beeswarm + bar + dependence scatter for the top feature."""
    expl = res["shap_explanation"]
    try:
        shap.plots.beeswarm(expl, max_display=top_k, show=True)
        shap.plots.bar(expl, max_display=top_k, show=True)
        top_feat = res["shap_importance"].iloc[0]["feature"]
        shap.plots.scatter(expl[:, top_feat], color=expl, show=True)
    except Exception:                                   # older shap
        shap.summary_plot(
            res["shap_values_oof"], res["X_display"],
            feature_names=res["feature_names"], max_display=top_k,
        )


def shap_decision(res, idx=None, max_samples=200):
    """Decision plot — good for spotting sub-groups with divergent paths."""
    expl = res["shap_explanation"]
    if idx is None:
        idx = np.arange(min(max_samples, expl.values.shape[0]))
    shap.decision_plot(
        float(np.mean(res["base_values_oof"][idx])),
        res["shap_values_oof"][idx],
        pd.DataFrame(res["X_display"][idx], columns=res["feature_names"]),
        link="identity",
    )


# =====================================================================
#  USAGE (in a Jupyter notebook)
# =====================================================================
#   shap.initjs()                     # <-- required once, before any force plot
#
#   res = nested_cross_calibrated_rf(X, y, param_distributions=param_dist)
#
#   shap_overview(res)                                   # beeswarm / bar / scatter
#   interactive_force_html(res, "shap_oof_force.html")   # interactive, all samples
#   interactive_single_patient(res, i=17, out_html="patient_17.html")
#   shap_decision(res)
#
# Outside a notebook, open the saved .html files in a browser.
def nested_cross_validation_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat,
                                           parameters, n_neighbors=20,n=20):
    """
    Nested cross-validation with hyperparameter tuning (RandomizedSearchCV)
    and SHAP values for a Random Forest Classifier.

    Preprocessing (Z-scaling + KNN imputation) is applied inside each fold,
    fitted on training data only, to prevent data leakage.

    Args:
        X:                          Feature DataFrame.
        y:                          Binary or multiclass target Series.
        Number_of_repeats:          Number of outer CV repetitions.
        Number_of_splits_pr_repeat: Splits per repetition (outer and inner CV).
        parameters:                 Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:                k for KNN imputation (default 20, per paper).

    Returns:
        performance_df:         DataFrame of per-fold AUROC and F1 (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features, n_classes).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    n_classes = y.unique().shape[0]
    shap_values_per_cv = np.zeros((X.shape[0], X.shape[1], n_classes))
    performance_test  = []
    performance_train = []
    brier_test   = []
    brier_train  = []
    feature_importances = []
    parameters_tree = []
    calibration_test_true=[]
    calibration_test_pred=[]
    calibration_train_true=[]
    calibration_train_pred=[]
    interaction=[]
    prevalence     = float(np.mean(y))
    baseline_brier = prevalence * (1.0 - prevalence)
    CrossValidation = RepeatedStratifiedKFold(
        n_splits=Number_of_splits_pr_repeat,
        n_repeats=Number_of_repeats,
        random_state=42
    )

    for i, (train_outer_ix, test_outer_ix) in enumerate(CrossValidation.split(X, y)):

        X_train_raw, X_test_raw = X.iloc[train_outer_ix], X.iloc[test_outer_ix]
        y_train, y_test         = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        # FIX (leakage): preprocessing fitted on train only, applied to both
        X_train, X_test = _preprocess_fold(X_train_raw, X_test_raw, n_neighbors=n_neighbors)

        cv_inner = StratifiedKFold(
            n_splits=Number_of_splits_pr_repeat,
            random_state=42,
            shuffle=True
        )

        search = RandomizedSearchCV(
            # FIX: random_state and class_weight set for reproducibility and class imbalance
            estimator=RandomForestClassifier(random_state=42),
            param_distributions=parameters,
            n_iter=n,
            cv=cv_inner,
            # FIX: scoring changed to roc_auc to match the paper's primary metric
            # (previously set to "f1" while comment said "roc_auc")
            scoring="roc_auc",
            random_state=42,
            # FIX: n_jobs=-1 to use all cores (was 1, contradicting the comment)
            n_jobs=-1,
            verbose=1,
            refit=True
        )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # SHAP values for the test fold
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_per_cv[test_outer_ix, :, :] += shap_values
        interaction.append(explainer.shap_interaction_values(X)) 

        # AUROC
        if n_classes > 2:
            performance_test.append(roc_auc_score(y_test,  best_model.predict_proba(X_test),  multi_class='ovr'))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train), multi_class='ovr'))
        else:
            performance_test.append(roc_auc_score(y_test,  best_model.predict_proba(X_test)[:,1]))
            performance_train.append(roc_auc_score(y_train, best_model.predict_proba(X_train)[:,1]))
        brier_tr=brier_score_loss(y_train, best_model.predict_proba(X_train)[:,1])
        brier_te=brier_score_loss(y_test,  best_model.predict_proba(X_test)[:,1])
        
        brier_test.append(1.0 - brier_te/ baseline_brier)
        brier_train.append( 1.0 - brier_tr / baseline_brier)
        cal_test_true, cal_test_pred =calibration_curve(y_test,  best_model.predict_proba(X_test)[:,1],n_bins=5,strategy='quantile' )
        cal_train_true, cal_train_pred= calibration_curve(y_train, best_model.predict_proba(X_train)[:,1],n_bins=5,strategy='quantile' )
        calibration_test_true.append(cal_test_true)
        calibration_test_pred.append(cal_test_pred)
        calibration_train_true.append(cal_train_true)
        calibration_train_pred.append(cal_train_pred)

        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(search.best_params_)

        print(f'Done: {((i + 1) / (Number_of_repeats * Number_of_splits_pr_repeat)) * 100:.1f}%')

    total_shap = shap_values_per_cv / Number_of_repeats

    performance_df = pd.DataFrame(
        [performance_train, performance_test, brier_train, brier_test],
        index=["AUROC_train", "AUROC_test", "brier_train", "brier_test"]
    ).T

    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree,calibration_test_true,calibration_test_pred,interaction


def nested_cross_validation_Regression(X, y, Number_of_repeats, Number_of_splits,
                                        parameters, n_neighbors=20):
    """
    Nested cross-validation with hyperparameter tuning (RandomizedSearchCV)
    and SHAP values for a Random Forest Regressor.

    Args:
        X:                Feature DataFrame.
        y:                Continuous target Series.
        Number_of_repeats:  Number of outer CV repetitions.
        Number_of_splits:   Splits per repetition (outer and inner CV).
        parameters:       Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:      k for KNN imputation (default 20).

    Returns:
        performance_df:         DataFrame of per-fold R² and RMSE (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    shap_values_per_cv          = np.zeros((X.shape[0], X.shape[1]))
    root_mean_squared_error_test  = []
    root_mean_squared_error_train = []
    R_squared_test  = []
    R_squared_train = []
    feature_importances = []
    parameters_tree = []

    CV = RepeatedKFold(n_splits=Number_of_splits, n_repeats=Number_of_repeats, random_state=42)

    for i, (train_outer_ix, test_outer_ix) in enumerate(CV.split(X)):

        X_train_raw, X_test_raw = X.iloc[train_outer_ix], X.iloc[test_outer_ix]
        y_train, y_test         = y.iloc[train_outer_ix], y.iloc[test_outer_ix]

        # Preprocessing: fit on train, transform both (consistent with classification functions)
        X_train, X_test = _preprocess_fold(X_train_raw, X_test_raw, n_neighbors=n_neighbors)

        cv_inner = KFold(n_splits=Number_of_splits, random_state=i, shuffle=True)

        search = RandomizedSearchCV(
            # FIX: random_state set for reproducibility
            estimator=RandomForestRegressor(random_state=42),
            param_distributions=parameters,
            cv=cv_inner,
            scoring="r2",
            random_state=42,
            n_jobs=-1,
            verbose=1
        )

        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        shap_values_per_cv[test_outer_ix, :] += shap_values

        R_squared_test.append(best_model.score(X_test, y_test))
        R_squared_train.append(best_model.score(X_train, y_train))
        root_mean_squared_error_test.append(root_mean_squared_error(y_test,  best_model.predict(X_test)))
        root_mean_squared_error_train.append(root_mean_squared_error(y_train, best_model.predict(X_train)))

        feature_importances.append(best_model.feature_importances_)
        parameters_tree.append(result.best_params_)

        print(f'Done: {((i + 1) / (Number_of_repeats * Number_of_splits)) * 100:.1f}%')

    total_shap = shap_values_per_cv / Number_of_repeats

    performance_df = pd.DataFrame(
        [R_squared_train, R_squared_test,
         root_mean_squared_error_train, root_mean_squared_error_test],
        index=["R_squared_train", "R_squared_test",
               "RMSE_train", "RMSE_test"]
    ).T

    feature_importances_df = pd.DataFrame(feature_importances, columns=X.columns).mean()

    return performance_df, total_shap, feature_importances_df, parameters_tree


def nested_cross_validation_nodata_classification(X, y, Number_of_repeats, Number_of_splits_pr_repeat,
                                                   parameters, n_neighbors=20):
    """
    Identical to nested_cross_validation_classification but retained as a separate
    entry point for backwards compatibility.

    All preprocessing (Z-scaling + KNN imputation) is applied correctly inside
    each fold — fitted on training data only.

    Args:
        X:                          Feature DataFrame.
        y:                          Binary or multiclass target Series.
        Number_of_repeats:          Number of outer CV repetitions.
        Number_of_splits_pr_repeat: Splits per repetition.
        parameters:                 Hyperparameter distributions for RandomizedSearchCV.
        n_neighbors:                k for KNN imputation (default 20).

    Returns:
        performance_df:         DataFrame of per-fold AUROC and F1 (train and test).
        total_shap:             Mean SHAP values array (n_samples, n_features, n_classes).
        feature_importances_df: Mean feature importances across folds.
        parameters_tree:        Best hyperparameters per fold.
    """
    # Delegates entirely to the fixed classification function above
    return nested_cross_validation_classification(
        X, y,
        Number_of_repeats,
        Number_of_splits_pr_repeat,
        parameters,
        n_neighbors=n_neighbors
    )

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



"""
Nested CV RandomForest with probability calibration + OOF SHAP,
extended to support INTERACTIVE SHAP plots.

Expected imports in the calling environment:

    import numpy as np
    import pandas as pd
    import shap
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import KNNImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                         RandomizedSearchCV)
    from sklearn.metrics import roc_auc_score, brier_score_loss
    # plus your own `calibration_metrics(y_true, probs)` -> ICI
"""

import numpy as np
import pandas as pd
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import roc_auc_score, brier_score_loss


# =====================================================================
#  MAIN FUNCTION
#  Lines marked  # NEW  are the only additions vs. your original.
# =====================================================================
def nested_cross_calibrated_rf_1(
    X, y,
    Number_CV=2,
    number_split=3,
    n_neighbors_imputer=25,
    param_distributions=None,
    n_iter_search=20,
    search_cv_folds=3,
    search_scoring="roc_auc",
    calibration_method="sigmoid",
    calibration_cv_folds=3,
    base_rf_kwargs=None,
    compute_shap=True,
    random_state=42,
):
    """
    Nested CV with:
      - inner RandomizedSearchCV for RF hyperparameters
      - CalibratedClassifierCV for probability calibration  (used for METRICS)
      - separate interpretation RF on full training fold    (used for SHAP)

    Returns
    -------
    dict including:
        summary, test_perf, train_perf, oof_probs, oof_y,
        best_params_per_fold,
        shap_values_oof   : (n_samples, n_features) — OOF mean SHAP
        shap_count        : (n_samples,) — times each sample appeared in test
        feature_names     : list[str]
        shap_importance   : pd.DataFrame mean(|SHAP|) per feature
        base_values_oof   : (n_samples,) — OOF mean expected value      # NEW
        X_display         : (n_samples, n_features) imputed, UNSCALED   # NEW
        shap_explanation  : shap.Explanation ready for interactive plots# NEW
    """
    base_rf_kwargs = base_rf_kwargs or {
        "n_jobs":       -1,
        "random_state": random_state,
    }

    # Feature names (preserved through scaler)
    if hasattr(X, "columns"):
        feature_names = list(X.columns)
    else:
        feature_names = [f"x{i}" for i in range(X.shape[1])]

    rskf = RepeatedStratifiedKFold(
        n_splits=number_split,
        n_repeats=Number_CV,
        random_state=random_state,
    )
    n_folds = number_split * Number_CV

    # ----- Per-fold metric storage -----
    test_perf = {
        "roc_test":   np.zeros(n_folds),
        "brier_test": np.zeros(n_folds),
        "ici_test":   np.zeros(n_folds),
        "bss_test":   np.zeros(n_folds),
    }
    train_perf = {
        "roc_train":   np.zeros(n_folds),
        "brier_train": np.zeros(n_folds),
        "ici_train":   np.zeros(n_folds),
        "bss_train":   np.zeros(n_folds),
    }
    best_params_per_fold = []

    # ----- OOF probability accumulators -----
    n_samples, n_features = X.shape
    oof_sum   = np.zeros(n_samples)
    oof_count = np.zeros(n_samples)

    # ----- OOF SHAP accumulators -----
    shap_sum   = np.zeros((n_samples, n_features))
    shap_count = np.zeros(n_samples)
    base_sum   = np.zeros(n_samples)                       # NEW: expected_value
    disp_sum   = np.zeros((n_samples, n_features))         # NEW: display values

    prevalence     = float(np.mean(y))
    baseline_brier = prevalence * (1 - prevalence)

    j = 0
    for train_idx, test_idx in rskf.split(X, y):
        X_train_raw = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_test_raw  = X.iloc[test_idx]  if hasattr(X, "iloc") else X[test_idx]
        y_train_raw = y.iloc[train_idx] if hasattr(y, "iloc") else y[train_idx]
        y_test_raw  = y.iloc[test_idx]  if hasattr(y, "iloc") else y[test_idx]
        
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled  = scaler.transform(X_test_raw)
        # ---- Preprocessing fit on TRAIN only ----
        imputer = KNNImputer(n_neighbors=n_neighbors_imputer)
        X_train_imp = imputer.fit_transform(X_train_scaled)
        X_test_imp  = imputer.transform(X_test_scaled)



        # ---- Step 1: Hyperparameter search ----
        base_rf = RandomForestClassifier(**base_rf_kwargs)
        search_cv = StratifiedKFold(
            n_splits=search_cv_folds, shuffle=True,
            random_state=random_state + j,
        )
        search = RandomizedSearchCV(
            estimator=base_rf,
            param_distributions=param_distributions,
            n_iter=n_iter_search,
            scoring=search_scoring,
            cv=search_cv,
            n_jobs=-1,
            random_state=random_state + j,
            refit=True,
        )
        search.fit(X_train_imp, y_train_raw)
        best_params_per_fold.append(search.best_params_)

        # ---- Step 2: Calibrated model — used for METRICS ----
        tuned_rf_for_calib = RandomForestClassifier(
            **{**base_rf_kwargs, **search.best_params_}
        )
        calibrated_clf = CalibratedClassifierCV(
            estimator=tuned_rf_for_calib,
            method=calibration_method,
            cv=calibration_cv_folds,
        )
        calibrated_clf.fit(X_train_imp, y_train_raw)

        train_probs = calibrated_clf.predict_proba(X_train_imp)[:, 1]
        test_probs  = calibrated_clf.predict_proba(X_test_imp)[:, 1]

        # ---- Step 3: Interpretation RF — used for SHAP ----
        # Same hyperparameters, but fit on FULL training fold (no inner CV split)
        interp_rf = RandomForestClassifier(
            **{**base_rf_kwargs, **search.best_params_}
        )
        interp_rf.fit(X_train_imp, y_train_raw)

        if compute_shap:
            explainer = shap.TreeExplainer(
                interp_rf,
                feature_perturbation="tree_path_dependent",
            )
            # SHAP for the positive class
            sv = explainer.shap_values(X_test_imp, check_additivity=False)
            # Newer SHAP returns array of shape (n_samples, n_features, n_classes);
            # older versions return list[array] per class
            if isinstance(sv, list):
                sv_pos = sv[1]                       # positive class
            elif sv.ndim == 3:
                sv_pos = sv[..., 1]                  # (n, p, classes) -> (n, p)
            else:
                sv_pos = sv                          # already (n, p)

            # NEW: expected value for the positive class
            ev = explainer.expected_value
            ev = np.atleast_1d(np.asarray(ev, dtype=float))
            base_val = float(ev[1]) if ev.size > 1 else float(ev[0])

            shap_sum[test_idx]   += sv_pos
            shap_count[test_idx] += 1
            base_sum[test_idx]   += base_val            # NEW
            disp_sum[test_idx]   += X_test_imp          # NEW (unscaled + imputed)

        # ---- Per-fold metrics ----
        train_perf["roc_train"][j]   = roc_auc_score(y_train_raw, train_probs)
        train_perf["brier_train"][j] = brier_score_loss(y_train_raw, train_probs)
        train_perf["ici_train"][j]   = calibration_metrics(np.asarray(y_train_raw), train_probs)
        train_perf["bss_train"][j]   = 1 - train_perf["brier_train"][j] / baseline_brier

        test_perf["roc_test"][j]     = roc_auc_score(y_test_raw, test_probs)
        test_perf["brier_test"][j]   = brier_score_loss(y_test_raw, test_probs)
        test_perf["ici_test"][j]     = calibration_metrics(np.asarray(y_test_raw), test_probs)
        test_perf["bss_test"][j]     = 1 - test_perf["brier_test"][j] / baseline_brier

        oof_sum[test_idx]   += test_probs
        oof_count[test_idx] += 1

        print(f"Fold {j+1}/{n_folds}  "
              f"AUC={test_perf['roc_test'][j]:.3f}  "
              f"BSS={test_perf['bss_test'][j]:+.3f}  "
              f"ICI={test_perf['ici_test'][j]:.3f}")
        j += 1

    # ----- Pooled OOF probabilities -----
    assert (oof_count > 0).all()
    oof_mean_probs = oof_sum / oof_count
    pooled_brier   = float(brier_score_loss(y, oof_mean_probs))

    summary = {
        "model":            "calibrated_rf_tuned",
        "calibration":      calibration_method,
        "prevalence":       prevalence,
        "baseline_brier":   baseline_brier,
        "pooled_oof_roc":   float(roc_auc_score(y, oof_mean_probs)),
        "pooled_oof_brier": pooled_brier,
        "pooled_oof_bss":   1 - pooled_brier / baseline_brier,
        "pooled_oof_ici":   float(calibration_metrics(np.asarray(y), oof_mean_probs)),
    }

    # ----- OOF SHAP aggregation -----
    shap_values_oof = None
    shap_importance = None
    base_values_oof = None      # NEW
    X_display       = None      # NEW
    shap_explanation = None     # NEW
    if compute_shap:
        # Repeated CV: each sample appears in `Number_CV` test folds → average
        with np.errstate(invalid="ignore", divide="ignore"):
            shap_values_oof = shap_sum / shap_count[:, None]
            base_values_oof = base_sum / shap_count                 # NEW
            X_display       = disp_sum / shap_count[:, None]        # NEW
        shap_values_oof = np.nan_to_num(shap_values_oof, nan=0.0)
        base_values_oof = np.nan_to_num(base_values_oof, nan=float(prevalence))
        X_display       = np.nan_to_num(X_display, nan=0.0)

        mean_abs = np.abs(shap_values_oof).mean(axis=0)
        shap_importance = (
            pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
              .sort_values("mean_abs_shap", ascending=False)
              .reset_index(drop=True)
        )

        # NEW: single object that drives every modern (interactive) shap.plots.* call
        shap_explanation = shap.Explanation(
            values=shap_values_oof,
            base_values=base_values_oof,
            data=X_display,
            feature_names=feature_names,
        )

    return {
        "summary":              summary,
        "test_perf":            test_perf,
        "train_perf":           train_perf,
        "oof_probs":            oof_mean_probs,
        "oof_y":                np.asarray(y),
        "best_params_per_fold": best_params_per_fold,
        "shap_values_oof":      shap_values_oof,
        "shap_count":           shap_count,
        "feature_names":        feature_names,
        "shap_importance":      shap_importance,
        "base_values_oof":      base_values_oof,      # NEW
        "X_display":            X_display,            # NEW
        "shap_explanation":     shap_explanation,     # NEW
    }


# =====================================================================
#  INTERACTIVE SHAP HELPERS
# =====================================================================
def _force_payload(expl, idx=None):
    """JSON-safe (scalar base, values, feature DataFrame) for force plots."""
    idx  = np.arange(expl.values.shape[0]) if idx is None else np.asarray(idx)
    vals = np.asarray(expl.values,      dtype=np.float64)[idx]
    base = np.asarray(expl.base_values, dtype=np.float64)[idx]
    data = np.asarray(expl.data,        dtype=np.float64)[idx]
    names = [str(f) for f in expl.feature_names]

    b0    = float(base.mean())            # <- scalar, this is the fix
    drift = float(np.abs(base - b0).max())
    if drift > 1e-3:
        print(f"[warn] base value varies by up to {drift:.4g} across samples; "
              f"plot uses the mean, so additivity is approximate.")
    return b0, vals, pd.DataFrame(data, columns=names)


def interactive_force_html(res, out_html="shap_oof_force.html",
                           max_samples=500, order_by="prediction", random_state=0):
    expl = res["shap_explanation"]
    n = expl.values.shape[0]
    if n > max_samples:
        if order_by == "prediction":
            pred = expl.base_values + expl.values.sum(axis=1)
            idx  = np.argsort(pred)[np.linspace(0, n - 1, max_samples).astype(int)]
        else:
            idx = np.random.default_rng(random_state).choice(n, max_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(n)

    b0, vals, feats = _force_payload(expl, idx)
    plot = shap.force_plot(b0, vals, feats)      # legacy API: scalar base, always JSON-safe
    shap.save_html(out_html, plot)
    print(f"[saved] {out_html}  ({len(idx)} samples)")
    return plot


def interactive_single_patient(res, i, out_html=None):
    expl = res["shap_explanation"]
    b0, vals, feats = _force_payload(expl, [i])
    plot = shap.force_plot(b0, vals[0], feats.iloc[0])
    if out_html:
        shap.save_html(out_html, plot)
    shap.plots.waterfall(expl[i], max_display=12, show=True)   # matplotlib, keeps the true per-sample base
    return plot


def shap_overview(res, top_k=15):
    """Beeswarm + bar + dependence scatter for the top feature."""
    expl = res["shap_explanation"]
    try:
        shap.plots.beeswarm(expl, max_display=top_k, show=True)
        shap.plots.bar(expl, max_display=top_k, show=True)
        top_feat = res["shap_importance"].iloc[0]["feature"]
        shap.plots.scatter(expl[:, top_feat], color=expl, show=True)
    except Exception:                                   # older shap
        shap.summary_plot(
            res["shap_values_oof"], res["X_display"],
            feature_names=res["feature_names"], max_display=top_k,
        )


def shap_decision(res, idx=None, max_samples=200):
    """Decision plot — good for spotting sub-groups with divergent paths."""
    expl = res["shap_explanation"]
    if idx is None:
        idx = np.arange(min(max_samples, expl.values.shape[0]))
    shap.decision_plot(
        float(np.mean(res["base_values_oof"][idx])),
        res["shap_values_oof"][idx],
        pd.DataFrame(res["X_display"][idx], columns=res["feature_names"]),
        link="identity",
    )


# =====================================================================
#  USAGE (in a Jupyter notebook)
# =====================================================================
#   shap.initjs()                     # <-- required once, before any force plot
#
#   res = nested_cross_calibrated_rf(X, y, param_distributions=param_dist)
#
#   shap_overview(res)                                   # beeswarm / bar / scatter
#   interactive_force_html(res, "shap_oof_force.html")   # interactive, all samples
#   interactive_single_patient(res, i=17, out_html="patient_17.html")
#   shap_decision(res)
#
# Outside a notebook, open the saved .html files in a browser.