"""Group endpoints and source/target-only threshold selection."""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, roc_curve


def decision_endpoints(y,p,threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    pred = p >= threshold
    sens = float(pred[y == 1].mean()) if (y == 1).any() else np.nan
    spec = float((~pred[y == 0]).mean()) if (y == 0).any() else np.nan
    result = dict(balanced_accuracy=(sens+spec)/2, sensitivity=sens, specificity=spec,specificity_error=spec-.9)
    confidence = np.abs(p-.5)
    keep = np.argsort(-confidence,kind="stable")[:max(1,int(.8*len(y)))]
    result["risk_at_80pct_coverage"] = float(np.mean(pred[keep] != y[keep]))
    for prev in [.01,.1,.5]:
        denom = sens*prev+(1-spec)*(1-prev)
        result[f"ppv_prevalence_{prev:g}"] = sens*prev/denom if denom > 0 else np.nan
    return result


def endpoints(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    two = len(np.unique(y)) == 2
    result = dict(auroc=float(roc_auc_score(y,p)) if two else np.nan,
        brier=float(np.mean((p-y)**2)),
        auprc_constructed_prevalence=float(average_precision_score(y,p)) if two else np.nan,
        constructed_prevalence=float(y.mean()), n_groups=len(y))
    # Logistic calibration describes the constructed evaluation population only.
    if two and np.ptp(p) > 0:
        lr = LogisticRegression(C=1e6, solver="lbfgs",max_iter=1000).fit(logit(np.clip(p,1e-7,1-1e-7))[:,None], y)
        result.update(calibration_slope=float(lr.coef_[0,0]), calibration_intercept=float(lr.intercept_[0]))
    else:
        result.update(calibration_slope=np.nan, calibration_intercept=np.nan)
    result.update(decision_endpoints(y,p,threshold))
    return result


def quantile_threshold(negatives, specificity=.9, tolerance=False, confidence=.95):
    z = np.sort(np.asarray(negatives))
    n = len(z)
    if not n:
        return np.nan
    rank = int(binom.ppf(confidence,n,specificity))+1 if tolerance else int(np.ceil(specificity*n))
    if rank > n:
        return np.nan  # No finite observed order statistic supplies the requested bound.
    return float(np.nextafter(z[max(rank,1)-1], np.inf))


def ba_threshold(y, p):
    fpr,tpr,thresholds = roc_curve(y,p)
    # Stable tie rule: first (highest) threshold. Guard sklearn's +inf endpoint.
    return float(thresholds[np.argmax(tpr-fpr)])


def temperature(y,p):
    z = logit(np.clip(p,1e-7,1-1e-7))
    r = minimize_scalar(lambda logt: log_loss(y,expit(z/np.exp(logt)),labels=[0,1]),
        bounds=(np.log(.05),np.log(20)),method="bounded")
    return float(np.exp(r.x))
