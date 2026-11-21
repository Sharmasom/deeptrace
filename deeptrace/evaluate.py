"""Metrics with fake = 1 as the positive class. Thresholds come from validation only."""
import numpy as np
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, accuracy_score,
                             balanced_accuracy_score, precision_recall_fscore_support, confusion_matrix)


def youden(y, p):
    fpr, tpr, th = roc_curve(y, p)
    return float(np.clip(th[int(np.argmax(tpr - fpr))], 0, 1))


def metrics(y, p, thr):
    y, p = np.asarray(y).astype(int), np.asarray(p)
    out = dict(n=len(y), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()), threshold=thr)
    if len(set(y)) == 2:
        out.update(auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))
    pred = (p >= thr).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    out.update(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred) if len(set(y)) == 2 else np.nan,
               real_prec=pr[0], real_rec=rc[0], real_f1=f1[0], fake_prec=pr[1], fake_rec=rc[1], fake_f1=f1[1],
               cm=confusion_matrix(y, pred, labels=[0, 1]).tolist())
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def per_method_auc(df):
    """df: rows with label, p, method. AUC of each fake method against all reals."""
    real = df[df.label == 0]
    out = {}
    for m, g in df[df.label == 1].groupby("method"):
        d = np.r_[np.zeros(len(real)), np.ones(len(g))]
        out[m] = float(roc_auc_score(d, np.r_[real.p, g.p]))
    return out


def bootstrap_auc_diff(y, p_a, p_b, n=1000, seed=0):
    """Paired bootstrap 95% CI of AUC(b) - AUC(a) on the same test videos."""
    rng = np.random.default_rng(seed)
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        d.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    return float(np.mean(d)), [float(x) for x in np.quantile(d, [0.025, 0.975])]
