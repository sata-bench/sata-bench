"""metrics_utils.py

Utility functions for evaluating multi‑label classification performance on
select‑all‑that‑apply (SATA) style questions.

Each metric operates on *preds* and *labels* where both are lists/arrays of
strings. Each string contains one or more choice letters (e.g. "AB" or "C")
drawn from the global `CHOICES` constant.

The module reproduces the metrics introduced in the paper and exposes
helpers for performance mterics(exact match (EM), micro‑averaged precision/recall/F1, Hamming
Jaccard index (JI)), count bias metrics (CtACC, CtDif, CtDifAbs), and selection bias metrics (CKLD, RStd, RSD, SPD, …).

All functions are **pure** and do not mutate their inputs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #

#: The complete ordered set of valid answer choices.
CHOICES: str = "ABCDEFGHIJKLMNO"


# --------------------------------------------------------------------------- #
# Low‑level helpers
# --------------------------------------------------------------------------- #


def difference_between_lists(
    l1: List[str], l2: List[str]
) -> Tuple[List[str], List[str]]:
    """Return per‑sample symmetric set differences *l1 \ l2* and *l2 \ l1*."""
    diff_1, diff_2 = [], []
    for a, b in zip(l1, l2):
        diff_1.append("".join(sorted(set(a) - set(b))))
        diff_2.append("".join(sorted(set(b) - set(a))))
    return diff_1, diff_2

def calculate_metrics_per_label(
    preds: List[str], labels: List[str]
) -> Dict[str, Dict[str, float]]:
    """Return per‑label precision/recall/F1 and accuracy.

    Parameters
    ----------
    preds
        Model predictions. Each element is a string of capital letters.
    labels
        Gold answers. Same format as *preds*.

    Returns
    -------
    Dict mapping each label (e.g. ``"A"``) to a sub‑dict with keys
    ``accuracy``, ``precision``, ``recall`` and ``f1-score``.
    """

    metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
    y_pred: List[List[str]] = [list(p) for p in preds]
    y_true: List[List[str]] = [list(t) for t in labels]
    all_labels: List[str] = sorted({lbl for sample in y_true for lbl in sample})

    for label in all_labels:
        tp = fp = fn = tn = 0  # type: int

        for true, pred in zip(y_true, y_pred):
            true_set, pred_set = set(true), set(pred)

            tp += label in pred_set and label in true_set
            fp += label in pred_set and label not in true_set
            fn += label in true_set and label not in pred_set
            tn += label not in pred_set and label not in true_set

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0

        metrics[label] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
        }

    return metrics


# --------------------------------------------------------------------------- #
# Higher‑level selection bias metrics
# --------------------------------------------------------------------------- #

def rstd(
    preds: List[str], labels: List[str], *, choices: str = CHOICES
) -> float:
    """Standard deviation of per‑label *recall* (×100)."""
    report = calculate_metrics_per_label(preds, labels)
    recalls = [report[c]["recall"] * 100 for c in choices if c in report]
    return float(np.round(np.std(recalls), 4))


def rsd(
    preds: List[str], labels: List[str], *, choices: str = CHOICES
) -> float:
    """Relative standard deviation of per‑label accuracy/F1.

    Follows the original definition where only the F1‑variant is returned.
    """
    # Note: `report` swaps *preds*/*labels* to mimic original behaviour.
    report = calculate_metrics_per_label(labels, preds)

    overall_f1 = f1_score(preds, labels)

    accs, var_f1 = [], []
    for c in choices:
        if c not in report:
            continue

        # Accuracy of label *c* (conditioned on its support)
        correct = [int((p == l) and (l == c)) for p, l in zip(preds, labels)]
        support = [int(l == c) for l in labels]
        acc_c = sum(correct) / sum(support) if sum(support) else 0.0
        f1_c = report[c]["f1-score"]

        accs.append(acc_c)
        var_f1.append((f1_c - overall_f1) ** 2)

    mean_acc = sum(accs) / len(accs) if accs else 0.0
    rsd_f1 = (
        np.sqrt(np.mean(var_f1)) / overall_f1 if overall_f1 else -1.0
    )
    return float(np.round(rsd_f1, 4))

def ckld(preds: List[str], labels: List[str]) -> float:
    """Cross‑KL divergence between predicted and gold label frequencies."""
    preds_str, labels_str = "".join(preds), "".join(labels)
    support = min(len(preds_str), len(labels_str))

    pred_count = Counter(preds_str[:support])
    label_count = Counter(labels_str[:support])

    ckld_val = 0.0
    for choice, gold_freq in label_count.items():
        gold_r = gold_freq / support
        pred_r = pred_count.get(choice, 0) / support
        if gold_r:
            ckld_val += gold_r * np.log(gold_r / (pred_r or 1e-10))
    return float(np.round(ckld_val, 4))


def rckld(preds, labels): # preds and labels are matrices of True/False, [repeats, len(options)]
    pred_count = {
        i: sum(preds[:,i]) for i in range(len(preds[0]))
    }
    label_count = {
        i: sum(labels[:,i]) for i in range(len(preds[0]))
    }

    support = len(labels)
    
    rckld = 0
    for choice in label_count.keys():
        pred_r = pred_count[choice] / support
        label_r = label_count[choice] / support 
        
        if label_r != 0 :
            rckld +=  (1 - pred_r/label_r) * np.log(label_r / (pred_r + int(pred_r==0) * 1e-10))
    
    return np.round(rckld, 4)

def weighted_rckld(preds, labels): # preds and labels are matrices of True/False, [repeats, len(options)]
    N_choices_p = [len(pred) for pred in preds]
    N_choices_l = [len(label) for label in labels]
    assert N_choices_p == N_choices_l

    N_choices = set(N_choices_p)

    supports = {
        i: len([rep for rep in preds if len(rep) == i]) for i in N_choices
    }
    # print(supports)

    rckld_vals = {}
    weighted_rckld_vals = {}
    for i in N_choices:
        preds_sub = [rep for rep in preds if len(rep) == i]
        labels_sub = [rep for rep in labels if len(rep) == i]
        # print(i)
        # print (preds_sub)
        # print (labels_sub)
        rckld_vals[i] = rckld(np.asarray(preds_sub), np.asarray(labels_sub))
        weighted_rckld_vals[i] = rckld_vals[i] * supports[i]

    # print(rckld_vals)
    weighted_rckld = sum(weighted_rckld_vals.values()) / sum(supports.values())
    return round(weighted_rckld, 4)  


# --------------------------------------------------------------------------- #
# Classic SATA metrics
# --------------------------------------------------------------------------- #

def hamming_score(preds: List[str], labels: List[str]) -> float:
    """Jaccard index averaged across samples."""
    total = sum(len(set(t) & set(p)) / len(set(t) | set(p)) for t, p in zip(labels, preds))
    return total / len(labels)


def recall_score(preds: List[str], labels: List[str]) -> float:
    total = sum(len(set(t) & set(p)) / len(t) for t, p in zip(labels, preds))
    return total / len(labels)


def precision_score(preds: List[str], labels: List[str]) -> float:
    total = sum(len(set(t) & set(p)) / max(len(p), 1) for t, p in zip(labels, preds))
    return total / len(labels)


def f1_score(preds: List[str], labels: List[str]) -> float:
    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def exact_match(preds: List[str], labels: List[str]) -> float:
    return np.sum([i == j for i, j in zip(labels, preds) if len(i) != 0 ])/len([len(i) for i in preds if len(i) != 0])



# --------------------------------------------------------------------------- #
# Count bias metrics
# --------------------------------------------------------------------------- #

def CtDif(preds: List[str], labels: List[str]) -> float:
    """Mean signed difference in set sizes (pred − gold)."""
    return float(np.mean([len(p) - len(l) for p, l in zip(preds, labels)]))


def CtAcc(preds: List[str], labels: List[str]) -> float:
    """Accuracy of predicting the *exact* set size."""
    return float(np.mean([int(len(p) == len(l)) for p, l in zip(preds, labels)]))


def CtDifAbs(preds: List[str], labels: List[str]) -> float:
    """Mean absolute difference in set sizes."""
    return float(np.mean([abs(len(p) - len(l)) for p, l in zip(preds, labels)]))




# --------------------------------------------------------------------------- #
# Master aggregator
# --------------------------------------------------------------------------- #

def collect_metrics(
    preds: List[str],
    labels: List[str],
    maximum: List[str],
) -> Dict[str, float]:
    """Compute the full metric suite in one call.

    The *maximum* argument is a list whose *i*‑th element equals the last valid
    choice (e.g. "D") for the *i*‑th example. It is used to build indicator
    matrices of the appropriate width for SPD / RCKLD‑based scores.
    """
    print( preds, labels)
    d = {
        "EM": exact_match(preds, labels),
        "Precision": precision_score(preds, labels),
        "Recall": recall_score(preds, labels),
        "JI": hamming_score(preds, labels),
        "F1": f1_score(preds, labels),
        "CKLD": ckld(preds, labels),
        "RSD": rsd(preds, labels),
        "RStd": rstd(preds, labels),
    }

    d["CtDif"] = CtDif(preds, labels)
    d["CtAcc"] = CtAcc(preds, labels)
    d["CtDifAbs"] = CtDifAbs(preds, labels)
    # Build indicator matrices for SPD
    ps, gts = [], []
    for i,j, k in zip(preds, labels, maximum):
        #print(i,j,k)
        base1 = [False] * (CHOICES.index(k) + 1)
        for c in range(len(base1)):
            if CHOICES[c] in i:
                base1[c] = True
        ps.append(base1)
        base2 = [False] * (CHOICES.index(k) + 1)
        for c in range(len(base1)):
            if CHOICES[c] in j:
                base2[c] = True
        gts.append(base2)

    d["SPD"] = weighted_rckld(ps, gts)


    return d


