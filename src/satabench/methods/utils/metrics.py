import numpy as np
from collections import defaultdict, Counter


# Convert a string of letters to a set of characters
def to_set(s):
    return set(s)

# Hamming Score
def hamming_score(preds, labels):
    scores = []
    for pred, label in zip(preds, labels):
        pred_set, label_set = to_set(pred), to_set(label)
        intersection = len(pred_set & label_set)
        union = len(pred_set | label_set)
        score = intersection / union if union > 0 else 0
        scores.append(score)
    return np.mean(scores)

# Length Difference Mean
def length_dif(preds, labels):
    return np.mean([len(pred) - len(label) for pred, label in zip(preds, labels)])

# Length Difference Standard Deviation
def length_std(preds, labels):
    return np.std([len(pred) - len(label) for pred, label in zip(preds, labels)])

# Absolute Length Difference
def length_abs(preds, labels):
    return np.mean([abs(len(pred) - len(label)) for pred, label in zip(preds, labels)])

# Recall Standard Deviation
def rstd(preds, labels, CHOICES="ABCDEFGHIJKLMNO"):
    report = calculate_metrics_per_label(preds, labels)
    recalls = []
    for choice in CHOICES:
        if choice in report.keys():
            recalls.append(report[choice]['recall'] * 100)
    return np.round(np.std(recalls), 4)

# Relative Standard Deviation for Accuracy and F1
def rsd(preds, labels, CHOICES="ABCDEFGHIJKLMNO"):
    report = calculate_metrics_per_label(labels, preds)
    acc, f1 = exact_match(preds, labels), f1_score(preds, labels)
    
    accs, f1s = [], []
    for choice in CHOICES:
        if choice in report.keys():
            choice_corr = [1 if pred == label and choice in label else 0 for pred, label in zip(preds, labels)]
            choice_support = [1 if choice in label else 0 for label in labels]
            acc_choice = sum(choice_corr) / sum(choice_support) if sum(choice_support) != 0 else 0.0
            f1_choice = report[choice]['f1-score']
            accs.append(acc_choice)
            f1s.append((f1_choice - f1) ** 2)
    
    acc = sum(accs) / len(accs) if len(accs) > 0 else 0
    accs = [(x - acc) ** 2 for x in accs]
    rsd_acc = np.sqrt(np.mean(accs)) / acc if acc != 0 else -1
    rsd_f1 = np.sqrt(np.mean(f1s)) / f1 if f1 != 0 else -1
    
    return np.round(rsd_acc, 4), np.round(rsd_f1, 4)

def rckld(preds, labels, CHOICES="ABCDEFGHIJKLMNO"):
    pred_count = {choice: sum(choice in pred for pred in preds) for choice in CHOICES}
    label_count = {choice: sum(choice in label for label in labels) for choice in CHOICES}
    
    support = len(labels)
    rckld = 0
    
    for choice in label_count.keys():
        pred_r = pred_count[choice] / support
        label_r = label_count[choice] / support
        if label_r != 0:
            rckld += (1 - pred_r / label_r) * np.log(label_r / (pred_r + (pred_r == 0) * 1e-10))
    
    return np.round(rckld, 4)

def ckld(preds, labels):
    preds, labels = ''.join(preds), ''.join(labels)
    pred_count = Counter(preds[:min(len(preds), len(labels))])
    label_count = Counter(labels[:min(len(preds), len(labels))])
    pred_count = {i:pred_count[i] for i in label_count}
    support = len(labels)
    
    ckld = 0
    for choice in label_count.keys():
        pred_r = pred_count[choice] / support
        label_r = label_count[choice] / support 
        
        if label_r != 0 :
            #print(label_r, np.log(label_r / (pred_r + int(pred_r==0) * 1e-10)))
            ckld +=  label_r * np.log(label_r / (pred_r + int(pred_r==0) * 1e-10))
    
    return np.round(ckld, 4)

# Function to calculate precision, recall, and f1-score for each label
def calculate_metrics_per_label(preds, labels):
    metrics = defaultdict(dict)
    y_pred = [list(y) for y in preds]
    y_true = [list(y) for y in labels]
    all_labels = list(set([j for i in y_true for j in i])) 
    for label in all_labels:
        TP = FP = FN = TN = 0
        for true_labels, pred_labels in zip(y_true, y_pred):
            true_set = set(true_labels)
            pred_set = set(pred_labels)
            
            if label in pred_set and label in true_set:
                TP += 1  # True Positive
            if label in pred_set and label not in true_set:
                FP += 1  # False Positive
            if label in true_set and label not in pred_set:
                FN += 1  # False Negative
            if label not in pred_set and label not in true_set:
                TN += 1  # True Negative

        # Calculate Precision, Recall and F1
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        # Calculate Accuracy
        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        
        metrics[label]['accuracy'] = accuracy
        metrics[label]['precision'] = precision
        metrics[label]['recall'] = recall
        metrics[label]['f1-score'] = f1_score
    
    return metrics

def exact_match(preds, labels):
    return np.sum([i == j for i, j in zip(labels, preds) if len(i) != 0 ])/len([len(i) for i in preds if len(i) != 0])

def length_exact_match(preds, labels):
    return np.sum([len(i) == len(j) for i, j in zip(labels, preds) if len(i) != 0]) / len([len(i) for i in preds if len(i) != 0])

def length_dif(preds, labels):
    return np.mean([len(i) - len(j) for i, j in zip(preds, labels)])

def length_std(preds, labels):
    return np.std([len(i) - len(j) for i, j in zip(preds, labels)])

def length_abs(preds, labels):
    return np.mean([abs(len(i) - len(j)) for i, j in zip(preds, labels)])

def f1_score(preds, labels):

    precision = precision_score(preds, labels)
    recall = recall_score(preds, labels)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

def recall_score(preds, labels):
    y_pred = [list(y) for y in preds]
    y_true = [list(y) for y in labels]
    n = len(y_true)
    total_recall = 0
    for true_labels, pred_labels in zip(y_true, y_pred):
        intersection = len(set(true_labels).intersection(set(pred_labels)))
        total_recall += intersection / len(true_labels)
        #print(intersection , len(true_labels))
    return total_recall / n


def precision_score(preds, labels):
    y_pred = [list(y) for y in preds]
    y_true = [list(y) for y in labels]
    
    n = len(y_true)
    total_precision = 0
    for true_labels, pred_labels in zip(y_true, y_pred):
        intersection = len(set(true_labels).intersection(set(pred_labels)))
        total_precision += intersection / max(len(pred_labels), 1)
        #print(total_precision)
    return total_precision / n

def difference_between_lists(l1, l2):
    result_1 = []
    result_2 = []
    
    for item1, item2 in zip(l1, l2):
        # Find characters in l1 but not in l2
        diff1 = ''.join(sorted(set(item1) - set(item2)))
        # Find characters in l2 but not in l1
        diff2 = ''.join(sorted(set(item2) - set(item1)))
        
        result_1.append(diff1)
        result_2.append(diff2)
    
    return result_1, result_2