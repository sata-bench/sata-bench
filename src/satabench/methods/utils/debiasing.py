# Adapted from PriDe algorithm arxiv.org/pdf/2309.03882
from itertools import permutations
import numpy as np


# Compute prior bias for SATA setting
def compute_prior_token_bias(observed, permuted_indices=None):
    observed = np.array(observed)
    observed = observed / (observed.sum(axis=1, keepdims=True) + 1e-10) # normalize raw probs to unit sum

    # here we change to a flexible num_options to get permuted_indices
    num_options = observed.shape[1]
    if permuted_indices is None:
        permuted_indices = generate_permuted_indices(num_options)

    if observed.shape[0] != observed.shape[1]:
        raise ValueError("observed.shape[0] != observed.shape[1]")

    debiased = gather_probs(observed, permuted_indices)
    debiased = np.mean(debiased, axis=1)

    prior = softmax(np.log(observed + 1e-10).mean(axis=0))
    return observed, debiased, prior

def generate_permuted_indices(num_options):
    """
    Generate cyclic permutations for any number of options.
    
    Args:
        num_options (int): Number of options (shape[1] of observed).
        
    Returns:
        list of tuples: List of cyclic permutations.
    """
    return [tuple((i + j) % num_options for i in range(num_options)) for j in range(num_options)]


def gather_probs(observed, permuted_indices=None):
    if permuted_indices is None:
        permuted_indices = sorted(permutations(range(observed.shape[1])))
    assert len(permuted_indices) == observed.shape[0]
    gathered_probs = [[] for _ in range(observed.shape[1])]

    for pdx, indices in enumerate(permuted_indices):
        for idx, index in enumerate(indices):
            gathered_probs[index].append(observed[pdx, idx])
    return gathered_probs


def cantor_expansion(p):
    n = len(p)
    code = 0
    for i in range(n):
        smaller_count = sum(1 for j in range(i+1, n) if p[j] < p[i])
        if smaller_count > 0:
            code += smaller_count * factorial(n - i - 1)
    return code


def factorial(num):
    if num == 0:
        return 1
    return num * factorial(num - 1)


def softmax(x):
    x = np.array(x)
    x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    x = x / (np.sum(x, axis=-1, keepdims=True) + 1e-10)
    return x