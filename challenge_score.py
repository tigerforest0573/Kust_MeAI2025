import numpy as np
def calculate_challenge_score(pred_probs, true_labels, alpha=0.05, random_seed=42):
    pred_probs = np.array(pred_probs)
    true_labels = np.array(true_labels)
    
    if len(pred_probs) != len(true_labels):
        raise ValueError("1")
    n = len(pred_probs)
    if n == 0:
        return 0.0
    
    total_true_positives = np.sum(true_labels)
    if total_true_positives == 0:
        return 0.0
    
    k = int(np.floor(alpha * n))
    if k == 0:
        k = 1
    
    sorted_indices = np.argsort(-pred_probs)
    sorted_probs = pred_probs[sorted_indices]
    sorted_labels = true_labels[sorted_indices]
    
    if k < n:
        boundary_prob = sorted_probs[k - 1]
        tie_indices = np.where(sorted_probs == boundary_prob)[0]
        if np.any(tie_indices < k) and np.any(tie_indices >= k):
            num_trials = 100
            scores = []
            rng = np.random.default_rng(random_seed)
            for _ in range(num_trials):
                shuffled_tie_indices = rng.permutation(tie_indices)
                new_sorted_indices = []
                tie_idx_ptr = 0
                for i in range(n):
                    if i not in tie_indices:
                        new_sorted_indices.append(i)
                    else:
                        new_sorted_indices.append(shuffled_tie_indices[tie_idx_ptr])
                        tie_idx_ptr += 1
                top_k_labels = sorted_labels[new_sorted_indices[:k]]
                top_true_positives = np.sum(top_k_labels)
                scores.append(top_true_positives / total_true_positives)
            return np.mean(scores)
    
    top_k_labels = sorted_labels[:k]
    top_true_positives = np.sum(top_k_labels)
    return top_true_positives / total_true_positives