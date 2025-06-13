import torch.nn.functional as F

def diversity_loss(param_sets):
    """
    Penalize similarity among expert augmentation parameter vectors.
    param_sets: list of Tensors, each of shape (4,)
    """
    loss = 0.0
    count = 0
    for i in range(len(param_sets)):
        for j in range(i + 1, len(param_sets)):
            sim = F.cosine_similarity(param_sets[i], param_sets[j], dim=0) #TODO: try other similarity metrics OR MAYBE NORMALIZING?
            loss += sim
            count += 1
    return loss / count if count > 0 else 0.0
