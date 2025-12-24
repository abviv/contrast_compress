import torch.nn.functional as F
import torch
import torch.nn as nn

def compute_pairwise_distances(batch):
    batch_size = batch.size(0)
    distances = torch.cdist(batch.view(batch_size, -1), batch.view(batch_size, -1), p=2)
    return distances


def apply_cosine_similarity(batch, factor=0.2):
    """Compute pairwise Euclidean distances in local coordinate system wrt to the trajectories.
    args:
        factor: Determines how much weight we should give to ADE calc between trajectories
    """
    pairwise_distances = compute_pairwise_distances(batch)

    # Compute direction similarities
    dir_start = batch[:, 0, :]
    dir_end = batch[:, -1, :]
    directions = dir_end - dir_start
    endpoint_similarities = F.cosine_similarity(directions.unsqueeze(1), directions.unsqueeze(0), dim=2)
    # Combine Euclidean distance and direction similarity with weighted average
    # similarities = 0.8 * direction_similarities + 0.2 * (1 / (1 + pairwise_distances))
    similarities = 1 / (1 + pairwise_distances * factor) * endpoint_similarities
    return similarities


def apply_fft_similarity_efficient(batch):
    """Compute pairwise similarities between trajectories using FFT magnitudes in an efficient manner.

    This function:
      1. Computes FFT of each trajectory in the batch.
      2. Takes the magnitude of FFT coefficients and reduces dimensionality by keeping only the first half.
      3. Normalizes the flattened FFT magnitudes for scale invariance.
      4. Computes the cosine similarity matrix via a vectorized matrix multiplication.

    Args:
        batch (torch.Tensor): Batch of trajectories of shape (batch_size, seq_len, dim)

    Returns:
        torch.Tensor: Pairwise similarity matrix of shape (batch_size, batch_size)
    """
    batch_size, seq_len, dim = batch.shape

    # Compute FFT along the sequence dimension
    fft_batch = torch.fft.fft(batch, dim=1)  # shape: (batch_size, seq_len, dim)
    fft_magnitudes = torch.abs(fft_batch)

    # For real inputs, use only the first half of FFT coefficients (due to conjugate symmetry)
    half_seq_len = seq_len // 2 + 1
    fft_magnitudes = fft_magnitudes[:, :half_seq_len, :]

    # Flatten the reduced FFT magnitudes to shape (batch_size, half_seq_len * dim)
    fft_magnitudes_flat = fft_magnitudes.reshape(batch_size, -1)

    # Normalize each flattened vector to unit length
    eps = 1e-8
    norms = fft_magnitudes_flat.norm(p=2, dim=1, keepdim=True) + eps
    normalized = fft_magnitudes_flat / norms

    # Compute cosine similarity using matrix multiplication.
    similarities = normalized @ normalized.t()

    return similarities


def generate_triplets(batch, similarity_threshold=0.7, cfg=None, mining_strategy='random_mining'):
    """Generate triplets for triplet loss training using different mining strategies.
    
    Args:
        batch: Tensor of shape (batch_size, seq_len, dim) containing trajectories
        similarity_threshold: Threshold for determining positive/negative pairs
        cfg: Config object containing similarity function and other parameters
        mining_strategy: Mining strategy to use ('random_mining', 'hard_mining', or 'semi_hard_mining')
    
    Returns:
        List of triplet tuples (anchor, positive, negative) based on similarity threshold
    """

    if cfg.args.similarity_function == "fft_similarity":
        similarities = apply_fft_similarity_efficient(batch)
    elif cfg.args.similarity_function == "cosine_similarity":
        similarities = apply_cosine_similarity(batch, cfg.args.cosine_load_factor)
    else:
        raise ValueError(f"Invalid similarity function: {cfg.args.similarity_function}")

    triplets = []
    batch_size = batch.size(0)
    counter = 0
    
    if mining_strategy == "hard_mining":
        print("Hard Mining")
        for i in range(batch_size):
            positive_indices = torch.where(similarities[i] >= similarity_threshold)[0]
            positive_indices = positive_indices[positive_indices != i]
            negative_indices = torch.where(similarities[i] < similarity_threshold)[0]

            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Hard Mining: 
                # Hardest positive: least similar among positives
                # Hardest negative: most similar among negatives
                positive_similarities = similarities[i][positive_indices]
                negative_similarities = similarities[i][negative_indices]
                
                # Get hardest positive (minimum similarity among positives)
                hardest_positive_idx = positive_indices[torch.argmin(positive_similarities)]
                
                # Get hardest negative (maximum similarity among negatives)
                hardest_negative_idx = negative_indices[torch.argmax(negative_similarities)]

                anchor = batch[i]
                positive = batch[hardest_positive_idx]
                negative = batch[hardest_negative_idx]
                triplets.append((anchor, positive, negative))
                counter += 1
    
    elif mining_strategy == "semi_hard_mining":
        print("Semi-Hard Mining")
        # Precompute all pairwise distances once
        distance_matrix = compute_pairwise_distances(batch)
        
        for i in range(batch_size):
            positive_indices = torch.where(similarities[i] >= similarity_threshold)[0]
            positive_indices = positive_indices[positive_indices != i]
            negative_indices = torch.where(similarities[i] < similarity_threshold)[0]

            if len(positive_indices) > 0:
                # Select random positive
                positive_idx = positive_indices[torch.randint(len(positive_indices), (1,)).item()]
                anchor_pos_distance = distance_matrix[i, positive_idx]

                # Vectorized distance lookup for negatives
                anchor_neg_distances = distance_matrix[i, negative_indices]
                
                # Semi-hard condition (vectorized)
                semi_hard_mask = (anchor_pos_distance < anchor_neg_distances) & \
                                (anchor_neg_distances < anchor_pos_distance + cfg.args.margin)
                semi_hard_negatives = negative_indices[semi_hard_mask]

                if len(semi_hard_negatives) > 0:
                    # Random semi-hard negative
                    neg_idx = semi_hard_negatives[torch.randint(len(semi_hard_negatives), (1,)).item()]
                elif len(negative_indices) > 0:  # Fallback
                    neg_idx = negative_indices[torch.argmax(similarities[i][negative_indices])]
                else:
                    continue  # Skip if no negatives
                
                triplets.append((batch[i], batch[positive_idx], batch[neg_idx]))
                counter += 1
        
    elif mining_strategy == "random_mining":
        print("Random Mining")
        for i in range(batch_size):
            positive_indices = torch.where(similarities[i] >= similarity_threshold)[0]
            positive_indices = positive_indices[positive_indices != i]
            negative_indices = torch.where(similarities[i] < similarity_threshold)[0]

            if len(positive_indices) > 0 and len(negative_indices) > 0:
                positive_idx = positive_indices[torch.randint(len(positive_indices), (1,)).item()]
                negative_idx = negative_indices[torch.randint(len(negative_indices), (1,)).item()]
                anchor = batch[i]
                positive = batch[positive_idx]
                negative = batch[negative_idx]
                triplets.append((anchor, positive, negative))
                counter += 1
    else:
        raise ValueError(f"Invalid mining strategy: {mining_strategy}")
    
    print(f"Total length of the triplets: {len(triplets)}")
    
    return triplets


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with mining.
    """
    def __init__(self, margin=0.2, emb_norm=True):
        super().__init__()
        self.margin = margin
        self.emb_norm = emb_norm

    def forward(self, anchor, positive, negative):
        # Normalize embeddings to unit sphere
        if self.emb_norm:
            anchor = F.normalize(anchor, p=2, dim=1)
            positive = F.normalize(positive, p=2, dim=1)
            negative = F.normalize(negative, p=2, dim=1)

        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)

        loss = F.relu(distance_positive - distance_negative + self.margin)
        
        return loss.mean()