import torch
from typing import Optional
from typing import Union, List


def calculate_accuracy(guess_index: torch.Tensor, label_index: torch.Tensor) -> float:
    # Create mask for valid labels (excluding non_label_index)
    valid_mask = label_index >= 0
    if not valid_mask.all():
        if valid_mask.any():
            label_index = label_index[valid_mask]
            guess_index = guess_index[valid_mask]
        else:
            # logger.warning("All labels invalid, returning zero accuracy")
            return 0.0

    accuracy = (guess_index == label_index).float().mean().item()
    return accuracy


def calculate_topk_accuracy(
    outputs: torch.Tensor,
    label_index: torch.Tensor,
    k: Union[int, List[int]] = [3],
) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Calculate top-k accuracy for given k values.

    Args:
        outputs: Model outputs tensor of shape (batch_size, n_timesteps, n_classes)
        label_index: True label indices tensor
        k: Either int or list of ints for top-k values

    Returns:
        Tensor of shape (batch_size, n_timesteps) if k is int,
        or list of such tensors if k is a list
    """
    # Handle single k value
    if isinstance(k, int):
        k_list = [k]
        return_single = True
    else:
        k_list = k
        return_single = False

    n_classes = outputs.shape[-1]
    max_k = min(max(k_list), n_classes)

    # Get top-k indices for the maximum k needed
    topk_indices = torch.topk(outputs, k=max_k, dim=-1).indices

    results = []
    for k_val in k_list:
        # Expand label_index to match dimensions if needed
        if label_index.dim() == 1:
            label_expanded = label_index.unsqueeze(1).expand(-1, outputs.shape[1])
        else:
            label_expanded = label_index

        # Check if true labels are in top-k predictions
        is_in_topk = (topk_indices[:, :, :k_val] == label_expanded.unsqueeze(-1)).any(
            dim=-1
        )

        results.append(is_in_topk.float())

    if return_single:
        return results[0]
    return results


def _confidence(softmax_outputs: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Extract confidence scores for given indices efficiently."""
    _ = indices.device
    dtype = softmax_outputs.dtype
    batch_size, n_timesteps, n_classes = softmax_outputs.shape

    # Create valid mask once
    valid_mask = (indices >= 0) & (indices < n_classes)

    if not valid_mask.any():
        return torch.zeros_like(indices, dtype=dtype)

    if indices.dim() == 1:
        # Vectorized expansion - more efficient than manual expand
        indices_2d = indices.unsqueeze(1).expand(-1, n_timesteps)
        valid_mask_2d = valid_mask.unsqueeze(1).expand(-1, n_timesteps)
    else:
        indices_2d = indices
        valid_mask_2d = valid_mask

    # Use advanced indexing - more efficient than gather for this case
    result = torch.zeros_like(indices_2d, dtype=dtype)
    if valid_mask_2d.any():
        batch_idx, time_idx = torch.where(valid_mask_2d)
        class_idx = indices_2d[valid_mask_2d]
        result[batch_idx, time_idx] = softmax_outputs[batch_idx, time_idx, class_idx]

    return result if indices.dim() > 1 else result.squeeze(1)


def calculate_confidence(
    outputs: torch.Tensor, indices: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Calculate confidence scores with optional caching."""
    # Cache softmax if called repeatedly with same outputs

    softmax_outputs = torch.softmax(outputs, dim=-1)

    if indices is None:
        indices = torch.argmax(softmax_outputs, dim=-1)
        return softmax_outputs.max(dim=-1)[0]
    elif isinstance(indices, list):  # recursive
        return [_confidence(softmax_outputs, idx) for idx in indices]
    else:
        return _confidence(softmax_outputs, indices)
