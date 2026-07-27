# https://github.com/yuunnn-w/RWKV_Pytorch
import torch
import torch.nn.functional as F

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    Sample from the logits output by the model.

    Args:
        out (torch.Tensor): Logits tensor output by the model, with shape [Batch, vocab_size].
        temperature (float): Temperature parameter used to control sampling diversity; defaults to 1.0.
        top_p (float): Top-p truncation parameter used to stabilize and control the sampling probability distribution; defaults to 0.8.

    Returns:
        torch.Tensor: Sampling result with shape [Batch, 1], where each element is the index of the token sampled for a sample.
    """
    # Ensure top_p and temperature are both non-negative
    top_p = max(0.0, min(1.0, top_p))
    temperature = max(0.0, temperature)

    # Convert out to a probability distribution
    probs = F.softmax(out, dim=-1)

    # Truncate the probability distribution according to top_p
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumulative_probs > top_p).float()
    cutoff_index = torch.argmax(cutoff_mask * torch.arange(cutoff_mask.shape[-1], device=cutoff_mask.device).float(), dim=-1)
    cutoff_values = sorted_probs.gather(-1, cutoff_index.unsqueeze(-1)).squeeze(-1)
    probs = torch.where(probs < cutoff_values.unsqueeze(-1), torch.zeros_like(probs), probs)

    # Apply temperature scaling to the probability distribution
    if temperature != 1.0:
        probs = torch.pow(probs, 1.0 / temperature)

    # Normalize the probability distribution
    probs /= torch.sum(probs, dim=-1, keepdim=True)

    # If top_p is 0, pick the position with the highest probability; otherwise sample randomly from the distribution
    if top_p != 0:
        sampled_indices = torch.multinomial(probs, num_samples=1)
    else:
        sampled_indices = torch.argmax(probs, dim=-1, keepdim=True)
        

    return sampled_indices
