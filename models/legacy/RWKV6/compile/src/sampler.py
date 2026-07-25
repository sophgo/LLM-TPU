# https://github.com/yuunnn-w/RWKV_Pytorch
import torch
import torch.nn.functional as F

def sample_logits(out: torch.Tensor, temperature: float = 1.0, top_p: float = 0.8) -> torch.Tensor:
    """
    对模型输出的logits进行采样。

    Args:
        out (torch.Tensor): 模型输出的logits张量,形状为[Batch, vocab_size]。
        temperature (float): 温度参数,用于调节采样的多样性,默认为1.0。
        top_p (float): Top-p截断参数,用于稳定和控制采样概率分布,默认为0.8。

    Returns:
        torch.Tensor: 采样结果,形状为[Batch, 1],每个元素表示一个样本中采样得到的词的索引。
    """
    # 确保top_p和temperature都是非负值
    top_p = max(0.0, min(1.0, top_p))
    temperature = max(0.0, temperature)

    # 将out转换为概率分布
    probs = F.softmax(out, dim=-1)

    # 根据top_p截断概率分布
    sorted_probs, _ = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff_mask = (cumulative_probs > top_p).float()
    cutoff_index = torch.argmax(cutoff_mask * torch.arange(cutoff_mask.shape[-1], device=cutoff_mask.device).float(), dim=-1)
    cutoff_values = sorted_probs.gather(-1, cutoff_index.unsqueeze(-1)).squeeze(-1)
    probs = torch.where(probs < cutoff_values.unsqueeze(-1), torch.zeros_like(probs), probs)

    # 对概率分布进行温度调节
    if temperature != 1.0:
        probs = torch.pow(probs, 1.0 / temperature)

    # 归一化概率分布
    probs /= torch.sum(probs, dim=-1, keepdim=True)

    # 如果top_p为0,则选择概率最大的位置;否则按照概率分布随机采样
    if top_p != 0:
        sampled_indices = torch.multinomial(probs, num_samples=1)
    else:
        sampled_indices = torch.argmax(probs, dim=-1, keepdim=True)
        

    return sampled_indices
