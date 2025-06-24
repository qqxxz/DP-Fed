import torch
from options import parse_args
from torch import autograd

args = parse_args()

def compute_fisher_diag(model, dataloader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # Calculate output log probabilities
        log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

        for i, label in enumerate(labels):
            log_prob = log_probs[i, label]

            # Calculate first-order derivatives (gradients)
            model.zero_grad()
            grad1 = autograd.grad(log_prob, model.parameters(), create_graph=True, retain_graph=True)

            # Update Fisher diagonal elements
            for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                fisher_diag_value.add_(grad_value.detach() ** 2)
                
            # Free up memory by removing computation graph
            del log_prob, grad1

        # Release CUDA memory
        # torch.cuda.empty_cache()

    # Calculate the mean value
    num_samples = len(dataloader.dataset)
    fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
        normalized_fisher_diag.append(normalized_fisher_value)

    return normalized_fisher_diag

def compute_noise_multiplier(target_epsilon, target_delta, global_epoch, local_epoch, batch_size, client_data_sizes):
    """
    手动计算噪声乘子 (noise_multiplier)。

    Args:
        target_epsilon (float): 目标隐私预算 ε。
        target_delta (float): 目标隐私失败概率 δ。
        global_epoch (int): 全局训练轮数。
        local_epoch (int): 每个客户端的本地训练轮数。
        batch_size (int): 批量大小。
        client_data_sizes (list): 每个客户端的数据集大小。

    Returns:
        float: 计算出的噪声乘子。
    """
    total_data_size = sum(client_data_sizes)
    sampling_probability = batch_size / total_data_size
    steps = global_epoch * local_epoch

    # 手动计算噪声乘子
    noise_multiplier = 1.0
    return noise_multiplier