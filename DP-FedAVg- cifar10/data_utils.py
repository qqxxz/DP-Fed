import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
from typing import Dict, Tuple
import logging

def get_epsilon(
    num_examples: int,
    batch_size: int,
    noise_multiplier: float,
    epochs: int,
    delta: float = 1e-2
) -> float:
    """返回当前训练设置下的 epsilon 值，使用 Opacus RDP 会计
    
    Args:
        num_examples: 训练样本总数
        batch_size: 批次大小
        noise_multiplier: 噪声乘数
        epochs: 训练轮数
        delta: 隐私参数 delta (default: 1e-2)
    
    Returns:
        Privacy budget epsilon
    """
    try:
        from opacus.accountants import RDPAccountant
    
        # 计算采样概率和步数
        sampling_probability = batch_size / num_examples
        steps = int(epochs * num_examples / batch_size)
    
        # 初始化 RDP 会计
        accountant = RDPAccountant()
        
        # 计算每一步的隐私预算
        for _ in range(steps):
            accountant.step(
                noise_multiplier=noise_multiplier,
                sample_rate=sampling_probability
            )
    
        # 获取当前 epsilon 值
        epsilon = accountant.get_epsilon(delta=delta)
        return epsilon
        
    except Exception as e:
        # 如果 Opacus 出错，回退到 TensorFlow Privacy
        print(f"Opacus 计算出错: {e}，尝试使用 TensorFlow Privacy...")
        from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
        
        # 计算每个 epoch 中的迭代次数
        steps = epochs * num_examples // batch_size
        
        # 直接使用 compute_dp_sgd_privacy 计算 epsilon 值
        epsilon, _ = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
            n=num_examples,
            batch_size=batch_size,
            noise_multiplier=noise_multiplier,
            epochs=epochs,
            delta=delta
        )
        
        return epsilon

def create_heterogeneous_svhn_datasets(
    alpha: float = 1.0,
    num_clients: int = 100,
    batch_size: int = 32,
    test_batch_size: int = 1024
) -> Tuple[Dict[str, Dataset], DataLoader]:
    """Create heterogeneous SVHN datasets using Dirichlet distribution.
    
    Args:
        alpha: Concentration parameter for Dirichlet distribution (smaller alpha means more heterogeneous)
        num_clients: Number of clients
        batch_size: Batch size for training
        test_batch_size: Batch size for testing
    
    Returns:
        Tuple of (client_datasets, test_loader)
    """
    # Data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
    ])
    
    # Load training data
    train_dataset = torchvision.datasets.SVHN(
        root='./data',
        split='train',
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.SVHN(
        root='./data',
        split='test',
        download=True,
        transform=transform
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    
    # Get labels and create indices
    labels = train_dataset.labels
    indices = np.arange(len(labels))
    
    # Create client datasets using Dirichlet distribution
    client_data = {}
    min_size = 0
    K = 10  # Number of classes
    
    while min_size < 10:  # Ensure each client has at least 10 samples
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = indices[labels == k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < len(train_dataset) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    # Create client datasets
    for i in range(num_clients):
        client_indices = idx_batch[i]
        client_dataset = torch.utils.data.Subset(train_dataset, client_indices)
        client_data[f'client_{i}'] = client_dataset
        logging.info(f'Client {i} dataset size: {len(client_indices)}')
        
        # Verify dataset format
        sample_data, sample_label = client_dataset[0]
        logging.info(f'Client {i} sample shape: {sample_data.shape}, label: {sample_label}')
    
    return client_data, test_loader

def create_heterogeneous_cifar10_datasets(
    alpha: float = 1.0,
    num_clients: int = 100,
    batch_size: int = 32,
    test_batch_size: int = 1024,
    transform=None
) -> Tuple[Dict[str, Dataset], DataLoader]:
    """Create heterogeneous CIFAR-10 datasets using Dirichlet distribution.
    
    Args:
        alpha: Concentration parameter for Dirichlet distribution (smaller alpha means more heterogeneous)
        num_clients: Number of clients
        batch_size: Batch size for training
        test_batch_size: Batch size for testing
        transform: Data transformation pipeline
    
    Returns:
        Tuple of (client_datasets, test_loader)
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    # Load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False
    )
    
    # Get labels and create indices
    labels = np.array(train_dataset.targets)
    indices = np.arange(len(labels))
    
    # Create client datasets using Dirichlet distribution
    client_data = {}
    min_size = 0
    K = 10  # Number of classes
    
    while min_size < 10:  # Ensure each client has at least 10 samples
        idx_batch = [[] for _ in range(num_clients)]
        for k in range(K):
            idx_k = indices[labels == k]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < len(train_dataset) / num_clients) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
    
    # Create client datasets
    for i in range(num_clients):
        client_indices = idx_batch[i]
        client_dataset = Subset(train_dataset, client_indices)
        client_data[f'client_{i}'] = client_dataset
        logging.info(f'Client {i} dataset size: {len(client_indices)}')
    
    return client_data, test_loader