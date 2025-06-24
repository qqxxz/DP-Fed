# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trains and evaluates EMNIST."""

import argparse
import logging
import os
import torch
import numpy as np
import pandas as pd
from models import create_model
from training_loop import TrainingLoop
from data_utils import create_heterogeneous_cifar10_datasets, get_epsilon
import sys
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Train SVHN with federated learning')
    
    # Training parameters
    parser.add_argument('--clients_per_round', type=int, default=10,
                      help='Number of clients participating in each round')
    parser.add_argument('--client_epochs_per_round', type=int, default=4,
                      help='Number of local epochs for each client')
    parser.add_argument('--client_batch_size', type=int, default=64,
                      help='Batch size for client training')
    parser.add_argument('--total_rounds', type=int, default=40,
                      help='Total number of communication rounds')
    
    # Optimizer parameters
    parser.add_argument('--client_lr', type=float, default=0.01,
                      help='Learning rate for client training')
    
    # Differential privacy parameters
    parser.add_argument('--clip_norm', type=float, default=0.2,
                      help='Clip L2 norm')
    parser.add_argument('--noise_multiplier', type=float, default=0.72,
                      help='Noise multiplier for DP algorithm')
    parser.add_argument('--target_epsilon', type=float, default=16,
                      help='Target privacy budget (epsilon). Training will stop if exceeded.')
    
    # Data heterogeneity parameters
    parser.add_argument('--alpha', type=float, default=100,
                        help='Dirichlet distribution parameter for data heterogeneity')
    
    # Other parameters
    parser.add_argument('--output_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DP-FedAvg_output_epi16_alpha100'),
                      help='Directory for writing experiment output')
    parser.add_argument('--seed', type=int, default=123,
                      help='Random seed')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f'Output directory: {args.output_dir}')
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'training.log')),
            logging.StreamHandler(stream=sys.stdout)
        ]
    )
    
    # Configure handlers
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)
            handler.flush = lambda: None
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 数据集的均值和标准差
    ])

    client_data, test_loader = create_heterogeneous_cifar10_datasets(
        alpha=args.alpha,
        num_clients=args.clients_per_round,
        batch_size=args.client_batch_size,
        transform=transform
    )
    
    # Calculate initial privacy budget
    initial_epsilon = get_epsilon(
        num_examples=len(client_data['client_0']),  # 使用第一个客户端的数据集大小
        batch_size=args.client_batch_size,
        noise_multiplier=args.noise_multiplier,
        epochs=args.client_epochs_per_round
    )
    logging.info(f'Initial privacy budget (epsilon): {initial_epsilon:.2f}')
    
    if args.target_epsilon is not None:
        logging.info(f'Target privacy budget (epsilon): {args.target_epsilon:.2f}')
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model().to(device)
    
    # Create training loop
    training_loop = TrainingLoop(
        model=model,
        train_data=client_data,
        test_data=test_loader,
        device=device,
        clients_per_round=args.clients_per_round,
        client_epochs_per_round=args.client_epochs_per_round,
        client_batch_size=args.client_batch_size,
        clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,
        total_rounds=args.total_rounds,
        target_epsilon=args.target_epsilon,
        client_lr=args.client_lr
    )
    
    # Train model
    test_accuracies, actual_rounds, final_epsilon, metrics_df = training_loop.train()
    
    # Save results
    np.save(os.path.join(args.output_dir, 'test_accuracies.npy'), test_accuracies)
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))
    
    # Save metrics to Excel
    metrics_df.to_excel(os.path.join(args.output_dir, 'training_metrics.xlsx'), index=False)
    
    # Save experiment configuration
    config = vars(args)
    config['initial_epsilon'] = initial_epsilon
    config['final_epsilon'] = final_epsilon
    config['actual_rounds'] = actual_rounds
    np.save(os.path.join(args.output_dir, 'config.npy'), config)
    
    logging.info(f'Training completed after {actual_rounds} rounds')
    logging.info(f'Final privacy budget (epsilon): {final_epsilon:.2f}')
    logging.info(f'Final test accuracy: {test_accuracies[-1]:.4f}')

    # Check data distribution
    for client_id, dataset in client_data.items():
        labels = [label for _, label in dataset]
        unique_labels, counts = np.unique(labels, return_counts=True)
        logging.info(f"Client {client_id} class distribution: {dict(zip(unique_labels, counts))}")

if __name__ == '__main__':
    main()
