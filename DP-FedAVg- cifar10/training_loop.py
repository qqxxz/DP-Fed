import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
from data_utils import get_epsilon
import copy

class ClientIDShuffler:
    """Shuffles client IDs for training."""
    def __init__(self, clients_per_round: int, client_ids: List[str]):
        self.clients_per_round = clients_per_round
        self.client_ids = client_ids
        self.current_epoch = 0
        self.current_index = 0
    
    def get_client_ids(self, round_idx: int) -> List[str]:
        """Get client IDs for the current round."""
        if self.current_index + self.clients_per_round > len(self.client_ids):
            # 重新打乱客户端ID
            np.random.shuffle(self.client_ids)
            self.current_index = 0
            self.current_epoch += 1
        
        # 获取当前轮的客户端
        selected_clients = self.client_ids[self.current_index:self.current_index + self.clients_per_round]
        self.current_index += self.clients_per_round
        return selected_clients

class TrainingLoop:
    """Federated learning training loop."""
    def __init__(
        self,
        model: nn.Module,
        train_data: Dict[str, torch.utils.data.Dataset],
        test_data: torch.utils.data.DataLoader,
        device: torch.device,
        clients_per_round: int,
        client_epochs_per_round: int,
        client_batch_size: int,
        clip_norm: float,
        noise_multiplier: float,
        total_rounds: int,
        target_epsilon: Optional[float] = None,
        client_lr: float = 0.01
    ):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.clients_per_round = clients_per_round
        self.client_epochs_per_round = client_epochs_per_round
        self.client_batch_size = client_batch_size
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.total_rounds = total_rounds
        self.target_epsilon = target_epsilon
        self.client_lr = client_lr
        self.client_ids = list(train_data.keys())
        self.client_shuffler = ClientIDShuffler(clients_per_round, self.client_ids)

    def train_round(self, round_idx: int) -> Dict[str, float]:
        """Train one round of federated learning.
        
        Args:
            round_idx: Current round index
            
        Returns:
            Dictionary containing metrics for this round
        """
        # Get clients for this round
        selected_clients = self.client_shuffler.get_client_ids(round_idx)
        logging.info(f"Round {round_idx + 1}: Selected clients: {selected_clients}")
        
        # Train each client
        client_metrics = []
        client_updates = []
        total_samples = 0
        
        for client_id in selected_clients:
            try:
                client_result = self.train_client(client_id)
                client_metrics.append(client_result)
                client_updates.append(client_result)
                total_samples += client_result['samples']
            except Exception as e:
                logging.error(f"Error training client {client_id}: {str(e)}")
                continue
        
        if not client_metrics:
            logging.error("No clients were successfully trained in this round")
            return {
                'train_loss': float('nan'),
                'train_accuracy': float('nan'),
                'test_loss': float('nan'),
                'test_accuracy': float('nan')
            }
        
        # Aggregate client updates with sample weights
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                weighted_updates = []
                for update in client_updates:
                    weight = update['samples'] / total_samples
                    weighted_updates.append(update['updates'][name] * weight)
                avg_update = torch.stack(weighted_updates).sum(0)
                param.data.add_(avg_update)
        
        # Aggregate metrics
        round_metrics = {
            'train_loss': np.mean([m['train_loss'] for m in client_metrics]),
            'train_accuracy': np.mean([m['train_accuracy'] for m in client_metrics]),
            'test_loss': np.mean([m['test_loss'] for m in client_metrics]),
            'test_accuracy': np.mean([m['test_accuracy'] for m in client_metrics])
        }
        
        # Log round metrics
        logging.info(f"Round {round_idx + 1} metrics:")
        logging.info(f"  Train Loss: {round_metrics['train_loss']:.4f}")
        logging.info(f"  Train Accuracy: {round_metrics['train_accuracy']:.4f}")
        logging.info(f"  Test Loss: {round_metrics['test_loss']:.4f}")
        logging.info(f"  Test Accuracy: {round_metrics['test_accuracy']:.4f}")
        
        return round_metrics

    def train_client(self, client_id: str) -> Dict[str, float]:
        """Train a single client.
        
        Args:
            client_id: ID of the client to train
            
        Returns:
            Dictionary containing training metrics and model updates
        """
        try:
            # Get client data
            client_data = self.train_data[client_id]
            num_samples = len(client_data)  # Get number of samples
            
            client_loader = DataLoader(
                client_data,
                batch_size=self.client_batch_size,
                shuffle=True,
                num_workers=0  # Disable multiprocessing to avoid potential issues
            )
            
            # Create client model
            client_model = copy.deepcopy(self.model)
            client_model.train()
            
            # Setup optimizer
            optimizer = optim.SGD(client_model.parameters(), lr=self.client_lr)
            
            # Train client model
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for epoch in range(self.client_epochs_per_round):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total = 0
                
                for batch_idx, (data, target) in enumerate(client_loader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)
                        
                        optimizer.zero_grad()
                        output = client_model(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()
                        
                        # Clip gradients
                        torch.nn.utils.clip_grad_norm_(client_model.parameters(), self.clip_norm)
                        
                        # Add noise to gradients
                        for param in client_model.parameters():
                            if param.grad is not None:
                                noise = torch.randn_like(param.grad) * self.noise_multiplier * self.clip_norm
                                param.grad.add_(noise)
                        
                        optimizer.step()
                        
                        # Update metrics
                        epoch_loss += loss.item()
                        _, predicted = output.max(1)
                        epoch_total += target.size(0)
                        epoch_correct += predicted.eq(target).sum().item()
                        
                        # Log batch metrics for debugging
                        if batch_idx % 10 == 0:
                            batch_acc = predicted.eq(target).sum().item() / target.size(0)
                            logging.debug(f"Client {client_id} - Batch {batch_idx} - Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")
                        
                    except Exception as e:
                        logging.error(f"Error in batch {batch_idx} for client {client_id}: {str(e)}")
                        continue
                
                # Calculate epoch metrics
                epoch_loss = epoch_loss / len(client_loader)
                epoch_acc = epoch_correct / epoch_total
                logging.debug(f"Client {client_id} - Epoch {epoch} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
                
                train_loss += epoch_loss
                train_correct += epoch_correct
                train_total += epoch_total
            
            # Calculate average metrics
            train_loss /= self.client_epochs_per_round
            train_accuracy = train_correct / train_total
            
            # Evaluate on test set
            test_loss, test_accuracy = self.evaluate(client_model)
            
            # Calculate model updates
            updates = {}
            with torch.no_grad():
                for name, param in client_model.named_parameters():
                    updates[name] = param.data - self.model.state_dict()[name]
            
            # Log detailed metrics
            logging.info(f"Client {client_id} completed - Loss: {train_loss:.4f}, Acc: {train_accuracy:.4f}, "
                        f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}, Samples: {num_samples}")
            
            return {
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'updates': updates,
                'samples': num_samples
            }
            
        except Exception as e:
            logging.error(f"Error training client {client_id}: {str(e)}")
            raise

    def evaluate(self, model: nn.Module) -> Tuple[float, float]:
        """Evaluate model on test set.
        
        Args:
            model: Model to evaluate
            
        Returns:
            Tuple of (test_loss, test_accuracy)
        """
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                test_loss += F.cross_entropy(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Log batch metrics for debugging
                batch_acc = predicted.eq(target).sum().item() / target.size(0)
                logging.debug(f"Evaluation batch - Loss: {test_loss:.4f}, Acc: {batch_acc:.4f}")
        
        test_loss /= len(self.test_data)
        test_accuracy = correct / total
        
        logging.debug(f"Evaluation complete - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}")
        return test_loss, test_accuracy

    def train(self) -> Tuple[List[float], int, float, pd.DataFrame]:
        """训练联邦学习模型"""
        test_accuracies = []
        metrics = {
            'round': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epsilon': []
        }
        
        # 初始化累计隐私预算
        total_epsilon = 0
        
        print("\n" + "="*50)
        print("Starting Federated Learning Training")
        print("="*50 + "\n")
        
        for round_idx in tqdm(range(self.total_rounds), desc="Training Rounds"):
            print(f"\nRound {round_idx + 1}/{self.total_rounds}")
            print("-"*30)
            
            # 选择客户端
            selected_clients = self.client_shuffler.get_client_ids(round_idx)
            print(f"Selected clients: {selected_clients}")
            
            # 训练选中的客户端
            client_updates = []
            round_train_loss = 0
            round_train_acc = 0
            total_samples = 0
            
            print(f"Starting training for {len(selected_clients)} clients...")
            for client_idx, client_id in enumerate(selected_clients):
                try:
                    print(f"\nTraining client {client_id} ({client_idx + 1}/{len(selected_clients)})")
                    # 为每个客户端创建新的模型实例并移动到正确的设备
                    client_model = type(self.model)().to(self.device)
                    client_model.load_state_dict({k: v.to(self.device) for k, v in self.model.state_dict().items()})
                    
                    # 训练客户端
                    updates = self.train_client(client_id)
                    
                    # 确保更新也在正确的设备上
                    updates['updates'] = {k: v.to(self.device) for k, v in updates['updates'].items()}
                    client_updates.append(updates)
                    round_train_loss += updates['train_loss'] * updates['samples']
                    round_train_acc += updates['train_accuracy'] * updates['samples']
                    total_samples += updates['samples']
                    
                    print(f"Client {client_id} completed - Loss: {updates['train_loss']:.4f}, Acc: {updates['train_accuracy']:.4f}")
                    
                    # 清理内存
                    del client_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    print(f"Error training client {client_id}: {str(e)}")
                    continue
            
            if not client_updates:
                print(f"No successful client updates in round {round_idx + 1}")
                continue
                
            print("\nComputing weighted average of client updates...")
            
            # 计算加权平均客户端更新
            avg_updates = {}
            for k in client_updates[0]['updates']:
                weighted_updates = torch.stack([
                    update['updates'][k] * update['samples'] 
                    for update in client_updates
                ])
                avg_updates[k] = weighted_updates.sum(0) / total_samples
            
            # 更新服务器模型
            with torch.no_grad():
                for k, v in self.model.state_dict().items():
                    v = v.to(self.device)
                    avg_updates[k] = avg_updates[k].to(self.device)
                    v += avg_updates[k]
                    if not torch.cuda.is_available():
                        v = v.cpu()
            
            # 确保模型在正确的设备上
            self.model = self.model.to(self.device)
            
            # 评估模型
            test_loss, test_acc = self.evaluate(self.model)
            test_accuracies.append(test_acc)
            
            # 计算当前轮的隐私预算
            round_epsilon = get_epsilon(
                num_examples=sum(len(self.train_data[client_id]) for client_id in selected_clients),
                batch_size=self.client_batch_size,
                noise_multiplier=self.noise_multiplier,
                epochs=self.client_epochs_per_round,
                delta=1e-2
            )
            
            # 累计隐私预算
            total_epsilon += round_epsilon
            
            # 记录指标
            metrics['round'].append(round_idx + 1)
            metrics['train_loss'].append(round_train_loss / total_samples)
            metrics['train_acc'].append(round_train_acc / total_samples)
            metrics['test_loss'].append(test_loss)
            metrics['test_acc'].append(test_acc)
            metrics['epsilon'].append(total_epsilon)
            
            # 输出每轮结果
            print("\n" + "="*50)
            print(f"Round {round_idx + 1}/{self.total_rounds} Summary:")
            print("-"*50)
            print(f"Training Metrics:")
            print(f"  Loss: {metrics['train_loss'][-1]:.4f}")
            print(f"  Accuracy: {metrics['train_acc'][-1]:.4f}")
            print(f"Test Metrics:")
            print(f"  Loss: {metrics['test_loss'][-1]:.4f}")
            print(f"  Accuracy: {metrics['test_acc'][-1]:.4f}")
            print(f"Privacy Budget:")
            print(f"  Current Round Epsilon: {round_epsilon:.4f}")
            print(f"  Total Epsilon: {total_epsilon:.4f}")
            if self.target_epsilon is not None:
                print(f"  Remaining Budget: {self.target_epsilon - total_epsilon:.4f}")
            print("="*50 + "\n")
            
            # 检查是否达到目标隐私预算
            if self.target_epsilon is not None and total_epsilon > self.target_epsilon:
                print(f"Reached target privacy budget at round {round_idx + 1}")
                break
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(metrics)
        
        return test_accuracies, len(metrics['round']), total_epsilon, metrics_df 