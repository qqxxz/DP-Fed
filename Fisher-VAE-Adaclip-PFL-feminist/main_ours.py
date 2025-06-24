import os
import torch
from torch.utils.data import DataLoader
from options import parse_args
from data import *
from net import get_model
from tqdm import tqdm
from tqdm.auto import trange, tqdm
import sys
import random
from FederatedClient import FederatedClient
from VAEManager import VAEManager
import torch.nn as nn
import torch.optim as optim
import copy
import tensorflow as tf
import numpy as np
from tensorflow_privacy.privacy.dp_query.quantile_adaptive_clip_sum_query import QuantileAdaptiveClipSumQuery
from tensorflow_privacy.privacy.dp_query.normalized_query import NormalizedQuery
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy_lib
from typing import Any, NamedTuple, Optional
from result_logger import ResultLogger  # 导入 ResultLogger 类


args = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
num_clients = args.num_clients
local_epoch = args.local_epoch
global_epoch = args.global_epoch
batch_size = args.batch_size
target_epsilon = args.target_epsilon
target_delta = args.target_delta
clipping_bound = args.clipping_bound
dataset = 'FEMNIST'  # 修改为 FEMNIST 数据集
user_sample_rate = args.user_sample_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAE = args.VAE
VAE_d = args.VAE_d
VAE_z = args.VAE_z
model = args.model
model_output_dim = args.model_output_dim
class_num = 62  # FEMNIST 有 62 个类别
VAE_client_num = args.VAE_client_num
VAE_comm_round = args.VAE_comm_round
client_select = args.client_select
VAE_local_epoch = args.VAE_local_epoch
VAE_batch_size = args.VAE_batch_size
VAE_re = args.VAE_re
VAE_kl = args.VAE_kl
VAE_ce = args.VAE_ce
VAE_x_ce = args.VAE_x_ce
logger = ResultLogger(output_dir="results", log_file="ours_results.log")
cumulative_epsilon = 0

# 设置更新缩放因子，防止梯度爆炸
UPDATE_SCALING_FACTOR = 1 / (user_sample_rate * num_clients)

if args.store == True:
    saved_stdout = sys.stdout
    file = open(
        f'./txt/{args.dirStr}/'
        f'dataset {args.dataset} '
        f'--num_clients {args.num_clients} '
        f'--local_epoch {args.local_epoch} '
        f'--global_epoch {args.global_epoch} '
        f'--batch_size {args.batch_size} '
        f'--target_epsilon {args.target_epsilon} '
        f'--target_delta {args.target_delta} '
        f'--clipping_bound {args.clipping_bound} '
        f'--fisher_threshold {args.fisher_threshold} '
        f'--lambda_1 {args.lambda_1} '
        f'--lambda_2 {args.lambda_2} '
        f'--lr {args.lr} '
        f'--alpha {args.alpha} '
        f'--VAE {args.VAE} '
        f'--VAE_d {args.VAE_d} '
        f'--VAE_z {args.VAE_z} '
        f'--model {args.model} '
        f'--model_output_dim {args.model_output_dim} '
        f'--class_num {args.class_num} '
        f'--VAE_client_num {args.VAE_client_num} '
        f'--VAE_comm_round {args.VAE_comm_round} '
        f'--client_select {args.client_select} '
        f'--VAE_local_epoch {args.VAE_local_epoch} '
        f'--VAE_batch_size {args.VAE_batch_size} '
        f'--VAE_re {args.VAE_re} '
        f'--VAE_kl {args.VAE_kl} '
        f'--VAE_ce {args.VAE_ce} '
        f'--VAE_x_ce {args.VAE_x_ce} '
        '.txt', 'a'
    )
    sys.stdout = file

def get_epsilon(
        num_examples,
        batch_size,
        noise_multiplier,
        epochs,
        delta=1e-2
    ) -> float:
        """返回当前训练设置下的 epsilon 值，使用 Opacus RDP 会计"""
        try:
            from opacus.accountants import RDPAccountant
            
            # 计算采样概率和步数
            sampling_probability = batch_size / num_examples
            steps = int(epochs * num_examples / batch_size)
            
            # 初始化 RDP 会计
            accountant = RDPAccountant()
            
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
def tf_tensor_to_torch(tensors):
    if isinstance(tensors, list):
        return [torch.tensor(t.numpy()) for t in tensors]
    else:
        return torch.tensor(tensors.numpy())

def torch_tensor_to_tf(tensors):
    if isinstance(tensors, list):
        return [tf.convert_to_tensor(t.detach().cpu().numpy(), dtype=tf.float32) for t in tensors]
    else:
        return tf.convert_to_tensor(tensors.detach().cpu().numpy(), dtype=tf.float32)

def adaptive_clip_noise_params(
    noise_multiplier: float,
    expected_clients_per_round: float,
    clipped_count_stddev: Optional[float] = None,
) -> tuple[float, float]:
  if noise_multiplier > 0.0:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.05 * expected_clients_per_round

    if noise_multiplier >= 2 * clipped_count_stddev:
      raise ValueError(
          f'clipped_count_stddev = {clipped_count_stddev} (defaults to '
          '0.05 * `expected_clients_per_round` if not specified) is too low '
          'to achieve the desired effective `noise_multiplier` '
          f'({noise_multiplier}). You must either increase '
          '`clipped_count_stddev` or decrease `noise_multiplier`.'
      )

    value_noise_multiplier = (
        noise_multiplier**-2 - (2 * clipped_count_stddev) ** -2
    ) ** -0.5

    added_noise_factor = value_noise_multiplier / noise_multiplier
    if added_noise_factor >= 2:
      warnings.warn(
          f'A significant amount of noise ({added_noise_factor:.2f}x) has to '
          'be added for record aggregation to achieve the desired effective '
          f'`noise_multiplier` ({noise_multiplier}). If you are manually '
          'specifying `clipped_count_stddev` you may want to increase it. Or '
          'you may need more `expected_clients_per_round`.'
      )
  else:
    if clipped_count_stddev is None:
      clipped_count_stddev = 0.0
    value_noise_multiplier = 0.0

  return value_noise_multiplier, clipped_count_stddev

def main():
    mean_acc_s = []
    acc_matrix = []

    if dataset == 'FEMNIST':  # 加载 FEMNIST 数据集
        clients_train_loaders, clients_test_loaders, client_data_sizes = get_FEMNIST(num_clients)
        num_classes = 62
        model_input_channels = 1
    else:
        print('Undefined dataset')
        assert 1 == 0

    # 统一初始化模型
    clients_models = [get_model(args, num_classes=num_classes, model_input_channels=model_input_channels) for _ in range(num_clients)]
    global_model = get_model(args, num_classes=num_classes, model_input_channels=model_input_channels)

    for client_model in clients_models:
        client_model.load_state_dict(global_model.state_dict())

    # ----------------- Step 1: VAEManager训练 & 共享数据生成 -----------------
    vae_manager = VAEManager(
        args=args,
        train_dataloader=clients_train_loaders,
        test_dataloader=clients_test_loaders,
        device=device
    )
    client_vae_list = vae_manager.setup_vae_client_list()
    vae_manager.setup_vae_server()
    global_share_dataset1, global_share_dataset2, global_share_data_y = vae_manager.share_data_step()

    # ----------------- Step 2: 联邦训练主循环 -----------------
    for epoch in trange(global_epoch):
        sampled_client_indices = random.sample(range(num_clients), max(1, int(user_sample_rate * num_clients)))
        sampled_clients_models = [clients_models[i] for i in sampled_client_indices]
        sampled_clients_train_loaders = [clients_train_loaders[i] for i in sampled_client_indices]
        sampled_clients_test_loaders = [clients_test_loaders[i] for i in sampled_client_indices]
        clients = []
        client_indices = []
        for local_idx, client_idx in enumerate(sampled_client_indices):
            client = FederatedClient(
                model=sampled_clients_models[local_idx],
                train_loader=sampled_clients_train_loaders[local_idx],
                test_loader=sampled_clients_test_loaders[local_idx],
                device=device,
                global_model=global_model,
                args=args
            )
            clients.append(client)
            client_indices.append(client_idx)
        clients_model_updates = []
        clients_accuracies = []

        for idx, client in enumerate(clients):
            if not args.store:
                tqdm.write(f'[Epoch {epoch}] Sampled global client indices: {client_indices}')
                tqdm.write(f'client:{idx+1}/{len(clients)}')
            global_client_idx = client_indices[idx]
            client_vae = client_vae_list[global_client_idx]
            vae_local_train_mixed_dataloader = client_vae.construct_mix_dataloader(global_share_dataset1,
                                                                                   global_share_dataset2,
                                                                                   global_share_data_y)
            update = client.local_update(vae_local_train_mixed_dataloader)
            if update is not None:
                clients_model_updates.append(update)
            else:
                print(f"客户端 {idx} 的更新为空，跳过。")
            accuracy = client.test()
            clients_accuracies.append(accuracy)

        print(clients_accuracies)
        mean_acc_s.append(sum(clients_accuracies) / len(clients_accuracies))
        acc_matrix.append(clients_accuracies)

        ####------- Server Aggregator-------#####
        sample_client_num=len(sampled_client_indices)
        N = sum([client_data_sizes[i] for i in sampled_client_indices])
        clip = args.clipping_bound if epoch == 0 else new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()
        if epoch==0:##-------如果是第一轮聚合-------##
            # 初始化自适应裁剪查询
            value_noise_multiplier, clipped_count_stddev = adaptive_clip_noise_params(
                noise_multiplier=args.noise_multiplier,
                expected_clients_per_round=sample_client_num,
                clipped_count_stddev=None)
            query = QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=clip,
                noise_multiplier=value_noise_multiplier,
                target_unclipped_quantile=args.target_unclipped_quantile,
                learning_rate=args.adaptive_clip_learning_rate,
                clipped_count_stddev=clipped_count_stddev,
                expected_num_records=sample_client_num,
                geometric_update=True)
            # 用NormalizedQuery包裹，实现平均值聚合
            query = NormalizedQuery(query, denominator=sample_client_num)
            # 初始化全局状态（第一次聚合时）
            global_state = query.initial_global_state()
            sample_params = query.derive_sample_params(global_state) # 获取采样参数
            # 初始化样本状态（累加器）
            example_update = torch_tensor_to_tf(clients_model_updates[0])  # 获取一个样例更新
            sample_state = query.initial_sample_state(example_update)  # 使用实际结构生成状态
            tf_clients_model_updates = []
            for update in clients_model_updates:
                tf_update = torch_tensor_to_tf(update)  # update 是 List[Tensor]
                tf_clients_model_updates.append(tf_update)
            # 预处理并累加所有客户端更新
            for update in tf_clients_model_updates:
                record = query.preprocess_record(sample_params, update)
                sample_state = query.accumulate_preprocessed_record(sample_state, record)
            # 聚合并加噪输出
            result, new_global_state, _ = query.get_noised_result(sample_state, global_state)
            clip = new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()

            # 检查是否超过最大裁剪阈值
            if args.max_clip_threshold > 0 and clip > args.max_clip_threshold:
                logger.log_result(f"裁剪阈值 {clip:.4f} 超过最大限制，将其限制为 {args.max_clip_threshold}")
                # 仅更新clip变量，不尝试修改state
                clip = args.max_clip_threshold

            print("\n更新后的裁剪范数:", clip)
            pytorch_update = [tf_tensor_to_torch(t) for t in result]
            pytorch_update = [update.to(device) for update in pytorch_update]  # 确保在同一设备
                
            # 添加更新缩放因子，防止梯度爆炸
            pytorch_update = [update * UPDATE_SCALING_FACTOR for update in pytorch_update]
                

            # Apply the aggregated updates to the global model parameters
            with torch.no_grad():
                for global_param, update in zip(global_model.parameters(), pytorch_update):
                    global_param.add_(update)
            
            epsilon = get_epsilon(
                num_examples=N,
                batch_size=batch_size,
                noise_multiplier=value_noise_multiplier,
                epochs=args.local_epoch,
                delta=args.target_delta
            )
            print(f"第{epoch}轮累计使用到的隐私预算为 ε ≈ {epsilon:.4f}")
            if epsilon >= args.target_epsilon:
                print(f"已达到隐私预算 ε={epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break
            

        else:##-------第二轮及之后的聚合-------##
            # 初始化自适应裁剪查询
            query = QuantileAdaptiveClipSumQuery(
                initial_l2_norm_clip=clip,
                noise_multiplier=value_noise_multiplier,
                target_unclipped_quantile=args.target_unclipped_quantile,
                learning_rate=args.adaptive_clip_learning_rate,
                clipped_count_stddev=clipped_count_stddev,
                expected_num_records=sample_client_num,
                geometric_update=True)
            # 用NormalizedQuery包裹，实现平均值聚合
            query = NormalizedQuery(query, denominator=sample_client_num)
            # 初始化全局状态（第一次聚合时）
            global_state = query.initial_global_state()
            sample_params = query.derive_sample_params(global_state) # 获取采样参数
            # 初始化样本状态（累加器）
            example_update = torch_tensor_to_tf(clients_model_updates[0])  # 获取一个样例更新
            sample_state = query.initial_sample_state(example_update)  # 使用实际结构生成状态
            tf_clients_model_updates = []
            for update in clients_model_updates:
                tf_update = torch_tensor_to_tf(update)  # update 是 List[Tensor]
                tf_clients_model_updates.append(tf_update)
            # 预处理并累加所有客户端更新
            for update in tf_clients_model_updates:
                record = query.preprocess_record(sample_params, update)
                sample_state = query.accumulate_preprocessed_record(sample_state, record)
            # 聚合并加噪输出
            result, new_global_state, _ = query.get_noised_result(sample_state, global_state)
            clip = new_global_state.numerator_state.sum_state.l2_norm_clip.numpy()
            print("\n更新后的裁剪范数:", clip)

            # 检查是否超过最大裁剪阈值
            if args.max_clip_threshold > 0 and clip > args.max_clip_threshold:
                logger.log_result(f"裁剪阈值 {clip:.4f} 超过最大限制，将其限制为 {args.max_clip_threshold}")
                # 仅更新clip变量，不尝试修改state
                clip = args.max_clip_threshold

            pytorch_update = [tf_tensor_to_torch(t) for t in result]
            pytorch_update = [update.to(device) for update in pytorch_update]  # 确保在同一设备
                
            # 添加更新缩放因子，防止梯度爆炸
            pytorch_update = [update * UPDATE_SCALING_FACTOR for update in pytorch_update]
                
            with torch.no_grad():
                for global_param, update in zip(global_model.parameters(), pytorch_update):
                    global_param.add_(update)
                    
            epsilon = get_epsilon(
                num_examples=N,
                batch_size=batch_size,
                noise_multiplier=value_noise_multiplier,
                epochs=args.local_epoch,
                delta=args.target_delta
            )
            print(f"第{epoch}轮累计使用到的隐私预算为 ε ≈ {epsilon:.4f}")

            
            cumulative_epsilon = accountant.get_epsilon(delta=args.target_delta)
            
            if cumulative_epsilon >= args.target_epsilon:
                print(f"已达到隐私预算 ε={cumulative_epsilon:.4f}，停止训练。")
                torch.save(global_model.state_dict(), f"early_stop_model_e{epoch}.pt")
                break

        # 记录结果到日志文件
        log_content = (
            f"Timestamp: {datetime.datetime.now()}\n"
            f"Dataset: {dataset}\n"
            f"Alpha: {args.alpha}\n"
            f"Epsilon: {epsilon}\n"
            f"Epsilon cumulative: {cumulative_epsilon}\n"
            f"Clipping Bound: {clipping_bound}\n"
            f"Noise Multiplier: {value_noise_multiplier}\n"
            f"Accuracy: {', '.join(f'{acc:.4f}' for acc in clients_accuracies)}\n"
            f"Mean Accuracy: {mean_acc_s[-1]:.4f}\n"
            f"Epoch: {epoch}\n"
        )

        # 记录每轮训练的结果
        logger.log_to_file(content=log_content)


    # 生成任务ID，包含 dataset, dir_alpha, target_epsilon
    ID = f"{dataset}_{args.dir_alpha}_{target_epsilon}"

    # 记录总结信息到日志文件
    summary_log_content = (
        f"===============================================================\n"
        f"task_ID : {ID}\n"
        f"main_yxy\n"
        f"mean accuracy : {mean_acc_s}\n"
        f"acc matrix : {torch.tensor(acc_matrix)}\n"
        f"===============================================================\n"
    )
    logger.log_to_file(content=summary_log_content)

if __name__ == '__main__':
    main()

