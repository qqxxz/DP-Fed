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
        
print(get_epsilon(50000, 64, 0.71, 4, 1e-2))