import os
from datetime import datetime

class ResultLogger:
    def __init__(self, output_dir="results", log_file="results.log"):
        """
        初始化结果记录器
        Args:
            output_dir (str): 保存结果的目录
            log_file (str): 保存结果的日志文件名
        """
        self.output_dir = output_dir
        self.log_file = log_file
        self.log_path = os.path.join(output_dir, log_file)

        # 创建目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def log_result(self, dataset, alpha, epsilon, clipping_bound, noise_multiplier, accuracy, mean_accuracy, epoch, notes=""):
        """
        记录结果到日志文件
        Args:
            dataset (str): 数据集名称
            alpha (float): Alpha 值
            epsilon (float): 消耗的隐私预算
            clipping_bound (float): 裁剪阈值
            noise_multiplier (float): 噪声乘子
            accuracy (list): 每个客户端的准确率
            mean_accuracy (float): 平均准确率
            epoch (int): 当前轮次
            notes (str): 备注信息
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_content = (
            f"Timestamp: {timestamp}, Dataset: {dataset}, Alpha: {alpha}, Epsilon: {epsilon}, "
            f"Clipping Bound: {clipping_bound}, Noise Multiplier: {noise_multiplier}, "
            f"Accuracy: {accuracy}, Mean Accuracy: {mean_accuracy}, Epoch: {epoch}, Notes: {notes}"
        )
        self.log_to_file(content=log_content)

    def log_to_file(self, content=""):
        """
        记录结果到日志文件
        Args:
            content (str): 要记录的内容
        """
        with open(self.log_path, "a") as f:
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {content}\n")