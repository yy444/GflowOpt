"""训练辅助工具

提供在训练过程中使用的各种辅助功能：
1. 移动平均损失计算
2. 最佳模型保存和恢复
3. 训练进度日志记录
"""

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque


class LossTracker:
    """跟踪和计算训练过程中的损失移动平均
    
    参数:
        window_size: 移动平均窗口大小
        smooth_factor: 平滑因子(0到1之间)，值越大平滑效果越强
    """
    def __init__(self, window_size: int = 10, smooth_factor: float = 0.9):
        self.window_size = window_size
        self.smooth_factor = smooth_factor
        self.losses = deque(maxlen=window_size)
        self.best_loss = float('inf')
        self.best_smoothed_loss = float('inf')
        self.best_epoch = 0
        self.current_epoch = 0
        self.smoothed_loss = None
    
    def update(self, loss: float, epoch: Optional[int] = None) -> Dict[str, float]:
        """更新损失跟踪器
        
        参数:
            loss: 当前批次/轮次的损失值
            epoch: 当前轮次(可选)
            
        返回:
            包含当前损失统计的字典:
                - loss: 当前损失
                - avg_loss: 移动平均损失
                - best_loss: 最佳损失
                - is_best: 是否是最佳损失
        """
        self.losses.append(loss)
        if epoch is not None:
            self.current_epoch = epoch
        
        # 计算简单移动平均
        avg_loss = sum(self.losses) / len(self.losses)
        
        # 计算指数移动平均(平滑处理)
        if self.smoothed_loss is None:
            self.smoothed_loss = loss
        else:
            self.smoothed_loss = self.smooth_factor * self.smoothed_loss + (1 - self.smooth_factor) * loss
        
        # 判断是否为最佳损失
        is_best = False
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_epoch = self.current_epoch
            is_best = True
        
        # 判断是否为最佳平滑损失
        if self.smoothed_loss < self.best_smoothed_loss:
            self.best_smoothed_loss = self.smoothed_loss
        
        return {
            'loss': loss,
            'avg_loss': avg_loss,
            'smooth_loss': self.smoothed_loss,
            'best_loss': self.best_loss,
            'is_best': is_best
        }


class CheckpointManager:
    """管理模型检查点的保存和加载
    
    参数:
        output_dir: 保存检查点的目录
        max_to_keep: 最多保留的检查点数量
    """
    def __init__(self, output_dir: str, max_to_keep: int = 3):
        self.output_dir = output_dir
        self.max_to_keep = max_to_keep
        self.checkpoints = []  # [(step, filename, metric), ...]
        os.makedirs(output_dir, exist_ok=True)
    
    def save_checkpoint(self, step: int, params: Any, metric: float, prefix: str = 'checkpoint') -> str:
        """保存检查点
        
        参数:
            step: 当前步数/轮次
            params: 要保存的参数
            metric: 相关指标(通常是验证损失)
            prefix: 检查点文件名前缀
            
        返回:
            保存的检查点文件路径
        """
        filename = f"{prefix}_{step:06d}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存参数
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        # 更新检查点列表
        self.checkpoints.append((step, filename, metric))
        
        # 按指标值排序(升序)
        self.checkpoints.sort(key=lambda x: x[2])
        
        # 如果超过最大保留数量，删除最差的检查点
        if len(self.checkpoints) > self.max_to_keep:
            _, worst_filename, _ = self.checkpoints.pop()
            worst_filepath = os.path.join(self.output_dir, worst_filename)
            if os.path.exists(worst_filepath):
                os.remove(worst_filepath)
        
        return filepath
    
    def save_best(self, step: int, params: Any, metric: float, filename: str = 'best_model.pkl') -> str:
        """保存最佳模型
        
        参数:
            step: 当前步数/轮次
            params: 要保存的参数
            metric: 相关指标(通常是验证损失)
            filename: 保存文件名
            
        返回:
            保存的文件路径
        """
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存参数
        with open(filepath, 'wb') as f:
            pickle.dump(params, f)
        
        print(f"保存最佳模型，步数: {step}, 指标: {metric:.6f}, 路径: {filepath}")
        
        return filepath
    
    def load_checkpoint(self, filename: str) -> Any:
        """加载检查点
        
        参数:
            filename: 检查点文件名
            
        返回:
            加载的参数
        """
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        
        return params
    
    def load_best(self, filename: str = 'best_model.pkl') -> Any:
        """加载最佳模型
        
        参数:
            filename: 模型文件名
            
        返回:
            加载的参数
        """
        return self.load_checkpoint(filename)
    
    def get_best_checkpoint(self) -> Tuple[int, str, float]:
        """获取最佳检查点信息
        
        返回:
            (步数, 文件名, 指标值)
        """
        if not self.checkpoints:
            return None
        
        return self.checkpoints[0]  # 因为按指标值排序(升序)，第一个是最好的


def exponential_moving_average(params, new_params, decay: float = 0.999):
    """计算参数的指数移动平均
    
    参数:
        params: 当前EMA参数
        new_params: 新的模型参数
        decay: EMA衰减率(0到1之间)
        
    返回:
        更新后的EMA参数
    """
    return jax.tree_map(lambda p, np: p * decay + np * (1 - decay), params, new_params)


def create_training_state(model, rng_key, config):
    """创建初始训练状态
    
    参数:
        model: 模型
        rng_key: 随机种子
        config: 训练配置
        
    返回:
        初始化的训练状态
    """
    # 示例：如何结构化训练状态
    # 具体实现取决于模型和训练逻辑
    loss_tracker = LossTracker(window_size=config.get('loss_window_size', 10))
    
    return {
        'step': 0,
        'optimizer': None,  # 需要具体实现
        'params': None,     # 需要具体实现
        'loss_tracker': loss_tracker,
        'rng_key': rng_key
    } 