"""学习率调度器工具

包含用于动态调整学习率的各种调度器：
1. ReduceLROnPlateau - 当损失不再改善时降低学习率
2. CosineAnnealingLR - 余弦退火学习率调度器
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple, Union

class ReduceLROnPlateau:
    """当损失不再改善时，降低学习率的调度器
    
    参数:
        initial_lr: 初始学习率
        factor: 学习率降低的因子，学习率将乘以这个因子 (0..1)
        patience: 无改善后需要等待多少个轮次才降低学习率
        min_lr: 学习率的下限
        threshold: 认为有改善的阈值
        threshold_mode: 'rel' 或 'abs'，相对或绝对变化
        cooldown: 降低学习率后等待多少轮次再次检查
        verbose: 是否打印学习率变化
    """
    def __init__(
        self,
        initial_lr=1e-3,
        factor=0.1,
        patience=10,
        min_lr=1e-6,
        threshold=1e-4,
        threshold_mode='rel',
        cooldown=0,
        verbose=True
    ):
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.verbose = verbose
        
        # 内部状态
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.waiting = 0
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
        self.is_better = self._init_is_better_fn()
        self.loss_history = deque(maxlen=50)  # 保存最近的损失值，用于平滑
    
    def _init_is_better_fn(self):
        """初始化判断损失是否改善的函数"""
        if self.threshold_mode == 'rel':
            return lambda a, best: a < best * (1 - self.threshold)
        else:  # 'abs'
            return lambda a, best: a < best - self.threshold
    
    def step(self, metrics, optimizer):
        """更新学习率
        
        参数:
            metrics: 当前的评估指标(损失值)
            optimizer: Optax优化器实例
        
        返回:
            更新后的优化器
        """
        # 更新损失历史并计算平滑损失
        self.loss_history.append(metrics)
        current_loss = np.mean(self.loss_history)
        
        # 检查是否冷却结束
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        # 检查损失是否改善
        if self.is_better(current_loss, self.best_loss):
            self.best_loss = current_loss
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # 检查是否超过耐心，需要降低学习率
        if self.num_bad_epochs > self.patience:
            # 计算新的学习率
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            
            # 如果学习率确实改变了
            if self.current_lr - new_lr > 1e-8:
                self.current_lr = new_lr
                
                # 使用新的学习率创建新的优化器
                optimizer = optax.chain(
                    optax.clip_by_global_norm(1.0),  # 可选：梯度裁剪
                    optax.adam(learning_rate=self.current_lr)
                )
                
                if self.verbose:
                    print(f"学习率降低到: {self.current_lr:.6f}")
                
                # 重置冷却计数器和坏轮计数器
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
        
        return optimizer


class CosineAnnealingLR:
    """余弦退火学习率调度器
    
    参数:
        initial_lr: 初始学习率
        T_max: 半周期长度
        eta_min: 最小学习率
        last_epoch: 上次迭代的批次（-1表示从头开始）
        verbose: 是否打印学习率变化
    """
    def __init__(
        self,
        initial_lr=1e-3,
        T_max=1000,
        eta_min=0,
        last_epoch=-1,
        verbose=True
    ):
        self.initial_lr = initial_lr
        self.T_max = T_max
        self.eta_min = eta_min
        self.last_epoch = last_epoch
        self.verbose = verbose
        
        # 内部状态
        self.current_lr = initial_lr
        self._step_count = last_epoch + 1
        self._last_update = 0
    
    def step(self, metrics=None, optimizer=None):
        """更新学习率
        
        参数:
            metrics: 不需要，为了与ReduceLROnPlateau保持一致的接口
            optimizer: Optax优化器实例
        
        返回:
            更新后的优化器
        """
        self._step_count += 1
        
        # 计算新的学习率
        if self._step_count - 1 == 0:
            new_lr = self.initial_lr
        else:
            new_lr = self.eta_min + (self.initial_lr - self.eta_min) * (
                1 + np.cos(np.pi * (self._step_count - 1) / self.T_max)
            ) / 2
        
        # 如果学习率确实改变了
        if abs(self.current_lr - new_lr) > 1e-8:
            self.current_lr = new_lr
            
            # 使用新的学习率创建新的优化器
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),  # 可选：梯度裁剪
                optax.adam(learning_rate=self.current_lr)
            )
            
            # 只有在显著变化时才打印（为了减少日志）
            if self.verbose and (self._step_count - self._last_update > 10 or new_lr < 1e-5):
                print(f"学习率更新为: {self.current_lr:.6f}")
                self._last_update = self._step_count
        
        return optimizer


class DummyLRScheduler:
    """伪学习率调度器，用于保持固定学习率"""
    def __init__(self, initial_lr=1e-3):
        self.current_lr = initial_lr
    
    def step(self, metrics=None, optimizer=None):
        """不做任何改变，直接返回原始优化器"""
        return optimizer


def create_lr_scheduler(
    scheduler_type: str,
    initial_lr: float,
    **kwargs
) -> Union[ReduceLROnPlateau, CosineAnnealingLR, DummyLRScheduler]:
    """创建学习率调度器
    
    参数:
        scheduler_type: 调度器类型，可选 'reduce_on_plateau', 'cosine', 'none'
        initial_lr: 初始学习率
        **kwargs: 传递给调度器的参数
    
    返回:
        学习率调度器实例
    """
    if scheduler_type == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            initial_lr=initial_lr,
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6),
            threshold=kwargs.get('threshold', 1e-3),
            threshold_mode=kwargs.get('threshold_mode', 'rel'),
            cooldown=kwargs.get('cooldown', kwargs.get('patience', 10) // 2),
            verbose=kwargs.get('verbose', True)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            initial_lr=initial_lr,
            T_max=kwargs.get('T_max', 1000),
            eta_min=kwargs.get('min_lr', 0),
            verbose=kwargs.get('verbose', True)
        )
    else:  # 'none' or any other value
        return DummyLRScheduler(initial_lr=initial_lr) 