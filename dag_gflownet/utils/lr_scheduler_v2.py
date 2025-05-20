"""学习率调度器工具

包含用于动态调整学习率的各种调度器：
1. ReduceLROnPlateau - 当损失不再改善时降低学习率
2. CosineAnnealingLR - 余弦退火学习率调度器
3. EnhancedReduceLROnPlateau - 增强版的ReduceLROnPlateau，能更好地处理损失抖动
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
    
    def step(self, metrics, optimizer=None):
        """更新学习率
        
        参数:
            metrics: 当前的评估指标(损失值)
            optimizer: Optax优化器实例
        
        返回:
            当前的学习率
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
                
                if self.verbose:
                    print(f"学习率降低到: {self.current_lr:.6f}")
                
                # 重置冷却计数器和坏轮计数器
                self.cooldown_counter = self.cooldown
                self.num_bad_epochs = 0
        
        return self.current_lr


class EnhancedReduceLROnPlateau:
    """增强版的ReduceLROnPlateau，具有更好的抗损失抖动能力
    
    特性:
    1. 使用指数移动平均(EMA)平滑损失值，减少噪声影响
    2. 检测损失稳定性，通过计算变异系数判断损失波动情况
    3. 自动处理无效损失值(NaN/Inf)
    4. 支持根据损失趋势动态调整学习率
    
    参数:
        initial_lr: 初始学习率
        factor: 学习率降低的因子 (0..1)
        patience: 无改善后需要等待多少个轮次才降低学习率
        min_lr: 学习率的下限
        smooth_factor: EMA平滑因子 (0..1)，越大表示历史损失的权重越大
        stability_window: 检测损失稳定性时使用的窗口大小
        stability_threshold: 判断损失稳定的阈值，变异系数低于此值认为稳定
        trend_detection: 是否启用损失趋势检测
        allow_small_fluctuation: 允许的小幅波动比例，相对于最佳损失
        verbose: 是否打印详细日志
    """
    def __init__(
        self,
        initial_lr=1e-3,
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        smooth_factor=0.9,
        stability_window=5,
        stability_threshold=0.15,
        trend_detection=True,
        allow_small_fluctuation=0.02,
        verbose=True
    ):
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.smooth_factor = smooth_factor
        self.stability_window = stability_window
        self.stability_threshold = stability_threshold
        self.trend_detection = trend_detection
        self.allow_small_fluctuation = allow_small_fluctuation
        self.verbose = verbose
        
        # 内部状态
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.best_smooth_loss = float('inf')
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
        self.last_update_epoch = -1
        
        # 损失历史和EMA值
        self.loss_history = deque(maxlen=max(100, stability_window * 3))
        self.ema_loss = None
        
    def _update_ema(self, value):
        """更新指数移动平均"""
        if np.isnan(value) or np.isinf(value) or value <= 0:
            if self.verbose:
                print(f"警告: 跳过无效损失值 {value}")
            return self.ema_loss if self.ema_loss is not None else 0.0
            
        if self.ema_loss is None:
            self.ema_loss = value
        else:
            self.ema_loss = self.smooth_factor * self.ema_loss + (1 - self.smooth_factor) * value
        return self.ema_loss
        
    def is_loss_stable(self):
        """检查最近的损失是否稳定"""
        if len(self.loss_history) < self.stability_window:
            return True  # 数据不足，假定稳定
            
        # 获取最近的损失值进行分析
        recent_losses = list(self.loss_history)[-self.stability_window:]
        
        # 过滤掉可能的无效值
        valid_losses = [loss for loss in recent_losses 
                        if not np.isnan(loss) and not np.isinf(loss) and loss > 0]
        
        if len(valid_losses) < 3:
            return False  # 有效数据不足，认为不稳定
            
        # 计算变异系数 (标准差/均值)
        mean_loss = np.mean(valid_losses)
        std_loss = np.std(valid_losses)
        
        # 避免除零错误
        if mean_loss == 0:
            return False
            
        variation = std_loss / mean_loss
        return variation < self.stability_threshold
    
    def detect_loss_trend(self):
        """检测损失趋势
        
        返回:
            1: 上升趋势 (变差)
            0: 无明显趋势
            -1: 下降趋势 (变好)
        """
        if len(self.loss_history) < 3:
            return 0  # 数据不足，无法判断趋势
            
        # 获取最近的EMA损失值
        if hasattr(self, 'ema_history') and len(self.ema_history) >= 3:
            recent = list(self.ema_history)[-3:]
        else:
            # 如果没有EMA历史，使用原始损失计算趋势
            recent = list(self.loss_history)[-3:]
            
        # 计算差分
        diffs = np.diff(recent)
        
        # 如果两次差分都是正值且显著，认为是上升趋势
        if np.all(diffs > 0) and np.mean(diffs) > 0.01 * recent[0]:
            return 1
            
        # 如果两次差分都是负值且显著，认为是下降趋势
        if np.all(diffs < 0) and np.mean(diffs) < -0.01 * recent[0]:
            return -1
            
        # 否则无明显趋势
        return 0
    
    def step(self, loss, optimizer=None, epoch=None):
        """更新学习率
        
        参数:
            loss: 当前损失值
            optimizer: 优化器(可选)，仅用于兼容性
            epoch: 当前训练轮次(可选)，用于更新策略
            
        返回:
            当前学习率
        """
        # 保存原始损失值
        self.loss_history.append(float(loss))
        
        # 计算平滑损失值 (EMA)
        smooth_loss = self._update_ema(loss)
        
        # 如果存在冷却期，减少计数器
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr
        
        # 检查损失是否改善 (允许小幅波动)
        has_improved = False
        
        if smooth_loss < self.best_smooth_loss * (1 + self.allow_small_fluctuation):
            # 如果确实有显著改善，更新最佳值
            if smooth_loss < self.best_smooth_loss:
                self.best_smooth_loss = smooth_loss
                if not hasattr(self, 'ema_history'):
                    self.ema_history = deque(maxlen=20)
                self.ema_history.append(smooth_loss)
                
            has_improved = True
            self.num_bad_epochs = 0
        else:
            # 损失未改善
            self.num_bad_epochs += 1
            if hasattr(self, 'ema_history'):
                self.ema_history.append(smooth_loss)
        
        # 记录最佳原始损失，用于显示
        if loss < self.best_loss and not np.isnan(loss) and not np.isinf(loss) and loss > 0:
            self.best_loss = loss
        
        # 策略1: 常规耐心策略
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr(optimizer, reason="连续无改善")
            self.num_bad_epochs = 0
            return self.current_lr
            
        # 策略2: 趋势检测 (当启用时)
        if self.trend_detection and epoch is not None and epoch > 0:
            # 检查是否距离上次更新有足够间隔
            if epoch - self.last_update_epoch >= 3:
                # 检查损失稳定性和趋势
                is_stable = self.is_loss_stable()
                trend = self.detect_loss_trend()
                
                # 如果损失不稳定且有明显上升趋势，降低学习率
                if not is_stable and trend > 0:
                    self._reduce_lr(optimizer, reason="损失不稳定且趋势上升", factor=0.7)
                    return self.current_lr
        
        return self.current_lr
    
    def _reduce_lr(self, optimizer=None, reason="常规调度", factor=None):
        """降低学习率"""
        if factor is None:
            factor = self.factor
            
        new_lr = max(self.current_lr * factor, self.min_lr)
        
        # 检查是否有实质性变化
        if self.current_lr - new_lr <= 1e-8:
            return
            
        self.current_lr = new_lr
        self.cooldown_counter = self.patience // 2  # 设置冷却期
        
        if self.verbose:
            print(f"学习率降低到: {self.current_lr:.6f} (原因: {reason})")


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
            当前的学习率
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
            
            # 只有在显著变化时才打印（为了减少日志）
            if self.verbose and (self._step_count - self._last_update > 10 or new_lr < 1e-5):
                print(f"学习率更新为: {self.current_lr:.6f}")
                self._last_update = self._step_count
        
        return self.current_lr


class DummyLRScheduler:
    """伪学习率调度器，用于保持固定学习率"""
    def __init__(self, initial_lr=1e-3):
        self.current_lr = initial_lr
    
    def step(self, metrics=None, optimizer=None):
        """不做任何改变，直接返回当前学习率"""
        return self.current_lr


def create_lr_scheduler(
    scheduler_type: str,
    initial_lr: float,
    **kwargs
) -> Union[ReduceLROnPlateau, CosineAnnealingLR, EnhancedReduceLROnPlateau, DummyLRScheduler]:
    """创建学习率调度器
    
    参数:
        scheduler_type: 调度器类型，可选 'reduce_on_plateau', 'cosine', 'enhanced_plateau', 'none'
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
    elif scheduler_type == 'enhanced_plateau':
        return EnhancedReduceLROnPlateau(
            initial_lr=initial_lr,
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            min_lr=kwargs.get('min_lr', 1e-6),
            smooth_factor=kwargs.get('smooth_factor', 0.9),
            stability_window=kwargs.get('stability_window', 5),
            stability_threshold=kwargs.get('stability_threshold', 0.15),
            trend_detection=kwargs.get('trend_detection', True),
            allow_small_fluctuation=kwargs.get('allow_small_fluctuation', 0.02),
            verbose=kwargs.get('verbose', True)
        )
    elif scheduler_type == 'cosine':
        return CosineAnnealingLR(
            initial_lr=initial_lr,
            T_max=kwargs.get('max_iters', 1000),
            eta_min=kwargs.get('min_lr', 0),
            verbose=kwargs.get('verbose', True)
        )
    else:  # 'none' or any other value
        return DummyLRScheduler(initial_lr=initial_lr) 