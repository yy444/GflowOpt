import os
import time
import json
import numpy as np
import jax.numpy as jnp
from typing import Dict, Any, Optional

from dag_gflownet.utils import io

# 添加一个辅助函数用于将JAX数组转换为Python原生类型
def _convert_to_serializable(obj):
    """将对象转换为可JSON序列化的类型
    
    特别处理JAX数组和NumPy数组
    """
    import jax
    
    if isinstance(obj, (jnp.ndarray, np.ndarray)):
        # 将数组转换为Python列表
        return obj.tolist()
    elif hasattr(obj, 'item') and callable(obj.item):
        # 处理scalar array
        try:
            return obj.item()
        except:
            return float(obj)
    elif isinstance(obj, dict):
        # 递归处理字典
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # 递归处理列表和元组
        return [_convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        # 这些类型已经可以序列化
        return obj
    else:
        # 其他类型转为字符串
        return str(obj)


class TrainingStateSaver:
    """训练状态管理器，用于保存和加载完整训练状态"""

    def __init__(self, output_dir: str):
        """初始化训练状态管理器
        
        Args:
            output_dir: 输出目录根路径
        """
        self.output_dir = output_dir
        self.train_state_dir = os.path.join(output_dir, 'train_state')
        os.makedirs(self.train_state_dir, exist_ok=True)
    
    def save(self, 
             iteration: int, 
             params: Any, 
             state: Any, 
             optimizer: Any, 
             replay: Any, 
             lr_scheduler: Any, 
             loss_tracker: Any, 
             epsilon: float,
             args: Optional[Dict] = None) -> int:
        """保存完整的训练状态
        
        Args:
            iteration: 当前迭代次数
            params: 模型参数
            state: 模型状态
            optimizer: 优化器
            replay: 回放缓冲区
            lr_scheduler: 学习率调度器
            loss_tracker: 损失跟踪器
            epsilon: 当前探索率
            args: 命令行参数
            
        Returns:
            int: 保存状态的时间戳
        """
        # 生成时间戳作为唯一标识
        timestamp = int(time.time())
        
        # 保存模型参数
        io.save(os.path.join(self.train_state_dir, f'model_{timestamp}.npz'), params=params.online)
        
        # 保存回放缓冲区
        replay.save(os.path.join(self.train_state_dir, f'replay_{timestamp}.npz'))
        
        # 保存其他训练状态
        training_info = {
            'iteration': iteration,
            'timestamp': timestamp,
            'epsilon': _convert_to_serializable(epsilon),
            'lr': _convert_to_serializable(lr_scheduler.current_lr),
            'best_loss': _convert_to_serializable(loss_tracker.best_loss),
            'best_epoch': _convert_to_serializable(loss_tracker.best_epoch),
        }
        
        # 保存训练信息到JSON文件
        info_path = os.path.join(self.train_state_dir, f'training_info_{timestamp}.json')
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=4)
        
        # 如果提供了命令行参数，也保存它们
        if args is not None:
            args_path = os.path.join(self.train_state_dir, f'args_{timestamp}.json')
            with open(args_path, 'w') as f:
                json_args = vars(args).copy()
                # 将不可序列化的参数转换为字符串或可序列化类型
                for k, v in json_args.items():
                    json_args[k] = _convert_to_serializable(v)
                json.dump(json_args, f, indent=4)
        
        print(f"训练状态已保存，当前迭代: {iteration}, 时间戳: {timestamp}")
        print(f"- 模型参数: {os.path.join(self.train_state_dir, f'model_{timestamp}.npz')}")
        print(f"- 回放缓冲区: {os.path.join(self.train_state_dir, f'replay_{timestamp}.npz')}")
        print(f"- 训练信息: {info_path}")
        
        return timestamp
    
    def list_checkpoints(self) -> Dict[int, Dict]:
        """列出所有可用的检查点
        
        Returns:
            Dict[int, Dict]: 时间戳到训练信息的映射
        """
        checkpoints = {}
        
        # 查找所有训练信息文件
        info_files = [f for f in os.listdir(self.train_state_dir) if f.startswith('training_info_') and f.endswith('.json')]
        
        for info_file in info_files:
            # 解析时间戳
            timestamp = int(info_file.split('_')[-1].split('.')[0])
            
            # 加载训练信息
            info_path = os.path.join(self.train_state_dir, info_file)
            with open(info_path, 'r') as f:
                training_info = json.load(f)
            
            checkpoints[timestamp] = training_info
        
        return checkpoints
    
    def get_latest_checkpoint(self) -> Optional[Dict]:
        """获取最新的检查点信息
        
        Returns:
            Optional[Dict]: 最新的检查点信息，如果没有检查点则返回None
        """
        checkpoints = self.list_checkpoints()
        
        if not checkpoints:
            return None
        
        # 按时间戳排序
        latest_timestamp = max(checkpoints.keys())
        return checkpoints[latest_timestamp]
    
    def load(self, timestamp: Optional[int] = None) -> Dict:
        """加载训练状态
        
        Args:
            timestamp: 要加载的检查点时间戳，如果不指定则加载最新的
            
        Returns:
            Dict: 包含训练状态信息的字典
        """
        # 如果未指定时间戳，查找最新的训练状态
        if timestamp is None:
            checkpoint_info = self.get_latest_checkpoint()
            
            if checkpoint_info is None:
                raise FileNotFoundError(f"训练状态目录中没有找到训练信息文件: {self.train_state_dir}")
            
            timestamp = checkpoint_info['timestamp']
            print(f"使用最新的训练状态，时间戳: {timestamp}")
        
        # 加载训练信息
        info_file = os.path.join(self.train_state_dir, f'training_info_{timestamp}.json')
        if not os.path.exists(info_file):
            raise FileNotFoundError(f"训练信息文件不存在: {info_file}")
        
        with open(info_file, 'r') as f:
            training_info = json.load(f)
        
        # 设置模型和回放缓冲区的路径
        training_info['model_path'] = os.path.join(self.train_state_dir, f'model_{timestamp}.npz')
        training_info['replay_path'] = os.path.join(self.train_state_dir, f'replay_{timestamp}.npz')
        
        # 检查文件是否存在
        if not os.path.exists(training_info['model_path']):
            raise FileNotFoundError(f"模型文件不存在: {training_info['model_path']}")
        
        if not os.path.exists(training_info['replay_path']):
            raise FileNotFoundError(f"回放缓冲区文件不存在: {training_info['replay_path']}")
        
        # 可选：加载参数文件
        args_file = os.path.join(self.train_state_dir, f'args_{timestamp}.json')
        if os.path.exists(args_file):
            with open(args_file, 'r') as f:
                training_info['args'] = json.load(f)
        
        print(f"训练状态已加载，迭代次数: {training_info['iteration']}, 学习率: {training_info['lr']}")
        return training_info


# 辅助函数，方便直接导入使用
def save_training_state(output_dir, iteration, params, state, optimizer, replay, lr_scheduler, loss_tracker, epsilon, args=None):
    """保存完整的训练状态，以便恢复训练
    
    Args:
        output_dir: 输出目录
        iteration: 当前迭代次数
        params: 模型参数
        state: 模型状态
        optimizer: 优化器
        replay: 回放缓冲区
        lr_scheduler: 学习率调度器
        loss_tracker: 损失跟踪器
        epsilon: 当前探索率
        args: 命令行参数
        
    Returns:
        int: 保存状态的时间戳
    """
    saver = TrainingStateSaver(output_dir)
    return saver.save(iteration, params, state, optimizer, replay, lr_scheduler, loss_tracker, epsilon, args)


def load_training_state(state_dir, timestamp=None):
    """加载训练状态
    
    Args:
        state_dir: 训练状态目录
        timestamp: 指定的时间戳，如果为None则加载最新的
        
    Returns:
        dict: 包含恢复的训练状态信息
    """
    saver = TrainingStateSaver(state_dir)
    return saver.load(timestamp) 