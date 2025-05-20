import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import json
import sys
from pathlib import Path
from tqdm import trange, tqdm
from argparse import ArgumentParser
import optax
import time
import wandb
from dag_gflownet.utils import io
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from scipy.spatial.distance import cdist
from jax import grad, jit, value_and_grad

# 导入学习率调度器
try:
    from gfnproxy.dag_gflownet.utils.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, create_lr_scheduler
    print("成功从包导入学习率调度器")
except ImportError:
    from dag_gflownet.utils.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, create_lr_scheduler
    print("从当前目录导入学习率调度器")

# 添加当前目录到Python路径
sys.path.append(os.path.abspath("."))

try:
    from gfnproxy.proxy_model import ProxyModel
    print("成功从包导入ProxyModel")
except ImportError:
    # 如果包导入失败，尝试直接导入
    from proxy_model import ProxyModel
    print("从当前目录导入ProxyModel")

from dag_gflownet.gflownet_c import DAGGFlowNet
from dag_gflownet.env_v4 import GFlowNetDAGEnv
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils import io
from dag_gflownet.utils.jraph_utils import to_graphs_tuple

# 添加标准化器类用于处理y值
class Normalizer:
    """用于对目标值进行标准化的类"""
    
    def __init__(self, method='standard', scale_factor=1.0):
        """
        初始化标准化器
        
        Args:
            method: 标准化方法，可选 'standard'(标准化), 'minmax'(最小-最大归一化), 'log'(对数变换), 'exp'(指数放缩)
            scale_factor: 缩放因子，可以用来调整标准化后的数值范围
        """
        self.method = method
        self.scale_factor = scale_factor
        self.params = {}
        self.is_fitted = False
    
    def fit(self, data):
        """计算标准化参数"""
        if self.method == 'standard':
            self.params['mean'] = np.mean(data)
            self.params['std'] = np.std(data)
            # Avoid division by zero error
            if self.params['std'] == 0:
                self.params['std'] = 1.0
                
        elif self.method == 'minmax':
            self.params['min'] = np.min(data)
            self.params['max'] = np.max(data)
            # Avoid division by zero error
            if self.params['max'] == self.params['min']:
                self.params['max'] = self.params['min'] + 1.0
        
        elif self.method == 'exp':
            # 指数放缩，先平移到正数范围，再计算缩放因子
            min_val = np.min(data)
            if min_val <= 0:
                self.params['shift'] = abs(min_val) + 1.0
            else:
                self.params['shift'] = 0.0
            # 计算数据的范围以确定合适的缩放因子
            data_range = np.max(data + self.params['shift']) - np.min(data + self.params['shift'])
            self.params['exp_base'] = max(1.1, min(2.0, 10.0 / data_range))
                
        elif self.method == 'log':
            # 对数变换，确保所有值为正
            min_val = np.min(data)
            if min_val <= 0:
                self.params['shift'] = abs(min_val) + 1.0
            else:
                self.params['shift'] = 0.0
                
        else:
            raise ValueError(f"不支持的标准化方法: {self.method}")
            
        self.is_fitted = True
        return self
    
    def transform(self, data):
        """对数据进行标准化"""
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合，请先调用fit方法")
            
        if self.method == 'standard':
            return self.scale_factor * ((data - self.params['mean']) / self.params['std'])
            
        elif self.method == 'minmax':
            # 最小-最大归一化，放缩到[0,scale_factor]范围
            return self.scale_factor * ((data - self.params['min']) / (self.params['max'] - self.params['min']))
        
        elif self.method == 'exp':
            # 使用指数放缩，可以放大差异
            shifted_data = data + self.params['shift']
            return self.scale_factor * (np.power(self.params['exp_base'], shifted_data) - 1.0)
            
        elif self.method == 'log':
            return self.scale_factor * np.log(data + self.params['shift'])
    
    def inverse_transform(self, data):
        """逆标准化，将标准化的数据转换回原始尺度"""
        if not self.is_fitted:
            raise ValueError("标准化器尚未拟合，请先调用fit方法")
            
        if self.method == 'standard':
            return (data / self.scale_factor) * self.params['std'] + self.params['mean']
            
        elif self.method == 'minmax':
            return (data / self.scale_factor) * (self.params['max'] - self.params['min']) + self.params['min']
        
        elif self.method == 'exp':
            # 指数逆变换
            exp_data = (data / self.scale_factor) + 1.0
            return np.log(exp_data) / np.log(self.params['exp_base']) - self.params['shift']
            
        elif self.method == 'log':
            return np.exp(data / self.scale_factor) - self.params['shift']
    
    def fit_transform(self, data):
        """拟合并转换数据"""
        return self.fit(data).transform(data)
    
    def save(self, path):
        """保存标准化器参数"""
        state = {
            'method': self.method,
            'params': self.params,
            'is_fitted': self.is_fitted,
            'scale_factor': self.scale_factor
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path):
        """加载标准化器参数"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.method = state['method']
        self.params = state['params']
        self.is_fitted = state['is_fitted']
        self.scale_factor = state.get('scale_factor', 1.0)  # 向后兼容
        return self

def load_gflownet_model(model_path):
    """加载GFlowNet模型"""
    print(f"加载GFlowNet模型: {model_path}")
    
    if os.path.exists(model_path):
        # 加载模型参数
        params = io.load(model_path)
        print("GFlowNet模型参数加载成功")
        return params
    else:
        raise ValueError(f"GFlowNet模型文件不存在: {model_path}")

def generate_samples(args, gflownet, env, params, num_samples=1000):
    """使用GFlowNet生成样本"""
    print(f"使用GFlowNet生成 {num_samples} 个样本...")
    
    normalization = jnp.array(1.)
    # 初始化随机种子
    key = jax.random.PRNGKey(args.seed)
    
    # 准备训练数据
    X = []  # 图特征
    y = []  # 真实分数
    
    # 分批生成样本，但只跟踪总样本数
    samples_collected = 0
    max_steps = num_samples * 1000  # 设置最大步数限制
    step_count = 0
    
    observations = env.reset()
    observations['graph'] = to_graphs_tuple(observations['adjacency'])
    
    # 使用进度条
    with tqdm(total=num_samples, desc="生成样本", ncols=100) as pbar:
        while samples_collected < num_samples and step_count < max_steps:
            # 获取当前状态信息
            score = observations['score']
            
            # 执行动作
            actions, key, _, embedding = gflownet.act(params.online, key, observations, 0.5, normalization)
            #print(observations["num_edges"])
            observations, _, dones, _ = env.step(np.asarray(actions))
            
            # 确保下一步图信息可用
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            
            # 转换embedding为jax格式并处理完成的样本
            embedding = jnp.array(embedding)
            
            # 处理每个环境的样本
            for i in range(len(dones)):
                if dones[i]:
                    # 只添加完成的样本
                    X.append(embedding[i])
                    y.append(score[i])
                    samples_collected += 1
                    pbar.update(1)
                    
                    # 如果已收集足够样本，提前退出
                    if samples_collected >= num_samples:
                        break
            
            step_count += 1
            
            # 每隔一定步数打印进度（可选）
            if step_count % 100 == 0:
                print(f"已执行 {step_count} 步，收集 {samples_collected}/{num_samples} 个样本")
    
    # 转换为numpy/jax数组
    X = jnp.stack(X[:num_samples])  # 使用stack而不是array
    y = np.array(y[:num_samples])
    
    print(f"成功生成 {X.shape[0]} 个样本，用了 {step_count} 步")
    print(f"分数范围: [{y.min():.4f}, {y.max():.4f}], 平均值: {y.mean():.4f}")
    
    return X, y

# 定义一个简化版的学习率调度器，直接包含在当前文件中
class SimpleReduceLR:
    """简化的学习率调度器：当损失不再有效改善时降低学习率
    
    参数:
        initial_lr: 初始学习率
        factor: 学习率降低的因子，学习率将乘以这个因子 (0..1)
        patience: 无改善后需要等待多少个轮次才降低学习率
        min_lr: 学习率的下限
        threshold: 认为有改善的阈值，相对于最佳损失
        verbose: 是否打印学习率变化
    """
    def __init__(
        self,
        initial_lr=1e-3,
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        threshold=0.001,
        verbose=True
    ):
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.verbose = verbose
        
        # 内部状态
        self.current_lr = initial_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.last_epoch = -1
    
    def step(self, current_loss, epoch=None):
        """更新学习率
        
        参数:
            current_loss: 当前的损失值
            epoch: 当前轮次(可选)，用于打印
            
        返回:
            当前学习率
        """
        # 如果提供了epoch参数
        self.last_epoch = epoch if epoch is not None else self.last_epoch + 1
        
        # 检查当前损失是否足够好
        is_improved = False
        
        # 绝对改善数量
        improvement = self.best_loss - current_loss
        
        # 相对改善比例
        relative_improvement = improvement / self.best_loss if self.best_loss > 0 else 0
        
        # 如果有显著改善，更新最佳损失
        if relative_improvement > self.threshold:
            self.best_loss = current_loss
            self.wait_count = 0
            is_improved = True
        else:
            # 没有足够的改善，增加等待计数
            self.wait_count += 1
            
        # 如果等待计数超过耐心值，降低学习率
        if self.wait_count >= self.patience:
            # 计算新的学习率
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait_count = 0  # 重置等待计数
            
            # 打印信息
            if self.verbose and old_lr - self.current_lr > 1e-8:
                print(f"第 {self.last_epoch} 轮：学习率从 {old_lr:.6f} 降低到 {self.current_lr:.6f}")
                
        return self.current_lr

def train_proxy_model(args, gflownet, X, y):
    """训练代理模型"""
    print("开始训练代理模型...")
    
    # 初始化wandb（如果启用）
    if args.use_wandb:
        wandb.init(project=args.wandb_project, 
                   name=args.wandb_run_name or f"proxy_model_{time.strftime('%Y%m%d_%H%M%S')}",
                   config=vars(args))
    
    # 创建并应用标准化器处理y值
    print(f"对目标值进行 {args.normalization_method} 标准化...")
    normalizer = Normalizer(method=args.normalization_method, scale_factor=args.norm_scale_factor)
    y_normalized = normalizer.fit_transform(y)
    
    print(f"标准化后的分数范围: [{y_normalized.min():.4f}, {y_normalized.max():.4f}], 平均值: {y_normalized.mean():.4f}")
    
    # 创建代理模型实例
    proxy_model = ProxyModel(
        gflownet=gflownet,
        hidden_dims=args.proxy_hidden_dims,
        learning_rate=args.proxy_lr,
        weight_decay=args.proxy_weight_decay
    )
    
    # 初始化模型参数
    key = jax.random.PRNGKey(args.seed)
    proxy_params = proxy_model.init(key, args.batch_size)
    
    # 创建简化的学习率调度器
    lr_scheduler = SimpleReduceLR(
        initial_lr=args.proxy_lr,
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
        threshold=args.lr_threshold,
        verbose=True
    )
    
    # 当前学习率
    current_lr = args.proxy_lr
    
    # 创建优化器
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.grad_clip),  # 梯度裁剪，防止梯度爆炸
        optax.adam(
            learning_rate=current_lr,
            b1=0.9,  # Adam的beta1参数
            b2=0.999,  # Adam的beta2参数
            eps=1e-8  # 数值稳定性参数
        )
    )
    opt_state = optimizer.init(proxy_params)
    
    # 划分训练集和验证集
    indices = np.random.permutation(len(X))
    train_size = int(len(X) * 0.8)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_X = [X[i] for i in train_indices]
    train_y = y_normalized[train_indices]
    train_y_orig = y[train_indices]  # 存储原始范围的y值，用于计算原始尺度的误差
    val_X = [X[i] for i in val_indices]
    val_y = y_normalized[val_indices]
    val_y_orig = y[val_indices]  # 存储原始范围的y值，用于计算原始尺度的误差
    
    print(f"训练集: {len(train_X)} 个样本，验证集: {len(val_X)} 个样本")
    
    # 训练循环
    print(f"开始训练，共 {args.num_epochs} 轮...")
    best_loss = float('inf')
    best_params = None
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_loss_orig': [],  # 原始尺度的训练损失
        'val_loss_orig': [],    # 原始尺度的验证损失
        'lr': []
    }
    
    # 创建总体进度条
    epoch_bar = tqdm(range(args.num_epochs), desc="训练进度", position=0, leave=True, ncols=100)
    
    # 计算原始尺度的MSE损失
    def compute_orig_mse(params, X_batch, y_batch_normalized, y_batch_orig):
        # 获取模型预测（在标准化空间）
        pred_normalized = proxy_model.network.apply(params, None, jnp.array(X_batch), is_training=False)
        # 逆变换到原始空间
        pred_orig = normalizer.inverse_transform(np.array(pred_normalized))
        # 计算MSE
        return np.mean((pred_orig - y_batch_orig) ** 2)
    
    for epoch in epoch_bar:
        # 训练阶段
        epoch_loss = 0
        epoch_loss_orig = 0
        num_batches = 0
        
        # 随机打乱训练数据
        batch_indices = np.random.permutation(len(train_X))
        
        # 创建批次进度条
        train_bar = tqdm(range(0, len(train_X), args.batch_size), 
                         desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]", 
                         position=1, leave=False, ncols=100)
        
        # 批次训练
        for i in train_bar:
            batch_idx = batch_indices[i:i + args.batch_size]
            batch_X = [train_X[j] for j in batch_idx]
            batch_y = train_y[batch_idx]
            batch_y_orig = train_y_orig[batch_idx]  # 原始范围的y值
            
            # 把batch_X转为jax格式
            batch_X = jnp.array(batch_X)
            # 把batch_y转为jax格式
            batch_y = jnp.array(batch_y)
            
            # 计算损失和梯度
            loss, grads = proxy_model.loss_and_grad(proxy_params, batch_X, batch_y)
            
            # 计算原始尺度的损失（用于显示，不参与训练）
            orig_loss = compute_orig_mse(proxy_params, batch_X, batch_y, batch_y_orig)
            
            # 更新优化器，采用当前学习率
            optimizer = optax.chain(
                optax.clip_by_global_norm(args.grad_clip),
                optax.adam(
                    learning_rate=current_lr,
                    b1=0.9,
                    b2=0.999,
                    eps=1e-8
                )
            )
            
            # 第一个批次需要初始化优化器状态
            if num_batches == 0:
                opt_state = optimizer.init(proxy_params)
            
            updates, opt_state = optimizer.update(grads, opt_state, proxy_params)
            proxy_params = optax.apply_updates(proxy_params, updates)
            
            epoch_loss += loss
            epoch_loss_orig += orig_loss
            num_batches += 1
            
            # 更新进度条，同时显示标准化空间和原始空间的损失
            train_bar.set_postfix({
                'loss': f"{loss:.4f}", 
                'orig_loss': f"{orig_loss:.4f}",
                'lr': f"{current_lr:.6f}"
            })
        
        # 计算训练集平均损失
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_orig = epoch_loss_orig / num_batches
        history['train_loss'].append(float(avg_train_loss))
        history['train_loss_orig'].append(float(avg_train_loss_orig))
        
        # 验证阶段
        val_loss = 0
        val_loss_orig = 0
        val_batches = 0
        
        # 创建验证进度条
        val_bar = tqdm(range(0, len(val_X), args.batch_size), 
                       desc=f"Epoch {epoch+1}/{args.num_epochs} [Valid]", 
                       position=1, leave=False, ncols=100)
        
        # 批次验证
        for i in val_bar:
            end_idx = min(i + args.batch_size, len(val_X))
            batch_X = val_X[i:end_idx]
            batch_y = val_y[i:end_idx]
            batch_y_orig = val_y_orig[i:end_idx]  # 原始范围的y值
            
            # 计算验证损失
            loss = proxy_model.loss_fn(proxy_params, batch_X, batch_y)
            
            # 计算原始尺度的验证损失
            orig_loss = compute_orig_mse(proxy_params, batch_X, batch_y, batch_y_orig)
            
            val_loss += loss
            val_loss_orig += orig_loss
            val_batches += 1
            
            # 更新进度条，同时显示标准化空间和原始空间的损失
            val_bar.set_postfix({
                'loss': f"{loss:.4f}",
                'orig_loss': f"{orig_loss:.4f}"
            })
        
        # 计算验证集平均损失
        avg_val_loss = val_loss / val_batches
        avg_val_loss_orig = val_loss_orig / val_batches
        history['val_loss'].append(float(avg_val_loss))
        history['val_loss_orig'].append(float(avg_val_loss_orig))
        
        history['lr'].append(float(current_lr))
        
        # 更新学习率 - 简化版本
        current_lr = lr_scheduler.step(avg_val_loss, epoch=epoch)
        
        # 更新总体进度条，同时显示标准化空间和原始空间的损失
        epoch_bar.set_postfix({
            'train_loss': f"{avg_train_loss:.4f}", 
            'val_loss': f"{avg_val_loss:.4f}",
            'orig_loss': f"{avg_val_loss_orig:.4f}", 
            'lr': f"{current_lr:.6f}"
        })
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_loss_orig': avg_train_loss_orig,
                'val_loss_orig': avg_val_loss_orig,
                'learning_rate': current_lr
            })
        
        # 打印进度
        if (epoch + 1) % args.log_every == 0:
            print(f"Epoch {epoch+1}/{args.num_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                  f"Orig Loss: {avg_val_loss_orig:.4f}, LR: {current_lr:.6f}")
        
        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_params = proxy_params
            print(f"发现新的最佳模型，验证损失: {best_loss:.4f}, 原始尺度损失: {avg_val_loss_orig:.4f}")
            
            if args.use_wandb:
                wandb.run.summary["best_val_loss"] = best_loss
                wandb.run.summary["best_val_loss_orig"] = avg_val_loss_orig
    
    print(f"训练完成！最佳验证损失: {best_loss:.4f}")
    
    # 关闭wandb
    if args.use_wandb:
        wandb.finish()
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, "training_history.pkl")
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"训练历史已保存至 {history_path}")
    
    # 保存标准化器
    normalizer_path = os.path.join(args.output_dir, "normalizer.pkl")
    normalizer.save(normalizer_path)
    print(f"标准化器已保存至 {normalizer_path}")
    
    return best_params, normalizer


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
      # 创建环境
    print("创建GFlowNet环境...")
    scorer, data, graph = get_scorer(args, rng=np.random.RandomState(args.seed))
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )  
    # 加载GFlowNet模型
    gflownet_params = load_gflownet_model(args.gflownet_model_path)
    
    # 创建GFlowNet实例
    gflownet = DAGGFlowNet(
        delta=args.delta,
        update_target_every=args.update_target_every,
        dataset_size=1
    )

    replay_capacity = 10000


    replay = ReplayBuffer(
        replay_capacity,
        num_variables=env.num_variables
    )
    
    from numpy.random import default_rng
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)


    lr = 1e-4
    
    optimizer = optax.adam(lr)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['adjacency'],
        replay.dummy['mask'],
        gflownet_params
    )    

    
    # 生成样本
    X, y = generate_samples(
        args,
        gflownet,
        env,
        params,
        num_samples=args.num_samples
    )
    
    # 训练代理模型
    proxy_params, normalizer = train_proxy_model(args, gflownet, X, y)
    
    # 保存代理模型
    output_file = os.path.join(args.output_dir, "proxy_model.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(proxy_params, f)
    print(f"代理模型已保存至 {output_file}")

# 修改predict方法来使用标准化器，添加到ProxyModel类中
def modified_predict(self, params, graph, mask=None, normalizer=None):
    """预测样本的得分，可选使用标准化器进行逆变换"""
    # 获取GFlowNet特征
    features = graph
    features = jnp.array(features)      
    # 预测分数
    pred_score = self.network.apply(params, None, features, is_training=False)
    
    # 如果提供了标准化器，进行逆变换
    if normalizer is not None and normalizer.is_fitted:
        pred_score = normalizer.inverse_transform(np.array(pred_score))
        
    return pred_score

# 修补ProxyModel类，添加改进后的predict方法
ProxyModel.predict = modified_predict

def gradient_optimize_embedding(proxy_model, proxy_params, initial_embedding, normalizer=None, 
                             num_steps=50, learning_rate=0.01, verbose=True):
    """
    使用梯度下降优化表征以最大化代理模型预测的分数
    
    参数:
        proxy_model: 代理模型实例
        proxy_params: 代理模型参数
        initial_embedding: 初始表征
        normalizer: 标准化器(可选)
        num_steps: 梯度下降迭代次数
        learning_rate: 学习率
        verbose: 是否显示进度
    
    返回:
        optimized_embedding: 优化后的表征
        final_score: 最终分数
    """
    # 定义一个函数，返回负的分数(我们要最大化分数，相当于最小化负分数)
    def loss_fn(embedding):
        # 将embedding转为适合模型输入的形式
        embedding_array = jnp.array([embedding])
        # 获取预测分数(不使用inverse_transform，因为我们只关心梯度方向)
        score = proxy_model.network.apply(proxy_params, None, embedding_array, is_training=False)[0]
        # 返回负分数作为损失函数
        return -score
    
    # 创建求值和梯度的函数
    value_and_grad_fn = jit(value_and_grad(loss_fn))
    
    # 将初始表征转为jax数组
    embedding = jnp.array(initial_embedding)
    
    if verbose:
        print(f"开始基于梯度优化表征...")
    
    # 梯度下降优化
    for step in range(num_steps):
        # 计算当前损失和梯度
        loss_val, grad_val = value_and_grad_fn(embedding)
        
        # 更新表征 (梯度下降)
        embedding = embedding - learning_rate * grad_val
        
        if verbose and (step + 1) % 10 == 0:
            # 计算当前实际分数(使用标准化器进行逆变换)
            current_score = -loss_val
            if normalizer is not None and normalizer.is_fitted:
                # 转换回numpy以便使用标准化器
                current_score_np = np.array([current_score])
                current_score = normalizer.inverse_transform(current_score_np)[0]
            
            print(f"  步骤 {step+1}/{num_steps}: 分数 = {current_score:.6f}")
    
    # 计算最终分数
    final_loss = loss_fn(embedding)
    final_score = -final_loss
    
    # 如果有标准化器，将分数转换回原始尺度
    if normalizer is not None and normalizer.is_fitted:
        final_score_np = np.array([final_score])
        final_score = normalizer.inverse_transform(final_score_np)[0]
    
    if verbose:
        print(f"表征优化完成。最终分数: {final_score:.6f}")
    
    # 将优化后的表征转换回numpy数组
    optimized_embedding = np.array(embedding)
    
    return optimized_embedding, final_score


from scipy.spatial.distance import cdist

def select_diverse_top_samples(X_pool, y_pred, top_k=10, diversity_threshold=0.5, metric='euclidean'):
    """
    选择得分最高且彼此之间保持多样性的样本
    
    参数:
        X_pool: 样本池
        y_pred: 预测分数
        top_k: 要选择的样本数量
        diversity_threshold: 样本间最小距离阈值
        metric: 距离度量方式，'euclidean'或'cosine'
    """
    # 按分数从高到低排序
    sorted_indices = np.argsort(y_pred)[::-1]
    
    selected_indices = []
    selected_X = []
    
    # 贪婪选择过程
    for idx in sorted_indices:
        # 如果是第一个样本，直接添加
        if len(selected_indices) == 0:
            selected_indices.append(idx)
            selected_X.append(X_pool[idx])
            continue
        
        # 计算当前样本与已选样本的距离
        current_X = X_pool[idx]
        
        if metric == 'cosine':
            # 余弦相似度转距离 (1 - 相似度)
            distances = 1 - cdist([current_X], selected_X, metric='cosine')[0]
            # 余弦相似度越高表示越相似，所以需要检查是否小于(1-阈值)
            too_similar = any(sim > (1 - diversity_threshold) for sim in distances)
        else:  # 欧氏距离
            distances = cdist([current_X], selected_X, metric='euclidean')[0]
            # 欧氏距离越小表示越相似
            too_similar = any(dist < diversity_threshold for dist in distances)
        
        # 只有当与已选样本都保持足够距离时，才添加该样本
        if not too_similar:
            selected_indices.append(idx)
            selected_X.append(current_X)
        
        # 如果已选择足够数量，结束循环
        if len(selected_indices) >= top_k:
            break
    
    # 如果未找到足够的多样性样本，输出警告
    if len(selected_indices) < top_k:
        print(f"警告: 只找到 {len(selected_indices)} 个多样性样本，少于请求的 {top_k} 个")
    
    # 返回结果
    top_X = np.array([X_pool[i] for i in selected_indices])
    top_y_pred = y_pred[selected_indices]
    
    return selected_indices, top_X, top_y_pred

def optimize_with_proxy(args):
    """
    使用代理模型和GFlowNet协同寻找高分表征和图结构
    
    流程：
    1. 加载训练好的代理模型和GFlowNet模型
    2. 使用GFlowNet生成一批样本和对应的表征
    3. 使用代理模型从这些样本中找出预测得分最高的表征
    4. 在GFlowNet生成的表征中找到与高分表征最相似的表征
    5. 使用这些相似表征对应的图结构作为起点，生成新的样本
    6. 重复步骤3-5，不断优化得到更高分的表征和图结构
    """
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print(f"代理模型优化 - 输出目录: {args.output_dir}")
    print("="*80 + "\n")
    
    # 步骤1: 创建环境和加载模型
    print("步骤 1/7: 创建GFlowNet环境...")
    scorer, data, graph = get_scorer(args, rng=np.random.RandomState(args.seed))
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )
    
    # 加载GFlowNet模型
    print("\n步骤 2/7: 加载GFlowNet模型...")
    gflownet_params = load_gflownet_model(args.gflownet_model_path)
    
    # 创建GFlowNet实例
    print("\n步骤 3/7: 初始化GFlowNet...")
    gflownet = DAGGFlowNet(
        delta=args.delta,
        update_target_every=args.update_target_every,
        dataset_size=1
    )
    
    # 初始化GFlowNet
    replay_capacity = 10000
    replay = ReplayBuffer(
        replay_capacity,
        num_variables=env.num_variables
    )
    
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    optimizer = optax.adam(1e-4)
    params, state = gflownet.init(
        subkey,
        optimizer,
        replay.dummy['adjacency'],
        replay.dummy['mask'],
        gflownet_params
    )
    
    # 加载代理模型
    print("\n步骤 4/7: 加载训练好的代理模型...")
    proxy_model = ProxyModel(
        gflownet=gflownet,
        hidden_dims=args.proxy_hidden_dims,
        learning_rate=args.proxy_lr,
        weight_decay=args.proxy_weight_decay
    )
    
    # 加载代理模型参数
    proxy_params_path = args.proxy_model_path
    if not os.path.exists(proxy_params_path):
        raise ValueError(f"代理模型参数文件不存在: {proxy_params_path}")
    
    with open(proxy_params_path, 'rb') as f:
        proxy_params = pickle.load(f)
    print(f"代理模型参数加载成功: {proxy_params_path}")
    
    # 加载标准化器(如果有)
    normalizer = None
    normalizer_path = os.path.join(os.path.dirname(proxy_params_path), "normalizer.pkl")
    if os.path.exists(normalizer_path):
        try:
            normalizer = Normalizer().load(normalizer_path)
            print(f"标准化器加载成功: {normalizer_path}")
        except Exception as e:
            print(f"加载标准化器失败: {e}")
            print("将不使用标准化器")
    
    # 步骤2: 使用GFlowNet生成初始样本
    print("\n步骤 5/7: 使用GFlowNet生成初始样本...")
    X_initial, y_initial, matrices_initial = generate_samples_with_embeddings(
        args,
        gflownet,
        env,
        params,
        num_samples=args.initial_samples
    )
    
    print(f"初始样本: {len(X_initial)}个, 分数范围: [{np.min(y_initial):.4f}, {np.max(y_initial):.4f}], 平均值: {np.mean(y_initial):.4f}")
    
    # 保存所有迭代的最佳样本
    all_best_samples = []
    
    # 创建迭代优化日志
    optimization_log = {
        "iterations": [],
        "best_score_per_iter": [],
        "mean_score_per_iter": [],
        "best_overall_score": float('-inf'),
        "final_best_samples": []
    }
    
    # 步骤3-7: 迭代优化过程
    print("\n步骤 6/7: 开始迭代优化...")
    best_score_overall = float('-inf')
    best_sample_overall = None
    
    # 添加基于梯度的优化参数
    gradient_steps = getattr(args, 'gradient_steps', 50)
    gradient_lr = getattr(args, 'gradient_lr', 0.01)
    
    # 创建目录保存每次迭代的结果
    for iteration in range(args.optimization_iterations):
        print(f"\n====== 迭代 {iteration+1}/{args.optimization_iterations} ======")
        
        iter_dir = os.path.join(args.output_dir, f"iteration_{iteration+1}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # 使用代理模型预测所有样本的得分
        if iteration == 0:
            # 第一次迭代使用GFlowNet生成的初始样本
            X_pool = X_initial
            y_pred = []
            
            # 分批预测以避免内存问题
            batch_size = args.batch_size
            for i in range(0, len(X_pool), batch_size):
                end_idx = min(i + batch_size, len(X_pool))
                batch_X = np.array([X_pool[j] for j in range(i, end_idx)])
                
                # 使用代理模型预测得分
                batch_pred = proxy_model.predict(proxy_params, batch_X, normalizer=normalizer)
                y_pred.extend(batch_pred)
            
            y_pred = np.array(y_pred)
            
            # 原始代码
            # top_indices = np.argsort(y_pred)[-args.top_k:][::-1]
            # top_X = np.array([X_pool[i] for i in top_indices])
            # top_y_pred = y_pred[top_indices]

            # 替换为多样性选择
            selected_indices, top_X, top_y_pred = select_diverse_top_samples(
                X_pool, 
                y_pred, 
                top_k=args.top_k,
                diversity_threshold=args.diversity_threshold,  # 需要添加这个参数
                metric=args.diversity_metric  # 需要添加这个参数
            )

            print(f"代理模型预测的Top {args.top_k} 样本得分范围: [{top_y_pred.min():.4f}, {top_y_pred.max():.4f}]")
            
            # 为每个顶部样本应用基于梯度的优化
            optimized_embeddings = []
            optimized_scores = []
            print("\n对顶部样本应用基于梯度的优化...")
            
            for i, embedding in enumerate(top_X):
                print(f"\n优化表征 #{i+1} (初始预测分数: {top_y_pred[i]:.4f}):")
                opt_embedding, opt_score = gradient_optimize_embedding(
                    proxy_model, 
                    proxy_params, 
                    embedding, 
                    normalizer,
                    num_steps=gradient_steps,
                    learning_rate=gradient_lr
                )
                print(f"  优化后分数: {opt_score:.4f} (增加: {opt_score - top_y_pred[i]:.4f})")
                optimized_embeddings.append(opt_embedding)
                optimized_scores.append(opt_score)
            
            # 替换为优化后的表征
            top_X = np.array(optimized_embeddings)
            top_y_pred = np.array(optimized_scores)
            
            # 计算每个GFlowNet样本到最高分样本的相似度(使用余弦相似度或欧式距离)
            if args.similarity_metric == 'cosine':
                # 余弦相似度 (值越大越相似)
                similarities = 1 - cdist(top_X, X_initial, metric='cosine')
            else:
                # 欧氏距离 (值越小越相似)，需要转换为相似度
                dist = cdist(top_X, X_initial, metric='euclidean')
                max_dist = np.max(dist)
                similarities = 1 - (dist / max_dist if max_dist > 0 else dist)
            
            # 对每个高分表征，找出最相似且不重复的k个GFlowNet样本
            start_matrices = []
            for i in range(len(top_X)):
                # 第1步：对初始矩阵进行去重，避免完全相同的图结构
                unique_matrices = {}  # 使用字典跟踪唯一矩阵
                unique_indices = []   # 存储唯一矩阵的索引
                
                for idx, matrix in enumerate(matrices_initial):
                    # 生成矩阵的哈希键（将矩阵转换为字符串）
                    matrix_key = str(np.array(matrix).flatten())
                    
                    # 如果这个矩阵之前没见过，记录它
                    if matrix_key not in unique_matrices:
                        unique_matrices[matrix_key] = idx
                        unique_indices.append(idx)
                
                # 将列表索引转为JAX数组索引
                unique_indices_array = jnp.array(unique_indices)
                unique_X = X_initial[unique_indices_array]
                unique_y = y_initial[unique_indices]
                
                print(f"初始样本去重: {len(matrices_initial)} -> {len(unique_indices)} 个唯一图结构")
                
                # 第2步：计算高分表征与去重后表征的相似度
                if args.similarity_metric == 'cosine':
                    # 余弦相似度 (值越大越相似)
                    current_similarities = 1 - cdist([top_X[i]], unique_X, metric='cosine')[0]
                else:
                    # 欧氏距离转换为相似度
                    dist = cdist([top_X[i]], unique_X, metric='euclidean')[0]
                    max_dist = np.max(dist) if np.max(dist) > 0 else 1.0
                    current_similarities = 1 - (dist / max_dist)
                
                # 第3步：找出最相似的k个样本索引
                sim_indices = np.argsort(current_similarities)[-args.num_similar_samples:][::-1]
                
                # 获取这些样本的实际分数和原始索引
                sim_scores = unique_y[sim_indices]
                orig_indices = [unique_indices[idx] for idx in sim_indices]
                
                # 打印相似样本信息
                print(f"高分表征 #{i+1} (优化后分数: {top_y_pred[i]:.4f}):")
                for j, (idx, orig_idx, sim_score) in enumerate(zip(sim_indices, orig_indices, sim_scores)):
                    print(f"  相似样本 #{j+1}: 相似度 = {current_similarities[idx]:.4f}, "
                        f"实际得分 = {sim_score:.4f}, 原始索引 = {orig_idx}")
                
                # 获取最相似的样本的图结构
                for idx in orig_indices:
                    # 从存储的邻接矩阵列表中获取矩阵
                    start_matrices.append(matrices_initial[idx])
        else:
            # 后续迭代使用前一轮的高分样本作为起点
            start_matrices = [sample['adjacency'] for sample in all_best_samples[-args.num_similar_samples:]]
            print(f"使用上一轮的 {len(start_matrices)} 个高分样本作为种子")
        
        # 使用选定的图结构作为起点，生成新的样本
        print(f"\n使用 {len(start_matrices)} 个图结构作为起点生成新样本...")
        X_new, y_new, matrices_new = generate_samples_from_matrices(
            args,
            gflownet,
            env,
            params,
            start_matrices,
            num_samples=args.samples_per_iteration
        )

        # 评估新样本的得分
        print(f"新生成样本: {len(X_new)}个, 分数范围: [{np.min(y_new):.4f}, {np.max(y_new):.4f}], 平均值: {np.mean(y_new):.4f}")

        # 准备全局唯一结构表 - 合并两个去重步骤
        global_matrix_keys = {}
        for sample in all_best_samples:
            matrix_key = str(np.array(sample['adjacency']).flatten())
            global_matrix_keys[matrix_key] = True

        # 找出本次迭代的最佳且结构不同的样本
        sorted_indices = np.argsort(y_new)[::-1]  # 按分数从高到低排序
        iter_best_X = []
        iter_best_y = []
        iter_best_matrices = []
        added_count = 0

        # 从分数最高的开始遍历，选择结构互不相同的top_k个样本
        for idx in sorted_indices:
            # 高效生成哈希键 - 只做一次转换
            matrix_arr = np.array(matrices_new[idx])
            matrix_key = str(matrix_arr.flatten())
            
            # 如果这个结构之前没见过，添加到结果中
            if matrix_key not in global_matrix_keys:
                # 同时添加到迭代结果和全局结构表
                global_matrix_keys[matrix_key] = True
                added_count += 1
                
                # 保存当前样本信息
                x = X_new[idx]
                y = y_new[idx]
                matrix = matrices_new[idx]
                
                # 添加到当前迭代的最佳列表
                iter_best_X.append(x)
                iter_best_y.append(y)
                iter_best_matrices.append(matrix)
                
                # 更新全局最优
                if y > best_score_overall:
                    best_score_overall = y
                    best_sample_overall = {
                        'iteration': iteration + 1,
                        'rank': len(iter_best_X),
                        'score': float(y),
                        'embedding': x,
                        'adjacency': matrix
                    }
                
                # 添加到所有最佳样本列表
                all_best_samples.append({
                    'iteration': iteration + 1,
                    'rank': len(iter_best_X),
                    'score': float(y),
                    'embedding': x,
                    'adjacency': matrix
                })
                
                # 如果已收集足够的唯一结构，停止
                if len(iter_best_X) >= args.top_k:
                    break

        # 如果找不到足够多的唯一结构，输出警告
        if len(iter_best_X) < args.top_k:
            print(f"警告: 只找到 {len(iter_best_X)} 个唯一结构，少于请求的 {args.top_k} 个")

        # 转换为正确的数组格式 - 确保只执行一次转换
        iter_best_X = np.array(iter_best_X)
        iter_best_y = np.array(iter_best_y)

        print(f"选择了 {len(iter_best_X)} 个结构各异的高分样本，得分范围: [{np.min(iter_best_y):.4f}, {np.max(iter_best_y):.4f}]")
        print(f"当前迭代中添加了 {added_count} 个新的唯一结构")

        # 按分数排序
        all_best_samples.sort(key=lambda x: x['score'], reverse=True)

        # 限制保存的样本数量
        if len(all_best_samples) > args.max_saved_samples:
            all_best_samples = all_best_samples[:args.max_saved_samples]
    
    # 步骤8: 汇总所有迭代的结果
    print("\n步骤 7/7: 生成最终结果报告...")
    
    # 按分数排序
    all_best_samples.sort(key=lambda x: x['score'], reverse=True)
    
    # 截取前N个最佳样本
    final_best_samples = all_best_samples[:args.final_top_n]


    # 转换NumPy数组为Python列表，以便JSON序列化
    for sample in final_best_samples:
        if 'embedding' in sample and isinstance(sample['embedding'], (np.ndarray, jnp.ndarray)):
            sample['embedding'] = sample['embedding'].tolist()
        if 'adjacency' in sample and isinstance(sample['adjacency'], (np.ndarray, jnp.ndarray)):
            sample['adjacency'] = sample['adjacency'].tolist()
    
    optimization_log["final_best_samples"] = final_best_samples
    
    # 确保所有数值都是JSON可序列化的
    for key in ["best_score_per_iter", "mean_score_per_iter"]:
        optimization_log[key] = [float(x) if isinstance(x, (np.number, jnp.number)) else x 
                                for x in optimization_log[key]]
    
    # 保存最终结果
    with open(os.path.join(args.output_dir, 'final_best_samples.pkl'), 'wb') as f:
        pickle.dump(final_best_samples, f)
    
    # 保存最终优化日志
    with open(os.path.join(args.output_dir, 'optimization_log.json'), 'w') as f:
        json.dump(optimization_log, f, indent=2)
    
    # 输出最终最佳样本信息
    print("\n全局最佳样本信息:")
    for i, sample in enumerate(final_best_samples):
        # 确保邻接矩阵是NumPy数组
        try:
            if isinstance(sample['adjacency'], list):
                adj_matrix = np.asarray(sample['adjacency'], dtype=float)
            else:
                adj_matrix = sample['adjacency']
                
            score = sample['score']
            iteration = sample['iteration']
            
            # 计算边数和平均权重
            edge_count = np.sum(adj_matrix > 0)
            avg_weight = np.mean(adj_matrix[adj_matrix > 0]) if edge_count > 0 else 0
            
            print(f"全局第{i+1}名 (得分: {score:.4f}, 迭代: {iteration}):")
            print(f"  边数: {edge_count}")
            print(f"  平均权重: {avg_weight:.4f}")
        except Exception as e:
            print(f"全局第{i+1}名 (得分: {sample['score']:.4f}, 迭代: {sample['iteration']}):")
            print(f"  无法处理邻接矩阵: {str(e)}")
    
    # 打印最终结果
    print("\n" + "="*80)
    print("最终结果")
    print("="*80)
    print(f"\n找到的前 {len(final_best_samples)} 个最高得分样本:")
    for i, sample in enumerate(final_best_samples):
        print(f"样本 {i+1}: 得分 = {sample['score']:.6f} (迭代 {sample['iteration']})")
    
    print(f"\n所有结果已保存至: {args.output_dir}")

def generate_samples_with_embeddings(args, gflownet, env, params, num_samples=1000):
    """使用GFlowNet生成样本并返回表征、分数和邻接矩阵"""
    print(f"使用GFlowNet生成 {num_samples} 个样本...")
    
    normalization = jnp.array(1.)
    # 初始化随机种子
    key = jax.random.PRNGKey(args.seed)
    
    # 准备数据
    X = []  # 表征
    y = []  # 真实分数
    matrices = []  # 邻接矩阵
    
    # 分批生成样本
    samples_collected = 0
    max_steps = num_samples * 100  # 设置最大步数限制
    step_count = 0
    
    observations = env.reset()
    observations['graph'] = to_graphs_tuple(observations['adjacency'])
    
    # 使用进度条
    with tqdm(total=num_samples, desc="生成样本", ncols=100) as pbar:
        while samples_collected < num_samples and step_count < max_steps:
            # 获取当前状态信息
            score = observations['score']
            adjacency = observations['adjacency']
            
            # 执行动作
            actions, key, _, embedding = gflownet.act(params.online, key, observations, 1., normalization)
            observations, _, dones, _ = env.step(np.asarray(actions))
            
            # 确保下一步图信息可用
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            
            # 转换embedding为jax格式
            embedding = jnp.array(embedding)
            
            # 处理每个环境的样本
            for i in range(len(dones)):
                if dones[i]:
                    # 只添加完成的样本
                    X.append(embedding[i])
                    y.append(score[i])
                    matrices.append(adjacency[i])
                    samples_collected += 1
                    pbar.update(1)
                    
                    # 如果已收集足够样本，提前退出
                    if samples_collected >= num_samples:
                        break
            
            step_count += 1
            
            # 每隔一定步数打印进度
            if step_count % 100 == 0:
                print(f"已执行 {step_count} 步，收集 {samples_collected}/{num_samples} 个样本")
    
    # 转换为numpy/jax数组
    X = jnp.stack(X[:num_samples])
    y = np.array(y[:num_samples])
    
    print(f"成功生成 {X.shape[0]} 个样本，用了 {step_count} 步")
    print(f"分数范围: [{y.min():.4f}, {y.max():.4f}], 平均值: {y.mean():.4f}")
    
    return X, y, matrices

def generate_samples_from_matrices(args, gflownet, env, params, target_matrices, num_samples=1000):
    """从空图开始，先构建到目标邻接矩阵，然后继续生成直到完成状态
    
    参数:
        args: 参数配置
        gflownet: GFlowNet模型
        env: 环境实例
        params: 模型参数
        target_matrices: 目标邻接矩阵列表
        num_samples: 要生成的样本总数
    
    返回:
        X: 样本表征
        y: 样本分数
        matrices: 样本邻接矩阵
    """
    print(f"从 {len(target_matrices)} 个目标邻接矩阵生成 {num_samples} 个样本...")
    
    # 初始化数据收集容器
    X = []  # 表征
    y = []  # 真实分数
    matrices = []  # 邻接矩阵
    
    # 每个目标矩阵生成多少样本
    samples_per_matrix = max(1, num_samples // len(target_matrices))
    
    # 为每个目标矩阵生成样本
    for matrix_idx, target_matrix in enumerate(target_matrices):

        print(f"处理目标矩阵 {matrix_idx+1}/{len(target_matrices)}...")
        
        # 计算需要为此矩阵生成的样本数
        samples_to_generate = min(samples_per_matrix, num_samples - len(X))
        if samples_to_generate <= 0:
            break
            
        # 记录当前矩阵的样本收集数量
        samples_collected = 0
        attempts = 0
        max_attempts = 10  # 每个矩阵尝试的最大次数
        
        # 使用进度条
        with tqdm(total=samples_to_generate, desc=f"矩阵 {matrix_idx+1}", ncols=100) as pbar:
            while samples_collected < samples_to_generate and attempts < max_attempts:
                # 重置环境，获取初始空图
                observations = env.reset()
                observations['graph'] = to_graphs_tuple(observations['adjacency'])
                
                # 跟踪当前构建过程的矩阵和状态
                current_matrices = observations['adjacency'].copy()  # 所有环境的当前矩阵
                build_steps = 0
                max_build_steps = 200  # 最大构建步骤数
                reached_targets = [False] * env.num_envs  # 每个环境是否达到目标矩阵
                
                # 初始化随机种子供GFlowNet使用
                key = jax.random.PRNGKey(args.seed + matrix_idx + attempts)
                normalization = jnp.array(1.)
                
                # 构建过程
                while build_steps < max_build_steps:
                    # 为每个环境确定动作
                    actions = []
                    for env_idx in range(env.num_envs):
                        if reached_targets[env_idx]:
                            # 已达到目标矩阵，使用GFlowNet策略探索
                            # 动作会在后面统一生成
                            pass
                        else:
                            # 判断当前环境是否已达到目标矩阵
                            if np.array_equal(current_matrices[env_idx], target_matrix):

                                reached_targets[env_idx] = True
                                #print(f"环境 {env_idx} 已达到目标矩阵结构，切换到GFlowNet探索模式...")
                    
                    # 在执行动作前获取当前状态的嵌入和分数
                    observations['graph'] = to_graphs_tuple(observations['adjacency'])
                    _, _, _, current_embeddings = gflownet.act(params.online, jax.random.PRNGKey(0), observations, 0., normalization)
                    current_scores = observations.get('score', np.zeros(env.num_envs))
                    
                    # 确定下一步动作
                    if all(reached_targets):
                        #print("所有环境都已达到目标矩阵，使用GFlowNet生成动作...")
                        # 所有环境都达到了目标矩阵，使用GFlowNet生成动作
                        observations['graph'] = to_graphs_tuple(observations['adjacency'])
                        actions, key, _, _ = gflownet.act(params.online, key, observations, 0.8, normalization)
                    else:
                        # 有环境尚未达到目标矩阵，逐个确定动作
                        actions = np.zeros(env.num_envs, dtype=np.int32)
                        for env_idx in range(env.num_envs):
                            if reached_targets[env_idx]:
                                # 已达到目标，使用GFlowNet策略
                                observations['graph'] = to_graphs_tuple(observations['adjacency'])
                                env_actions, key, _, _ = gflownet.act(params.online, key, observations, 0.8, normalization)
                                actions[env_idx] = env_actions[env_idx]
                            else:
                                # 未达到目标，确定性构建
                                actions[env_idx] = determine_next_action(current_matrices[env_idx], target_matrix, env)
                    
                    # 保存执行动作前的邻接矩阵（对于收集完成样本很重要）
                    pre_action_matrices = current_matrices.copy()
                                    
                    # 执行动作
                    next_observations, rewards, dones, _ = env.step(np.asarray(actions))

                    #print(next_observations['adjacency'])
                    # 检查每个环境是否完成，收集完成的样本
                    for env_idx in range(env.num_envs):
                        if dones[env_idx]:
                            # 使用动作执行前的嵌入、分数和矩阵（因为环境可能已经重置）
                            X.append(jnp.array(current_embeddings[env_idx]))
                            y.append(current_scores[env_idx] + rewards[env_idx])  # 添加最后一步的奖励
                            matrices.append(pre_action_matrices[env_idx].copy())
                            samples_collected += 1
                            pbar.update(1)
                            # 重要修改: 当环境完成并重置时，对应的reached_targets也应重置
                            reached_targets[env_idx] = False
                            # 如果已收集足够样本，提前退出
                            if samples_collected >= samples_to_generate:
                                break
                    
                    # 如果已收集足够样本，提前退出
                    if samples_collected >= samples_to_generate:
                        break
                    
                    # 更新观察和当前矩阵
                    observations = next_observations
                    observations['graph'] = to_graphs_tuple(observations['adjacency'])
                    current_matrices = observations['adjacency'].copy()
                    
                    build_steps += 1
                
                attempts += 1
                
                # 如果未能收集到足够样本，给出警告
                if samples_collected < samples_per_matrix and attempts >= max_attempts:
                    print(f"警告: 矩阵 {matrix_idx+1} 未能收集到足够样本，只收集到 {samples_collected}/{samples_per_matrix}")
    
    # 确保我们有足够的样本
    if len(X) < num_samples:
        print(f"警告: 只生成了 {len(X)} 个样本，少于请求的 {num_samples} 个")
    
    # 如果生成的样本超过请求数量，截取请求数量
    if len(X) > num_samples:
        X = X[:num_samples]
        y = y[:num_samples]
        matrices = matrices[:num_samples]
    
    # 转换为jax/numpy数组
    X = jnp.stack(X) if X else jnp.array([])
    y = np.array(y)
    

    
    if len(X) > 0:
        print(f"成功生成 {X.shape[0]} 个样本")
        print(f"分数范围: [{y.min():.4f}, {y.max():.4f}], 平均值: {y.mean():.4f}")
    else:
        print("未能生成任何样本")
    
    return X, y, matrices

def determine_next_action(current_matrix, target_matrix, env):
    """确定从当前矩阵到目标矩阵的下一步操作
    
    参数:
        current_matrix: 当前邻接矩阵
        target_matrix: 目标邻接矩阵
        env: 环境实例，用于确定操作格式
        
    返回:
        action: 下一步操作，格式为整数值(i*num_variables + j)
    """
    # 获取矩阵的维度（节点数）
    num_variables = current_matrix.shape[0]
    
    # 找出当前矩阵和目标矩阵之间的差异
    for i in range(num_variables):
        for j in range(num_variables):
            # 如果目标有这条边但当前没有，添加边(i,j)
            if target_matrix[i, j] > 0 and current_matrix[i, j] == 0:
                # 转换为环境接受的动作格式: source*num_variables + target
                return i * num_variables + j
    
    # 如果没有找到差异（已经相同或只需要移除边），这种情况不应该出现
    # 因为我们在外部逻辑中已经检查了是否达到目标矩阵
    num_variables = current_matrix.shape[0]
    print("警告: 未找到当前矩阵与目标矩阵的差异，但尚未检测到已达到目标")
    return num_variables * num_variables

def get_adjacency_matrix(env, sample_idx, matrices=None):
    """
    获取样本的邻接矩阵
    
    参数:
        env: 环境实例
        sample_idx: 样本索引
        matrices: 可选，预先存储的邻接矩阵列表
        
    返回:
        adjacency_matrix: 邻接矩阵
    """
    if matrices is not None and sample_idx < len(matrices):
        # 如果提供了预先存储的矩阵列表，直接从中获取
        return matrices[sample_idx]
    else:
        # 否则尝试从环境中重置并获取
        # 注意：这只在sample_idx小于num_envs时有效
        observations = env.reset()
        if sample_idx < len(observations['adjacency']):
            return observations['adjacency'][sample_idx]
        else:
            raise ValueError(f"索引 {sample_idx} 超出范围，环境只有 {len(observations['adjacency'])} 个并行环境。"
                           f"请提供预先存储的矩阵列表或减少索引值。")

def set_adjacency_matrix(env, observations, adjacency_matrix):
    """设置环境的邻接矩阵"""
    # 这个函数需要根据实际环境实现来设置邻接矩阵
    # 这里提供一个简单的实现示例
    observations['adjacency'] = np.array([adjacency_matrix] * env.num_envs)
    return observations

if __name__ == "__main__":
    parser = ArgumentParser(description='从GFlowNet训练数据训练代理模型或使用代理模型优化')
    
    # 添加子命令解析器
    subparsers = parser.add_subparsers(help='操作模式', dest='mode')
    
    # 训练代理模型的子命令
    train_parser = subparsers.add_parser('train', help='训练代理模型')
    
    # 数据路径
    train_parser.add_argument('--gflownet_model_path', type=str, required=True,
                        help='GFlowNet模型参数文件路径')
    train_parser.add_argument('--output_dir', type=str, default='output/proxy',
                        help='代理模型输出目录路径')
    
    # 环境参数
    train_parser.add_argument('--num_envs', type=int, default=8,
                        help='并行环境数量')
    train_parser.add_argument('--scorer_kwargs', type=str, default='{}',
                        help='评分器参数')
    train_parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                        help='图先验分布')
    train_parser.add_argument('--prior_kwargs', type=str, default='{}',
                        help='图先验参数')
    
    # 训练参数
    train_parser.add_argument('--num_samples', type=int, default=1000,
                        help='用于训练的样本数量')
    train_parser.add_argument('--num_epochs', type=int, default=100,
                        help='训练轮数')
    train_parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    train_parser.add_argument('--proxy_hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='代理模型隐藏层维度')
    train_parser.add_argument('--proxy_lr', type=float, default=1e-4,
                        help='代理模型学习率')
    train_parser.add_argument('--proxy_weight_decay', type=float, default=1e-5,
                        help='代理模型权重衰减')
    train_parser.add_argument('--log_every', type=int, default=10,
                        help='每多少轮记录一次')
    train_parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪阈值，防止梯度爆炸')
    
    # 添加标准化方法参数
    train_parser.add_argument('--normalization_method', type=str, default='standard',
                        choices=['standard', 'minmax', 'log', 'exp'],
                        help='目标值标准化方法: standard (标准化), minmax (最小-最大归一化), log (对数变换), exp (指数放缩)')
    train_parser.add_argument('--norm_scale_factor', type=float, default=10.0,
                        help='标准化后的缩放因子，可以放大损失值，使训练更稳定')
    
    # 学习率调度参数
    train_parser.add_argument('--lr_patience', type=int, default=5,
                        help='学习率调度器的耐心值，连续多少轮无改善后降低学习率')
    train_parser.add_argument('--lr_factor', type=float, default=0.5,
                        help='学习率降低系数，每次降低为原来的多少倍')
    train_parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='最小学习率下限')
    train_parser.add_argument('--lr_threshold', type=float, default=0.001,
                        help='认为有效改善的阈值，相对于最佳损失的比例')
    
    # Weights & Biases参数
    train_parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用Weights & Biases跟踪训练进度')
    train_parser.add_argument('--wandb_project', type=str, default='gfnproxy',
                        help='Weights & Biases项目名称')
    train_parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases运行名称')
    
    # GFlowNet参数
    train_parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber损失的delta值')
    train_parser.add_argument('--update_target_every', type=int, default=1000,
                        help='目标网络更新频率')
    
    # 对比学习参数
    train_parser.add_argument('--contrastive_lambda', type=float, default=0.1,
                        help='对比学习损失的权重系数')
    train_parser.add_argument('--temperature', type=float, default=0.5,
                        help='对比学习的温度参数，控制相似度分布的平滑度')
    
    # 其他参数
    train_parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    train_parser.add_argument('--num_workers', type=int, default=4,
                        help='工作进程数量')
    train_parser.add_argument('--mp_context', type=str, default='spawn',
                        help='多进程上下文')
    
    # 添加基于梯度的优化参数
    train_parser.add_argument('--gradient_steps', type=int, default=50,
                    help='基于梯度优化表征的迭代次数')
    train_parser.add_argument('--gradient_lr', type=float, default=0.01,
                    help='基于梯度优化表征的学习率')
    
    # 优化模式的子命令
    optimize_parser = subparsers.add_parser('optimize', help='使用代理模型优化')
    
    # 数据路径
    optimize_parser.add_argument('--gflownet_model_path', type=str, required=True,
                        help='GFlowNet模型参数文件路径')
    optimize_parser.add_argument('--proxy_model_path', type=str, required=True,
                        help='代理模型参数文件路径')
    optimize_parser.add_argument('--output_dir', type=str, default='output/proxy_optimize',
                        help='优化结果输出目录路径')
    
    # 环境参数
    optimize_parser.add_argument('--num_envs', type=int, default=8,
                        help='并行环境数量')
    optimize_parser.add_argument('--scorer_kwargs', type=str, default='{}',
                        help='评分器参数')
    optimize_parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                        help='图先验分布')
    optimize_parser.add_argument('--prior_kwargs', type=str, default='{}',
                        help='图先验参数')
    
    # 代理模型参数
    optimize_parser.add_argument('--proxy_hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='代理模型隐藏层维度')
    optimize_parser.add_argument('--proxy_lr', type=float, default=1e-4,
                        help='代理模型学习率')
    optimize_parser.add_argument('--proxy_weight_decay', type=float, default=1e-5,
                        help='代理模型权重衰减')
    
    # 优化参数
    optimize_parser.add_argument('--initial_samples', type=int, default=5000,
                        help='用于初始化的样本数量')
    optimize_parser.add_argument('--optimization_iterations', type=int, default=5,
                        help='优化迭代次数')
    optimize_parser.add_argument('--samples_per_iteration', type=int, default=5000,
                        help='每次迭代生成的样本数量')
    optimize_parser.add_argument('--batch_size', type=int, default=64,
                        help='批处理大小')
    optimize_parser.add_argument('--top_k', type=int, default=10,
                        help='每次迭代选择的顶部样本数量')
    optimize_parser.add_argument('--num_similar_samples', type=int, default=5,
                        help='每个高分表征选择的相似样本数量')
    optimize_parser.add_argument('--similarity_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='表征相似度度量方式: cosine (余弦相似度), euclidean (欧氏距离)')
    optimize_parser.add_argument('--max_saved_samples', type=int, default=100,
                        help='最多保存的样本数量')
    optimize_parser.add_argument('--final_top_n', type=int, default=10,
                        help='最终返回的顶部样本数量')
    optimize_parser.add_argument('--diversity_threshold', type=float, default=1e-5,
                      help='样本多样性最小距离阈值')
    optimize_parser.add_argument('--diversity_metric', type=str, default='euclidean',
                      choices=['euclidean', 'cosine'],
                      help='多样性距离度量方式: euclidean (欧氏距离), cosine (余弦距离)')
    optimize_parser.add_argument('--graph_diversity_threshold', type=float, default=0.5,
                      help='图结构多样性阈值，Jaccard相似度高于此值的图被认为相似')
    # GFlowNet参数 
    optimize_parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber损失的delta值')
    optimize_parser.add_argument('--update_target_every', type=int, default=1000,
                        help='目标网络更新频率')
    
    # 其他参数
    optimize_parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    optimize_parser.add_argument('--num_workers', type=int, default=4,
                        help='工作进程数量')
    optimize_parser.add_argument('--mp_context', type=str, default='spawn',
                        help='多进程上下文')
    
    # 图类型参数 - 共享于两个子命令
    for subparser in [train_parser, optimize_parser]:
        graph_subparsers = subparser.add_subparsers(help='图类型', dest='graph')
        
        er_lingauss = graph_subparsers.add_parser('erdos_renyi_lingauss')
        er_lingauss.add_argument('--num_variables', type=int, required=True,
                                help='变量数量')
        er_lingauss.add_argument('--num_edges', type=int, required=True,
                                help='平均边数')
        er_lingauss.add_argument('--num_samples', type=int, required=True,
                                help='样本数量')
        
        sachs_continuous = graph_subparsers.add_parser('sachs_continuous')
        sachs_intervention = graph_subparsers.add_parser('sachs_interventional')
        asia_intervention = graph_subparsers.add_parser('asia_interventional')
        asia_intervention_bic = graph_subparsers.add_parser('asia_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('sachs_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('alarm_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('child_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('hailfinder_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('win95pts_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('formed_custom')
        asia_intervention_bic = graph_subparsers.add_parser('property_custom')
        asia_intervention_bic = graph_subparsers.add_parser('sports_custom')

    args = parser.parse_args()
    
    # 根据模式执行不同的操作
    if args.mode == 'train':
        main(args)
    elif args.mode == 'optimize':
        optimize_with_proxy(args)
    else:
        parser.print_help()