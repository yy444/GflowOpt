# DAG-GFlowNet 训练工具

本目录包含了一些用于辅助训练 DAG-GFlowNet 模型的工具函数和类。

## 学习率调度器 (`lr_scheduler.py`)

提供了动态调整学习率的调度器，以帮助模型更好地收敛：

### 可用的调度器

1. **ReduceLROnPlateau**: 当损失停止改善时，降低学习率
2. **CosineAnnealingLR**: 余弦退火学习率调度策略
3. **DummyLRScheduler**: 固定学习率（不进行调整）

### 使用方法

```python
from dag_gflownet.utils.lr_scheduler import create_lr_scheduler

# 创建学习率调度器
lr_scheduler = create_lr_scheduler(
    scheduler_type='reduce_on_plateau',  # 'reduce_on_plateau', 'cosine', 'none'
    initial_lr=1e-3,
    factor=0.5,              # 学习率降低因子（reduce_on_plateau）
    patience=10,             # 等待轮次（reduce_on_plateau）
    min_lr=1e-6,             # 最小学习率
    threshold=1e-3,          # 改善阈值（reduce_on_plateau）
    T_max=1000,              # 半周期长度（cosine）
    verbose=True             # 是否打印学习率变化
)

# 在训练循环中使用
for iteration in range(num_iterations):
    # 训练步骤...
    loss = train_step()
    
    # 更新学习率（每 N 轮检查一次）
    if iteration % lr_check_freq == 0:
        new_optimizer = lr_scheduler.step(loss, optimizer)
        if new_optimizer is not optimizer:
            optimizer = new_optimizer
            # 更新优化器状态...
```

## 训练工具 (`training_utils.py`)

提供了跟踪训练进度和管理模型检查点的工具：

### LossTracker

用于计算和跟踪训练损失的移动平均：

```python
from dag_gflownet.utils.training_utils import LossTracker

# 创建损失跟踪器
loss_tracker = LossTracker(window_size=10, smooth_factor=0.9)

# 在训练循环中使用
for epoch in range(num_epochs):
    # 训练步骤...
    loss = train_step()
    
    # 更新损失跟踪器
    loss_stats = loss_tracker.update(loss, epoch)
    
    # 获取损失统计
    avg_loss = loss_stats['avg_loss']       # 移动平均损失
    smooth_loss = loss_stats['smooth_loss'] # 平滑损失
    is_best = loss_stats['is_best']         # 是否为最佳损失
    
    if is_best:
        # 保存最佳模型...
```

### CheckpointManager

用于管理模型检查点的保存和加载：

```python
from dag_gflownet.utils.training_utils import CheckpointManager

# 创建检查点管理器
checkpoint_manager = CheckpointManager(
    output_dir='./checkpoints',
    max_to_keep=3  # 最多保留的检查点数量
)

# 保存定期检查点
checkpoint_manager.save_checkpoint(
    step=iteration,
    params=model_params,
    metric=validation_loss,
    prefix='checkpoint'
)

# 保存最佳模型
if is_best_model:
    checkpoint_manager.save_best(
        step=iteration,
        params=model_params,
        metric=validation_loss
    )

# 加载最佳模型
best_params = checkpoint_manager.load_best()
```

## 使用示例

完整的训练流程示例：

```python
import jax
import optax
from dag_gflownet.utils.lr_scheduler import create_lr_scheduler
from dag_gflownet.utils.training_utils import LossTracker, CheckpointManager

# 初始化
key = jax.random.PRNGKey(seed)
optimizer = optax.adam(1e-3)
loss_tracker = LossTracker()
checkpoint_manager = CheckpointManager('./output')

# 创建学习率调度器
lr_scheduler = create_lr_scheduler('reduce_on_plateau', 1e-3)

# 训练循环
for iteration in range(num_iterations):
    # 训练步骤
    loss = train_step()
    
    # 更新损失跟踪器
    loss_stats = loss_tracker.update(loss, iteration)
    
    # 更新学习率
    if iteration % 100 == 0:
        optimizer = lr_scheduler.step(loss_stats['smooth_loss'], optimizer)
    
    # 保存检查点
    if iteration % 1000 == 0:
        checkpoint_manager.save_checkpoint(iteration, params, loss)
    
    # 保存最佳模型
    if loss_stats['is_best']:
        checkpoint_manager.save_best(iteration, params, loss)
``` 