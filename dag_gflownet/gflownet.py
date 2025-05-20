import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple
from functools import partial
from jax import grad, random, vmap, jit
from jax import grad, random, jit, tree_util, lax

from dag_gflownet.nets.gflownet_v2 import gflownet
from dag_gflownet.utils.gflownet import uniform_log_policy, detailed_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice
from dag_gflownet.utils.jraph_utils import to_graphs_tuple


DAGGFlowNetParameters = namedtuple('DAGGFlowNetParameters', ['online', 'target'])
DAGGFlowNetState = namedtuple('DAGGFlowNetState', ['optimizer', 'key', 'steps'])

class DAGGFlowNet:
    def __init__(self, model=None, delta=1., update_target_every=1000, dataset_size=1, 
                 batch_size=None, contrastive_lambda=0.01, temperature=0.5, 
                 gradient_clip_value=1.0, embed_dim=128, dropout_rate=0.1, 
                 init_scale=1.0, logit_clip=5.0, use_layer_norm=True):
        """初始化DAG GFlowNet
        
        参数:
            model: GFlowNet模型，如果为None则创建默认模型
            delta: DAG-GFN delta超参数
            update_target_every: 目标网络更新频率
            dataset_size: 数据集大小
            batch_size: 批次大小
            contrastive_lambda: 对比损失权重
            temperature: 对比学习温度参数
            gradient_clip_value: 梯度裁剪阈值
            embed_dim: 嵌入维度，默认为128
            dropout_rate: Dropout比率，默认0.1
            init_scale: 初始化缩放因子，默认1.0
            logit_clip: logits裁剪范围，默认5.0
            use_layer_norm: 是否使用层归一化，默认True
        """
        self.embed_dim = embed_dim
        self.dropout_rate = dropout_rate
        self.init_scale = init_scale
        self.logit_clip = logit_clip
        self.use_layer_norm = use_layer_norm
        
        if model is None:
            model = gflownet(
                embed_dim=embed_dim, 
                dropout_rate=dropout_rate,
                init_scale=init_scale,
                logit_clip=logit_clip,
                use_layer_norm=use_layer_norm
            )

        self.model = model
        self.delta = delta
        self.update_target_every = update_target_every
        self.dataset_size = dataset_size
        # 对比学习参数
        self.contrastive_lambda = contrastive_lambda  # 对比损失权重
        self.temperature = temperature  # 温度系数，控制分布的平滑度
        # 梯度裁剪值
        self.gradient_clip_value = gradient_clip_value

        self._optimizer = None
        self._steps = jnp.array(0)  # 用于跟踪训练步数


    def contrastive_loss(self, current_rep, next_rep, batch_size):
        """计算对比学习损失
        
        参数:
            current_rep: 当前状态的表示, shape: [batch_size, embedding_dim]
            next_rep: 下一个状态的表示, shape: [batch_size, embedding_dim]
            batch_size: 批次大小
            
        返回:
            contrastive_loss: 对比损失
        """
        # 正则化表示向量
        current_rep = current_rep / (jnp.sqrt(jnp.sum(current_rep**2, axis=1, keepdims=True) + 1e-8) + 1e-8)
        next_rep = next_rep / (jnp.sqrt(jnp.sum(next_rep**2, axis=1, keepdims=True) + 1e-8) + 1e-8)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        sim_matrix = jnp.matmul(current_rep, next_rep.T) / self.temperature
        
        # 添加数值稳定性：防止过大和过小的值
        sim_matrix = jnp.clip(sim_matrix, -20.0, 20.0)
        
        # 创建标签矩阵 - 对角线为正样本
        labels = jnp.eye(batch_size)
        
        # 计算InfoNCE loss的稳定版本
        # 使用logsumexp避免数值溢出
        log_prob = sim_matrix - jax.nn.logsumexp(sim_matrix, axis=1, keepdims=True)
        
        # 计算正样本损失 (最大化正样本的相似度)
        pos_loss = jnp.sum(labels * log_prob) / batch_size
        
        # 检查并替换NaN值
        pos_loss = jnp.where(jnp.isnan(pos_loss), 0.0, pos_loss)
        
        # 总对比损失 (取负数，因为我们要最大化正样本相似度)
        contrastive_loss = -pos_loss
        
        return contrastive_loss


    def trajectory_contrastive_loss(self, current_rep, next_rep, trajectory_ids, temperature=None):
        """计算同轨迹对比学习损失
        
        仅在同一轨迹的样本之间计算对比损失，不同轨迹之间不进行对比
        
        参数:
            current_rep: 当前状态的表示 [batch_size, embedding_dim]
            next_rep: 下一状态的表示 [batch_size, embedding_dim]
            trajectory_ids: 样本所属的轨迹ID [batch_size]
            temperature: 温度参数，控制分布的平滑度（如果为None则使用self.temperature）
        
        返回:
            对比学习损失
        """
        if temperature is None:
            temperature = self.temperature
        
        # 正则化表示向量
        current_rep = current_rep / (jnp.sqrt(jnp.sum(current_rep**2, axis=1, keepdims=True) + 1e-8) + 1e-8)
        next_rep = next_rep / (jnp.sqrt(jnp.sum(next_rep**2, axis=1, keepdims=True) + 1e-8) + 1e-8)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        sim_matrix = jnp.matmul(current_rep, next_rep.T) / temperature
        
        # 添加数值稳定性
        sim_matrix = jnp.clip(sim_matrix, -20.0, 20.0)
        
        # 创建轨迹掩码：仅在同一轨迹内计算对比损失
        # mask[i, j] = 1 表示样本i和样本j属于同一轨迹
        trajectory_mask = (trajectory_ids[:, None] == trajectory_ids[None, :]).astype(jnp.float32)
        
        # 创建对角线掩码（自身掩码）
        identity_mask = jnp.eye(current_rep.shape[0], dtype=jnp.float32)
        
        
        # 对于轨迹外的样本，设置一个极小的相似度值，使其在softmax中的贡献几乎为0
        # 使用掩码将轨迹外样本的相似度设为极小值
        masked_sim_matrix = jnp.where(
            trajectory_mask, 
            sim_matrix,  # 保留同轨迹内的相似度
            -1e9  # 将不同轨迹间的相似度设为极小值
        )
        
        # 计算对比损失 (InfoNCE loss)
        exp_sim = jnp.exp(masked_sim_matrix)
        
        # 对于每个样本i，计算其与所有样本的相似度的softmax
        # 用于归一化，只考虑同轨迹内的样本
        sum_exp = jnp.sum(exp_sim, axis=1)
        
        # 获取正样本的相似度 (对角线元素)
        pos_sim = jnp.diag(sim_matrix)
        
        # 计算正样本的log概率，相对于所有可能的样本（包括负样本）
        # 这里的关键是分母包含了所有同轨迹内的样本，包括正样本和负样本
        log_prob = pos_sim - jnp.log(sum_exp + 1e-9)
        
        # 对于没有同轨迹样本的情况，忽略损失计算
        valid_samples = jnp.sum(trajectory_mask, axis=1) > 1  # 至少有一个同轨迹样本（不包括自己）
        log_prob = jnp.where(valid_samples, log_prob, 0.0)
        
        # 计算平均损失
        num_valid_samples = jnp.sum(valid_samples)
        contrastive_loss = jnp.where(
            num_valid_samples > 0,
            -jnp.sum(log_prob) / (num_valid_samples + 1e-9),
            0.0  # 如果没有有效样本，返回0损失
        )
        
        return contrastive_loss


    def loss(self, params, target_params, key, samples, normalization, steps):
        @partial(jax.vmap, in_axes=0)
        def _loss(log_pi_t, log_pi_tp1, actions, num_edges, scores):
            # 计算detailed balance loss前先处理输入以增加数值稳定性
            log_pi_t = jnp.clip(log_pi_t, -30.0, 30.0)
            log_pi_tp1 = jnp.clip(log_pi_tp1, -30.0, 30.0)
            
            # 如果分数过大，可能导致不稳定性，进行缩放
            scores = jnp.clip(scores, -100.0, 100.0)
            
            # Compute the sub-trajectory balance loss
            loss, logs = detailed_balance_loss(
                        log_pi_t,
                        log_pi_tp1,
                        actions,
                        scores,
                        num_edges,
                        normalization=self.dataset_size,
                        delta=self.delta, 
                                                )
            
            # 检查并替换损失中的NaN值
            loss = jnp.where(jnp.isnan(loss), 0.0, loss)
            
            # 限制损失最大值，避免梯度爆炸
            loss = jnp.clip(loss, -100.0, 100.0)

            return (loss, logs)
        
        subkey1, subkey2, subkey3 = random.split(key, 3)
        
        # 尝试计算GFlowNet策略和全局表示
        try:
            # 在训练时传递RNG key，设置is_training=True
            log_pi_t, log_pi_t_global = self.model.apply(params, subkey1, samples['graph'], samples['mask'], normalization, is_training=True)
            log_pi_tp1, log_pi_tp1_global = self.model.apply(target_params, subkey2, samples['next_graph'], samples['next_mask'], normalization, is_training=True)
        except Exception as e:
            # 如果失败，返回一个保守的损失值
            dummy_loss = jnp.array(10.0)
            return dummy_loss, {'error': str(e)}

        # 获取图的表示 (使用全局表示作为状态表示)
        current_rep = log_pi_t_global   # 当前状态的表示 [batch_size, embedding_dim]
        next_rep = log_pi_tp1_global    # 下一状态的表示 [batch_size, embedding_dim]
        
        # 计算批次大小
        batch_size = current_rep.shape[0]
        
        # 使用渐进式对比学习权重调整
        # 随着训练进行，减少对比学习的重要性
        # 根据用户要求，保持对比学习权重恒定，不随训练步数减少
        # adaptive_cl_lambda = self.contrastive_lambda * jnp.exp(-steps / 10000.0)
        # adaptive_cl_lambda = jnp.clip(adaptive_cl_lambda, 0.01, self.contrastive_lambda)
        adaptive_cl_lambda = self.contrastive_lambda  # 保持恒定权重
        
        # 使用轨迹ID进行同轨迹对比学习
        if 'trajectory_ids' in samples:
            # 获取轨迹ID
            trajectory_ids = samples['trajectory_ids']
            # 在同一轨迹内进行对比学习
            cl_loss = self.trajectory_contrastive_loss(current_rep, next_rep, trajectory_ids)
        else:
            # 如果没有轨迹ID信息，退化为标准对比学习
            cl_loss = self.contrastive_loss(current_rep, next_rep, batch_size)
        
        # 检查并替换对比损失中的NaN值
        cl_loss = jnp.where(jnp.isnan(cl_loss), 0.0, cl_loss)
        
        # 限制对比损失大小，避免梯度爆炸
        cl_loss = jnp.clip(cl_loss, 0.0, 10.0)
        
        outputs = _loss(log_pi_t, log_pi_tp1,
            samples['actions'], samples['num_edges'], samples['delta_scores'],)
        balance_loss, logs = tree_util.tree_map(partial(jnp.mean, axis=0), outputs)
        
        # 检查并替换平衡损失中的NaN值
        balance_loss = jnp.where(jnp.isnan(balance_loss), 0.0, balance_loss)
        
        # 合并平衡损失和对比学习损失，使用自适应权重
        total_loss = balance_loss + adaptive_cl_lambda * cl_loss
        
        # 最终限制总损失范围，确保数值稳定性
        total_loss = jnp.clip(total_loss, -100.0, 100.0)
        
        # 记录额外的日志信息
        logs.update({
            'log_pi_t': log_pi_t,
            'error': outputs[1]['error'],  
            'balance_loss': balance_loss,
            'contrastive_loss': cl_loss,
            'cl_lambda': adaptive_cl_lambda,
            'total_loss': total_loss,
            'steps': steps
        })
        return (total_loss, logs)


    def init(self, key, optimizer, graph, mask, gflownet_params=None):
        """初始化GFlowNet
        
        参数:
            key: JAX随机种子
            optimizer: 优化器（可以是None，将使用默认优化器）
            graph: 图结构
            mask: 掩码
            gflownet_params: 预训练参数（可选）
            
        返回:
            (params, state): 初始化的参数和状态
        """
        # Set the optimizer
        key, subkey = random.split(key)
        
        # 如果没有提供优化器，使用默认的优化器配置
        if optimizer is None:
            # 添加梯度裁剪和学习率调度器
            # 创建具有预热和余弦衰减的学习率调度器
            base_lr = 1e-4  # 基础学习率
            min_lr = 1e-6   # 最小学习率
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=0.0,          # 预热开始学习率
                peak_value=base_lr,      # 预热后峰值学习率
                warmup_steps=1000,       # 预热步数
                decay_steps=50000,       # 衰减总步数
                end_value=min_lr,        # 最终最小学习率
            )
            
            # 使用AdamW优化器，添加权重衰减以防止过拟合
            optimizer_chain = [
                optax.clip_by_global_norm(self.gradient_clip_value),  # 梯度裁剪
                optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),      # Adam优化器
                optax.add_decayed_weights(weight_decay=1e-5),         # 权重衰减（类似L2正则化）
                optax.scale_by_schedule(schedule),                     # 学习率调度
                optax.scale(-1.0),                                     # 负号用于最小化
            ]
            
            # 构建优化器链
            self._optimizer = optax.chain(
                *optimizer_chain,
                optax.zero_nans()  # 处理NaN，置零
            )
        else:
            # 使用用户提供的优化器，但仍添加梯度裁剪和NaN处理
            self._optimizer = optax.chain(
                optax.clip_by_global_norm(self.gradient_clip_value),
                optimizer,
                optax.zero_nans()
            )

        if gflownet_params is not None:
            online_params = gflownet_params['params']
        else:
            # 初始化网络参数，使用稳定的初始化方法
            try:
                # 传递RNG键，因为现在模型需要它用于初始化
                key, init_key = random.split(key)
                online_params = self.model.init(init_key, graph, mask, jnp.array(1.), is_training=True)
            except Exception as e:
                print(f"模型初始化失败: {e}")
                # 使用更小的图尝试初始化
                small_graph = jraph.GraphsTuple(
                    nodes=jnp.zeros((1,), dtype=jnp.int32),
                    edges=jnp.zeros((1,), dtype=jnp.int32),
                    globals=jnp.zeros((1,)),
                    receivers=jnp.zeros((1,), dtype=jnp.int32),
                    senders=jnp.zeros((1,), dtype=jnp.int32),
                    n_node=jnp.array([1]),
                    n_edge=jnp.array([1])
                )
                small_mask = jnp.ones((1, 1, 2))
                key, small_init_key = random.split(key)
                online_params = self.model.init(small_init_key, small_graph, small_mask, jnp.array(1.), is_training=True)
            
        # 创建参数，初始时target和online相同
        params = DAGGFlowNetParameters(
            online=online_params,
            target=online_params
        )
        
        # 初始化优化器状态和随机种子
        state = DAGGFlowNetState(
            optimizer=self.optimizer.init(online_params),
            key=key,
            steps=jnp.array(0),
        )
        
        print(f"GFlowNet初始化完成，嵌入维度: {self.embed_dim}, 使用层归一化: {self.use_layer_norm}")
        return (params, state)

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples, normalization):
        key, subkey = random.split(state.key)
        
        # 使用value_and_grad而不是grad，这样我们可以获取损失值和梯度
        (loss_value, logs), grads = jax.value_and_grad(self.loss, has_aux=True)(
            params.online, params.target, subkey, samples, normalization, state.steps)
        
        # 梯度裁剪以增加训练稳定性 - 正确的使用方式
        # 计算全局梯度范数并使用它手动裁剪
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, self.gradient_clip_value / (grad_norm + 1e-6))
        grads = tree_util.tree_map(lambda g: g * scale, grads)

        # 检查梯度中的NaN/Inf，并替换为0
        grads = tree_util.tree_map(
            lambda g: jnp.where(jnp.logical_or(jnp.isnan(g), jnp.isinf(g)), jnp.zeros_like(g), g),
            grads
        )

        # Update the online params
        updates, opt_state = self.optimizer.update(
            grads,
            state.optimizer,
            params.online
        )
        state = DAGGFlowNetState(optimizer=opt_state, key=key, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)
        
        if self.update_target_every > 0:
            # 使用指数移动平均更新目标网络参数，而不是周期性硬更新
            decay = jnp.minimum(state.steps / 1000.0, 0.99)  # 渐进式增加EMA权重
            target_params = tree_util.tree_map(
                lambda target, online: decay * target + (1 - decay) * online,
                params.target, 
                online_params
            )
        else:
            target_params = params.target
            
        params = DAGGFlowNetParameters(online=online_params, target=target_params)

        # 添加额外监控指标
        grad_norm = optax.global_norm(grads)
        logs.update({
            'grad_norm': grad_norm,
            'loss_value': loss_value,
            'decay': decay if self.update_target_every > 0 else 0.0,
            'embed_dim': self.embed_dim,  # 记录当前的嵌入维度
            'large_gradient': grad_norm > 10.0,  # 监控大梯度
        })

        return (params, state, logs)

    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon, normalization):
        masks = observations['mask'].astype(jnp.float32)
        adjacencies = observations['adjacency'].astype(jnp.float32)

        graphs = observations['graph']

        batch_size = adjacencies.shape[0]
        key, subkey1, subkey2, rng_key = random.split(key, 4)

        # 在推理时传递RNG key，设置is_training=False
        log_pi, log_pi_global = self.model.apply(params, rng_key, graphs, masks, normalization, is_training=False)

        # Get uniform policy
        log_uniform = uniform_log_policy(masks)

        # Mixture of GFlowNet policy and uniform policy
        is_exploration = random.bernoulli(
            subkey1, p=1. - epsilon, shape=(batch_size, 1))
        log_pi = jnp.where(is_exploration, log_uniform, log_pi)

        # Sample actions
        actions = batch_random_choice(subkey2, jnp.exp(log_pi), masks)

        logs = {
            'is_exploration': is_exploration.astype(jnp.int32),
        }
        return (actions, key, logs, log_pi_global)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                               'GFlowNet, you must call `DAGGFlowNet.init` first.')
        return self._optimizer
