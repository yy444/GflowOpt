import jax.numpy as jnp
import haiku as hk
import optax
import jax

from collections import namedtuple
from functools import partial
from jax import grad, random, vmap, jit
from jax import grad, random, jit, tree_util, lax

from dag_gflownet.nets.gflownet import gflownet
from dag_gflownet.utils.gflownet_v2 import uniform_log_policy, detailed_balance_loss
from dag_gflownet.utils.jnp_utils import batch_random_choice
from dag_gflownet.utils.jraph_utils import to_graphs_tuple


DAGGFlowNetParameters = namedtuple('DAGGFlowNetParameters', ['online', 'target'])
DAGGFlowNetState = namedtuple('DAGGFlowNetState', ['optimizer', 'key', 'steps'])

class DAGGFlowNet:
    def __init__(self, model=None, delta=1., update_target_every=1000, dataset_size=1, batch_size=None, contrastive_lambda=0.1, temperature=0.5):
        if model is None:
            model = gflownet()

        self.model = model
        self.delta = delta
        self.update_target_every = update_target_every
        self.dataset_size = dataset_size
        # 对比学习参数
        self.contrastive_lambda = contrastive_lambda  # 对比损失权重
        self.temperature = temperature  # 温度系数，控制分布的平滑度

        self._optimizer = None


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
        current_rep = current_rep / jnp.sqrt(jnp.sum(current_rep**2, axis=1, keepdims=True) + 1e-8)
        next_rep = next_rep / jnp.sqrt(jnp.sum(next_rep**2, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        sim_matrix = jnp.matmul(current_rep, next_rep.T) / self.temperature
        
        # 创建标签矩阵 - 对角线为正样本
        labels = jnp.eye(batch_size)
        
        # 计算对比损失 (InfoNCE loss)
        # 对角线上为正样本的log概率，其他位置为负样本的log概率
        exp_sim_matrix = jnp.exp(sim_matrix)
        # 避免数值不稳定性
        log_prob = sim_matrix - jnp.log(jnp.sum(exp_sim_matrix, axis=1, keepdims=True))
        
        # 计算正样本损失 (最大化正样本的相似度)
        pos_loss = jnp.sum(labels * log_prob) / batch_size
        
        # 总对比损失 (取负数，因为我们要最大化正样本相似度)
        contrastive_loss = -pos_loss
        
        return contrastive_loss


    def loss(self, params, target_params, key, samples, normalization):
        @partial(jax.vmap, in_axes=0)
        def _loss(log_pi_t, log_pi_tp1, actions, num_edges, scores):
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

            return (loss, logs)
        
        subkey1, subkey2, subkey3 = random.split(key, 3)
        log_pi_t, log_pi_t_global = self.model.apply(params, samples['graph'], samples['mask'], normalization)
        log_pi_tp1, log_pi_tp1_global = self.model.apply(target_params, samples['next_graph'], samples['next_mask'], normalization)

        # 获取图的表示 (使用全局表示作为状态表示)
        current_rep = log_pi_t_global   # 当前状态的表示 [batch_size, embedding_dim]
        next_rep = log_pi_tp1_global    # 下一状态的表示 [batch_size, embedding_dim]
        
        # 计算批次大小
        batch_size = current_rep.shape[0]
        '''
        # 使用轨迹ID进行同轨迹对比学习
        if 'trajectory_ids' in samples:
            # 获取轨迹ID
            trajectory_ids = samples['trajectory_ids']
            # 在同一轨迹内进行对比学习
            cl_loss = self.trajectory_contrastive_loss(current_rep, next_rep, trajectory_ids)
        else:
            # 如果没有轨迹ID信息，退化为标准对比学习
        '''
        cl_loss = self.contrastive_loss(current_rep, next_rep, batch_size)
        
        outputs = _loss(log_pi_t, log_pi_tp1,
            samples['actions'], samples['num_edges'], samples['delta_scores'],)
        balance_loss, logs = tree_util.tree_map(partial(jnp.mean, axis=0), outputs)
        #total_loss = self.contrastive_lambda * cl_loss
        # 合并平衡损失和对比学习损失
        total_loss = balance_loss + self.contrastive_lambda * cl_loss
        
        logs.update({
            'log_pi_t': log_pi_t,
            'error': outputs[1]['error'],  # Leave "error" unchanged
            'balance_loss': balance_loss,
            'contrastive_loss': cl_loss,
            'total_loss': total_loss
        })
        return (total_loss, logs)


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
        current_rep = current_rep / jnp.sqrt(jnp.sum(current_rep**2, axis=1, keepdims=True) + 1e-8)
        next_rep = next_rep / jnp.sqrt(jnp.sum(next_rep**2, axis=1, keepdims=True) + 1e-8)
        
        # 计算相似度矩阵 [batch_size, batch_size]
        sim_matrix = jnp.matmul(current_rep, next_rep.T) / temperature
        
        # 创建轨迹掩码：仅在同一轨迹内计算对比损失
        # mask[i, j] = 1 表示样本i和样本j属于同一轨迹
        trajectory_mask = (trajectory_ids[:, None] == trajectory_ids[None, :]).astype(jnp.float32)
        
        # 创建对角线掩码（自身掩码）
        identity_mask = jnp.eye(current_rep.shape[0], dtype=jnp.float32)
        
        # 对角线为正样本 (当前状态和对应的下一状态)
        positive_mask = identity_mask
        
        # 计算轨迹内的负样本掩码 (同一轨迹内，但不是自身配对的状态)
        intra_negative_mask = trajectory_mask * (1 - identity_mask)
        
        # 计算轨迹外的负样本掩码 (不同轨迹的状态)
        inter_negative_mask = 1 - trajectory_mask
        
        # 对于轨迹外的样本，设置一个极小的相似度值，使其在softmax中的贡献几乎为0
        # 使用掩码将轨迹外样本的相似度设为极小值
        masked_sim_matrix = jnp.where(
            trajectory_mask, 
            sim_matrix,  # 保留同轨迹内的相似度
            -1e9  # 将不同轨迹间的相似度设为极小值
        )
        
        # 计算对比损失 (InfoNCE loss)
        # 对角线上为正样本的log概率
        exp_sim = jnp.exp(masked_sim_matrix)
        
        # 对于每个样本i，计算其与所有样本的相似度的softmax
        # sum_exp用于归一化，只考虑同轨迹内的样本
        sum_exp = jnp.sum(exp_sim, axis=1, keepdims=True)
        
        # 计算正样本（对角线元素）的概率的对数
        pos_exp = jnp.exp(jnp.diag(sim_matrix))
        log_prob = jnp.log(pos_exp / (jnp.sum(exp_sim, axis=1) + 1e-9))
        
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


    @partial(jit, static_argnums=(0,))
    def act(self, params, key, observations, epsilon, normalization):
        masks = observations['mask'].astype(jnp.float32)
        adjacencies = observations['adjacency'].astype(jnp.float32)

        graphs = observations['graph']

        batch_size = adjacencies.shape[0]
        key, subkey1, subkey2 = random.split(key, 3)

        # Get the GFlowNet policy
        '''
        log_pi = vmap(self.model.apply, in_axes=(None, 0, 0))(
            params,
            adjacencies,
            masks
        )
        '''

        log_pi, log_pi_global = self.model.apply(params, graphs, masks, normalization)

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

    @partial(jit, static_argnums=(0,))
    def step(self, params, state, samples, normalization):
        key, subkey = random.split(state.key)

        grads, logs = grad(self.loss, has_aux=True)(params.online, params.target, subkey, samples, normalization)

        # Update the online params
        updates, opt_state = self.optimizer.update(
            grads,
            state.optimizer,
            params.online
        )
        state = DAGGFlowNetState(optimizer=opt_state, key=key, steps=state.steps + 1)
        online_params = optax.apply_updates(params.online, updates)
        if self.update_target_every > 0:
            target_params = optax.periodic_update(
                online_params,
                params.target,
                state.steps,
                self.update_target_every
            )
        else:
            target_params = params.target
        params = DAGGFlowNetParameters(online=online_params, target=target_params)

        return (params, state, logs)

    def init(self, key, optimizer, graph, mask, gflownet_params=None):
        # Set the optimizer
        key, subkey = random.split(key)

        self._optimizer = optax.chain(optimizer, optax.zero_nans())

        if gflownet_params is not None:
            online_params = gflownet_params['params']
        else:
            online_params = self.model.init(subkey, graph, mask, jnp.array(1.))
        params = DAGGFlowNetParameters(
            online=online_params,
            target=online_params
        )
        state = DAGGFlowNetState(
            optimizer=self.optimizer.init(online_params),
            key=key,
            steps=jnp.array(0),
        )
        return (params, state)

    @property
    def optimizer(self):
        if self._optimizer is None:
            raise RuntimeError('The optimizer is not defined. To train the '
                               'GFlowNet, you must call `DAGGFlowNet.init` first.')
        return self._optimizer
