import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pickle
from typing import Dict, Tuple, Any, Optional
from functools import partial
from dag_gflownet.utils.jraph_utils import to_graphs_tuple

class ProxyModel:
    """代理模型，使用GFlowNet的图特征来预测样本得分"""
    
    def __init__(
        self,
        gflownet: Any,  # GFlowNet模型实例
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        初始化代理模型
        
        Args:
            gflownet: GFlowNet模型实例，用于提取图特征
            hidden_dims: 隐藏层维度
            learning_rate: 学习率
            weight_decay: 权重衰减
        """
        self.gflownet = gflownet
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # 定义网络
        def _network(graph_features, is_training=True):
            # 使用GFlowNet提取的图特征
            features = graph_features
            
            # MLP预测得分
            for dim in self.hidden_dims:
                features = hk.Linear(dim)(features)
                features = jax.nn.relu(features)

            
            # 输出单一得分值
            score = hk.Linear(1)(features)
            return score.squeeze(-1)
        
        self.network = hk.transform(_network)
        
        # 初始化优化器
        self.optimizer = optax.adamw(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
    

    
    def init(self, key, batch_size):
        """初始化模型参数"""
        # 创建一个简单的虚拟图用于初始化
        dummy_graph = jnp.zeros((batch_size, 128))  # 5节点的空图
        # 获取GFlowNet特征维度
        dummy_features = jnp.array(dummy_graph)
        # 初始化代理模型参数
        params = self.network.init(key, dummy_features)
        
        return params
    
    def predict(self, params, graph, mask=None):
        """预测样本的得分"""
        # 获取GFlowNet特征
        features = graph
        features = jnp.array(features)      
        # 预测分数
        return self.network.apply(params, None, features, is_training=False)
    
    @partial(jax.jit, static_argnums=(0,))
    def loss_fn(self, params, batch_X, batch_y):
        """计算损失函数"""
        # 获取图和掩码
        graphs = [x for x in batch_X]
        
        # 获取GFlowNet特征

        features =jnp.array(graphs)
        
        # 预测分数
        pred_scores = jax.vmap(lambda f: self.network.apply(params, None, f, is_training=True))(features)
        
        # 均方误差损失
        mse_loss = jnp.mean((pred_scores - batch_y) ** 2)
        
        return mse_loss
    
    def loss_and_grad(self, params, batch_X, batch_y):
        """计算损失和梯度"""
        return jax.value_and_grad(self.loss_fn)(params, batch_X, batch_y)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, params, opt_state, graphs, masks, true_scores):
        """执行一步训练"""
        loss, grads = jax.value_and_grad(self.loss_fn)(params, graphs, masks, true_scores)
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        
        return new_params, new_opt_state, {'loss': loss}

def save_proxy_model(save_path, params):
    """保存代理模型参数"""
    with open(save_path, 'wb') as f:
        pickle.dump(params, f)

def load_proxy_model(load_path):
    """加载代理模型参数"""
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    return params 