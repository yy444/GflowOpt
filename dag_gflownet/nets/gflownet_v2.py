import jax.numpy as jnp
import haiku as hk
import jraph

from jax import lax, nn
import jax

from dag_gflownet.nets.transformers import TransformerBlock
from dag_gflownet.utils.gflownet import log_policy
from dag_gflownet.utils.jraph_utils import edge_features_to_dense


def gflownet(embed_dim=128, dropout_rate=0.1, init_scale=1.0, logit_clip=5.0, use_layer_norm=True):
    """创建GFlowNet网络，支持自定义嵌入维度
    
    参数:
        embed_dim: 嵌入维度，默认为128
        dropout_rate: Dropout比率，用于正则化，默认0.1
        init_scale: 初始化缩放因子，控制权重初始化幅度，默认1.0
        logit_clip: logits裁剪范围，限制输出幅度，默认5.0
        use_layer_norm: 是否使用层归一化，默认True
    """
    @hk.transform
    def _gflownet(graphs, masks, normalization, is_training=True):
        batch_size, num_variables = masks.shape[:2]
        edge_masks = jnp.ones(graphs.edges.shape, dtype=jnp.float32)

        # 使用更稳定的初始化方法
        node_embeddings = hk.Embed(
            num_variables, 
            embed_dim=embed_dim,
            w_init=hk.initializers.VarianceScaling(init_scale, "fan_in", "truncated_normal")
        )
        edge_embedding = hk.get_parameter(
            'edge_embed', 
            shape=(1, embed_dim),
            init=hk.initializers.VarianceScaling(init_scale, "fan_in", "truncated_normal")
        )
        
        # 增加全局特征维度，使其与嵌入维度匹配
        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),
            globals=jnp.zeros((graphs.n_node.shape[0], embed_dim)),
        )

        # 添加初始层归一化以稳定训练
        if use_layer_norm:
            nodes_normalized = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='initial_node_norm'
            )(graphs.nodes)
            
            edges_normalized = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='initial_edge_norm'
            )(graphs.edges)
            
            graphs = graphs._replace(
                nodes=nodes_normalized,
                edges=edges_normalized
            )

        # 添加安全的残差连接和维度处理
        @jraph.concatenated_args
        def update_node_fn(features):
            # 获取输入特征维度
            feature_dim = features.shape[-1]
            
            # 投影到标准嵌入维度
            x_proj = hk.Linear(embed_dim, name='node_proj')(features)
            
            # 使用Swish激活函数的MLP
            hidden = hk.nets.MLP(
                [embed_dim, embed_dim], 
                activation=jax.nn.swish, 
                name='node'
            )(x_proj)
            
            # 仅在训练时应用Dropout
            if is_training and dropout_rate > 0:
                hidden = hk.dropout(hk.next_rng_key(), dropout_rate, hidden)
            
            # 投影回原始维度以支持残差连接
            output = hk.Linear(feature_dim, name='node_out_proj')(hidden)
            
            # 残差连接
            return features + output 

        @jraph.concatenated_args
        def update_edge_fn(features):
            feature_dim = features.shape[-1]
            
            # 投影到标准嵌入维度
            x_proj = hk.Linear(embed_dim, name='edge_proj')(features)
            
            # 使用Swish激活函数的MLP
            hidden = hk.nets.MLP(
                [embed_dim, embed_dim], 
                activation=jax.nn.swish, 
                name='edge'
            )(x_proj)
            
            # 仅在训练时应用Dropout
            if is_training and dropout_rate > 0:
                hidden = hk.dropout(hk.next_rng_key(), dropout_rate, hidden)
            
            # 投影回原始维度以支持残差连接
            output = hk.Linear(feature_dim, name='edge_out_proj')(hidden)
            
            # 残差连接
            return features + output

        @jraph.concatenated_args
        def update_global_fn(features):
            feature_dim = features.shape[-1]
            
            # 投影到标准嵌入维度
            x_proj = hk.Linear(embed_dim, name='global_proj')(features)
            
            # 使用Swish激活函数的MLP
            hidden = hk.nets.MLP(
                [embed_dim, embed_dim], 
                activation=jax.nn.swish, 
                name='global'
            )(x_proj)
            
            # 仅在训练时应用Dropout
            if is_training and dropout_rate > 0:
                hidden = hk.dropout(hk.next_rng_key(), dropout_rate, hidden)
            
            # 投影回原始维度以支持残差连接
            output = hk.Linear(feature_dim, name='global_out_proj')(hidden)
            
            # 残差连接
            return features + output

        # 创建图神经网络
        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        
        # 第一次GNN传递
        features = graph_net(graphs)
        
        # 中间层归一化
        if use_layer_norm:
            nodes_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='mid_node_norm'
            )(features.nodes)
            
            edges_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='mid_edge_norm'
            )(features.edges)
            
            globals_norm = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='mid_global_norm'
            )(features.globals)
            
            features = features._replace(
                nodes=nodes_norm,
                edges=edges_norm,
                globals=globals_norm
            )
        
        # 第二次GNN传递增强特征表示
        features = graph_net(features)

        # 重塑节点特征用于自注意力
        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        
        # 保存原始特征用于残差连接
        original_node_features = node_features
        original_feature_dim = original_node_features.shape[-1]
        
        # 根据嵌入维度动态调整注意力头数和大小
        key_size = max(16, embed_dim // 8)
        num_heads = max(4, embed_dim // key_size)
        
        # 线性投影生成查询、键、值
        node_features = hk.Linear(embed_dim * 3, name='qkv_projection')(node_features)
        queries, keys, values = jnp.split(node_features, 3, axis=2)

        # 多头自注意力
        attention_output = hk.MultiHeadAttention(
            num_heads=num_heads,
            key_size=key_size,
            w_init_scale=init_scale,
            name='self_attention'
        )(queries, keys, values)
        
        # 仅在训练时应用dropout
        if is_training and dropout_rate > 0:
            attention_output = hk.dropout(hk.next_rng_key(), dropout_rate, attention_output)
        
        # 将注意力输出投影回原始维度以支持残差连接
        attention_projected = hk.Linear(
            original_feature_dim, 
            name='attention_output_proj'
        )(attention_output)
        
        # 注意力残差连接
        node_features = original_node_features + attention_projected

        # 最终层归一化
        if use_layer_norm:
            nodes_final = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='final_node_norm'
            )(node_features)
            
            globals_final = hk.LayerNorm(
                axis=-1, create_scale=True, create_offset=True, name='final_global_norm'
            )(features.globals[:batch_size])
            
            features = features._replace(
                nodes=nodes_final,
                globals=globals_final
            )
        else:
            features = features._replace(
                nodes=node_features,
                globals=features.globals[:batch_size]
            )

        # 生成动作logits
        senders = hk.nets.MLP(
            [embed_dim, embed_dim], 
            activation=jax.nn.swish, 
            name='senders'
        )(features.nodes)
        
        receivers = hk.nets.MLP(
            [embed_dim, embed_dim], 
            activation=jax.nn.swish, 
            name='receivers'
        )(features.nodes)

        # 批量矩阵乘法生成边的logits
        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
        
        # 归一化logits以稳定训练
        logits = logits / jnp.sqrt(float(embed_dim))
        logits = logits.reshape(batch_size, -1)
        
        # 生成停止动作的logits
        hidden_dim = max(32, embed_dim // 2)
        stop = hk.nets.MLP(
            [embed_dim, hidden_dim, 1], 
            activation=jax.nn.swish, 
            name='stop'
        )(features.globals)
        
        # 使用tanh限制logits范围，防止梯度爆炸
        logits = jnp.tanh(logits) * logit_clip
        stop = jnp.tanh(stop) * logit_clip
        
        # 获取全局表示用于对比学习或其他目的
        global_representation = features.globals

        # 返回策略logits和全局表示
        return log_policy(logits * normalization, stop * normalization, masks), global_representation

    return _gflownet