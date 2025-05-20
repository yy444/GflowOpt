import jax.numpy as jnp
import haiku as hk
import jraph

from jax import lax, nn
import jax

from dag_gflownet.nets.transformers import TransformerBlock
from dag_gflownet.utils.gflownet import log_policy
from dag_gflownet.utils.jraph_utils import edge_features_to_dense


def gflownet():
    
    @hk.transform
    def _gflownet(graphs, masks, normalization, is_training=True):  # 添加 is_training 参数
        batch_size, num_variables = masks.shape[:2]
        edge_masks = jnp.ones(graphs.edges.shape, dtype=jnp.float32)

        # Embedding of the nodes & edges
        node_embeddings = hk.Embed(
            num_variables,
            embed_dim=258,
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")  # 使用 VarianceScaling 初始化
        )
        edge_embedding = hk.get_parameter(
            'edge_embed',
            shape=(1, 258),
            init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")  # 使用 VarianceScaling 初始化
        )

        graphs = graphs._replace(
            nodes=node_embeddings(graphs.nodes),
            edges=jnp.repeat(edge_embedding, graphs.edges.shape[0], axis=0),
            globals=jnp.zeros((graphs.n_node.shape[0], 128)),  # 增加全局特征维度
        )

        # 在输入 embedding 后添加 LayerNorm
        graphs = graphs._replace(
            nodes=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='input_node_norm')(graphs.nodes),
            edges=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='input_edge_norm')(graphs.edges),
        )

        # Define graph network updates
        @jraph.concatenated_args
        def update_node_fn(features):
            hidden = hk.nets.MLP([128, 128], name='node')(features)
            if is_training:  # 训练时使用 dropout
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)  # 添加 dropout
            return hidden

        @jraph.concatenated_args
        def update_edge_fn(features):
            hidden = hk.nets.MLP([128, 128], name='edge')(features)
            if is_training:
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)
            return hidden

        @jraph.concatenated_args
        def update_global_fn(features):
            hidden = hk.nets.MLP([128, 128], name='global')(features)
            if is_training:
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)
            return hidden

        graph_net = jraph.GraphNetwork(
            update_edge_fn=update_edge_fn,
            update_node_fn=update_node_fn,
            update_global_fn=update_global_fn,
        )
        features = graph_net(graphs)

        # Reshape the node features, and project into keys, queries & values
        node_features = features.nodes[:batch_size * num_variables]
        node_features = node_features.reshape(batch_size, num_variables, -1)
        node_features = hk.Linear(128 * 3, name='projection')(node_features)

        queries, keys, values = jnp.split(node_features, 3, axis=2)

        # Self-attention layer
        node_features = hk.MultiHeadAttention(
            num_heads=4,
            key_size=32,
            w_init_scale=1.0,  # 降低 w_init_scale
            name='self_attention'
        )(queries, keys, values)

        # 在 attention 后添加 LayerNorm
        node_features = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='attention_norm')(node_features)


        # Replace the node & global features
        features = features._replace(
            nodes=node_features,
            globals=hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name='global_norm')(features.globals[:batch_size])
        )

        senders = hk.nets.MLP(
            [128, 258, 128],
            name='senders',
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")  # 使用 VarianceScaling 初始化
        )(features.nodes)
        senders = jnp.tanh(senders)  * 2.5 # 对 senders 进行激活
        receivers = hk.nets.MLP(
            [128, 258, 128],
            name='receivers',
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")  # 使用 VarianceScaling 初始化
        )(features.nodes)
        receivers = jnp.tanh(receivers) * 2.5
        
        logits = lax.batch_matmul(senders, receivers.transpose(0, 2, 1))
        logits = logits / jnp.sqrt(128.0) # 对 logits 进行缩放
        logits = logits.reshape(batch_size, -1)
        stop = hk.nets.MLP(
            [128, 64, 1],  # 增加 stop MLP 的层数
            name='stop',
            w_init=hk.initializers.VarianceScaling(1.0, "fan_in", "truncated_normal")  # 使用 VarianceScaling 初始化
        )(features.globals)
        stop = jnp.tanh(stop) * 2.5  # 对 stop logits 进行裁剪

        return log_policy(logits * normalization, stop * normalization, masks), features.globals

    return _gflownet