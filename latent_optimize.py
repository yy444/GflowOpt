import jax
import jax.numpy as jnp
import numpy as np
import pickle
import os
import datetime
from pathlib import Path
from tqdm import trange, tqdm
from argparse import ArgumentParser
import optax

from gfnproxy.proxy_model import ProxyModel, load_proxy_model
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils import io
from dag_gflownet.utils.jraph_utils import to_graphs_tuple

def optimize_in_latent_space(args, gflownet, proxy_model, proxy_params, env):
    """在GFlowNet隐空间中使用梯度上升优化"""
    print("开始在隐空间中优化...")
    
    # 初始化随机种子
    key = jax.random.PRNGKey(args.seed)
    
    # 获取GFlowNet的编码器函数（从图到隐空间）
    # 注意：这里我们使用GFlowNet的编码器函数，具体实现可能需要根据您的GFlowNet结构调整
    def encode_to_latent(graph, mask=None):
        """将图编码到隐空间"""
        # 使用GFlowNet的编码器，如果有专门的编码器函数可以直接调用
        # 这里假设gflownet.encoder是可用的函数
        return gflownet.encoder(gflownet.params.online, graph, mask)
    
    # 从隐空间解码到图结构
    # 这个函数在实际应用中需要根据GFlowNet的具体结构来实现
    def decode_from_latent(embedding):
        """从隐空间解码为图结构"""
        # 这个函数需要根据具体的GFlowNet实现来调整
        # 这里提供一个示例实现
        
        # 步骤1：从隐空间表示初始化一个空图
        observations = env.reset(batch_size=1)
        
        # 步骤2：使用隐空间表示指导图的构建
        done = False
        
        while not done:
            # 准备图特征
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            mask = observations.get('mask')
            
            # 使用隐空间表示和当前图状态决定下一步动作
            # 这里需要根据GFlowNet的具体实现调整
            current_embedding = jnp.concatenate([
                embedding,
                encode_to_latent(observations['graph'], mask)
            ])
            
            # 使用隐空间表示预测动作
            logits = gflownet.latent_to_action(gflownet.params.online, current_embedding, mask)
            
            # 确定性选择最高概率的动作
            action = jnp.argmax(logits)
            
            # 执行动作
            next_observations, _, dones, _ = env.step(np.asarray([action]))
            
            done = dones[0]
            observations = next_observations
        
        # 返回最终构建的图
        observations['graph'] = to_graphs_tuple(observations['adjacency'])
        return observations['graph'], observations.get('mask')
    
    # 定义优化目标函数
    def score_function(embedding, proxy_params):
        """计算隐空间表示对应的分数"""
        # 解码隐空间表示为图结构
        graph, mask = decode_from_latent(embedding)
        
        # 使用代理模型评分
        score = proxy_model.predict(proxy_params, graph, mask)
        return score
    
    # 使用JAX的自动微分计算梯度
    value_and_grad_fn = jax.value_and_grad(score_function)
    
    # 实现adam优化器（可选使用其他优化器）
    if args.optimizer == 'adam':
        optimizer = optax.adam(learning_rate=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optax.sgd(learning_rate=args.learning_rate)
    else:
        raise ValueError(f"不支持的优化器: {args.optimizer}")
    
    # 优化过程
    results = []
    
    # 多个起点并行优化
    for i in range(args.num_starting_points):
        print(f"优化起点 {i+1}/{args.num_starting_points}")
        
        # 随机初始化或从已有图采样初始化
        if args.init_strategy == 'random':
            # 随机初始化隐空间表示
            key, subkey = jax.random.split(key)
            current_embedding = jax.random.normal(subkey, (args.embedding_dim,))
            
            # 标准化嵌入向量
            if args.normalize_embedding:
                current_embedding = current_embedding / jnp.linalg.norm(current_embedding)
        
        elif args.init_strategy == 'sample':
            # 从GFlowNet采样图并编码
            observations = env.reset(batch_size=1)
            
            # 生成完整图
            done = False
            while not done:
                observations['graph'] = to_graphs_tuple(observations['adjacency'])
                mask = observations.get('mask')
                
                # 使用GFlowNet策略选择动作
                key, subkey = jax.random.split(key)
                logits, _ = gflownet.policy(gflownet.params.online, observations['graph'], mask)
                
                # 采样动作
                key, subkey = jax.random.split(key)
                action = jax.random.categorical(subkey, logits)
                
                # 执行动作
                next_observations, _, dones, _ = env.step(np.asarray([action]))
                
                done = dones[0]
                observations = next_observations
            
            # 编码为隐空间表示
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            current_embedding = encode_to_latent(
                observations['graph'], 
                observations.get('mask')
            )
        
        # 初始化优化器状态
        opt_state = optimizer.init(current_embedding)
        
        # 记录优化轨迹
        trajectory = [current_embedding]
        scores = []
        
        # 评估初始分数
        initial_score = score_function(current_embedding, proxy_params)
        scores.append(initial_score)
        
        print(f"  初始分数: {initial_score:.4f}")
        
        # 梯度上升优化
        best_embedding = current_embedding
        best_score = initial_score
        
        for step in trange(args.optimization_steps):
            # 计算梯度
            score, grad = value_and_grad_fn(current_embedding, proxy_params)
            
            # 梯度上升（目标是最大化分数）
            updates, opt_state = optimizer.update(-grad, opt_state)  # 负梯度用于最大化
            current_embedding = optax.apply_updates(current_embedding, updates)
            
            # 如果需要，保持嵌入在单位球上
            if args.normalize_embedding:
                current_embedding = current_embedding / jnp.linalg.norm(current_embedding)
            
            # 记录轨迹
            if step % args.save_trajectory_every == 0:
                trajectory.append(current_embedding)
            
            # 评估当前分数
            current_score = score_function(current_embedding, proxy_params)
            scores.append(current_score)
            
            # 更新最佳嵌入
            if current_score > best_score:
                best_score = current_score
                best_embedding = current_embedding
            
            # 打印进度
            if (step + 1) % args.log_every == 0:
                print(f"  步骤 {step+1}: 当前分数 = {current_score:.4f}, 最佳分数 = {best_score:.4f}")
        
        # 记录优化结果
        results.append({
            'final_embedding': current_embedding,
            'best_embedding': best_embedding,
            'final_score': scores[-1],
            'best_score': best_score,
            'trajectory': trajectory,
            'scores': scores,
            'starting_point': i + 1
        })
        
        print(f"  优化起点 {i+1} 完成。最终分数: {scores[-1]:.4f}, 最佳分数: {best_score:.4f}")
    
    # 对结果排序，选择最好的
    results.sort(key=lambda x: x['best_score'], reverse=True)
    
    # 解码最好的隐空间表示为图结构
    best_results = []
    
    print("\n开始解码最佳隐空间表示...")
    for i, result in enumerate(results[:args.top_k]):
        # 解码最佳嵌入为图结构
        graph, mask = decode_from_latent(result['best_embedding'])
        score = proxy_model.predict(proxy_params, graph, mask)
        
        # 记录完整结果
        best_results.append({
            'embedding': result['best_embedding'],
            'graph': graph,
            'mask': mask,
            'score': score,
            'rank': i + 1,
            'optimization_history': {
                'scores': result['scores'],
                'trajectory_samples': result['trajectory']
            }
        })
        
        print(f"最佳表示 #{i+1}: 分数 = {score:.4f}")
    
    # 保存结果
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(args.output_dir, f"latent_optimization_results_{timestamp}.pkl")
    
    with open(output_file, 'wb') as f:
        pickle.dump({
            'best_results': best_results,
            'all_results': results,
            'args': vars(args)
        }, f)
    
    print(f"已将优化结果保存至 {output_file}")
    return best_results

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    
    # 创建环境
    print("创建环境...")
    scorer, data, graph = get_scorer(args, rng=np.random.RandomState(args.seed))
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )
    
    # 加载GFlowNet模型
    print(f"加载GFlowNet模型: {args.gflownet_model_path}")
    gflownet = DAGGFlowNet(
        delta=args.delta,
        update_target_every=args.update_target_every,
        dataset_size=1
    )
    
    # 加载参数
    if os.path.exists(args.gflownet_model_path):
        params = io.load(args.gflownet_model_path)
        gflownet.params = params
        print("GFlowNet模型参数加载成功")
    else:
        raise ValueError(f"GFlowNet模型文件不存在: {args.gflownet_model_path}")
    
    # 加载代理模型
    print(f"加载代理模型: {args.proxy_model_path}")
    
    if os.path.exists(args.proxy_model_path):
        proxy_params = load_proxy_model(args.proxy_model_path)
        
        # 创建代理模型实例
        proxy_model = ProxyModel(
            gflownet=gflownet,
            hidden_dims=args.proxy_hidden_dims,
            learning_rate=args.proxy_lr,
            weight_decay=args.proxy_weight_decay
        )
        
        print("代理模型参数加载成功")
    else:
        raise ValueError(f"代理模型文件不存在: {args.proxy_model_path}")
    
    # 在GFlowNet隐空间中优化
    optimize_in_latent_space(args, gflownet, proxy_model, proxy_params, env)
    print("隐空间优化完成！")

if __name__ == "__main__":
    parser = ArgumentParser(description='在GFlowNet隐空间中优化表示')
    
    # 数据和模型路径
    parser.add_argument('--gflownet_model_path', type=str, required=True,
                        help='GFlowNet模型参数文件路径')
    parser.add_argument('--proxy_model_path', type=str, required=True,
                        help='代理模型参数文件路径')
    parser.add_argument('--output_dir', type=str, default='output/latent_optimization',
                        help='输出目录路径')
    
    # 隐空间优化参数
    parser.add_argument('--embedding_dim', type=int, required=True,
                        help='隐空间表示维度')
    parser.add_argument('--init_strategy', type=str, default='random',
                        choices=['random', 'sample'],
                        help='隐空间表示初始化策略')
    parser.add_argument('--num_starting_points', type=int, default=10,
                        help='随机初始化的起点数量')
    parser.add_argument('--optimization_steps', type=int, default=1000,
                        help='每个起点的优化步数')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='优化学习率')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'sgd'],
                        help='优化器类型')
    parser.add_argument('--normalize_embedding', action='store_true',
                        help='是否在每步后标准化嵌入向量')
    parser.add_argument('--top_k', type=int, default=5,
                        help='保留前K个最高分的优化结果')
    parser.add_argument('--log_every', type=int, default=100,
                        help='每多少步记录一次')
    parser.add_argument('--save_trajectory_every', type=int, default=10,
                        help='每多少步保存一次轨迹')
    
    # 环境参数
    parser.add_argument('--num_envs', type=int, default=1,
                        help='并行环境数量')
    parser.add_argument('--scorer_kwargs', type=str, default='{}',
                        help='评分器参数')
    parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                        help='图先验分布')
    
    # GFlowNet参数
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber损失的delta值')
    parser.add_argument('--update_target_every', type=int, default=1000,
                        help='目标网络更新频率')
    
    # 代理模型参数
    parser.add_argument('--proxy_hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='代理模型隐藏层维度')
    parser.add_argument('--proxy_lr', type=float, default=1e-4,
                        help='代理模型学习率')
    parser.add_argument('--proxy_weight_decay', type=float, default=1e-5,
                        help='代理模型权重衰减')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='工作进程数量')
    parser.add_argument('--mp_context', type=str, default='spawn',
                        help='多进程上下文')
    
    # 图类型参数
    subparsers = parser.add_subparsers(help='图类型', dest='graph')
    
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
                            help='变量数量')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
                            help='平均边数')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
                            help='样本数量')
    
    sachs_continuous = subparsers.add_parser('sachs_continuous')
    sachs_intervention = subparsers.add_parser('sachs_interventional')
    asia_intervention = subparsers.add_parser('asia_interventional')
    asia_intervention_bic = subparsers.add_parser('asia_interventional_bic')
    
    args = parser.parse_args()
    
    main(args) 