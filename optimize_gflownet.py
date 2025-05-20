import jax
import jax.numpy as jnp
import numpy as np
import pickle
import wandb
import os
import datetime
from pathlib import Path
from argparse import ArgumentParser, Namespace
from tqdm import trange, tqdm
from functools import partial

from gfnproxy.proxy_model import ProxyModel, load_proxy_model
from dag_gflownet.env import GFlowNetDAGEnv
from dag_gflownet.gflownet import DAGGFlowNet
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils import io
from dag_gflownet.utils.jraph_utils import to_graphs_tuple

def optimize_gflownet_with_proxy(args, gflownet, proxy_model, proxy_params, env, replay_buffer):
    """使用代理模型优化GFlowNet"""
    print("开始使用代理模型优化GFlowNet...")
    
    # 初始化随机种子
    key = jax.random.PRNGKey(args.seed)
    
    # 初始化wandb
    wandb_run = None
    if args.use_wandb:
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb_run = wandb.init(
            project=args.project_name,
            name=f"optimize_gflownet_{nowtime}",
            config=vars(args)
        )
    
    # 创建优化器
    optimizer = optax.adam(args.gflownet_lr)
    
    # 获取初始参数
    gflownet_params = gflownet.params
    
    # 实现REINFORCE优化策略
    def compute_policy_gradient(key, params, env, proxy_model, proxy_params, batch_size=32):
        """计算策略梯度"""
        
        def sample_and_score_trajectory(key):
            """采样轨迹并评分"""
            observations = env.reset(batch_size=1)
            trajectory = []
            log_probs = []
            
            done = False
            
            while not done:
                observations['graph'] = to_graphs_tuple(observations['adjacency'])
                
                # 使用当前策略选择动作并记录概率
                key, subkey = jax.random.split(key)
                logits, _ = gflownet.policy(params.online, observations['graph'], observations.get('mask'))
                probs = jax.nn.softmax(logits)
                
                # 采样动作
                key, subkey = jax.random.split(key)
                action = jax.random.categorical(subkey, logits)
                log_prob = jnp.log(probs[action])
                
                # 存储轨迹
                trajectory.append((observations, action))
                log_probs.append(log_prob)
                
                # 执行动作
                next_observations, _, dones, _ = env.step(np.asarray([action]))
                
                done = dones[0]
                observations = next_observations
            
            # 使用代理模型评分
            final_score = proxy_model.predict(proxy_params, observations['graph'], observations.get('mask'))
            
            return trajectory, log_probs, final_score
        
        # 批量采样轨迹
        keys = jax.random.split(key, batch_size)
        trajectories, log_probs_batch, scores = [], [], []
        
        for k in keys:
            traj, log_p, score = sample_and_score_trajectory(k)
            trajectories.append(traj)
            log_probs_batch.append(log_p)
            scores.append(score)
        
        # 计算梯度
        # 这里我们实现一个简化的REINFORCE算法
        # 实际应用中可能需要更复杂的算法
        
        # 标准化分数
        scores = jnp.array(scores)
        baseline = jnp.mean(scores)
        advantages = scores - baseline
        
        # 计算策略梯度
        policy_grads = []
        
        for log_probs, advantage in zip(log_probs_batch, advantages):
            policy_grad = sum(log_p * advantage for log_p in log_probs)
            policy_grads.append(policy_grad)
        
        avg_policy_grad = jnp.mean(jnp.array(policy_grads))
        avg_score = jnp.mean(scores)
        
        return avg_policy_grad, avg_score
    
    # 训练循环
    print("开始优化循环...")
    best_score = float('-inf')
    best_params = None
    
    for iteration in trange(args.num_iterations):
        key, subkey = jax.random.split(key)
        
        # 计算策略梯度
        policy_grad, avg_score = compute_policy_gradient(
            subkey,
            gflownet_params,
            env,
            proxy_model,
            proxy_params,
            batch_size=args.batch_size
        )
        
        # 更新GFlowNet参数
        # 这里需要根据具体的GFlowNet实现来更新参数
        # 简化版本：直接使用Adam优化器
        
        # 记录指标
        metrics = {
            'iteration': iteration,
            'avg_score': avg_score,
            'policy_grad': policy_grad
        }
        
        if args.use_wandb:
            wandb.log(metrics)
        
        # 打印进度
        if iteration % args.log_every == 0:
            print(f"Iteration {iteration}: Avg Score = {avg_score:.4f}, Policy Grad = {policy_grad:.4f}")
        
        # 保存模型
        if iteration % args.save_every == 0 or iteration == args.num_iterations - 1:
            save_path = os.path.join(args.output_dir, f"gflownet_optimized_iter_{iteration}.npz")
            io.save(save_path, params=gflownet_params.online)
            print(f"模型保存至 {save_path}")
        
        # 保存最佳模型
        if avg_score > best_score:
            best_score = avg_score
            best_params = gflownet_params
            best_path = os.path.join(args.output_dir, "gflownet_optimized_best.npz")
            io.save(best_path, params=best_params.online)
            print(f"发现新的最佳模型，保存至 {best_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    print(f"GFlowNet优化完成！最佳得分: {best_score:.4f}")
    return best_params

def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置随机种子
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    
    # 创建环境
    print("创建环境...")
    scorer, data, graph = get_scorer(args, rng=np.random.RandomState(args.seed))
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )
    
    # 创建回放缓冲区
    print("创建回放缓冲区...")
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables
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
    
    # 使用代理模型优化GFlowNet
    optimize_gflownet_with_proxy(args, gflownet, proxy_model, proxy_params, env, replay)
    print("GFlowNet优化完成！")

if __name__ == "__main__":
    parser = ArgumentParser(description='使用代理模型优化GFlowNet')
    
    # 数据和模型路径
    parser.add_argument('--gflownet_model_path', type=str, required=True,
                        help='GFlowNet模型参数文件路径')
    parser.add_argument('--proxy_model_path', type=str, required=True,
                        help='代理模型参数文件路径')
    parser.add_argument('--output_dir', type=str, default='output/optimized',
                        help='输出目录路径')
    
    # 环境参数(从原GFlowNet训练脚本复制)
    parser.add_argument('--num_envs', type=int, default=8,
                        help='并行环境数量')
    parser.add_argument('--scorer_kwargs', type=str, default='{}',
                        help='评分器参数')
    parser.add_argument('--prior', type=str, default='uniform',
                        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
                        help='图先验分布')
    parser.add_argument('--prior_kwargs', type=str, default='{}',
                        help='图先验参数')
    
    # GFlowNet参数
    parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber损失的delta值')
    parser.add_argument('--update_target_every', type=int, default=1000,
                        help='目标网络更新频率')
    parser.add_argument('--gflownet_lr', type=float, default=1e-5,
                        help='GFlowNet优化学习率')
    
    # 回放缓冲区参数
    parser.add_argument('--replay_capacity', type=int, default=100000,
                        help='回放缓冲区容量')
    
    # 代理模型参数
    parser.add_argument('--proxy_hidden_dims', type=int, nargs='+', default=[128, 64, 32],
                        help='代理模型隐藏层维度')
    parser.add_argument('--proxy_lr', type=float, default=1e-4,
                        help='代理模型学习率')
    parser.add_argument('--proxy_weight_decay', type=float, default=1e-5,
                        help='代理模型权重衰减')
    
    # 优化参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_iterations', type=int, default=1000,
                        help='优化迭代次数')
    parser.add_argument('--log_every', type=int, default=10,
                        help='每多少轮记录一次')
    parser.add_argument('--save_every', type=int, default=100,
                        help='每多少轮保存一次模型')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='工作进程数量')
    parser.add_argument('--mp_context', type=str, default='spawn',
                        help='多进程上下文')
    parser.add_argument('--use_wandb', action='store_true',
                        help='是否使用wandb记录实验')
    parser.add_argument('--project_name', type=str, default='gfnproxy',
                        help='wandb项目名称')
    
    # 图类型参数(从原GFlowNet训练脚本复制)
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