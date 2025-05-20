import jax.numpy as jnp
import numpy as np
import optax
import networkx as nx
import pickle
import jax
import wandb
import datetime
import json
from collections import deque
import os
import time

from tqdm import trange
from numpy.random import default_rng
from argparse import Namespace

from dag_gflownet.env_v4 import GFlowNetDAGEnv
from dag_gflownet.gflownet_c_b import DAGGFlowNet
from dag_gflownet.utils.replay_buffer import ReplayBuffer
from dag_gflownet.utils.factories import get_scorer
from dag_gflownet.utils.gflownet_v2 import posterior_estimate
from dag_gflownet.utils.metrics import expected_shd, expected_edges, threshold_metrics
from dag_gflownet.utils import io
from dag_gflownet.utils.jraph_utils import to_graphs_tuple
# 导入学习率调度器和训练工具
from dag_gflownet.utils.lr_scheduler import create_lr_scheduler
from dag_gflownet.utils.training_utils import LossTracker, CheckpointManager
# 导入训练状态管理模块
from dag_gflownet.utils.training_state import TrainingStateSaver, save_training_state, load_training_state

def main(args):
    # 检查是否从训练状态恢复
    training_info = None
    if args.resume_state_dir is not None:
        print(f"尝试从训练状态目录恢复: {args.resume_state_dir}")
        try:
            # 使用新的加载函数
            training_info = load_training_state(args.resume_state_dir, args.resume_timestamp)
            
            # 设置恢复参数
            args.pretrained_model = training_info['model_path']
            args.load_replay_buffer = training_info['replay_path']
            args.resume_iteration = training_info['iteration']
            
            # 可选：设置固定探索率
            if args.fixed_epsilon is None:
                args.fixed_epsilon = 1.0 - training_info.get('epsilon', 0.9)  # 默认使用0.9作为epsilon
                print(f"使用恢复的探索率: {args.fixed_epsilon}")
            
            # 可选：初始化学习率为恢复的值
            if hasattr(args, 'lr') and args.lr is None:
                args.lr = training_info.get('lr', 1e-4)  # 默认使用1e-4作为学习率
                print(f"使用恢复的学习率: {args.lr}")
        except Exception as e:
            print(f"警告: 无法从训练状态恢复: {e}")
            print("将继续使用命令行参数进行训练")
    
    rng = default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)

    # Create the environment
    scorer, data, graph = get_scorer(args, rng=rng)
    env = GFlowNetDAGEnv(
        num_envs=args.num_envs,
        scorer=scorer,
        num_workers=args.num_workers,
        context=args.mp_context
    )

    # Create the replay buffer
    replay = ReplayBuffer(
        args.replay_capacity,
        num_variables=env.num_variables
    )
    
    # 加载回放缓冲区（如果指定）
    if args.load_replay_buffer is not None and os.path.exists(args.load_replay_buffer):
        print(f"加载回放缓冲区: {args.load_replay_buffer}")
        replay = ReplayBuffer.load(args.load_replay_buffer)
        print(f"回放缓冲区加载成功，包含 {len(replay)} 个样本")

    # 处理对比学习参数
    contrastive_lambda = 0.0 if getattr(args, 'disable_contrastive', False) else getattr(args, 'contrastive_lambda', 0.1)
    normalization = jnp.array(1.)
    #normalization = jnp.array(data.shape[0])

    # Create the GFlowNet & initialize parameters
    gflownet = DAGGFlowNet(
        delta=args.delta,
        update_target_every=args.update_target_every, 
        dataset_size=normalization, #######
        contrastive_lambda=contrastive_lambda,
        temperature=args.temperature if hasattr(args, 'temperature') else 0.5
    )

    # 初始化优化器和学习率调度器
    initial_lr = args.lr if hasattr(args, 'lr') and args.lr is not None else config.lr
    
    # 根据命令行参数创建学习率调度器
    scheduler_type = args.lr_scheduler if hasattr(args, 'lr_scheduler') else 'none'
    lr_scheduler = create_lr_scheduler(
        scheduler_type=scheduler_type,
        initial_lr=initial_lr,
        factor=getattr(args, 'lr_factor', 0.5),
        patience=getattr(args, 'lr_patience', 10),
        min_lr=getattr(args, 'lr_min', 1e-6),
        threshold=getattr(args, 'lr_threshold', 1e-3),
        T_max=config.num_iterations // 4,  # 对于余弦退火调度器
        verbose=True
    )
    
    # 打印学习率调度器信息
    if scheduler_type == 'reduce_on_plateau':
        print(f"使用ReduceLROnPlateau学习率调度器，初始学习率: {initial_lr}")
    elif scheduler_type == 'cosine':
        print(f"使用余弦退火学习率调度器，初始学习率: {initial_lr}, T_max: {config.num_iterations // 4}")
    else:
        print(f"不使用学习率调度器，固定学习率: {initial_lr}")
    
    # 创建优化器
    optimizer = optax.adam(initial_lr)
    
    # 初始化或加载参数
    if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
        print(f"加载预训练模型: {args.pretrained_model}")
        
        # 使用io模块加载模型参数
        gflownet_params = io.load(args.pretrained_model)
        print("GFlowNet模型参数加载成功")
        
        # 创建优化器
        optimizer = optax.adam(initial_lr)
        
        # 初始化GFlowNet（传入预训练参数）
        params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['graph'],
            replay.dummy['mask'],
            gflownet_params
        )
        
        # 设置起始步数（如果需要）
        if args.resume_iteration > 0:
            state = state._replace(steps=jnp.array(args.resume_iteration))
            print(f"设置起始步数为 {args.resume_iteration}")
        
        print("预训练模型加载并初始化成功")
    else:
        print("未指定预训练模型，从头开始训练")
        # 创建优化器
        optimizer = optax.adam(initial_lr)
        
        # 初始化GFlowNet（不使用预训练参数）
        params, state = gflownet.init(
            subkey,
            optimizer,
            replay.dummy['graph'],
            replay.dummy['mask']
        )

    # 探索率调度
    if args.fixed_epsilon is not None:
        print(f"使用固定探索率: {args.fixed_epsilon}")
        # 创建一个始终返回固定值的函数
        def fixed_exploration_schedule(iteration):
            return jnp.array(1. - args.fixed_epsilon)
        exploration_schedule = jax.jit(fixed_exploration_schedule)
    else:
        print("使用线性探索率调度")
        exploration_schedule = jax.jit(optax.linear_schedule(
            init_value=jnp.array(0.),
            end_value=jnp.array(1. - args.min_exploration),
            transition_steps=config.num_iterations // 2,
            transition_begin=args.prefill,
        ))

    # 创建损失跟踪器和检查点管理器
    loss_tracker = LossTracker(window_size=200, smooth_factor=0.9)
    checkpoint_manager = CheckpointManager(
        output_dir=str(args.output_folder),
        max_to_keep=3
    )

    # Training loop
    # 如果继续训练，调整起始迭代次数
    start_iteration = args.resume_iteration if args.resume_iteration > 0 else 0
    total_iterations = args.prefill + config.num_iterations
    
    # 如果继续训练且起始迭代大于预填充阶段，跳过预填充
    if start_iteration > args.prefill:
        print(f"从第 {start_iteration} 轮继续训练，跳过预填充阶段")
        prefill_needed = False
    else:
        prefill_needed = True
        
    # 创建和恢复观察
    indices = None
    observations = env.reset()



    # 准备Wandb配置
    wandb_config = config.__dict__.copy() if hasattr(config, '__dict__') else {}
    
    # 添加学习率调度器配置
    if hasattr(args, 'lr_scheduler'):
        wandb_config.update({
            'lr_scheduler': args.lr_scheduler,
            'lr_patience': getattr(args, 'lr_patience', 10),
            'lr_factor': getattr(args, 'lr_factor', 0.5),
            'lr_min': getattr(args, 'lr_min', 1e-6),
            'lr_threshold': getattr(args, 'lr_threshold', 1e-3)
        })
    
    # 添加对比学习配置
    wandb_config.update({
        'contrastive_lambda': contrastive_lambda,
        'temperature': getattr(args, 'temperature', 0.5)
    })
    
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    wandb.init(project=config.project_name, config=wandb_config, name=nowtime, save_code=True)
    run_id = wandb.run.id
    #------------------------------

    
    # 用于跟踪最佳模型
    best_loss = float('inf')
    best_params = None
    
    best_loss = float('inf')
    best_params = None

    # 定义检查学习率的频率
    lr_check_freq = args.lr_monitor_freq if hasattr(args, 'lr_monitor_freq') else 10
    print(f"学习率检查频率: 每{lr_check_freq}轮")

    # 定义保存最佳模型的频率
    best_model_save_freq = 2000
    print(f"最佳模型保存频率: 每{best_model_save_freq}轮")

    # 否则，正常开始训练
    with trange(start_iteration, total_iterations, desc='Training') as pbar:
        for iteration in pbar:
            # Sample actions, execute them, and save transitions in the replay buffer
            epsilon = exploration_schedule(iteration)
            observations['graph'] = to_graphs_tuple(observations['adjacency'])
            actions, key, logs, log_pi_global = gflownet.act(params.online, key, observations, epsilon, normalization)
            #print("score:", observations['num_edges'])
            #print("score:", observations['score'])
            next_observations, delta_scores, dones, _ = env.step(np.asarray(actions))
            indices = replay.add(
                observations,
                actions,
                logs['is_exploration'],
                next_observations,
                delta_scores,
                dones,
                prev_indices=indices
            )
            observations = next_observations

            # 对于继续训练的情况，只有预填充完毕才执行模型更新
            should_update = (prefill_needed and iteration >= args.prefill) or (not prefill_needed)
            
            if should_update:
                # Update the parameters of the GFlowNet
                samples = replay.sample(batch_size=config.batch_size, rng=rng)
                params, state, logs = gflownet.step(params, state, samples, normalization)
                
                # 获取当前的损失
                current_loss = logs['total_loss'] if 'total_loss' in logs else logs['loss']
                
                # 提取各项损失组件
                balance_loss = logs.get('balance_loss', current_loss)
                contrastive_loss = logs.get('contrastive_loss', 0.0)
                
                # 更新损失跟踪器
                loss_stats = loss_tracker.update(current_loss, iteration)
                avg_loss = loss_stats['avg_loss']
                smooth_loss = loss_stats['smooth_loss']
                is_best = loss_stats['is_best']
                
                # 更新学习率（每lr_check_freq轮检查一次）
                if hasattr(args, 'lr_scheduler') and args.lr_scheduler != 'none' and iteration % lr_check_freq == 0 and iteration > start_iteration + 50:
                    # 使用损失的移动平均来更新学习率
                    new_optimizer = lr_scheduler.step(smooth_loss, optimizer)
                    
                    # 如果优化器发生变化，需要更新state
                    if new_optimizer is not optimizer:
                        optimizer = new_optimizer
                        # 创建一个新的state，保留原来的参数
                        # 这里我们需要检查state的数据结构
                        if hasattr(state, '_replace') and hasattr(state, 'optimizer_state'):
                            state = state._replace(optimizer_state=optimizer.init(params.online))
                        else:
                            # 如果state不是namedtuple或没有optimizer_state属性，可能需要其他方式初始化
                            print("警告：无法更新优化器状态，学习率可能不会被正确应用")
                
                # 追踪最佳模型，但不立即保存
                if is_best:
                    best_params = params.online
                    best_loss = avg_loss
                
                # 每best_model_save_freq轮保存一次最佳模型
                if iteration % best_model_save_freq == 0 and iteration > start_iteration:
                    if best_params is not None:
                        checkpoint_manager.save_best(
                            step=iteration,
                            params=best_params,
                            metric=best_loss,
                            filename='best_model.pkl'
                        )
                        print(f"第 {iteration} 轮 | 保存最佳模型，损失: {best_loss:.4f}")
                
                # 定期保存检查点（每save_freq轮或args.save_freq指定的频率）
                save_freq = getattr(args, 'save_freq', 1000)
                if iteration % save_freq == 0 and iteration > start_iteration:
                    checkpoint_manager.save_checkpoint(
                        step=iteration,
                        params=params.online,
                        metric=avg_loss,
                        prefix='checkpoint'
                    ) 
                
                # 定期保存完整训练状态（每save_state_freq轮）
                save_state_freq = getattr(args, 'save_state_freq', 5000)
                if iteration % save_state_freq == 0 and iteration > start_iteration:
                    # 创建训练状态管理器并保存状态
                    timestamp = save_training_state(
                        str(args.output_folder),
                        iteration,
                        params,
                        state,
                        optimizer,
                        replay,
                        lr_scheduler,
                        loss_tracker,
                        epsilon,
                        args
                    )
                
                # 更新进度条和wandb日志
                pbar.set_postfix(
                    loss=f"{current_loss:.4f}", 
                    avg_loss=f"{avg_loss:.4f}", 
                    smooth=f"{smooth_loss:.4f}",
                    cl_loss=f"{contrastive_loss:.4f}",  # 添加对比损失
                    lr=f"{lr_scheduler.current_lr:.6f}", 
                    epsilon=f"{epsilon:.2f}"
                )
                wandb.log({
                    'iteration': iteration, 
                    'loss': current_loss,
                    'avg_loss': avg_loss,
                    'smooth_loss': smooth_loss,
                    'learning_rate': lr_scheduler.current_lr,
                    'epsilon': epsilon,
                    'balance_loss': balance_loss,  # 平衡损失
                    'contrastive_loss': contrastive_loss,   # 对比损失
                    'total_loss': current_loss       # 总损失
                })
                
                # 如果连续100轮损失没有明显下降，且使用了学习率调度器，则打印学习率信息
                if hasattr(args, 'lr_scheduler') and args.lr_scheduler != 'none' and iteration % 100 == 0:
                    print(f"第 {iteration} 轮 | 学习率: {lr_scheduler.current_lr:.6f} | "
                          f"总损失: {current_loss:.4f} | 平衡损失: {balance_loss:.4f} | "
                          f"对比损失: {contrastive_loss:.4f} | 对比权重: {contrastive_lambda:.3f}")
    
    # 使用最佳参数替换最终参数
    if best_params is not None:
        params = params._replace(online=best_params)
        print(f"使用最佳模型参数，最佳损失: {loss_tracker.best_loss:.4f}")
    
    wandb.finish()

    # Evaluate the posterior estimate
    posterior, _ = posterior_estimate(
        gflownet,
        params.online,
        env,
        key,
        normalization, 
        num_samples=args.num_samples_posterior,
        desc='Sampling from posterior',
    )

    # Compute the metrics
    ground_truth = nx.to_numpy_array(graph, weight=None)
    results = {
        'expected_shd': expected_shd(posterior, ground_truth),
        'expected_edges': expected_edges(posterior),
        **threshold_metrics(posterior, ground_truth)
    }

    # Save model, data & results
    args.output_folder.mkdir(exist_ok=True)
    with open(args.output_folder / 'arguments.json', 'w') as f:
        json.dump(vars(args), f, default=str)
    data.to_csv(args.output_folder / 'data.csv')
    with open(args.output_folder / 'graph.pkl', 'wb') as f:
        pickle.dump(graph, f)
    io.save(args.output_folder / 'model.npz', params=params.online)
    replay.save(args.output_folder / 'replay_buffer.npz')
    np.save(args.output_folder / 'posterior.npy', posterior)
    with open(args.output_folder / 'results.json', 'w') as f:
        json.dump(results, f, default=list)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import json

    parser = ArgumentParser(description='DAG-GFlowNet for Strucure Learning.')

    # Environment
    environment = parser.add_argument_group('Environment')
    environment.add_argument('--num_envs', type=int, default=8,
        help='Number of parallel environments (default: %(default)s)')
    environment.add_argument('--scorer_kwargs', type=json.loads, default='{}',
        help='Arguments of the scorer.')
    environment.add_argument('--prior', type=str, default='uniform',
        choices=['uniform', 'erdos_renyi', 'edge', 'fair'],
        help='Prior over graphs (default: %(default)s)')
    environment.add_argument('--prior_kwargs', type=json.loads, default='{}',
        help='Arguments of the prior over graphs.')

    # Optimization
    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--lr', type=float, default=1e-3,
        help='Learning rate (default: %(default)s)')
    optimization.add_argument('--delta', type=float, default=1.,
        help='Value of delta for Huber loss (default: %(default)s)')
    optimization.add_argument('--batch_size', type=int, default=32,
        help='Batch size (default: %(default)s)')
    optimization.add_argument('--num_iterations', type=int, default=100_000,
        help='Number of iterations (default: %(default)s)')
    
    # 学习率调度器
    lr_scheduler = parser.add_argument_group('Learning Rate Scheduler')
    lr_scheduler.add_argument('--lr_scheduler', type=str, default='none',
        choices=['none', 'reduce_on_plateau', 'cosine'],
        help='学习率调度器类型，可以选择不使用(none)、根据损失自适应(reduce_on_plateau)或余弦退火(cosine) (default: %(default)s)')
    
    # ReduceLROnPlateau 参数
    reduce_lr = parser.add_argument_group('ReduceLROnPlateau Scheduler')
    reduce_lr.add_argument('--lr_patience', type=int, default=10,
        help='学习率降低前等待的轮次数 (default: %(default)s)')
    reduce_lr.add_argument('--lr_factor', type=float, default=0.5,
        help='学习率降低的因子，每次降低为原来的多少倍 (default: %(default)s)')
    reduce_lr.add_argument('--lr_threshold', type=float, default=1e-3,
        help='认为有改善的阈值，损失需要下降多少才算有改善 (default: %(default)s)')
    
    # 通用学习率调度器参数
    lr_common = parser.add_argument_group('Common LR Scheduler Parameters')
    lr_common.add_argument('--lr_min', type=float, default=1e-6,
        help='最小学习率，学习率不会低于此值 (default: %(default)s)')
    lr_common.add_argument('--lr_monitor_freq', type=int, default=100,
        help='检查并更新学习率的频率（每多少轮一次）(default: %(default)s)')
    
    # 检查点和模型保存
    checkpoint = parser.add_argument_group('Checkpointing')
    checkpoint.add_argument('--save_freq', type=int, default=1000,
        help='保存检查点的频率（每多少轮保存一次）(default: %(default)s)')
    checkpoint.add_argument('--max_checkpoints', type=int, default=3,
        help='最多保留的检查点数量 (default: %(default)s)')

    # Replay buffer
    replay = parser.add_argument_group('Replay Buffer')
    replay.add_argument('--replay_capacity', type=int, default=100_000,
        help='Capacity of the replay buffer (default: %(default)s)')
    replay.add_argument('--prefill', type=int, default=1000,
        help='Number of iterations with a random policy to prefill '
             'the replay buffer (default: %(default)s)')
    
    # Exploration
    exploration = parser.add_argument_group('Exploration')
    exploration.add_argument('--min_exploration', type=float, default=0.1,
        help='Minimum value of epsilon-exploration (default: %(default)s)')
    exploration.add_argument('--update_epsilon_every', type=int, default=10,
        help='Frequency of update for epsilon (default: %(default)s)')
    
    # Miscellaneous
    misc = parser.add_argument_group('Miscellaneous')
    misc.add_argument('--num_samples_posterior', type=int, default=1000,
        help='Number of samples for the posterior estimate (default: %(default)s)')
    misc.add_argument('--update_target_every', type=int, default=1000,
        help='Frequency of update for the target network (default: %(default)s)')
    misc.add_argument('--seed', type=int, default=0,
        help='Random seed (default: %(default)s)')
    misc.add_argument('--num_workers', type=int, default=4,
        help='Number of workers (default: %(default)s)')
    misc.add_argument('--mp_context', type=str, default='spawn',
        help='Multiprocessing context (default: %(default)s)')
    misc.add_argument('--output_folder', type=Path, default='output',
        help='Output folder (default: %(default)s)')
    # 添加加载预训练模型相关参数
    misc.add_argument('--pretrained_model', type=str, default=None,
        help='加载预训练模型的路径 (.npz 文件)，用于继续训练')
    misc.add_argument('--load_replay_buffer', type=str, default=None,
        help='加载回放缓冲区的路径 (.npz 文件)，保持训练经验连续性')
    misc.add_argument('--resume_iteration', type=int, default=0,
        help='继续训练的起始迭代次数，用于恢复训练状态')
    misc.add_argument('--fixed_epsilon', type=float, default=None,
        help='固定的探索率值（0到1之间），如不指定则使用探索率调度')
    misc.add_argument('--resume_state_dir', type=str, default=None,
        help='恢复训练的状态目录，包含完整训练状态')
    misc.add_argument('--resume_timestamp', type=str, default=None,
        help='恢复训练的时间戳，如不指定则使用最新的')
    misc.add_argument('--save_state_freq', type=int, default=5000,
        help='保存训练状态的频率（每多少轮保存一次）(default: %(default)s)')

    # 对比学习参数
    contrastive = parser.add_argument_group('Contrastive Learning')
    contrastive.add_argument('--contrastive_lambda', type=float, default=0.1,
        help='对比学习损失的权重系数 (default: %(default)s)')
    contrastive.add_argument('--temperature', type=float, default=0.5,
        help='对比学习温度参数，控制相似度分布的平滑度 (default: %(default)s)')
    contrastive.add_argument('--disable_contrastive', action='store_true', 
        help='禁用对比学习损失 (设置此标志将忽略contrastive_lambda值并将其设为0)')
    
    subparsers = parser.add_subparsers(help='Type of graph', dest='graph')

    # Erdos-Renyi Linear-Gaussian graphs
    er_lingauss = subparsers.add_parser('erdos_renyi_lingauss')
    er_lingauss.add_argument('--num_variables', type=int, required=True,
        help='Number of variables')
    er_lingauss.add_argument('--num_edges', type=int, required=True,
        help='Average number of edges')
    er_lingauss.add_argument('--num_samples', type=int, required=True,
        help='Number of samples')

    # Flow cytometry data (Sachs) with observational data
    sachs_continuous = subparsers.add_parser('sachs_continuous')

    # Flow cytometry data (Sachs) with interventional data
    sachs_intervention = subparsers.add_parser('sachs_interventional')

    sachs_intervention = subparsers.add_parser('asia_interventional')

    sachs_intervention = subparsers.add_parser('asia_interventional_bic')
    sachs_intervention = subparsers.add_parser('child_interventional_bic')

    sachs_intervention = subparsers.add_parser('alarm_interventional_bic')
    sachs_intervention = subparsers.add_parser('sachs_interventional_bic')
    sachs_intervention = subparsers.add_parser('hailfinder_interventional_bic')
    sachs_intervention = subparsers.add_parser('win95pts_interventional_bic')
    sachs_intervention = subparsers.add_parser('formed_custom')
    sachs_intervention = subparsers.add_parser('property_custom')
    sachs_intervention = subparsers.add_parser('sports_custom')

    args = parser.parse_args()


    config = Namespace(
        project_name='gfngnn_alarm_new',
        batch_size= 258,
        lr = 1e-4,
        num_iterations = 50000,
    )
    

    main(args)
