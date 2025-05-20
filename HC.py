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
# 导入pgmpy相关库
from pgmpy.estimators import HillClimbSearch, BDeuScore, BicScore
from pgmpy.models import BayesianNetwork
import pandas as pd

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


from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)
from pgmpy.estimators import StructureScore
class HillClimbSearch(StructureEstimator):
    """
    Class for heuristic hill climb searches for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )
        potential_new_edges = sorted(potential_new_edges)
        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = [var for var in old_parents if var != X]
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

    def estimate(
        self,
        scoring_method="k2score",
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        epsilon=1e-50,
        max_iter=1e6,
        show_progress=True,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None

        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data)
        >>> best_model = est.estimate(scoring_method=BicScore(data))
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        OutEdgeView([('B', 'J'), ('A', 'J')])
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        OutEdgeView([('J', 'A'), ('B', 'J')])
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }
        if (
            (
                isinstance(scoring_method, str)
                and (scoring_method.lower() not in supported_methods)
            )
        ) and (not isinstance(scoring_method, StructureScore)):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        if self.use_cache:
            score_fn = ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        # Step 1.2: Check the start_dag
        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)
        elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag

        if show_progress and config.SHOW_PROGRESS:
            iteration = trange(int(max_iter))
        else:
            iteration = range(int(max_iter))

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        for _ in iteration:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )
            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_operation)

        # Step 3: Return if no more improvements or maximum iterations reached.
        return current_model
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
    matrices = []  # 添加邻接矩阵收集列表

    # 分批生成样本，但只跟踪总样本数
    samples_collected = 0
    max_steps = num_samples * 10  # 设置最大步数限制
    step_count = 0
    
    observations = env.reset()
    observations['graph'] = to_graphs_tuple(observations['adjacency'])
    
    # 使用进度条
    with tqdm(total=num_samples, desc="生成样本", ncols=100) as pbar:
        while samples_collected < num_samples and step_count < max_steps:
            # 获取当前状态信息
            score = observations['score']
            adjacency = observations['adjacency']  # 获取邻接矩阵

            # 执行动作
            actions, key, _, embedding = gflownet.act(params.online, key, observations, 1., normalization)
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
                    matrices.append(adjacency[i])  # 保存邻接矩阵
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
    X = jnp.stack(X[:num_samples])
    y = np.array(y[:num_samples])
    
    # 计算唯一图结构的数量 - 优化版本
    # 使用元组作为字典键，比字符串哈希更可靠且更高效
    unique_dict = {}
    
    for i, matrix in enumerate(matrices[:num_samples]):
        # 转换为numpy数组，然后转为元组作为字典键
        matrix_np = np.array(matrix)
        matrix_tuple = tuple(map(tuple, matrix_np))
        
        # 如果这个结构是新的，记录它
        if matrix_tuple not in unique_dict:
            unique_dict[matrix_tuple] = i
    
    # 获取唯一矩阵的索引
    unique_indices = list(unique_dict.values())
    
    print(f"成功生成 {X.shape[0]} 个样本")
    print(f"包含 {len(unique_indices)} 个唯一图结构")
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
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
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
    使用代理模型和GFlowNet协同寻找高分表征和图结构，并结合爬山法优化贝叶斯网络结构
    
    流程：
    1. 加载训练好的代理模型和GFlowNet模型
    2. 使用GFlowNet生成一批样本和对应的表征
    3. 使用代理模型从这些样本中找出预测得分最高的表征
    4. 在GFlowNet生成的表征中找到与高分表征最相似的表征
    5. 使用这些相似表征对应的图结构作为爬山法的初始矩阵
    6. 使用爬山法优化贝叶斯网络结构
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
    

    print("\n步骤 6/7: 对唯一嵌入预测得分并进行后续处理...")

    # --- 1. 准备数据并找到唯一嵌入及其对应的原始信息 ---
    if isinstance(X_initial, list):
        X_np = np.stack([np.array(x) for x in X_initial])
    else:
        X_np = np.array(X_initial) # Assume already NumPy array or convertible

    if isinstance(y_initial, list):
        y_np = np.array(y_initial)
    else:
        y_np = np.array(y_initial) # Assume already NumPy array or convertible

    # 确保 matrices_initial 是列表 (通常是)
    if not isinstance(matrices_initial, list):
         # 如果不是列表，尝试转换，但这可能需要根据具体类型调整
         try:
             matrices_initial_list = list(matrices_initial)
             print("警告：matrices_initial 不是列表，已尝试转换。")
         except TypeError:
             print("错误：matrices_initial 无法转换为列表。")
             # 可能需要退出或采取其他错误处理
             return # 或者 exit()
    else:
        matrices_initial_list = matrices_initial


    if len(X_np) != len(y_np) or len(X_np) != len(matrices_initial_list):
         print(f"错误: 初始数据长度不匹配! Embeddings: {len(X_np)}, Scores: {len(y_np)}, Matrices: {len(matrices_initial_list)}")
         return # 或者 exit()

    print(f"  原始样本总数: {len(X_np)}")

    try:
        # 找到唯一的嵌入 (unique_X) 和它们首次出现时的索引 (first_indices)
        unique_X, first_indices = np.unique(X_np, axis=0, return_index=True)
        print(f"  唯一嵌入数量: {len(unique_X)}")

        # 使用首次出现的索引提取唯一嵌入对应的 *原始* 分数和 *原始* 邻接矩阵
        unique_y_initial = y_np[first_indices]
        unique_matrices_initial = [matrices_initial_list[i] for i in first_indices] # 获取对应的唯一矩阵
        print(f"  已提取唯一嵌入对应的 {len(unique_y_initial)} 个原始分数和 {len(unique_matrices_initial)} 个邻接矩阵。")

    except Exception as e:
        print(f"  错误：使用 np.unique 查找唯一嵌入时出错: {e}")
        print("  无法继续仅对唯一值进行分析。")
        return # 或者 exit()

    # --- 2. 分批预测唯一嵌入的得分 ---
    unique_y_pred_list = []
    batch_size = args.batch_size # Or a suitable batch size for prediction
    num_unique_samples = len(unique_X)

    with tqdm(total=num_unique_samples, desc="预测唯一嵌入", ncols=100) as pbar:
        for i in range(0, num_unique_samples, batch_size):
            end_idx = min(i + batch_size, num_unique_samples)
            batch_unique_X = unique_X[i:end_idx]

            # 使用代理模型预测得分
            batch_pred = proxy_model.predict(proxy_params, batch_unique_X, normalizer=normalizer)
            unique_y_pred_list.extend(batch_pred)
            pbar.update(len(batch_unique_X))

    # unique_y_pred 是唯一嵌入对应的 *预测* 分数
    unique_y_pred = np.array(unique_y_pred_list)
    print(f"  已获得 {len(unique_y_pred)} 个唯一嵌入的预测分数。")

    # --- 3. 从唯一的预测结果中选择多样性的 Top K 样本 ---
    # 注意：现在 select_diverse_top_samples 处理的是 unique_X 和 unique_y_pred
    print(f"\n步骤 7/7: 从 {len(unique_X)} 个唯一嵌入中选择 Top {args.top_k} 个多样性样本...")
    selected_unique_indices, top_unique_X, top_unique_y_pred = select_diverse_top_samples(
        unique_X,           # 输入唯一嵌入
        unique_y_pred,      # 输入唯一预测分数
        top_k=args.top_k,
        diversity_threshold=args.diversity_threshold,
        metric=args.diversity_metric
    )
    # selected_unique_indices 是 top_unique_X 在 unique_X 中的索引

    print(f"  代理模型预测的 Top {args.top_k} 个唯一多样性样本得分范围: [{top_unique_y_pred.min():.4f}, {top_unique_y_pred.max():.4f}]")

    # --- 4. (可选) 对选出的 Top K 唯一嵌入应用梯度优化 ---
    optimized_top_embeddings = []
    optimized_top_scores = []
    if args.gradient_steps > 0: # 检查是否需要优化
        print("\n步骤 8/8: 对 Top K 唯一样本应用基于梯度的优化...")
        for i, embedding in enumerate(top_unique_X):
            initial_pred_score = top_unique_y_pred[i] # 获取这个唯一嵌入的预测分数
            print(f"\n  优化唯一表征 #{i+1} (初始预测分数: {initial_pred_score:.4f}):")
            opt_embedding, opt_score = gradient_optimize_embedding(
                proxy_model,
                proxy_params,
                embedding, # 优化选出的唯一嵌入
                normalizer,
                num_steps=args.gradient_steps,
                learning_rate=args.gradient_lr
            )
            print(f"    优化后分数: {opt_score:.4f} (增加: {opt_score - initial_pred_score:.4f})")
            optimized_top_embeddings.append(opt_embedding)
            optimized_top_scores.append(opt_score)

        # 更新 top 变量为优化后的结果
        final_top_X = np.array(optimized_top_embeddings)
        final_top_y_pred = np.array(optimized_top_scores)
        print("  梯度优化完成。")
    else:
        # 如果不进行优化，直接使用选择出的 top 样本
        final_top_X = top_unique_X
        final_top_y_pred = top_unique_y_pred # 预测分数保持不变
        print("\n跳过梯度优化步骤。")


    # --- 5. 为每个最终的 Top 嵌入找到最相似的初始 *唯一* 图结构 ---
    print(f"\n步骤 9/9: 为最终 Top {len(final_top_X)} 个嵌入寻找最相似的 {args.num_similar_samples} 个初始唯一图结构...")
    start_matrices = [] # 存储最终要返回的邻接矩阵列表

    # 计算最终 Top 嵌入与所有 *唯一* 初始嵌入之间的相似度/距离
    # 这是主要计算，只需执行一次
    if args.similarity_metric == 'cosine':
        # similarity shape: (len(final_top_X), len(unique_X))
        similarities_all = 1 - cdist(final_top_X, unique_X, metric='cosine')
    else: # euclidean
        # distance shape: (len(final_top_X), len(unique_X))
        distances_all = cdist(final_top_X, unique_X, metric='euclidean')
        # Normalize distances to similarities (0 to 1) for each row (top embedding)
        max_dist_per_row = np.max(distances_all, axis=1, keepdims=True)
        # Avoid division by zero if all distances are zero for a row
        max_dist_per_row[max_dist_per_row == 0] = 1.0
        similarities_all = 1 - (distances_all / max_dist_per_row)

    # 为每个 top 嵌入找出最相似的 unique 索引
    for i in range(len(final_top_X)):
        current_similarities = similarities_all[i] # 获取第 i 个 top 嵌入对所有 unique 嵌入的相似度

        # 找到相似度最高的 N 个 unique 样本的索引 (在 unique_X 中的索引)
        # argsort 从小到大排序，取最后 N 个是最大的，然后[::-1]反转得到从大到小
        sim_unique_indices = np.argsort(current_similarities)[-args.num_similar_samples:][::-1]

        print(f"  对于 Top 嵌入 #{i+1} (优化后预测分: {final_top_y_pred[i]:.4f}):")
        print(f"    最相似的 {args.num_similar_samples} 个 Unique 索引 (in unique_X): {sim_unique_indices}")
        print(f"    对应的相似度: {current_similarities[sim_unique_indices]}")

        # 获取这些最相似的 unique 索引对应的 *初始* 邻接矩阵
        similar_matrices_for_top_i = [unique_matrices_initial[idx] for idx in sim_unique_indices]

        # 将找到的矩阵添加到最终列表中
        start_matrices.extend(similar_matrices_for_top_i)
        # 如果需要确保 start_matrices 里的矩阵本身也是唯一的，可以在最后处理
        # start_matrices = list({tuple(map(tuple, np.array(m))): m for m in start_matrices}.values())

    # (Code from Step 9 - Finding similar matrices for each top embedding)
    # ... loop finishes, start_matrices is populated ...

    print(f"\n共找到 {len(start_matrices)} 个起始邻接矩阵（可能包含重复）。")

    # --- 去除 start_matrices 中的重复项 ---
    print("去除重复的起始邻接矩阵...")
    unique_start_matrices_dict = {}
    duplicates_found = 0
    for matrix in start_matrices:
        # 将 NumPy 数组转换为可哈希的元组形式作为字典键
        matrix_key = tuple(map(tuple, np.array(matrix))) # Ensure it's array then tuplefy

        if matrix_key not in unique_start_matrices_dict:
            unique_start_matrices_dict[matrix_key] = matrix # Store the original matrix object
        else:
            duplicates_found += 1

    # 从字典的值中获取唯一的邻接矩阵列表
    unique_start_matrices = list(unique_start_matrices_dict.values())
    print(f"去重后剩余 {len(unique_start_matrices)} 个唯一的起始邻接矩阵。 (移除了 {duplicates_found} 个重复项)")
    # ------------------------------------

    # ****** 后续代码应使用 unique_start_matrices ******
    # 例如，如果需要传递给其他函数或保存:
    # final_starting_points = unique_start_matrices

    print("优化和相似性搜索完成。")
    start_matrices = unique_start_matrices
    
    # 步骤7: 使用爬山法优化贝叶斯网络结构
    print("\n步骤 7/7: 使用爬山法优化贝叶斯网络结构...")
    
    # 保存结果容器
    hill_climbing_results = []
    
    # 准备数据，将数据转换为pandas DataFrame
    if isinstance(data, np.ndarray):
        data_df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        data_df = data
    else:
        raise TypeError("数据类型不支持，需要numpy数组或pandas DataFrame")
    
    # 确保列名是字符串形式
    if data_df.columns.dtype != 'object':
        data_df.columns = [f'X{i}' for i in range(data_df.shape[1])]
    
    # 设置评分器
    if hasattr(args, 'hill_climb_scoring') and args.hill_climb_scoring == 'bic':
        print("使用BIC评分函数...")
        scoring_method = BicScore(data_df)
    else:
        print("使用BDeu评分函数...")
        equivalent_sample_size = args.bdeu_equivalent_sample_size if hasattr(args, 'bdeu_equivalent_sample_size') else 10
        scoring_method = BDeuScore(data_df, equivalent_sample_size=equivalent_sample_size)
    
    # 创建爬山搜索器
    hc = HillClimbSearch(data_df)
    
    # 对每个初始矩阵进行爬山法优化
    for i, initial_matrix in enumerate(start_matrices):
        print(f"\n优化初始矩阵 {i+1}/{len(start_matrices)}...")
        
        # 将邻接矩阵转换为边列表和BN模型
        edges = []
        node_names = data_df.columns
        for j in range(initial_matrix.shape[0]):
            for k in range(initial_matrix.shape[1]):
                if initial_matrix[j, k] > 0:
                    # 使用列名作为节点名称
                    edges.append((node_names[j], node_names[k]))
        
        if not edges:
            print("初始矩阵没有边，跳过优化")
            continue
        
        try:
            # 创建初始DAG
            initial_model = BayesianNetwork()
            
            # 先添加所有节点，确保孤立节点也被包含
            # 获取数据框的列名作为节点
            nodes = list(data_df.columns)
            initial_model.add_nodes_from(nodes)
            
            # 然后添加边
            initial_model.add_edges_from(edges)
            
            # 使用pgmpy计算初始分数，而不是使用scorer
            initial_score = scoring_method.score(initial_model)
            print(f"初始矩阵得分(pgmpy): {initial_score:.4f}")
            
            # 设置爬山法参数
            max_iter = args.hill_climb_max_iter if hasattr(args, 'hill_climb_max_iter') else 100
            epsilon = args.hill_climb_epsilon if hasattr(args, 'hill_climb_epsilon') else 1e-4
            tabu_length = args.hill_climb_tabu_length if hasattr(args, 'hill_climb_tabu_length') else 100
            max_indegree = args.hill_climb_max_indegree if hasattr(args, 'hill_climb_max_indegree') else None
            
            # 设置黑名单和白名单
            black_list = None
            white_list = None
            
            
            # 使用爬山法搜索最优结构
            print(f"开始爬山搜索，最大迭代次数: {max_iter}, epsilon: {epsilon}, tabu_length: {tabu_length}")
            best_model = hc.estimate(
                start_dag=initial_model,
                scoring_method=scoring_method,
                max_iter=max_iter,
                epsilon=epsilon,
                tabu_length=tabu_length,
                max_indegree=max_indegree,
                black_list=black_list,
                white_list=white_list
            )
            
            # 获取优化后的边列表
            optimized_edges = list(best_model.edges())
            
            # 将优化后的边列表转换回邻接矩阵
            optimized_matrix = np.zeros_like(initial_matrix)
            for edge in optimized_edges:
                # 找到边对应的索引
                source_idx = np.where(node_names == edge[0])[0][0]
                target_idx = np.where(node_names == edge[1])[0][0]
                optimized_matrix[source_idx, target_idx] = 1
            
            # 使用pgmpy评分方法计算优化后分数，而不是使用scorer
            optimized_score = scoring_method.score(best_model)
            print(f"爬山法优化完成:")
            print(f"  初始得分(pgmpy): {initial_score:.4f}")
            print(f"  优化后得分(pgmpy): {optimized_score:.4f}")
            print(f"  分数提升: {optimized_score - initial_score:.4f}")
            print(f"  边的数量: {len(optimized_edges)}")
            
            # 保存结果，使用pgmpy的分数
            hill_climbing_results.append({
                'initial_matrix': initial_matrix.copy(),
                'initial_score': float(initial_score),
                'optimized_matrix': optimized_matrix.copy(),
                'optimized_score': float(optimized_score),
                'improvement': float(optimized_score - initial_score),
                'initial_edges': edges,
                'optimized_edges': optimized_edges,
                'node_names': list(node_names),
                'best_model': best_model  # 保存优化后的模型对象
            })
            
        except Exception as e:
            print(f"优化过程中出错: {e}")
            print("跳过当前矩阵")
    
    # 按优化后分数排序结果
    hill_climbing_results.sort(key=lambda x: x['optimized_score'], reverse=True)
    
    # 确保前N个结果是结构各不相同的
    max_top_structures = min(args.top_k, len(hill_climbing_results))
    diverse_top_results = []
    unique_structures = set()
    
    # 按分数排序遍历所有结果
    for result in hill_climbing_results:
        # 使用边的集合作为结构的唯一标识
        edges_tuple = tuple(sorted((str(edge[0]), str(edge[1])) for edge in result['optimized_edges']))
        
        # 如果是一个新的结构，添加到多样性结果中
        if edges_tuple not in unique_structures:
            unique_structures.add(edges_tuple)
            diverse_top_results.append(result)
            
            # 如果已收集足够的多样性结构，停止
            if len(diverse_top_results) >= max_top_structures:
                break
    
    print(f"找到 {len(diverse_top_results)} 个不同的优秀结构")
    
    # 保存优化结果
    results_path = os.path.join(args.output_dir, 'hill_climbing_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(hill_climbing_results, f)
    
    # 保存最佳结构(多个)
    if diverse_top_results:
        # 保存单个最佳结构文件(向后兼容)
        best_result = diverse_top_results[0]
        best_model_path = os.path.join(args.output_dir, 'best_bn_structure.json')
        
        # 构建可JSON序列化的结构，包含初始矩阵和评分
        best_structure = {
            'nodes': list(best_result['node_names']),
            'edges': [(str(edge[0]), str(edge[1])) for edge in best_result['optimized_edges']],
            'score': float(best_result['optimized_score']),
            'initial_score': float(best_result['initial_score']),
            'improvement': float(best_result['improvement']),
            'initial_matrix': best_result['initial_matrix'].tolist(),  # 添加初始矩阵
            'optimized_matrix': best_result['optimized_matrix'].tolist(),  # 添加优化后的矩阵
            'initial_edges': [(str(edge[0]), str(edge[1])) for edge in best_result['initial_edges']]  # 添加初始边列表
        }
        
        with open(best_model_path, 'w') as f:
            json.dump(best_structure, f, indent=2)
        
        # 保存多个最佳结构
        top_structures_path = os.path.join(args.output_dir, 'top_bn_structures.json')
        top_structures = []
        
        for i, result in enumerate(diverse_top_results):
            structure_info = {
                'rank': i + 1,
                'nodes': list(result['node_names']),
                'edges': [(str(edge[0]), str(edge[1])) for edge in result['optimized_edges']],
                'score': float(result['optimized_score']),
                'initial_score': float(result['initial_score']),
                'improvement': float(result['improvement']),
                'edge_count': len(result['optimized_edges']),
                'initial_matrix': result['initial_matrix'].tolist(),  # 添加初始矩阵
                'optimized_matrix': result['optimized_matrix'].tolist(),  # 添加优化后的矩阵
                'initial_edges': [(str(edge[0]), str(edge[1])) for edge in result['initial_edges']]  # 添加初始边列表
            }
            top_structures.append(structure_info)
        
        with open(top_structures_path, 'w') as f:
            json.dump(top_structures, f, indent=2)
        
        print(f"前 {len(diverse_top_results)} 个多样化贝叶斯网络结构已保存至 {top_structures_path}")
        
        # 如果数据集不太大，尝试拟合模型并保存完整BN模型
        try:
            if len(data_df) <= 10000:  # 限制数据集大小，避免内存问题
                print("拟合最佳贝叶斯网络模型...")
                # 创建目录保存模型文件
                model_dir = os.path.join(args.output_dir, 'best_model')
                os.makedirs(model_dir, exist_ok=True)
                
                # 使用pickle保存最佳模型
                best_model_file = os.path.join(model_dir, 'best_bn_model.pkl')
                with open(best_model_file, 'wb') as f:
                    pickle.dump(best_result['best_model'], f)
                print(f"最佳贝叶斯网络模型已保存至 {best_model_file}")
                
                # 保存前N个多样化模型
                for i, result in enumerate(diverse_top_results[:5]):  # 限制为前5个
                    model_file = os.path.join(model_dir, f'bn_model_rank{i+1}.pkl')
                    with open(model_file, 'wb') as f:
                        pickle.dump(result['best_model'], f)
                print(f"前 {min(5, len(diverse_top_results))} 个多样化贝叶斯网络模型已保存")
        except Exception as e:
            print(f"保存完整模型时出错: {e}")
        
        print(f"最佳贝叶斯网络结构已保存至 {best_model_path}")
    
    # 输出结果摘要
    print("\n" + "="*80)
    print("爬山法优化结果摘要")
    print("="*80)
    
    if not diverse_top_results:
        print("未能成功优化任何结构")
    else:
        print(f"\n找到的前 {len(diverse_top_results)} 个多样化高分结构:")
        for i, result in enumerate(diverse_top_results):
            print(f"\n结构 {i+1}:")
            print(f"  初始得分(pgmpy): {result['initial_score']:.4f}")
            print(f"  优化后得分(pgmpy): {result['optimized_score']:.4f}")
            print(f"  分数提升: {result['improvement']:.4f}")
            print(f"  边的数量: {len(result['optimized_edges'])}")
            print(f"  初始边数量: {len(result['initial_edges'])}")

            # 只打印前10条边
            print("  部分边:")
            for j, edge in enumerate(result['optimized_edges'][:10]):
                print(f"    {edge[0]} -> {edge[1]}")
            
            if len(result['optimized_edges']) > 10:
                print(f"    ... 还有 {len(result['optimized_edges'])-10} 条边未显示")
        
        # 计算结构间的差异性
        if len(diverse_top_results) > 1:
            print("\n结构间差异性分析:")
            for i in range(len(diverse_top_results)):
                for j in range(i+1, len(diverse_top_results)):
                    edges_i = set((str(e[0]), str(e[1])) for e in diverse_top_results[i]['optimized_edges'])
                    edges_j = set((str(e[0]), str(e[1])) for e in diverse_top_results[j]['optimized_edges'])
                    
                    # 计算Jaccard相似度: 交集大小/并集大小
                    intersection = len(edges_i.intersection(edges_j))
                    union = len(edges_i.union(edges_j))
                    similarity = intersection / union if union > 0 else 0
                    
                    print(f"  结构 {i+1} 和结构 {j+1} 的差异性: {(1-similarity):.2f} (共享 {intersection} 条边，共 {union} 条不同边)")
    
    print(f"\n所有结果已保存至: {args.output_dir}")
    
    return diverse_top_results

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
    max_steps = num_samples * 1000  # 设置最大步数限制
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
            actions, key, _, embedding = gflownet.act(params.online, key, observations, 0.9, normalization)
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
    
    # 计算唯一图结构的数量 - 优化版本
    # 使用元组作为字典键，比字符串哈希更可靠且更高效
    unique_dict = {}
    
    for i, matrix in enumerate(matrices[:num_samples]):
        # 转换为numpy数组，然后转为元组作为字典键
        matrix_np = np.array(matrix)
        matrix_tuple = tuple(map(tuple, matrix_np))
        
        # 如果这个结构是新的，记录它
        if matrix_tuple not in unique_dict:
            unique_dict[matrix_tuple] = i
    
    # 获取唯一矩阵的索引
    unique_indices = list(unique_dict.values())
    
    print(f"成功生成 {X.shape[0]} 个样本")
    print(f"包含 {len(unique_indices)} 个唯一图结构")
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
    optimize_parser.add_argument('--num_envs', type=int, default=16,
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
    optimize_parser.add_argument('--top_k', type=int, default=20,
                        help='每次迭代选择的顶部样本数量')
    optimize_parser.add_argument('--num_similar_samples', type=int, default=1,
                        help='每个高分表征选择的相似样本数量')
    optimize_parser.add_argument('--similarity_metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean'],
                        help='表征相似度度量方式: cosine (余弦相似度), euclidean (欧氏距离)')
    optimize_parser.add_argument('--max_saved_samples', type=int, default=100,
                        help='最多保存的样本数量')
    optimize_parser.add_argument('--diversity_threshold', type=float, default=1e-5,
                      help='样本多样性最小距离阈值')
    optimize_parser.add_argument('--diversity_metric', type=str, default='euclidean',
                      choices=['euclidean', 'cosine'],
                      help='多样性距离度量方式: euclidean (欧氏距离), cosine (余弦距离)')
    optimize_parser.add_argument('--graph_diversity_threshold', type=float, default=0.5,
                      help='图结构多样性阈值，Jaccard相似度高于此值的图被认为相似')
    
    # 梯度优化参数
    optimize_parser.add_argument('--gradient_steps', type=int, default=50,
                      help='梯度优化迭代次数')
    optimize_parser.add_argument('--gradient_lr', type=float, default=0.01,
                      help='梯度优化学习率')
                      
    # 爬山法参数
    optimize_parser.add_argument('--hill_climb_max_iter', type=int, default=10000,
                      help='爬山法最大迭代次数')
    optimize_parser.add_argument('--hill_climb_epsilon', type=float, default=1e-8,
                      help='爬山法收敛阈值')
    optimize_parser.add_argument('--hill_climb_tabu_length', type=int, default=100,
                      help='爬山法禁忌表长度')
    optimize_parser.add_argument('--hill_climb_max_indegree', type=int, default=None,
                      help='爬山法最大入度限制')
    optimize_parser.add_argument('--hill_climb_scoring', type=str, default='bic', choices=['bdeu', 'bic'],
                      help='爬山法评分函数: bdeu (BDeu评分), bic (BIC评分)')
    optimize_parser.add_argument('--bdeu_equivalent_sample_size', type=float, default=10,
                      help='BDeu评分的等效样本大小')
                      
    # GFlowNet参数 
    optimize_parser.add_argument('--delta', type=float, default=1.0,
                        help='Huber损失的delta值')
    optimize_parser.add_argument('--update_target_every', type=int, default=1000,
                        help='目标网络更新频率')
    
    # 其他参数
    optimize_parser.add_argument('--seed', type=int, default=0,
                        help='随机种子')
    optimize_parser.add_argument('--num_workers', type=int, default=16,
                        help='工作进程数量')
    optimize_parser.add_argument('--mp_context', type=str, default='fork',
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
        asia_intervention_bic = graph_subparsers.add_parser('child_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('alarm_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('win95pts_interventional_bic')
        asia_intervention_bic = graph_subparsers.add_parser('hailfinder_interventional_bic')
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