import numpy as np
import math

from numpy.random import default_rng
from jraph import GraphsTuple

from dag_gflownet.utils.jraph_utils import to_graphs_tuple


class ReplayBuffer:
    def __init__(self, capacity, num_variables):
        self.capacity = capacity
        self.num_variables = num_variables

        nbytes = math.ceil((num_variables ** 2) / 8)
        dtype = np.dtype([
            ('adjacency', np.uint8, (nbytes,)),
            ('num_edges', np.int_, (1,)),
            ('actions', np.int_, (1,)),
            ('is_exploration', np.bool_, (1,)),
            ('delta_scores', np.float_, (1,)),
            ('scores', np.float_, (1,)),
            ('mask', np.uint8, (nbytes,)),
            ('next_adjacency', np.uint8, (nbytes,)),
            ('next_mask', np.uint8, (nbytes,))
        ])
        self._replay = np.zeros((capacity,), dtype=dtype)
        self._index = 0
        self._is_full = False
        self._prev = np.full((capacity,), -1, dtype=np.int_)
        
        # 简化的轨迹跟踪系统
        self._next = np.full((capacity,), -1, dtype=np.int_)  # 样本的后继
        self._trajectory_id = np.full((capacity,), -1, dtype=np.int_)  # 样本所属轨迹ID
        self._trajectory_map = {}  # 轨迹ID -> 样本索引列表
        self._next_trajectory_id = 0  # 下一个轨迹ID
        
        # 当前活跃轨迹ID
        self._active_trajectory_id = -1
        self._active_trajectory = []
        
        # 跨批次采样状态
        self._pending_trajectory = None
        self._trajectory_position = 0
        
        # 预计算各轨迹的平均得分（用于基于分数采样）
        self._trajectory_scores = {}  # 轨迹ID -> 平均得分
        
        # 快速轨迹长度查询
        self._trajectory_lengths = {}  # 轨迹ID -> 长度
        
        # 缓存有效轨迹ID（长度>1的轨迹）
        self._valid_trajectories = set()  # 保存所有长度>1的轨迹ID

    def add(
            self,
            observations,
            actions,
            is_exploration,
            next_observations,
            delta_scores,
            dones,
            prev_indices=None
        ):
        indices = np.full((dones.shape[0],), -1, dtype=np.int_)
        if np.all(dones):
            return indices

        num_samples = np.sum(~dones)
        add_idx = np.arange(self._index, self._index + num_samples) % self.capacity
        self._is_full |= (self._index + num_samples >= self.capacity)
        self._index = (self._index + num_samples) % self.capacity
        indices[~dones] = add_idx

        data = {
            'adjacency': self.encode(observations['adjacency'][~dones]),
            'num_edges': observations['num_edges'][~dones],
            'actions': actions[~dones],
            'delta_scores': delta_scores[~dones],
            'mask': self.encode(observations['mask'][~dones]),
            'next_adjacency': self.encode(next_observations['adjacency'][~dones]),
            'next_mask': self.encode(next_observations['mask'][~dones]),

            # Extra keys for monitoring
            'is_exploration': is_exploration[~dones],
            'scores': observations['score'][~dones],
        }

        for name in data:
            shape = self._replay.dtype[name].shape
            self._replay[name][add_idx] = np.asarray(data[name].reshape(-1, *shape))
        
        # 处理轨迹关系 - 简化版本
        if prev_indices is not None:
            prev_valid = prev_indices[~dones]
            self._prev[add_idx] = prev_valid
            
            # 更新next关系，只用于遍历完整轨迹
            for i, prev_idx in enumerate(prev_valid):
                if prev_idx >= 0:
                    self._next[prev_idx] = add_idx[i]
            
            # 为每个样本分配轨迹ID
            for i, (prev_idx, curr_idx) in enumerate(zip(prev_valid, add_idx)):
                # 如果是新轨迹的起点
                if prev_idx < 0:
                    # 创建新轨迹
                    new_tid = self._next_trajectory_id
                    self._next_trajectory_id += 1
                    self._active_trajectory_id = new_tid
                    self._active_trajectory = [curr_idx]
                    
                    # 记录样本所属轨迹
                    self._trajectory_id[curr_idx] = new_tid
                    self._trajectory_map[new_tid] = [curr_idx]
                else:
                    # 继续已有轨迹
                    tid = self._trajectory_id[prev_idx]
                    
                    # 如果先前样本的轨迹ID已记录
                    if tid >= 0:
                        self._trajectory_id[curr_idx] = tid
                        if tid in self._trajectory_map:
                            # 添加到轨迹
                            self._trajectory_map[tid].append(curr_idx)
                            
                            # 如果轨迹长度变为2，加入有效轨迹集合
                            if len(self._trajectory_map[tid]) == 2:
                                self._valid_trajectories.add(tid)
                        else:
                            self._trajectory_map[tid] = [curr_idx]
                        
                        # 如果是当前活跃轨迹
                        if tid == self._active_trajectory_id:
                            self._active_trajectory.append(curr_idx)
            
            # 处理轨迹结束的情况
            done_indices = np.where(dones[~dones])[0]
            for done_i in done_indices:
                curr_idx = add_idx[done_i]
                tid = self._trajectory_id[curr_idx]
                
                # 如果轨迹已记录且是当前活跃轨迹
                if tid >= 0 and tid == self._active_trajectory_id:
                    # 计算轨迹平均得分
                    if tid in self._trajectory_map and len(self._trajectory_map[tid]) > 0:
                        scores = [self._replay['scores'][idx][0] for idx in self._trajectory_map[tid]]
                        self._trajectory_scores[tid] = np.mean(scores)
                        self._trajectory_lengths[tid] = len(self._trajectory_map[tid])
                        
                        # 确保正确记录有效轨迹
                        if len(self._trajectory_map[tid]) > 1:
                            self._valid_trajectories.add(tid)
                    
                    # 重置当前活跃轨迹
                    self._active_trajectory_id = -1
                    self._active_trajectory = []
        
        return indices

    def sample(self, batch_size, rng=default_rng()):
        """随机采样独立的样本"""
        indices = rng.choice(len(self), size=batch_size, replace=False)
        return self._get_samples_from_indices(indices)

    def sample_trajectories(self, batch_size, max_trajectory_length=None, rng=default_rng()):
        """
        高效的轨迹采样 - 直接根据轨迹ID采样，避免遍历所有轨迹
        """
        # 首先处理上一批次未完成的轨迹
        selected_indices = []
        
        if self._pending_trajectory is not None:
            remaining_indices = self._pending_trajectory[self._trajectory_position:]
            can_fit = min(len(remaining_indices), batch_size)
            selected_indices.extend(remaining_indices[:can_fit])
            
            self._trajectory_position += can_fit
            if self._trajectory_position >= len(self._pending_trajectory):
                self._pending_trajectory = None
                self._trajectory_position = 0
            
            if len(selected_indices) >= batch_size:
                return self._get_samples_from_indices(selected_indices[:batch_size])
        
        # 直接使用缓存的有效轨迹ID列表
        available_trajectories = list(self._valid_trajectories)
        
        if not available_trajectories:
            remaining = batch_size - len(selected_indices)
            if remaining > 0:
                valid_indices = np.arange(len(self))
                exclude = set(selected_indices)
                valid_choices = [i for i in valid_indices if i not in exclude]
                
                if valid_choices:
                    remaining_indices = rng.choice(
                        valid_choices, 
                        size=min(remaining, len(valid_choices)), 
                        replace=False
                    )
                    selected_indices.extend(remaining_indices)
            
            return self._get_samples_from_indices(selected_indices[:batch_size])
        
        # 打乱轨迹ID
        rng.shuffle(available_trajectories)
        
        remaining_space = batch_size - len(selected_indices)
        for tid in available_trajectories:
            trajectory = self._trajectory_map[tid]
            
            # 应用最大长度限制
            if max_trajectory_length is not None and len(trajectory) > max_trajectory_length:
                effective_trajectory = trajectory[:max_trajectory_length]
            else:
                effective_trajectory = trajectory
            
            # 处理轨迹太长的情况
            if len(effective_trajectory) > remaining_space:
                batch_portion = effective_trajectory[:remaining_space]
                selected_indices.extend(batch_portion)
                
                self._pending_trajectory = effective_trajectory
                self._trajectory_position = remaining_space
                
                return self._get_samples_from_indices(selected_indices)
            else:
                selected_indices.extend(effective_trajectory)
                remaining_space -= len(effective_trajectory)
                
                if remaining_space <= 0:
                    break
        
        # 如果所有轨迹都已处理但批次仍不满，随机填充
        if len(selected_indices) < batch_size:
            remaining = batch_size - len(selected_indices)
            valid_indices = np.arange(len(self))
            exclude = set(selected_indices)
            valid_choices = [i for i in valid_indices if i not in exclude]
            
            if valid_choices:
                remaining_indices = rng.choice(
                    valid_choices, 
                    size=min(remaining, len(valid_choices)), 
                    replace=False
                )
                selected_indices.extend(remaining_indices)
        
        return self._get_samples_from_indices(selected_indices[:batch_size])
    
    def sample_by_score(self, batch_size, high_score_prob=0.7, rng=default_rng()):
        """基于轨迹得分采样"""
        # 使用缓存的有效轨迹，而不是重新计算
        if len(self._trajectory_scores) < 2 or not self._valid_trajectories:
            return self.sample(batch_size, rng)
        
        # 只对已记录分数的有效轨迹排序
        valid_scored_trajectories = [tid for tid in self._valid_trajectories if tid in self._trajectory_scores]
        if len(valid_scored_trajectories) < 2:
            return self.sample(batch_size, rng)
            
        # 根据得分排序轨迹
        sorted_trajectories = sorted(
            valid_scored_trajectories,
            key=lambda tid: self._trajectory_scores[tid],
            reverse=True
        )
        
        # 将轨迹分为高分和低分两组
        split_point = max(1, int(len(sorted_trajectories) * 0.3))  # 前30%为高分
        high_score_trajectories = sorted_trajectories[:split_point]
        low_score_trajectories = sorted_trajectories[split_point:]
        
        # 打乱每组内的顺序
        rng.shuffle(high_score_trajectories)
        rng.shuffle(low_score_trajectories)
        
        selected_indices = []
        high_score_idx = 0
        low_score_idx = 0
        
        while len(selected_indices) < batch_size:
            # 决定选择高分还是低分轨迹
            if (low_score_idx >= len(low_score_trajectories)) or \
               (high_score_idx < len(high_score_trajectories) and rng.random() < high_score_prob):
                # 选择高分轨迹
                if high_score_idx < len(high_score_trajectories):
                    tid = high_score_trajectories[high_score_idx]
                    high_score_idx += 1
                    
                    # 获取轨迹样本
                    trajectory = self._trajectory_map.get(tid, [])
                    remaining = batch_size - len(selected_indices)
                    selected_indices.extend(trajectory[:remaining])
                    
                    if len(selected_indices) >= batch_size:
                        break
            else:
                # 选择低分轨迹
                if low_score_idx < len(low_score_trajectories):
                    tid = low_score_trajectories[low_score_idx]
                    low_score_idx += 1
                    
                    # 获取轨迹样本
                    trajectory = self._trajectory_map.get(tid, [])
                    remaining = batch_size - len(selected_indices)
                    selected_indices.extend(trajectory[:remaining])
                    
                    if len(selected_indices) >= batch_size:
                        break
            
            # 如果所有轨迹都已处理但仍不足
            if high_score_idx >= len(high_score_trajectories) and \
               low_score_idx >= len(low_score_trajectories):
                break
        
        # 如果仍不足，随机采样填充
        if len(selected_indices) < batch_size:
            remaining = batch_size - len(selected_indices)
            valid_indices = np.arange(len(self))
            exclude = set(selected_indices)
            valid_choices = [i for i in valid_indices if i not in exclude]
            
            if valid_choices:
                remaining_indices = rng.choice(
                    valid_choices, 
                    size=min(remaining, len(valid_choices)), 
                    replace=False
                )
                selected_indices.extend(remaining_indices)
        
        return self._get_samples_from_indices(selected_indices[:batch_size])

    def reset_trajectory_tracking(self):
        """重置轨迹跟踪状态"""
        self._pending_trajectory = None
        self._trajectory_position = 0

    def _get_samples_from_indices(self, indices):
        """从索引获取样本"""
        samples = self._replay[indices]

        adjacency = self.decode(samples['adjacency'], dtype=np.int_)
        next_adjacency = self.decode(samples['next_adjacency'], dtype=np.int_)

        # 获取样本对应的轨迹ID
        trajectory_ids = self._trajectory_id[indices]

        # Convert structured array into dictionary
        return {
            'adjacency': adjacency.astype(np.float32),
            'graph': to_graphs_tuple(adjacency),
            'num_edges': samples['num_edges'],
            'actions': samples['actions'],
            'delta_scores': samples['delta_scores'],
            'mask': self.decode(samples['mask']),
            'next_adjacency': next_adjacency.astype(np.float32),
            'next_graph': to_graphs_tuple(next_adjacency),
            'next_mask': self.decode(samples['next_mask']),
            'indices': indices,
            'trajectory_ids': trajectory_ids  # 添加轨迹ID信息
        }

    def __len__(self):
        return self.capacity if self._is_full else self._index

    @property
    def transitions(self):
        return self._replay[:len(self)]

    def save(self, filename):
        data = {
            'version': 6,  # 更新版本号
            'replay': self.transitions,
            'index': self._index,
            'is_full': self._is_full,
            'prev': self._prev,
            'next': self._next,
            'capacity': self.capacity,
            'num_variables': self.num_variables,
            'trajectory_id': self._trajectory_id,
            'trajectory_map': self._trajectory_map,
            'trajectory_scores': self._trajectory_scores,
            'next_trajectory_id': self._next_trajectory_id,
            'trajectory_lengths': self._trajectory_lengths,
            'valid_trajectories': list(self._valid_trajectories)  # 保存有效轨迹ID列表
        }
        np.savez_compressed(filename, **data)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            version = data['version'].item() if isinstance(data['version'], np.ndarray) else data['version']
            
            replay = cls(
                capacity=data['capacity'].item(),
                num_variables=data['num_variables'].item()
            )
            replay._index = data['index'].item()
            replay._is_full = data['is_full'].item()
            replay._prev = data['prev']
            replay._next = data['next']
            replay._replay[:len(replay)] = data['replay']
            
            if version >= 4:
                # 版本4及以上包含轨迹信息
                replay._trajectory_id = data['trajectory_id']
                if version >= 5:
                    # 版本5使用新的轨迹跟踪系统
                    replay._trajectory_map = data['trajectory_map'].item()
                    replay._trajectory_scores = data['trajectory_scores'].item()
                    replay._trajectory_lengths = data['trajectory_lengths'].item()
                    replay._next_trajectory_id = data['next_trajectory_id'].item()
                    
                    if version >= 6:
                        # 版本6包含有效轨迹缓存
                        replay._valid_trajectories = set(data['valid_trajectories'].tolist())
                    else:
                        # 从轨迹长度重建有效轨迹集合
                        replay._valid_trajectories = {tid for tid, length in replay._trajectory_lengths.items() if length > 1}
                else:
                    # 将版本4的轨迹数据转换为新格式
                    old_trajectories = data['trajectories'].item()
                    old_scores = data['trajectory_scores'].item()
                    replay._trajectory_map = {}
                    replay._trajectory_scores = {}
                    replay._trajectory_lengths = {}
                    replay._valid_trajectories = set()
                    
                    for tid, trajectory in old_trajectories.items():
                        replay._trajectory_map[tid] = trajectory
                        length = len(trajectory)
                        replay._trajectory_lengths[tid] = length
                        if length > 1:
                            replay._valid_trajectories.add(tid)
                        if tid in old_scores:
                            replay._trajectory_scores[tid] = old_scores[tid]
                    
                    replay._next_trajectory_id = data['next_trajectory_id'].item()
            else:
                # 旧版本，需要重建轨迹信息
                replay._rebuild_trajectory_info()
        
        return replay
    
    def _rebuild_trajectory_info(self):
        """从原始样本重建轨迹信息（用于旧版本兼容）"""
        # 重置轨迹跟踪数据
        self._next = np.full((self.capacity,), -1, dtype=np.int_)
        self._trajectory_id = np.full((self.capacity,), -1, dtype=np.int_)
        self._trajectory_map = {}
        self._trajectory_scores = {}
        self._trajectory_lengths = {}
        self._valid_trajectories = set()
        self._next_trajectory_id = 0
        
        # 重建next关系
        valid_indices = np.arange(len(self))
        for idx in valid_indices:
            prev_idx = self._prev[idx]
            if prev_idx >= 0:
                self._next[prev_idx] = idx
        
        # 找出所有轨迹起点
        start_indices = valid_indices[self._prev[valid_indices] == -1]
        
        # 构建每条轨迹
        for start_idx in start_indices:
            # 跟踪当前轨迹
            current_idx = start_idx
            trajectory = [current_idx]
            
            # 为轨迹分配ID
            tid = self._next_trajectory_id
            self._next_trajectory_id += 1
            
            # 记录样本所属轨迹
            self._trajectory_id[current_idx] = tid
            
            # 沿轨迹前进
            next_idx = self._next[current_idx]
            while next_idx >= 0:
                current_idx = next_idx
                trajectory.append(current_idx)
                self._trajectory_id[current_idx] = tid
                next_idx = self._next[current_idx]
            
            # 只记录长度大于1的轨迹
            if len(trajectory) > 1:
                self._trajectory_map[tid] = trajectory
                self._trajectory_lengths[tid] = len(trajectory)
                self._valid_trajectories.add(tid)  # 添加到有效轨迹集合
                
                # 计算轨迹平均得分
                scores = [self._replay['scores'][idx][0] for idx in trajectory]
                self._trajectory_scores[tid] = np.mean(scores)
        
        print(f"重建了 {len(self._trajectory_map)} 条轨迹信息，其中有效轨迹 {len(self._valid_trajectories)} 条")

    def encode(self, decoded):
        encoded = decoded.reshape(-1, self.num_variables ** 2)
        return np.packbits(encoded, axis=1)

    def decode(self, encoded, dtype=np.float32):
        decoded = np.unpackbits(encoded, axis=-1, count=self.num_variables ** 2)
        decoded = decoded.reshape(*encoded.shape[:-1], self.num_variables, self.num_variables)
        return decoded.astype(dtype)

    @property
    def dummy(self):
        shape = (1, self.num_variables, self.num_variables)
        graph = GraphsTuple(
            nodes=np.arange(self.num_variables),
            edges=np.zeros((1,), dtype=np.int_),
            senders=np.zeros((1,), dtype=np.int_),
            receivers=np.zeros((1,), dtype=np.int_),
            globals=None,
            n_node=np.full((1,), self.num_variables, dtype=np.int_),
            n_edge=np.ones((1,), dtype=np.int_),
        )
        adjacency = np.zeros(shape, dtype=np.float32)
        return {
            'adjacency': adjacency,
            'graph': graph,
            'num_edges': np.zeros((1,), dtype=np.int_),
            'actions': np.zeros((1,), dtype=np.int_),
            'delta_scores': np.zeros((1,), dtype=np.float_),
            'mask': np.zeros(shape, dtype=np.float32),
            'next_adjacency': adjacency,
            'next_graph': graph,
            'next_mask': np.zeros(shape, dtype=np.float32)
        }
