# cafe_torchrec/embedding.py

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict
import traceback

from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

try:
    from .hotsketch_manager import HotSketchManager
except ImportError:
    from hotsketch_manager import HotSketchManager


class CafeEmbeddingBagCollection(nn.Module):
    """
    使用 TorchRec EmbeddingBagCollection 实现 CAFE 嵌入逻辑。
    根据 HotSketch 的运行时重要性分数，动态将嵌入查找路由到“热”表或“哈希”表。
    """

    def __init__(
        self,
        embedding_dim: int,
        hot_feature_count: int,
        hash_table_size: int,
        hotsketch_manager: HotSketchManager,
        device: torch.device,
        hot_table_name: str = "hot_embeddings",
        hash_table_name: str = "hash_embeddings",
    ):
        super().__init__()

        if hot_feature_count <= 0 or hash_table_size <= 0:
            raise ValueError("hot_feature_count 和 hash_table_size 必须为正数。")

        self.embedding_dim = embedding_dim
        # C++ HotSketch 实现中热 ID 从 1 开始分配，因此 +1 留出空间
        # (注意: 如果已经在 C++ 中修复为从 0 开始，这里可能不需要 +1，但保留它通常是安全的)
        self.hot_feature_count = hot_feature_count + 1
        self.hash_table_size = hash_table_size
        self.manager = hotsketch_manager
        self.hot_table_name = hot_table_name
        self.hash_table_name = hash_table_name

        # 定义用于 KJT 和 EBC 路由的唯一特征键
        self._hot_feature_key = f"{hot_table_name}_feature_key"
        self._hash_feature_key = f"{hash_table_name}_feature_key"

        # 配置热表和哈希表
        hot_config = EmbeddingBagConfig(
            name=self.hot_table_name,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.hot_feature_count,
            feature_names=[self._hot_feature_key],
        )

        hash_config = EmbeddingBagConfig(
            name=self.hash_table_name,
            embedding_dim=self.embedding_dim,
            num_embeddings=self.hash_table_size,
            feature_names=[self._hash_feature_key],
        )

        self.ebc = EmbeddingBagCollection(
            tables=[hot_config, hash_config],
            device=device,
        )

        # 反向传播所需状态
        self._input_ids_for_update: Optional[np.ndarray] = None
        self._query_results_for_update: Optional[np.ndarray] = None
        self._needs_grad_processing = False

    def forward(self,
                feature_ids: torch.Tensor,
                offsets: torch.Tensor,
                per_sample_weights: Optional[torch.Tensor] = None
               ) -> torch.Tensor:
        """
        使用 HotSketch 和 EBC 执行动态嵌入查找。
        """
        if per_sample_weights is not None:
             raise NotImplementedError("CafeEmbeddingBagCollection 不支持 per_sample_weights。")
        
        self._needs_grad_processing = True

        # 确保输入在 CPU 上以进行 HotSketch 查询
        feature_ids_cpu = feature_ids.cpu() if feature_ids.device.type != 'cpu' else feature_ids
        offsets_cpu = offsets.cpu() if offsets.device.type != 'cpu' else offsets

        batch_size = len(offsets_cpu)
        num_indices_total = len(feature_ids_cpu)

        if batch_size <= 0 or num_indices_total == 0:
             self._input_ids_for_update = None
             self._query_results_for_update = None
             self._needs_grad_processing = False
             return torch.zeros((batch_size, self.embedding_dim), device=self.ebc.device)

        # 1. 查询 HotSketch
        feature_ids_np = feature_ids_cpu.numpy().astype(np.uint32)
        query_results_np = self.manager.query(feature_ids_np)
        query_results = torch.from_numpy(query_results_np).to(torch.int32)

        # 存储状态供反向传播使用
        self._input_ids_for_update = feature_ids_np
        self._query_results_for_update = query_results_np

        # 2. 分离热/冷 ID
        is_hot_mask_cpu = query_results < 0
        is_cold_mask_cpu = ~is_hot_mask_cpu

        hot_indices = torch.abs(query_results[is_hot_mask_cpu]).to(torch.long)
        
        cold_original_ids = query_results[is_cold_mask_cpu]
        cold_indices = (cold_original_ids.to(torch.long) % self.hash_table_size) if cold_original_ids.numel() > 0 else torch.tensor([], dtype=torch.long)

        # 合并为 KJT values，并确保在正确的设备上
        kjt_values = torch.cat((hot_indices, cold_indices)).to(self.ebc.device, non_blocking=True)

        # 3. 计算 KJT lengths
        offsets_with_end = torch.cat((offsets_cpu, torch.tensor([num_indices_total], dtype=torch.long)))
        sample_lengths = offsets_with_end[1:] - offsets_with_end[:-1]
        
        sample_indices_map = torch.repeat_interleave(
            torch.arange(batch_size, dtype=torch.long), 
            repeats=sample_lengths.to(torch.long)
        )
        
        hot_lengths = torch.bincount(sample_indices_map, weights=is_hot_mask_cpu.to(torch.int64), minlength=batch_size).to(torch.long)
        cold_lengths = torch.bincount(sample_indices_map, weights=is_cold_mask_cpu.to(torch.int64), minlength=batch_size).to(torch.long)

        # 4. 构造 KeyedJaggedTensor
        kjt_lengths = torch.cat((hot_lengths, cold_lengths)).to(self.ebc.device, non_blocking=True)
        kjt = KeyedJaggedTensor(
            keys=[self._hot_feature_key, self._hash_feature_key],
            values=kjt_values,
            lengths=kjt_lengths,
        )

        # 5. EBC 前向传播
        pooled_embeddings_keyed: KeyedTensor = self.ebc(kjt)

        # 6. 合并结果
        return pooled_embeddings_keyed[self._hot_feature_key] + pooled_embeddings_keyed[self._hash_feature_key]

    def update_and_migrate(self):
        """
        反向传播后逻辑：获取梯度，计算重要性，更新 Sketch，并执行权重迁移。
        """
        if not self._needs_grad_processing:
            return

        if self._input_ids_for_update is None or self._query_results_for_update is None:
            # print("  Warning: No forward pass data available stored for update. Skipping.")
            self._needs_grad_processing = False
            return

        scores_np = np.zeros_like(self._input_ids_for_update, dtype=np.float32)
        grad_map: Dict[np.uint32, float] = {}

        try:
            # --- Step 1 & 2: 获取梯度并计算范数 ---
            hot_table_weights = None
            hash_table_weights = None
            found_params = 0

            for name, param in self.ebc.named_parameters():
                is_hot = name == f"embedding_bags.{self.hot_table_name}.weight"
                is_hash = name == f"embedding_bags.{self.hash_table_name}.weight"

                if is_hot: hot_table_weights = param
                if is_hash: hash_table_weights = param

                if (is_hot or is_hash) and param.grad is not None:
                    found_params += 1
                    grad = param.grad

                    # --- [DEBUG] NaN/Inf check commented out for performance ---
                    # if torch.isnan(grad).any() or torch.isinf(grad).any():
                    #      print(f"\n🚨 [CRITICAL ERROR] NaN or Inf found in gradients for table: {name}!")
                    #      raise ValueError("NaN/Inf detected in gradients, stopping before crash.")
                    # -------------------------

                    is_target_mask = (self._query_results_for_update < 0) if is_hot else ~(self._query_results_for_update < 0)
                    original_ids_in_table = self._input_ids_for_update[is_target_mask]
                    query_results_in_table = self._query_results_for_update[is_target_mask]

                    if grad.is_sparse:
                        indices = grad._indices().squeeze().cpu().numpy().astype(np.int64)
                        norms = torch.norm(grad._values().detach(), p=2, dim=1).cpu().numpy()
                        table_idx_to_norm = dict(zip(indices, norms))

                        for i, original_id in enumerate(original_ids_in_table):
                            table_idx = np.int64(abs(query_results_in_table[i])) if is_hot else np.int64(original_id % self.hash_table_size)
                            if table_idx in table_idx_to_norm:
                                grad_map[original_id] = grad_map.get(original_id, 0.0) + table_idx_to_norm[table_idx]

                    elif not grad.is_sparse:
                        grad_dense = grad.detach()
                        num_rows = grad_dense.shape[0]
                        
                        for i, original_id in enumerate(original_ids_in_table):
                            if is_hot:
                                raw_query_val = query_results_in_table[i]
                                table_idx = np.int64(abs(raw_query_val))
                                
                                # --- [DEBUG] Bounds check commented out for performance ---
                                # if table_idx >= num_rows:
                                #     # ... (error printing logic)
                                #     raise ValueError("Hot table index out of bounds detected in Python!")
                                # --------------------

                            else:
                                table_idx = np.int64(original_id % num_rows)
                                
                                # --- [DEBUG] Bounds check commented out for performance ---
                                # if table_idx >= num_rows:
                                #     # ... (error printing logic)
                                #     raise ValueError("Hash table index out of bounds detected!")
                                # --------------------

                            # Basic protection against crash here if logic is wrong, though less likely now
                            if table_idx < num_rows:
                                norm = torch.norm(grad_dense[table_idx], p=2).item()
                                if norm > 0:
                                    grad_map[original_id] = grad_map.get(original_id, 0.0) + norm

            # if found_params < 2:
            #    pass # print("Warning: Could not find both hot and hash weight parameters.")

            for i, original_id in enumerate(self._input_ids_for_update):
                scores_np[i] = grad_map.get(original_id, 0.0)

        except Exception as e:
            print(f"Error during gradient processing: {e}")
            traceback.print_exc()
            self._needs_grad_processing = False
            return

        # --- Step 3: 更新 HotSketch 并获取迁移掩码 ---
        migration_mask_np = self.manager.update(self._input_ids_for_update, scores_np.astype(np.float32))
        migrated_indices = np.where(migration_mask_np == 1)[0]

        # --- Step 4: 执行迁移 (权重复制) ---
        if len(migrated_indices) > 0:
            migrated_original_ids_np = self._input_ids_for_update[migrated_indices]
            
            # 再次查询以获取最新的热特征 ID
            new_query_results_np = self.manager.query(migrated_original_ids_np.astype(np.uint32))
            
            # === [CRITICAL FIX] 严格过滤 ===
            # 只有当 query 返回负数 (确实是热特征) 且在有效范围内时，才进行迁移
            # 这直接修复了日志中观察到的 bug (原始 ID 被当成了热特征索引)
            valid_hot_mask = (new_query_results_np < 0) & (new_query_results_np >= -self.hot_feature_count)
            
            # if not np.all(valid_hot_mask):
            #      print(f"Warning: Filtered {np.sum(~valid_hot_mask)} invalid migration candidates.")

            # 应用过滤
            final_original_ids_np = migrated_original_ids_np[valid_hot_mask]
            final_hot_indices_np = np.abs(new_query_results_np[valid_hot_mask])
            
            if len(final_original_ids_np) > 0:
                hash_table_size_actual = hash_table_weights.shape[0] # 获取哈希表的实际大小
                source_hash_indices_np = final_original_ids_np % hash_table_size_actual
                target_device = self.ebc.device
                
                migrated_hot_indices = torch.from_numpy(final_hot_indices_np).to(torch.long).to(target_device)
                source_hash_indices = torch.from_numpy(source_hash_indices_np).to(torch.long).to(target_device)

                try:
                    # if hot_table_weights is None or hash_table_weights is None:
                    #      raise AttributeError("Migration failed: Weight parameters not found.")

                    # --- [PERFORMANCE] Removed explicit synchronize calls ---
                    # if self.ebc.device.type == 'cuda':
                    #    torch.cuda.synchronize()
                    # --------------------------------------------

                    with torch.no_grad():
                        source_vectors = hash_table_weights.detach()[source_hash_indices]
                        hot_table_weights[migrated_hot_indices] = source_vectors

                    # --- [PERFORMANCE] Removed explicit synchronize calls ---
                    # if self.ebc.device.type == 'cuda':
                    #    torch.cuda.synchronize()
                    # --------------------------------------------

                except Exception as e:
                    print(f"ERROR during weight copy: {e}")
                    traceback.print_exc()

        self._input_ids_for_update = None
        self._query_results_for_update = None
        self._needs_grad_processing = False

if __name__ == '__main__':
    print("Example usage requires setting up torch.distributed and a valid HotSketchManager.")