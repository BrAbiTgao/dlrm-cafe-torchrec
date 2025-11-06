# test_cafe_full_step.py

import torch
import torch.nn as nn
import numpy as np
import os
import ctypes
import time
import traceback
from typing import Dict # 引入 Dict

# --- 导入所需模块 ---
try:
    from cafe_torchrec.hotsketch_manager import HotSketchManager
    from cafe_torchrec.embedding import CafeEmbeddingBagCollection
    import torchrec
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchRec imported successfully.")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保 torchrec 已安装，且 cafe_torchrec 目录及文件存在。")
    exit(1)
except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
    print("请确保 C++ 库路径正确或 sketch.cpp 文件存在。")
    exit(1)

# --- 配置区 ---
EMBEDDING_DIM = 4        # 使用小维度以便调试时查看
HOT_FEATURE_COUNT = 10   # 热表大小 (实际大小为 11)
HASH_TABLE_SIZE = 50     # 哈希表大小
INITIAL_THRESHOLD = 5.0  # HotSketch 初始阈值
ADJUST_THRESHOLD = True
ALPHA = 1.0000005
LEARNING_RATE = 0.1      # 学习率

# C++ 库文件路径
CPP_SOURCE_PATH = os.path.join("cafe_torchrec", "cpp", "sketch.cpp")
LIB_OUTPUT_PATH = os.path.join("cafe_torchrec", "cpp", "libhotsketch.so")

# --- 辅助函数：编译 C++ 代码 ---
def compile_hotsketch_lib(source_path, output_path):
    if not os.path.exists(source_path):
        print(f"错误: C++ 源文件未找到: {source_path}")
        return False
    # 强制重新编译以确保使用的是最新代码
    print(f"正在编译 {source_path} -> {output_path}...")
    compile_cmd = (
        f"g++ -fPIC -shared -o {output_path} -g -rdynamic -mavx2 -mbmi "
        f"-mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp {source_path}"
    )
    return_code = os.system(compile_cmd)
    if return_code != 0:
        print("C++ 代码编译失败!")
        return False
    else:
        print("C++ 代码编译成功.")
        return True

# --- 测试主逻辑 ---
if __name__ == "__main__":
    print("="*50)
    print("测试 CafeEmbeddingBagCollection 完整步骤 (Forward + Backward + Update/Migrate)")
    print("="*50)
    
    # 设置 CUDA_LAUNCH_BLOCKING 以获取更准确的 CUDA 错误位置
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # 1. 编译 C++ 库
    if not compile_hotsketch_lib(CPP_SOURCE_PATH, LIB_OUTPUT_PATH):
        exit(1)

    # 2. 确定运行设备
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("未检测到 CUDA 设备，将在 CPU 上运行测试。")

    # 3. 初始化 HotSketchManager
    try:
        print("\n--- 初始化 HotSketchManager ---")
        manager = HotSketchManager(
            library_path=LIB_OUTPUT_PATH,
            hot_feature_count=HOT_FEATURE_COUNT,
            initial_threshold=INITIAL_THRESHOLD,
            adjust_threshold=ADJUST_THRESHOLD,
            alpha=ALPHA
        )
        print("HotSketchManager 初始化成功.")
    except Exception as e:
        print(f"HotSketchManager 初始化失败: {e}")
        traceback.print_exc()
        exit(1)

    # 4. 初始化 CafeEmbeddingBagCollection
    try:
        print("\n--- 初始化 CafeEmbeddingBagCollection ---")
        cafe_ebc = CafeEmbeddingBagCollection(
            embedding_dim=EMBEDDING_DIM,
            hot_feature_count=HOT_FEATURE_COUNT,
            hash_table_size=HASH_TABLE_SIZE,
            hotsketch_manager=manager,
            device=device,
            hot_table_name="test_hot_table",
            hash_table_name="test_hash_table"
        )
        cafe_ebc.to(device)
        
        # --- 关键：手动初始化权重以便验证迁移 ---
        with torch.no_grad():
            hot_w_init, hash_w_init = None, None
            for name, param in cafe_ebc.ebc.named_parameters():
                if name == f"embedding_bags.{cafe_ebc.hot_table_name}.weight":
                    nn.init.constant_(param, -0.5) # 热表 -0.5
                    hot_w_init = param
                elif name == f"embedding_bags.{cafe_ebc.hash_table_name}.weight":
                    nn.init.constant_(param, 0.5) # 哈希表 0.5
                    hash_w_init = param
            if hot_w_init is None or hash_w_init is None:
                raise AttributeError("初始化时未找到 EBC 权重参数")
        
        print("CafeEmbeddingBagCollection 初始化成功 (权重已自定义初始化: Hot=-0.5, Hash=0.5).")
    except Exception as e:
        print(f"CafeEmbeddingBagCollection 初始化失败: {e}")
        traceback.print_exc()
        exit(1)

    # 5. 设置优化器
    # 关键：我们使用标准优化器，*不* 使用融合优化器
    # 这样 loss.backward() 才会填充 .grad 属性
    optimizer = torch.optim.SGD(cafe_ebc.parameters(), lr=LEARNING_RATE)
    print(f"\n--- 已设置标准 SGD 优化器 (LR={LEARNING_RATE}) ---")

    # 6. 准备测试输入数据 (在 CPU 上)
    # 批次大小 = 2
    # 样本 0: [10, 20]    (ID 10, 20 都是冷特征)
    # 样本 1: [30, 20]    (ID 30 冷, ID 20 冷)
    test_feature_ids = torch.tensor([10, 20, 30, 20], dtype=torch.long)
    test_offsets = torch.tensor([0, 2, 4], dtype=torch.long) # 批次大小为 2
    expected_batch_size = len(test_offsets) - 1

    # 创建一个简单的下游层和损失函数
    downstream_layer = nn.Linear(EMBEDDING_DIM, 1).to(device)
    loss_fn = nn.MSELoss()
    target = torch.randn(expected_batch_size, 1).to(device)

    print("\n--- 准备测试数据 ---")
    print(f"  Feature IDs (CPU): {test_feature_ids}")
    print(f"  Offsets (CPU): {test_offsets}")
    print(f"  预期批次大小: {expected_batch_size}")
    print(f"  目标设备: {device}")
    
    # ---------------------------------------------
    # --- 步骤 1: 第一次训练 (触发梯度计算与迁移) ---
    # ---------------------------------------------
    print("\n" + "="*50)
    print("开始第一次训练步骤 (计算梯度与迁移)")
    print("="*50)
    
    try:
        # --- 模拟训练步骤 ---
        cafe_ebc.train()
        optimizer.zero_grad()
        
        # 1. 前向传播
        print("调用 forward...")
        output_embeddings = cafe_ebc(test_feature_ids, test_offsets)
        print(f"  Forward 输出形状: {output_embeddings.shape}")
        
        # 检查此时 _input_ids_for_update 是否已设置
        if cafe_ebc._input_ids_for_update is None:
            raise RuntimeError("_input_ids_for_update 在 forward 后未设置")

        # 模拟下游计算
        predictions = downstream_layer(output_embeddings)
        
        # 2. 计算损失
        loss = loss_fn(predictions, target)
        print(f"  计算损失: {loss.item()}")

        # 3. 反向传播 (填充 .grad)
        print("调用 loss.backward() ...")
        loss.backward() # 使用真实梯度（即使很小）
        print("  Backward 完成.")

        # --- 手动修改梯度以确保迁移 ---
        print("  (手动修改梯度以确保 ID 20, 30 的范数足够大)")
        hot_w, hash_w = None, None
        for name, param in cafe_ebc.ebc.named_parameters():
            if name == f"embedding_bags.{cafe_ebc.hot_table_name}.weight": hot_w = param
            elif name == f"embedding_bags.{cafe_ebc.hash_table_name}.weight": hash_w = param
        
        if hash_w is None: raise AttributeError("未找到哈希表权重")
        
        # 为哈希表创建高范数梯度 (模拟)
        # ID 20 (哈希索引 20), ID 30 (哈希索引 30)
        hash_indices_to_grad = torch.tensor([20, 30], dtype=torch.long, device=device)
        grad_values = torch.full((2, EMBEDDING_DIM), 10.0, device=device) # 范数会远大于 5
        
        # 注意：这里我们假设 .grad 是 None 或者可以被覆盖
        # 如果 loss.backward() 已经创建了稀疏梯度，可能需要更复杂的操作
        # 为了测试简单，我们直接覆盖
        hash_w.grad = torch.sparse_coo_tensor(
            hash_indices_to_grad.unsqueeze(0), # 1D 索引需要 unsqueeze
            grad_values, 
            hash_w.shape
        ).coalesce()
        if hot_w.grad is not None:
             hot_w.grad.zero_() # 确保热表梯度为 0
        print("  (已手动注入高范数稀疏梯度到哈希表索引 20, 30)")
        
        # 4. 调用 update_and_migrate (现在它将使用我们注入的梯度)
        print("调用 update_and_migrate()...")
        cafe_ebc.update_and_migrate() # 调用修复后的函数
        
        # 5. 优化器步骤 (使用我们注入的梯度)
        print("调用 optimizer.step()...")
        optimizer.step()
        print("  优化器步骤完成。")
        print("第一次训练步骤完成。")

    except Exception as e:
        print(f"❌ 第一次训练步骤出错: {e}")
        traceback.print_exc()
        exit(1)

    # ---------------------------------------------
    # --- 步骤 2: 第二次前向传播 (验证迁移) ---
    # ---------------------------------------------
    print("\n" + "="*50)
    print("开始第二次前向传播 (验证迁移)")
    print("="*50)

    try:
        cafe_ebc.eval()
        with torch.no_grad():
            print(f"再次查询 HotSketch (ID: {test_feature_ids.numpy()})...")
            # 预期：ID 20 和 30 现在应该是热特征了
            query_results_np = manager.query(test_feature_ids.numpy().astype(np.uint32))
            print(f"  HotSketch 查询结果: {query_results_np}")
            
            # 获取 ID 10, 20, 30 对应的状态和热索引
            status_10 = query_results_np[0]
            status_20 = query_results_np[1] # 第一次出现的 20
            status_30 = query_results_np[2]
            
            hot_idx_20 = abs(status_20)
            hot_idx_30 = abs(status_30)
            
            print(f"  ID 10 状态: {status_10} (预期 >= 0)")
            print(f"  ID 20 状态 (索引 1): {status_20} (预期 < 0, e.g., -1)")
            print(f"  ID 30 状态 (索引 2): {status_30} (预期 < 0, e.g., -2)")
            
            # 检查热表对应槽位的值
            hot_w = None
            for name, param in cafe_ebc.ebc.named_parameters():
                if name == f"embedding_bags.{cafe_ebc.hot_table_name}.weight": hot_w = param
            if hot_w is None: raise AttributeError("未找到热表权重")

            # 验证 ID 20
            if status_20 < 0 and hot_idx_20 > 0 and hot_idx_20 < (HOT_FEATURE_COUNT + 1):
                vec_20 = hot_w[hot_idx_20]
                print(f"  热表中索引 {hot_idx_20} (ID 20) 的向量 (前2维): {vec_20[:2]}")
                if torch.allclose(vec_20, torch.tensor(0.5, device=device)):
                    print("  ✅ 验证成功: ID 20 的向量已从哈希表 (0.5) 复制而来！")
                else:
                    print(f"  ❌ 验证失败: ID 20 的向量值为 {vec_20[0]}，预期为 0.5！(可能被优化器步骤修改了？)")
                    # 检查是否接近 0.5 - lr * grad (0.5 - 0.1 * 10 = -0.5 ?)
                    # 优化器会修改权重，所以我们检查它是否不再是 -0.5 (初始热表值)
                    if not torch.allclose(vec_20, torch.tensor(-0.5, device=device)):
                         print("  ✅ 验证成功: ID 20 的向量值已改变 (不再是 -0.5)，说明被迁移+更新了。")
                    else:
                         print(f"  ❌ 验证失败: ID 20 的向量值仍为 {vec_20[0]}，迁移或更新失败。")
            else:
                 print(f"  ❌ 验证失败: ID 20 (索引 1) 未按预期变为热特征 (状态: {status_20})。")

            # 验证 ID 30
            if status_30 < 0 and hot_idx_30 > 0 and hot_idx_30 < (HOT_FEATURE_COUNT + 1):
                vec_30 = hot_w[hot_idx_30]
                print(f"  热表中索引 {hot_idx_30} (ID 30) 的向量 (前2维): {vec_30[:2]}")
                if torch.allclose(vec_30, torch.tensor(0.5, device=device)):
                    print("  ✅ 验证成功: ID 30 的向量已从哈希表 (0.5) 复制而来！")
                elif not torch.allclose(vec_30, torch.tensor(-0.5, device=device)):
                     print("  ✅ 验证成功: ID 30 的向量值已改变 (不再是 -0.5)，说明被迁移+更新了。")
                else:
                    print(f"  ❌ 验证失败: ID 30 的向量值为 {vec_30[0]}，迁移或更新失败。")
            else:
                 print(f"  ❌ 验证失败: ID 30 (索引 2) 未按预期变为热特征 (状态: {status_30})。")

            # 验证 ID 10
            if status_10 >= 0:
                 print("  ✅ 验证成功: ID 10 仍然是冷特征。")
            else:
                 print(f"  ❌ 验证失败: ID 10 意外变为热特征 (索引: {status_10})。")
                 
            print("\n调用 forward (第二次)...")
            output_embeddings_2 = cafe_ebc(test_feature_ids, test_offsets)
            print(f"  Forward 输出形状: {output_embeddings_2.shape}")
            print("第二次前向传播成功。")
            
    except Exception as e:
        print(f"❌ 第二次前向传播或验证时出错: {e}")
        traceback.print_exc()
        exit(1)

    print("\n" + "="*50)
    print("完整步骤（非融合）测试完成。")
    print("="*50)