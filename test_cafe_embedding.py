# test_cafe_embedding.py

import torch
import torch.nn as nn
import numpy as np
import os
import ctypes
import time

# --- 导入我们需要测试的模块 ---
# 假设这些文件在 'cafe_torchrec' 子目录下
try:
    from cafe_torchrec.hotsketch_manager import HotSketchManager
    from cafe_torchrec.embedding import CafeEmbeddingBagCollection
    # 尝试导入 torchrec 以检查其是否安装
    import torchrec
    print(f"PyTorch Version: {torch.__version__}")
    print(f"TorchRec imported successfully.")
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保你已经正确安装了 torchrec 并且在正确的目录下运行此脚本。")
    print("确保 cafe_torchrec 目录下有 hotsketch_manager.py 和 embedding.py。")
    exit(1)
except FileNotFoundError as e:
    print(f"文件未找到错误: {e}")
    print("请确保 C++ 库路径正确或 sketch.cpp 文件存在于预期位置。")
    exit(1)


# --- 配置区 ---
EMBEDDING_DIM = 16       # 嵌入维度 (示例)
HOT_FEATURE_COUNT = 100  # 热特征槽位数量 (示例)
HASH_TABLE_SIZE = 1000   # 哈希表大小 (示例)
INITIAL_THRESHOLD = 5.0  # HotSketch 初始阈值
ADJUST_THRESHOLD = True  # 是否动态调整阈值
ALPHA = 1.0000005        # HotSketch 衰减因子

# C++ 库文件路径 (相对于脚本位置)
CPP_SOURCE_PATH = os.path.join("cafe_torchrec", "cpp", "sketch.cpp")
LIB_OUTPUT_PATH = os.path.join("cafe_torchrec", "cpp", "libhotsketch.so")

# --- 辅助函数：编译 C++ 代码 ---
def compile_hotsketch_lib(source_path, output_path):
    if not os.path.exists(source_path):
        print(f"错误: C++ 源文件未找到: {source_path}")
        return False
    # 如果库已存在，则跳过编译 (可按需修改为强制重新编译)
    # if os.path.exists(output_path):
    #     print(f"库文件已存在: {output_path}. 跳过编译.")
    #     return True

    print(f"正在编译 {source_path} -> {output_path}...")
    compile_cmd = (
        f"g++ -fPIC -shared -o {output_path} -g -rdynamic -mavx2 -mbmi "
        f"-mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp {source_path}"
    )
    print(f"运行编译命令: {compile_cmd}")
    return_code = os.system(compile_cmd)
    if return_code != 0:
        print("C++ 代码编译失败!")
        return False
    else:
        print("C++ 代码编译成功.")
        return True

# --- 测试主逻辑 ---
if __name__ == "__main__":
    print("开始测试 CafeEmbeddingBagCollection 初始化和前向传播...")

    # 1. 编译 C++ 库 (如果需要)
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
        print("初始化 HotSketchManager...")
        manager = HotSketchManager(
            library_path=LIB_OUTPUT_PATH,
            hot_feature_count=HOT_FEATURE_COUNT,
            initial_threshold=INITIAL_THRESHOLD,
            adjust_threshold=ADJUST_THRESHOLD,
            alpha=ALPHA
        )
        print("HotSketchManager 初始化成功.")
    except (FileNotFoundError, OSError, TypeError, ValueError) as e:
        print(f"HotSketchManager 初始化失败: {e}")
        exit(1)

    # 4. 初始化 CafeEmbeddingBagCollection
    try:
        print("初始化 CafeEmbeddingBagCollection...")
        cafe_ebc = CafeEmbeddingBagCollection(
            embedding_dim=EMBEDDING_DIM,
            hot_feature_count=HOT_FEATURE_COUNT,
            hash_table_size=HASH_TABLE_SIZE,
            hotsketch_manager=manager,
            device=device, # 将模块直接创建在目标设备上
            hot_table_name="test_hot_table",
            hash_table_name="test_hash_table"
        )
        # 将模型移到设备 (虽然 EBC 初始化时已指定 device，这确保整个 Module 在正确设备上)
        cafe_ebc.to(device)
        print("CafeEmbeddingBagCollection 初始化成功.")
        print("模型结构:")
        print(cafe_ebc)
        print("-" * 30)
    except Exception as e:
        print(f"CafeEmbeddingBagCollection 初始化失败: {e}")
        exit(1)

    # 5. (可选) 模拟一些更新，使 HotSketch 状态更有趣
    print("模拟一些 HotSketch 更新...")
    try:
        # 第一次更新，让 ID=20 变热
        update_ids_1 = np.array([10, 20, 30, 20, 40], dtype=np.uint32)
        update_scores_1 = np.array([2.0, 6.0, 1.0, 7.0, 8.0], dtype=np.float32) # > threshold
        manager.update(update_ids_1, update_scores_1)
        # 第二次更新，让 ID=40 变热，ID=10 接近但不超过
        update_ids_2 = np.array([50, 40, 10, 60], dtype=np.uint32)
        update_scores_2 = np.array([1.0, 9.0, 4.5, 3.0], dtype=np.float32)
        manager.update(update_ids_2, update_scores_2)
        print("模拟更新完成.")
    except Exception as e:
        print(f"模拟更新时出错: {e}")
        # 继续测试 forward

    # 6. 准备测试输入数据 (在 CPU 上)
    # 批次大小 = 3
    # 样本 0: [10, 20]    (ID 10 可能接近阈值, ID 20 应该是热的)
    # 样本 1: []          (空包)
    # 样本 2: [30, 40, 10] (ID 30 冷, ID 40 热, ID 10 可能接近阈值)
    test_feature_ids = torch.tensor([10, 20, 30, 40, 10], dtype=torch.long)
    test_offsets = torch.tensor([0, 2, 2, 5], dtype=torch.long) # 注意最后一个偏移量是总长度
    expected_batch_size = len(test_offsets) - 1

    print("-" * 30)
    print("准备测试输入:")
    print(f"  Feature IDs (CPU): {test_feature_ids}")
    print(f"  Offsets (CPU): {test_offsets}")
    print(f"  预期批次大小: {expected_batch_size}")
    print("-" * 30)

    # 7. 调用 forward 方法
    try:
        print("调用 CafeEmbeddingBagCollection.forward...")
        # 确保模型在评估模式（如果未来有 Dropout 等层）
        cafe_ebc.eval()
        with torch.no_grad(): # 测试前向传播不需要计算梯度
            output_embeddings = cafe_ebc(test_feature_ids, test_offsets)
        print("Forward 调用成功!")
        print("-" * 30)

        # 8. 检查输出
        print("检查输出:")
        print(f"  输出张量形状: {output_embeddings.shape}")
        print(f"  输出张量设备: {output_embeddings.device}")

        # 验证形状是否正确
        expected_shape = (expected_batch_size, EMBEDDING_DIM)
        if output_embeddings.shape == expected_shape:
            print("✅ 输出形状正确!")
        else:
            print(f"❌ 输出形状错误! 预期: {expected_shape}, 实际: {output_embeddings.shape}")

        # 验证设备是否正确
        if str(output_embeddings.device) == str(device):
             print("✅ 输出设备正确!")
        else:
             print(f"❌ 输出设备错误! 预期: {device}, 实际: {output_embeddings.device}")

    except Exception as e:
        print(f"❌ 调用 Forward 时发生错误: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误追踪信息
        exit(1)

    print("-" * 30)
    print("初步测试完成。")