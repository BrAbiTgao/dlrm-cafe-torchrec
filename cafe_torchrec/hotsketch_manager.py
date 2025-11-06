# cafe_torchrec/hotsketch_manager.py

import ctypes
import numpy as np
import os
from typing import Optional

class HotSketchManager:
    """
    管理 C++ HotSketch 共享库的交互。

    加载 C++ 库, 定义 ctypes 签名, 并提供 Python 接口
    来调用核心 HotSketch 操作 (init, query, update)。
    """

    def __init__(
        self,
        library_path: str,
        hot_feature_count: int,
        initial_threshold: float,
        adjust_threshold: bool,
        alpha: float
    ):
        """
        初始化 HotSketchManager 和底层的 C++ HotSketch 对象。

        Args:
            library_path (str): C++ 共享库 (.so) 路径。
            hot_feature_count (int): 热特征的目标数量 (容量 'n' 或 'lim')。
            initial_threshold (float): 区分冷热特征的初始分数阈值。
            adjust_threshold (bool): 是否动态调整阈值。
            alpha (float): 分数衰减因子。
        """
        if not os.path.exists(library_path):
            raise FileNotFoundError(f"HotSketch 库未找到: {library_path}。 "
                                    "请确保 sketch.cpp 已正确编译。")

        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as e:
            raise OSError(f"加载 HotSketch 库失败: {library_path}: {e}")

        self.hot_feature_count = hot_feature_count
        self.initial_threshold = initial_threshold
        self.adjust_threshold = int(adjust_threshold) # 转换为 C int
        self.alpha = alpha

        # --- 定义 C 函数 ---

        self._init_func = self.lib.init
        self._init_func.argtypes = [
            ctypes.c_int,      # n (hot_feature_count)
            ctypes.c_float,      # Threshold (阈值)
            ctypes.c_int,      # adjust_thres
            ctypes.c_double    # alp (alpha)
        ]
        self._init_func.restype = None

        # C++: int* batch_query(uint32_t *data, int len);
        self._batch_query = self.lib.batch_query
        self._batch_query.argtypes = [
            ctypes.POINTER(ctypes.c_uint), # data (特征 ID)
            ctypes.c_int                   # len
        ]
        self._batch_query.restype = ctypes.POINTER(ctypes.c_int) # 返回指向内部 'que' 缓冲区的指针

        # C++: int* batch_insert_val(uint32_t *data, float *v, int len);
        self._batch_insert_val = self.lib.batch_insert_val
        self._batch_insert_val.argtypes = [
            ctypes.POINTER(ctypes.c_uint), # data (特征 ID)
            ctypes.POINTER(ctypes.c_float),# v (分数)
            ctypes.c_int                   # len
        ]
        self._batch_insert_val.restype = ctypes.POINTER(ctypes.c_int) # 返回指向内部 'ins' 缓冲区的指针

        # --- 初始化 C++ HotSketch 对象 ---
        print(f"初始化 HotSketch: n={self.hot_feature_count}, "
              f"Threshold={self.initial_threshold}, adjust={self.adjust_threshold}, "
              f"alpha={self.alpha}")
        self._init_func(
            self.hot_feature_count,
            self.initial_threshold, # C++ init 接口需要 float 类型的阈值
            self.adjust_threshold,
            self.alpha
        )
        print("HotSketch 初始化完成。")


    def query(self, feature_ids: np.ndarray) -> np.ndarray:
        """
        查询 HotSketch 以获取特征的状态（冷/热）和索引。

        """
        if not isinstance(feature_ids, np.ndarray):
            raise TypeError("feature_ids 必须是 NumPy 数组。")
        if feature_ids.size == 0:
            return np.array([], dtype=np.int32)

        n = len(feature_ids)
        # 确保输入数组为 C 连续且类型为 uint32
        ids_c = np.ascontiguousarray(feature_ids, dtype=np.uint32)
        ids_ptr = ids_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))

        # 调用 C++ 函数
        result_ptr = self._batch_query(ids_ptr, n)

        # C++ 返回其内部静态缓冲区 'que' 的指针
        # 必须 .copy() 复制数据，防止缓冲区被覆盖
        result_array = np.ctypeslib.as_array(result_ptr, shape=(n,)).copy()

        return result_array.astype(np.int32) # 确保返回 int32 类型


    def update(self, feature_ids: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """
        使用特征 ID 及其重要性分数（梯度范数）更新 HotSketch。
        这将触发分数更新、衰减和潜在的特征迁移。

        """
        if not isinstance(feature_ids, np.ndarray) or not isinstance(scores, np.ndarray):
            raise TypeError("feature_ids 和 scores 必须是 NumPy 数组。")
        if feature_ids.shape != scores.shape:
            raise ValueError("feature_ids 和 scores 必须有相同的形状。")
        if feature_ids.size == 0:
            return np.array([], dtype=np.int32)

        n = len(feature_ids)
        # 确保输入数组为 C 连续且类型正确
        ids_c = np.ascontiguousarray(feature_ids, dtype=np.uint32)
        scores_c = np.ascontiguousarray(scores, dtype=np.float32) 

        ids_ptr = ids_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint))
        scores_ptr = scores_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # 调用 C++ 函数
        mask_ptr = self._batch_insert_val(ids_ptr, scores_ptr, n)

        # C++ 返回其内部静态缓冲区 'ins' 的指针 (迁移掩码)
        # 必须 .copy() 复制数据
        mask_array = np.ctypeslib.as_array(mask_ptr, shape=(n,)).copy()

        return mask_array.astype(np.int32)

# 测试脚本
if __name__ == '__main__':
    lib_path = os.path.join(os.path.dirname(__file__), 'cpp/libhotsketch.so')

    # 如果库不存在 尝试编译 C++ 代码
    cpp_file = os.path.join(os.path.dirname(__file__), 'cpp/sketch.cpp')
    if not os.path.exists(lib_path) and os.path.exists(cpp_file):
        print(f"正在编译 {cpp_file}...")
        compile_cmd = (
            f"g++ -fPIC -shared -o {lib_path} -g -rdynamic -mavx2 -mbmi "
            f"-mavx512bw -mavx512dq --std=c++17 -O3 -fopenmp {cpp_file}"
        )
        print(f"运行: {compile_cmd}")
        ret = os.system(compile_cmd)
        if ret != 0:
            print("编译失败!")
            exit(1)
        else:
            print("编译成功。")

    if os.path.exists(lib_path):
        # 测试参数
        hot_count = 100
        threshold = 5.0
        adjust = True
        decay_alpha = 1.0000005 # 匹配 sketch.cpp 中的默认值

        try:
            manager = HotSketchManager(lib_path, hot_count, threshold, adjust, decay_alpha)

            # --- 测试查询 (初始) ---
            test_ids = np.array([10, 20, 30, 10, 40], dtype=np.uint32)
            print(f"\n查询 IDs: {test_ids}")
            query_results = manager.query(test_ids)
            print(f"查询结果: {query_results}") # 初始状态，应全为冷特征 (正数)

            # --- 测试更新 ---
            test_scores = np.array([2.0, 6.0, 1.0, 7.0, 8.0], dtype=np.float32)
            print(f"\n使用分数更新: {test_scores}")
            migration_mask = manager.update(test_ids, test_scores)
            print(f"迁移掩码: {migration_mask}") # 分数 > 5 的特征可能会迁移

            # --- 再次测试查询 ---
            print(f"\n再次查询 IDs: {test_ids}")
            query_results_after_update = manager.query(test_ids)
            print(f"更新后查询结果: {query_results_after_update}") # 此时部分特征应变为热特征 (负数)

            # --- 模拟多次更新 (测试衰减或重置) ---
            print("\n运行多次更新...")
            for i in range(20): # 循环次数
                ids = np.random.randint(1, 1000, size=50, dtype=np.uint32)
                scores = np.random.rand(50).astype(np.float32) * 10
                manager.update(ids, scores)
            print("多次更新完成。")

            # --- 最终查询 ---
            print(f"\n多次更新后查询 IDs: {test_ids}")
            final_query_results = manager.query(test_ids)
            print(f"最终查询结果: {final_query_results}")

        except (FileNotFoundError, OSError, TypeError, ValueError) as e:
            print(f"\n测试时出错: {e}")
    else:
        print(f"\n无法运行示例: 库文件 {lib_path} 未找到或编译失败。")