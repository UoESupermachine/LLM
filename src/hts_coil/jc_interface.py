"""Jc 模型接口定义。

该模块定义了电磁求解与材料/数据驱动模型之间的稳定边界：
- 输入特征统一为 ``[T, B_mag, B_parallel, B_perpendicular]``。
- 输出统一为每段局部临界电流密度 ``Jc``（单位：A/m²）。

统一输出为 Jc 的好处是：
1) 与截面积、工程电流分布等几何信息解耦；
2) 便于后续替换为任意机器学习/深度学习模型；
3) 管线侧只需通过 ``Ic_local = Jc * area`` 完成换算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class JcModel(Protocol):
    """局部临界电流密度预测接口。

    Notes:
        * ``features`` 为形状 ``(n_segments, 4)`` 的二维数组。
        * 列顺序固定为 ``[T, B_mag, B_parallel, B_perpendicular]``。
        * 返回值为形状 ``(n_segments,)`` 或 ``(n_segments, 1)`` 的 ``Jc``，单位 A/m²。
    """

    def predict(self, features: np.ndarray) -> np.ndarray:
        """预测每段的局部临界电流密度 Jc（A/m²）。"""


@dataclass
class DummyJcModel:
    """用于联调的占位模型：输出常数 Jc。

    Args:
        jc_value: 常数临界电流密度，单位 A/m²。
    """

    jc_value: float = 3.0e10

    def predict(self, features: np.ndarray) -> np.ndarray:
        """返回与输入段数一致的常数 Jc 向量。"""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != 4:
            raise ValueError(
                "features must have shape (n_segments, 4) with columns "
                "[T, B_mag, B_parallel, B_perpendicular]."
            )

        n_segments = features.shape[0]
        return np.full((n_segments,), float(self.jc_value), dtype=float)
