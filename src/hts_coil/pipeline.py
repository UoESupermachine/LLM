"""HTS 线圈 Ic 估计流程。

该模块提供 ``estimate_coil_critical_current``，把电磁场计算与 Jc 预测模型连接起来，
形成可扩展的数据流：

1. 给定候选电流 I；
2. 计算每段局部磁场；
3. 组装模型特征 [T, B_mag, B_parallel, B_perpendicular]；
4. 调用 jc_model.predict 获取局部 Jc；
5. 按“最弱段”准则（默认）或占位 E-I 准则估计线圈 Ic。

后续可直接替换 jc_model 为深度学习模型，无需改动电磁求解部分。
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np

from .jc_interface import JcModel

Criterion = Literal["weakest_segment", "ei_placeholder"]


def estimate_coil_critical_current(
    current_candidates: np.ndarray,
    temperature: float,
    segment_area: np.ndarray,
    field_solver: Callable[[float], np.ndarray],
    jc_model: JcModel,
    criterion: Criterion = "weakest_segment",
    e_criterion_threshold: float = 1.0,
) -> float:
    """估计线圈临界电流 Ic。

    Args:
        current_candidates: 候选电流序列（A），建议单调递增。
        temperature: 线圈温度（K），当前实现假设各段一致。
        segment_area: 每段截面积数组（m²），形状为 ``(n_segments,)``。
        field_solver: 电磁场求解函数，输入候选电流 I（A），输出
            形状 ``(n_segments, 3)`` 的磁场分量 ``[B_mag, B_parallel, B_perpendicular]``（T）。
        jc_model: 满足 ``JcModel`` 接口的预测模型。
        criterion: 估计准则。
            - ``weakest_segment``：当任一段 ``I > Ic_local`` 时视为超限。
            - ``ei_placeholder``：E-I 判据占位实现，当前基于超限比例的简化指标。
        e_criterion_threshold: E-I 占位准则阈值，指标超过该值视为失超。

    Returns:
        估计得到的线圈临界电流 Ic（A）。

    Raises:
        ValueError: 输入维度错误或候选电流为空时抛出。
    """
    currents = np.asarray(current_candidates, dtype=float).reshape(-1)
    if currents.size == 0:
        raise ValueError("current_candidates must not be empty.")

    area = np.asarray(segment_area, dtype=float).reshape(-1)
    if area.ndim != 1 or area.size == 0:
        raise ValueError("segment_area must be a non-empty 1D array.")
    if np.any(area <= 0):
        raise ValueError("segment_area must contain strictly positive values.")

    last_safe_current = currents[0]

    for i_candidate in currents:
        local_b = np.asarray(field_solver(float(i_candidate)), dtype=float)
        if local_b.ndim != 2 or local_b.shape[1] != 3:
            raise ValueError(
                "field_solver must return shape (n_segments, 3) as "
                "[B_mag, B_parallel, B_perpendicular]."
            )
        if local_b.shape[0] != area.size:
            raise ValueError(
                "field_solver output segment count must match segment_area length."
            )

        features = np.column_stack(
            (
                np.full((area.size,), float(temperature), dtype=float),
                local_b,
            )
        )

        local_jc = np.asarray(jc_model.predict(features), dtype=float).reshape(-1)
        if local_jc.size != area.size:
            raise ValueError("jc_model.predict must return one Jc value per segment.")
        if np.any(local_jc <= 0):
            raise ValueError("jc_model.predict must return strictly positive Jc values.")

        local_ic = local_jc * area

        if criterion == "weakest_segment":
            passed = bool(np.all(i_candidate <= local_ic))
        elif criterion == "ei_placeholder":
            overload_ratio = np.maximum(i_candidate / local_ic - 1.0, 0.0)
            indicator = float(np.mean(overload_ratio))
            passed = indicator <= float(e_criterion_threshold)
        else:
            raise ValueError(f"Unsupported criterion: {criterion}")

        if passed:
            last_safe_current = i_candidate
        else:
            break

    return float(last_safe_current)
