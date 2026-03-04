# HTS Pancake Coil Physics-Guided Evaluator

一个用于高温超导（HTS）pancake 线圈临界电流快速估计的轻量框架：

- 使用 **Biot–Savart** 计算线圈内局部磁场。
- 将带材离散为小段，假设每段局部场近似均匀。
- 将磁场分解为带材局部坐标中的 `B_parallel` / `B_perpendicular`。
- 通过可插拔 `JcModel` 接口接入带材级深度学习模型。

## 几何与物理假设

- 采用单 pancake 几何。
- 电流分布暂按平均分配（匝间、宽度方向均匀分配）。
- **忽略带材厚度**（薄带近似），但**保留带材宽度**并参与离散与局部分解。
- 单位约定：
  - 长度：m
  - 电流：A
  - 磁场：T
  - 温度：K

## 项目结构

- `src/hts_coil/geometry.py`：线圈与带材参数
- `src/hts_coil/segments.py`：导体离散与电流元生成
- `src/hts_coil/field.py`：Biot–Savart 求场
- `src/hts_coil/frames.py`：局部坐标分解
- `src/hts_coil/jc_interface.py`：Jc 模型接口
- `src/hts_coil/pipeline.py`：总流程与 Ic 占位估计
- `examples/pancake_demo.py`：示例与 csv 导出

## 快速开始

```bash
python -m pip install -e .
python examples/pancake_demo.py
```

运行后会输出各段 `B_mag/B_parallel/B_perpendicular` 统计值，并导出：

- `outputs/segment_fields.csv`

## 接入你的 Jc 模型

实现如下接口：

```python
class YourModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        # features 列顺序: [temperature, B_mag, B_parallel, B_perpendicular]
        # 返回每个离散段的 Jc (A/m^2)
        ...
```

然后传给：

- `evaluate_segments(...)`
- `estimate_coil_critical_current(...)`

## 说明

当前 `estimate_coil_critical_current` 采用最弱段（minimum local Ic）作为占位准则，便于后续替换为更完整的 E-I 判据或迭代求解策略。


## FEM 对照用测试文件

已新增 `tests/test_pancake_field_solver.py`，可按可修改参数构建 pancake coil 模型并计算任意空间点磁场：

- `inner_radius`
- `radial_thickness`
- `radial_turns`
- `radial_gap`
- `axial_width`
- `axial_layers`
- `center`

返回每个点：
- `B_mag`（总磁场）
- `B_parallel = sqrt(Bx^2 + By^2)`（平行线圈平面）
- `B_perpendicular = |Bz|`（垂直线圈平面）

运行：

```bash
PYTHONPATH=src python tests/test_pancake_field_solver.py
```

或使用测试：

```bash
PYTHONPATH=src pytest -q tests/test_pancake_field_solver.py
```


## 100A 直流下求空间某点磁场

可以。已提供正式接口 `PancakeFieldModel` + `solve_point_fields`：

```python
from hts_coil import PancakeFieldModel, solve_point_fields

model = PancakeFieldModel(
    inner_radius=0.025,
    radial_thickness=0.0003,
    radial_turns=20,
    radial_gap=0.0003,
    axial_width=0.012,
    axial_layers=1,
    center=(0.0, 0.0, 0.0),
)

points = [
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.01),
    (0.03, 0.0, 0.0),
]

result = solve_point_fields(model=model, points=points, current_a=100.0, n_theta=180)
for row in result:
    print(row["B_mag"], row["B_parallel"], row["B_perpendicular"])
```

其中：
- `B_mag`：磁场总大小
- `B_parallel = sqrt(Bx^2 + By^2)`：平行线圈平面
- `B_perpendicular = |Bz|`：垂直线圈平面
