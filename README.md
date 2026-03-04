# LLM

## 几何假设

- **薄带近似（thin-tape approximation）**：计算时忽略导带厚度，只保留导带宽度。
- `TapeSpec.thickness` 为可选输入，仅用于记录参数，不参与场积分。
- `TapeSpec.width` 为必填且必须大于 0，用于离散化与局部坐标定义（如每匝的径向宽度坐标 `u ∈ [-width/2, width/2]`）。
