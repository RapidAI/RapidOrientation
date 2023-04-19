## Rapid Structure
- 该部分的功能主要针对文档类图像，包括文档图像分类、版面分析和表格识别。

### [版面分析](./docs/README_Layout.md)

### [表格识别](./docs/README_Table.md)

### [文档方向分类](./docs/README_Orientation.md)

### 整体流程
```mermaid
flowchart LR
    A[/文档图像/] --> B(文档方向分类 rapid_orientation) --> C(版面分析 rapid_layout) & D(表格识别 rapid_table) --> E(OCR识别 rapidocr_onnxruntime)
    E --> F(结构化输出)
```