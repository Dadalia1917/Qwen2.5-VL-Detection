# Qwen2.5-VL 目标检测程序

## 项目简介
本项目基于阿里巴巴通义实验室开发的Qwen2.5-VL视觉语言模型，专注于通用图像目标检测任务。通过该程序，用户可以快速对图像中的各种对象进行检测和定位，并获得精确的边界框坐标和对象标签。

## 更新说明
- 优化了输出目录结构，将不同类型的输出文件分类保存到独立子目录中
- 添加了YOLO格式标签文件的生成功能，便于后续模型训练
- 更新了图像处理路径，支持指定测试数据集目录
- 完善了GitHub上传配置，排除大型模型和数据文件

## 功能特点
- 使用Qwen2.5-VL模型进行图像目标检测
- 输出JSON格式的边界框坐标和标签
- 在图像上绘制边界框并保存结果
- 支持批量处理数据集中的图像
- 优化的提示词工程，提高检测准确性
- 针对纸箱(carton)等特定物体的检测优化
- 降低误检率，提高小物体检测精度

## 技术架构
- **核心模型**: Qwen2.5-VL（多模态大语言模型）
- **处理框架**: Python + PyTorch + Transformers
- **图像处理**: Pillow + JSON解析
- **批处理**: 多图像并发处理

## 优化策略

为了提高目标检测的准确性和降低误检率，我们采用了以下优化策略：

1. **提示词工程优化**：
   - 设计了更精确的系统提示词和用户提示词
   - 明确要求模型检测独立对象而非组合对象
   - 指定特定标签（如"carton"）用于纸箱类物体
   - 强调提供精确的边界框坐标
   - 要求模型只检测高置信度的对象，避免误检

2. **模型参数优化**：
   - 降低生成温度（temperature=0.1）以获得更确定的输出
   - 使用贪婪解码（do_sample=False）提高一致性
   - 调整top-p采样参数以平衡准确性和多样性

3. **后处理优化**：
   - 增强JSON解析的鲁棒性，处理各种格式问题
   - 标准化标签名称，统一"box"/"boxes"为"carton"
   - 验证和修正边界框坐标顺序
   - 过滤无效或格式错误的检测结果
   - 使用非极大值抑制（NMS）去除重复检测框

## 数据集设置方法

1. **数据集目录结构**：
   ```
   datasets/
   └── Data/
       ├── data.yaml
       ├── test/                    # 推理测试数据（只包含图片）
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       ├── train/                   # YOLO格式训练数据
       │   ├── classes.txt
       │   ├── data.yaml
       │   ├── images/
       │   │   ├── image1.jpg
       │   │   └── ...
       │   ├── labels/
       │   │   ├── image1.txt
       │   │   └── ...
       │   └── labels.cache
       └── val/                     # YOLO格式验证数据
           ├── classes.txt
           ├── data.yaml
           ├── images/
           │   ├── image1.jpg
           │   └── ...
           ├── labels/
           │   ├── image1.txt
           │   └── ...
           └── labels.cache
   ```

2. **数据准备**：
   - 将待检测的图像文件放入 `datasets/Data/test` 目录（用于推理）
   - `train/` 和 `val/` 目录包含YOLO格式的训练和验证数据
   - 支持的图像格式：`.jpg`, `.jpeg`, `.png`
   - 确保图像文件名不包含特殊字符

3. **数据质量要求**：
   - 图像清晰度高，避免模糊或过曝
   - 目标物体完整，避免严重遮挡
   - 图像大小适中，建议分辨率在 640x480 以上

## 任务说明

本项目专注于纸箱（carton）等特定物体的检测任务，主要应用场景包括：

1. **物流仓储**：自动识别和计数仓库中的纸箱
2. **生产线检测**：检测包装线上的产品纸箱
3. **库存管理**：统计货架上的纸箱数量和位置

## 代码修改步骤

1. **优化检测逻辑**：
   - 增强提示词工程，明确要求只检测高置信度对象
   - 添加非极大值抑制（NMS）算法去除重复检测框
   - 优化JSON解析和结果后处理流程

2. **降低误检率**：
   - 在提示词中明确要求避免误检和重复检测
   - 添加IoU阈值控制，去除重叠的检测框
   - 增强边界框坐标验证逻辑

3. **性能优化**：
   - 优化模型参数配置以提高准确性
   - 改进错误处理机制，提高程序稳定性

## 环境要求
- Python 3.8+
- CUDA GPU（推荐）
- 至少16GB内存
- 至少50GB硬盘空间

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保模型文件放置在 `models` 目录中
2. 将待检测的图像放置在 `datasets/Data/test` 目录中
3. 运行目标检测程序:
   ```bash
   python main.py
   ```

程序将自动处理`datasets/Data/test`目录中的所有图像文件，并将结果保存到`outputs`目录的相应子目录中。

## 输出文件说明
程序会在 `outputs` 目录中生成以下子目录和文件:

```
outputs/
├── images/
│   └── {image_name}_detection.jpg  # 带有边界框标注的图像
├── json/
│   └── {image_name}_detection.json # 解析后的检测结果(JSON格式)
├── labels/
│   └── {image_name}.txt           # YOLO格式的标签文件
└── raw/
    └── {image_name}_raw.txt        # 模型的原始输出
```

目录结构说明：
- `images/`: 保存带有边界框标注的检测结果图像
- `json/`: 保存解析后的检测结果(JSON格式)
- `labels/`: 保存YOLO格式的标签文件，便于后续模型训练
- `raw/`: 保存模型的原始输出文本

## JSON结果格式
```json
[
  {
    "bbox_2d": [x1, y1, x2, y2],
    "label": "object_name"
  }
]
```
其中:
- `bbox_2d`: 边界框坐标 [左上角x, 左上角y, 右下角x, 右下角y]
- `label`: 检测到的对象标签

## GitHub上传配置

为了便于代码版本管理和协作开发，本项目已配置好GitHub上传设置，自动排除大型文件和敏感数据：

### 排除上传的目录
- `datasets/`: 原始数据集目录（包含训练、验证和测试数据）
- `models/`: 模型文件目录（包含预训练模型权重）
- `outputs/`: 输出结果目录（包含检测结果和标签文件）

### .gitignore配置
项目根目录下的`.gitignore`文件已配置好排除规则，确保只上传核心代码和文档：
```
# Python生成文件
__pycache__/
*.py[oc]
build/
dist/
wheels/
*.egg-info

# 虚拟环境
.venv

# 排除大型数据目录
datasets/
models/
outputs/
```

### 上传步骤
1. 初始化Git仓库：
   ```bash
   git init
   ```
2. 添加所有文件：
   ```bash
   git add .
   ```
3. 提交更改：
   ```bash
   git commit -m "初始化项目版本"
   ```
4. 关联远程仓库并推送：
   ```bash
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

注意：在上传前请确保已获得必要的授权，并遵守数据隐私和知识产权相关规定。

## 参考资源
- 项目参考文章: https://blog.csdn.net/qq_42589613/article/details/151677057?spm=1001.2014.3001.5501
- 模型下载地址: https://modelscope.cn/models/Qwen/Qwen2.5-VL-3B-Instruct

## 扩展应用指南

本项目可通过修改提示词和数据集来适应不同的目标检测任务，如生活垃圾分类检测、车辆检测、烟雾检测等。

### 1. 更换提示词方法

要更改检测任务，需要修改`main.py`文件中的提示词：

1. **系统提示词（System Prompt）**：位于第28行，定义模型的角色和任务
   - 修改`system_prompt`变量中的内容，指定新的检测目标
   - 例如，对于垃圾分类任务，可改为："You are an expert in object detection, specifically for identifying different types of garbage in images..."

2. **用户提示词（User Prompt）**：位于第54行，提供具体的检测指令
   - 修改`user_prompt`变量中的内容，明确检测规则和输出格式
   - 根据新任务调整检测规则和标签要求

3. **标签标准化**：在`parse_detection_result`函数中（第250行左右），根据需要修改标签标准化逻辑
   - 更新标签映射规则以适应新任务
   - 例如，将"box"/"boxes"映射到"carton"的规则可以改为其他标签映射

### 2. 数据集更换与放置方式

1. **数据集目录结构**：
   ```
   datasets/
   └── Data/
       ├── data.yaml
       ├── test/                    # 推理测试数据（只包含图片）
       │   ├── image1.jpg
       │   ├── image2.png
       │   └── ...
       ├── train/                   # YOLO格式训练数据
       │   ├── classes.txt
       │   ├── data.yaml
       │   ├── images/             # 训练图像
       │   │   ├── image1.jpg
       │   │   └── ...
       │   ├── labels/             # YOLO格式标签
       │   │   ├── image1.txt
       │   │   └── ...
       │   └── labels.cache
       └── val/                     # YOLO格式验证数据
           ├── classes.txt
           ├── data.yaml
           ├── images/             # 验证图像
           │   ├── image1.jpg
           │   └── ...
           ├── labels/             # YOLO格式标签
           │   ├── image1.txt
           │   └── ...
           └── labels.cache
   ```

2. **数据集更换步骤**：
   - 推理数据：将待检测的图像直接放入`datasets/Data/test/`目录
   - 训练数据：将图像放入`datasets/Data/train/images/`，标签放入`datasets/Data/train/labels/`
   - 验证数据：将图像放入`datasets/Data/val/images/`，标签放入`datasets/Data/val/labels/`
   - 确保图像格式为`.jpg`、`.jpeg`或`.png`
   - YOLO标签格式：每个图像对应一个同名.txt文件

3. **数据集格式要求**：
   - 图像文件：建议分辨率在640x480以上，确保目标物体清晰可见
   - YOLO标签格式：`class_id center_x center_y width height`（归一化坐标）
   - 数据质量：避免模糊、过曝或严重遮挡的图像

### 3. 任务切换示例

以车辆检测任务为例：
1. 修改系统提示词为："You are an expert in object detection, specifically for identifying vehicles (cars, trucks, buses) in images..."
2. 修改用户提示词，将检测规则中的"cardboard boxes (cartons)"替换为"vehicles"
3. 更新标签标准化逻辑，将"carton"映射改为相应的车辆类别
4. 准备车辆检测数据集，按上述目录结构放置

## 注意事项
1. 首次运行时，程序会自动加载模型，这可能需要一些时间
2. 程序需要较大的显存，建议使用具有至少8GB显存的GPU
3. 如果遇到CUDA内存不足的问题，可以尝试减少批处理大小或使用CPU运行（性能会显著下降）
4. 检测精度依赖于Qwen2.5-VL模型的训练效果，对于特定场景可能需要进一步微调
5. 为获得最佳检测效果，请确保输入图像质量良好且目标物体清晰可见