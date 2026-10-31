深度学习框架目标检测算法YOLOV8如何训练电力场景安全行为安全带检测数据集 高空抛物检测数据集的训练及应用
@[toc]

数据集文字描述：

电力施工作业安全行为检测图像数据集，图片总共 2300左右，标注为voc(xml)格式，包含高空抛物，正确/未正确佩戴安全带，高处作业有/无人监护

表格描述:


### **电力施工作业安全行为检测图像数据集**

| 类别                     | 描述                           | 图像数量（约） | 标注格式 | 备注 |
|--------------------------|--------------------------------|----------------|-----------|------|
| 高空抛物                 | 包含识别高空抛物的行为         | 未知           | VOC (XML) | ， |
| 正确佩戴安全带           | 包含正确佩戴安全带的情况       |                |           |      |
| 未正确佩戴安全带         | 包含未正确佩戴安全带的情况     |                |           |      |
| 高处作业有人监护         | 包含高处作业时有监护人员在场   |                |           |      |
| 高处作业无人监护         | 包含高处作业时无监护人员在场   |                |           |      |

> **注意**：
>\
> - 所有图像都以VOC格式(XML文件)进行标注，这表示每张图片的标签中包含了边界框位置、类别名称等信息，非常适合用于目标检测算法的训练。

\\

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8fd118a4aaff4112b2c1d6f8a073bb7e.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d31f7c1d49c3468a9c4c74efa24e3797.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d34f7ba040834ac687b1e7a660fae07b.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bbad6d185b8d4a96bfe9871ba5228d06.png)
**电力施工作业安全行为检测**的 YOLOv8 模型，数据集约2300 张图像，VOC XML 格式， **高空抛物，正确/未正确佩戴安全带，高处作业有/无人监护** 5类行为。将从 **环境搭建 → 数据集转换 → 模型训练 → 推理 → 评估** 全流程详细指导。
以下文字及代码仅供参考学习使用。

---

## ✅ 一、系统环境准备（Windows/Linux）

### 1. CUDA 驱动安装

确保同学你的 GPU 支持 CUDA：

```bash
nvidia-smi
```

- 输出应显示驱动版本和 GPU 信息。
- 若无输出，请安装最新 NVIDIA 驱动和 CUDA Toolkit（推荐 CUDA 11.8 或 12.1）。

> 官网下载：[https://www.nvidia.com/Download/index.aspx](https://www.nvidia.com/Download/index.aspx)

---

### 2. 安装 Anaconda

从官网下载并安装 Anaconda：
> [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

验证安装：
```bash
conda --version
```

---

### 3. 创建 Python 虚拟环境

```bash
# 创建名为 yolov8_env 的环境，Python 3.9 推荐
conda create -n yolov8_env python=3.9

# 激活环境
conda activate yolov8_env
```

---

### 4. 安装必要依赖

```bash
# 安装 PyTorch（以 CUDA 11.8 为例）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或使用 pip（推荐）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 YOLOv8（Ultralytics 官方库）
pip install ultralytics

# 安装 OpenCV 和其他工具
pip install opencv-python numpy matplotlib lxml tqdm
```

✅ 验证安装：
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 应输出 True
```

---

## ✅ 二、数据集准备（VOC → YOLO 格式）

YOLOv8 要求数据为 **YOLO 格式**：每张图对应一个 `.txt` 文件，内容为：

```
class_id center_x center_y width height  # 归一化 [0,1]
```

### 1. VOC 数据结构示例

```
dataset/
├── JPEGImages/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Annotations/
│   ├── img1.xml
│   ├── img2.xml
│   └── ...
├── ImageSets/Main/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
```

### 2. 编写 VOC → YOLO 转换脚本

```python
# convert_voc_to_yolo.py
import os
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path

# 类别定义（按你的任务）
classes = ["throwing", "no_safety_belt", "safety_belt", "no_supervisor", "with_supervisor"]
class_to_id = {cls: idx for idx, cls in enumerate(classes)}

def convert_box(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def convert_annotation(xml_file, output_dir, img_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    if size is None:
        return
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    lines = []
    for obj in root.iter('object'):
        cls = obj.find('name').text.strip().lower()
        if cls not in class_to_id:
            continue
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        bb = convert_box((w, h), (xmin, xmax, ymin, ymax))
        lines.append(f"{class_to_id[cls]} {' '.join(f'{x:.6f}' for x in bb)}")

    # 写入 .txt 文件
    with open(os.path.join(output_dir, Path(xml_file).stem + '.txt'), 'w') as f:
        f.write('\n'.join(lines))

# 路径配置
voc_root = "dataset"
img_dir = os.path.join(voc_root, "JPEGImages")
ann_dir = os.path.join(voc_root, "Annotations")
output_dir = os.path.join(voc_root, "labels")

os.makedirs(output_dir, exist_ok=True)

for xml_file in glob(os.path.join(ann_dir, "*.xml")):
    convert_annotation(xml_file, output_dir, img_dir)

print("✅ VOC to YOLO 格式转换完成！")
```

---

### 3. 划分训练/验证/测试集

```python
# split_dataset.py
import os
from sklearn.model_selection import train_test_split
from glob import glob

img_files = glob("dataset/JPEGImages/*.jpg")
img_names = [os.path.basename(f).split('.')[0] for f in img_files]

train_names, test_names = train_test_split(img_names, test_size=0.2, random_state=42)
val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42)

def write_txt(names, filename):
    with open(filename, 'w') as f:
        f.write('\n'.join(names))

write_txt(train_names, "dataset/train.txt")
write_txt(val_names, "dataset/val.txt")
write_txt(test_names, "dataset/test.txt")

print("✅ 数据集划分完成：train/val/test")
```

---

### 4. 创建 `data.yaml`

```yaml
# data.yaml
train: dataset/train.txt
val: dataset/val.txt
test: dataset/test.txt

nc: 5  # 类别数量
names: ['throwing', 'no_safety_belt', 'safety_belt', 'no_supervisor', 'with_supervisor']
```

---

## ✅ 三、调用 YOLOv8 训练模型

### 1. 使用官方预训练模型进行训练

```python
# train.py
from ultralytics import YOLO

# 加载预训练模型（推荐使用 YOLOv8m 或 YOLOv8l 提升精度）
model = YOLO('yolov8m.pt')  # 也可用 yolov8n.pt, yolov8s.pt, yolov8l.pt

# 开始训练
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,           # 根据 GPU 显存调整（16适合16G显存）
    name='safety_det_v1',
    project='runs/safety',
    device=0,           # 使用 GPU 0，多卡可用 [0,1]
    workers=4,
    optimizer='AdamW',  # 可选 SGD, Adam, AdamW
    lr0=0.001,          # 初始学习率
    patience=10,        # 早停
    save_period=5,      # 每5轮保存一次
    val=True
)

print("✅ 训练完成！")
```

> 📌 模型会自动保存在 `runs/safety/safety_det_v1/weights/` 中，`best.pt` 是最佳模型。

---

## ✅ 四、推理代码（Inference）

```python
# infer.py
from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/safety/safety_det_v1/weights/best.pt')

# 推理单张图像
results = model('test_image.jpg', conf=0.5)  # 置信度阈值

# 显示结果
for r in results:
    im_array = r.plot()  # 绘制边界框和标签
    im = cv2.cvtColor(im_array, cv2.COLOR_RGB2BGR)
    cv2.imshow('Safety Detection', im)
    cv2.waitKey(0)

# 或保存结果
results[0].save(filename='result.jpg')
```

### 批量推理视频或文件夹

```python
# 推理整个文件夹
results = model('dataset/JPEGImages/', save=True, project='runs/predict', name='test')
```

---

## ✅ 五、评估模型性能

```python
# evaluate.py
from ultralytics import YOLO

# 加载最佳模型
model = YOLO('runs/safety/safety_det_v1/weights/best.pt')

# 在验证集上评估
metrics = model.val(
    data='data.yaml',
    split='val',        # 使用 val.txt
    batch=16,
    imgsz=640,
    device=0
)

# 输出指标
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p:.4f}")
print(f"Recall: {metrics.box.r:.4f}")

# 混淆矩阵保存在 runs/val/ 目录下
```

---

## ✅ 六、可视化训练过程

训练完成后，查看：

- `runs/safety/safety_det_v1/results.png`：loss、mAP、precision、recall 曲线
- `runs/safety/safety_det_v1/confusion_matrix.png`：分类混淆情况

你也可以使用 **TensorBoard**：

```bash
tensorboard --logdir runs/safety
```

---


