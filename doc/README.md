# 视障人群AI前方障碍物识别语音提醒系统

## 项目概述

本项目是一个基于**昇腾AI开发板**和**MindSpore框架**的智能障碍物识别系统，专门为视障人群设计。系统通过实时的摄像头输入、AI模型推理和语音播报，为视障人群提供前方障碍物的实时预警和距离提示。

### 核心特性

- **端侧推理**：基于昇腾NPU，无需网络连接，毫秒级推理延迟
- **实时语音播报**：即时识别障碍物并通过语音提醒用户
- **轻量化模型**：优化后的模型体积小，计算复杂度低，功耗友好
- **准确的距离估计**：基于视觉特征估计障碍物距离
- **多障碍物支持**：支持识别行人、柱子、树木、楼梯等多种障碍物
- **风险等级评估**：根据距离自动判断安全、警告、危险等风险等级

---

## 项目结构

```
obstacle_detection_system/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── main.py                   # 主程序入口
│   ├── camera/                   # 摄像头采集模块
│   │   └── camera.py
│   ├── preprocess/               # 图像预处理模块
│   │   └── preprocessor.py
│   ├── model/                    # 模型推理模块
│   │   └── inference.py
│   ├── detection/                # 障碍物检测模块
│   │   └── detector.py
│   ├── voice/                    # 语音播报模块
│   │   └── tts.py
│   └── utils/                    # 工具函数
│       └── config.py
├── config/
│   └── config.yaml               # 系统配置文件
├── models/                       # 模型文件存储目录
├── datasets/                     # 数据集存储目录
├── logs/                         # 日志输出目录
├── tests/                        # 单元测试
├── requirements.txt              # Python依赖列表
└── README.md                     # 项目文档
```

---

## 快速开始

### 1. 环境要求

- Python 3.8+
- 昇腾AI开发板或昇腾仿真环境
- 支持的操作系统：Linux、Windows、macOS

### 2. 安装依赖

```bash
# 安装Python依赖
pip install -r requirements.txt

# 安装MindSpore（推荐版本2.1+）
# 对于昇腾910 AI处理器
pip install mindspore

# 安装MindIE推理引擎（可选，用于性能优化）
# 详见昇腾官方文档
```

### 3. 配置系统

编辑 `config/config.yaml` 文件配置：

```yaml
camera:
  device_id: 0              # 摄像头设备ID
  frame_width: 640          # 帧宽度
  frame_height: 480         # 帧高度
  fps: 30                   # 帧率

voice:
  tts_engine: "pyttsx3"     # 语音引擎
  speaker_rate: 150         # 语速
  volume: 1.0               # 音量

detection:
  confidence_threshold: 0.5 # 置信度阈值
  dangerous_distance: 2.0   # 危险距离（米）
  warning_distance: 3.5     # 警告距离（米）
```

### 4. 运行系统

```bash
# 使用默认配置运行
python src/main.py

# 使用自定义配置运行
python src/main.py /path/to/custom/config.yaml
```

---

## 模块说明

### 摄像头模块 (`src/camera/camera.py`)

负责实时获取摄像头画面：

```python
from src.camera.camera import create_camera_from_config

camera = create_camera_from_config(config)
camera.initialize()

frame = camera.capture_frame()
# 处理frame...

camera.release()
```

**支持的功能**：
- 摄像头初始化和参数配置
- 实时帧采集
- 性能监控

### 图像预处理模块 (`src/preprocess/preprocessor.py`)

对输入图像进行标准化处理：

```python
from src.preprocess.preprocessor import create_preprocessor_from_config

preprocessor = create_preprocessor_from_config(config)

# 预处理图像，输出形状为 (1, 3, 640, 640)
processed_image = preprocessor.preprocess(frame)
```

**支持的功能**：
- 尺寸调整（保持长宽比）
- 归一化处理
- BGR转RGB
- 输出MindSpore格式

### 模型推理模块 (`src/model/inference.py`)

加载和运行AI模型：

```python
from src.model.inference import create_model_from_config

model = create_model_from_config(config)
model.initialize()

detections = model.inference(preprocessed_image)

model.release()
```

**支持的功能**：
- MindSpore框架支持
- MindIE推理引擎支持
- 昇腾芯片优化

### 检测模块 (`src/detection/detector.py`)

处理模型输出进行后处理：

```python
from src.detection.detector import create_detector_from_config

detector = create_detector_from_config(config)

# 处理一帧的检测
detections = detector.process_frame(raw_detections, height, width)

for det in detections:
    print(f"{det['class_name']} 距离 {det['distance']:.1f}m (风险: {det['risk_level'].name})")
```

**支持的功能**：
- 置信度过滤
- NMS非极大值抑制
- 距离估计
- 风险等级评估

### 语音模块 (`src/voice/tts.py`)

文本转语音播报：

```python
from src.voice.tts import create_announcer_from_config

announcer = create_announcer_from_config(config)
announcer.initialize()

# 播报障碍物信息
announcer.announce_obstacle("行人", 1.5, "danger")

# 播报安全提示
announcer.announce_safe()

announcer.stop()
```

**支持的功能**：
- pyttsx3 TTS引擎
- Edge TTS引擎
- 多语言支持（中文、英文等）
- 可定制的播报内容和语速

---

## 开发指南

### 添加新的障碍物类别

编辑 `src/detection/detector.py`：

```python
class ObstacleDetector:
    def __init__(self, ...):
        self.class_names = [
            "person",      # 0
            "pole",        # 1
            # ... 添加你的新类别
            "new_obstacle" # N
        ]
```

### 自定义推理引擎

编辑 `src/model/inference.py`，实现 `_init_custom_engine()` 方法：

```python
def _init_custom_engine(self) -> bool:
    # 你的自定义推理引擎初始化代码
    pass
```

### 调整距离估计算法

编辑 `src/detection/detector.py` 中的 `estimate_distance()` 方法。

---

## 测试

运行单元测试：

```bash
# 测试摄像头模块
python -m pytest tests/test_camera.py

# 测试预处理模块
python -m pytest tests/test_preprocess.py

# 测试全部测试
python -m pytest tests/
```

---

**最后更新**: 2025年10月19日
