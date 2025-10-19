"""
图像预处理模块
用于对输入图像进行标准化处理，为模型推理做准备
"""
import cv2
import numpy as np
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """图像预处理类"""
    
    def __init__(self,
                 input_size: Tuple[int, int] = (640, 640),
                 mean: List[float] = None,
                 std: List[float] = None,
                 normalize: bool = True,
                 convert_to_rgb: bool = True):
        """
        初始化图像预处理器
        
        Args:
            input_size: 模型输入尺寸 (H, W)
            mean: 归一化均值
            std: 归一化标准差
            normalize: 是否进行归一化
            convert_to_rgb: 是否进行BGR转RGB
        """
        self.input_size = input_size
        self.mean = np.array(mean or [0.485, 0.456, 0.406])
        self.std = np.array(std or [0.229, 0.224, 0.225])
        self.normalize = normalize
        self.convert_to_rgb = convert_to_rgb
        self.original_shape = None
        self.scale_factor = None
        
    def resize_image(self, image: np.ndarray, keep_aspect: bool = True) -> np.ndarray:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            keep_aspect: 是否保持长宽比
            
        Returns:
            np.ndarray: 调整后的图像
        """
        h, w = image.shape[:2]
        target_h, target_w = self.input_size
        
        if keep_aspect:
            # 计算缩放因子
            scale = min(target_w / w, target_h / h)
            self.scale_factor = scale
            
            # 计算新尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 调整大小
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # 创建目标尺寸的图像，填充为灰色（pad）
            padded = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
            y_offset = (target_h - new_h) // 2
            x_offset = (target_w - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            
            return padded
        else:
            # 直接调整尺寸
            self.scale_factor = (target_w / w, target_h / h)
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            return resized
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像 (H, W, C)，值范围0-255
            
        Returns:
            np.ndarray: 归一化后的图像，值范围为-2到2之间
        """
        # 转换为float32
        image = image.astype(np.float32) / 255.0
        
        # 应用均值和标准差
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        完整的预处理流程
        
        Args:
            image: 输入图像
            
        Returns:
            np.ndarray: 预处理后的图像，形状为 (1, 3, H, W)
        """
        try:
            # 保存原始形状
            self.original_shape = image.shape
            
            # BGR转RGB（OpenCV默认读取为BGR）
            if self.convert_to_rgb and len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸
            image = self.resize_image(image, keep_aspect=True)
            
            # 归一化
            if self.normalize:
                image = self.normalize_image(image)
            else:
                image = image.astype(np.float32) / 255.0
            
            # 转换为模型输入格式 (1, C, H, W)
            image = np.transpose(image, (2, 0, 1))  # (C, H, W)
            image = np.expand_dims(image, axis=0)    # (1, C, H, W)
            
            return image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {str(e)}")
            raise
    
    def postprocess_detections(self, 
                               detections: list,
                               confidence_threshold: float = 0.5) -> list:
        """
        后处理检测结果
        
        Args:
            detections: 原始检测结果列表 [(x1, y1, x2, y2, conf, class_id), ...]
            confidence_threshold: 置信度阈值
            
        Returns:
            list: 过滤后的检测结果
        """
        filtered = []
        
        for det in detections:
            if det[4] >= confidence_threshold:  # 检查置信度
                filtered.append(det)
        
        return filtered


def create_preprocessor_from_config(config: dict) -> ImagePreprocessor:
    """
    从配置字典创建预处理器
    
    Args:
        config: 包含预处理配置的字典
        
    Returns:
        ImagePreprocessor: 预处理器对象
    """
    preprocess_config = config.get('preprocess', {})
    
    preprocessor = ImagePreprocessor(
        input_size=tuple(preprocess_config.get('input_size', [640, 640])),
        mean=preprocess_config.get('mean', [0.485, 0.456, 0.406]),
        std=preprocess_config.get('std', [0.229, 0.224, 0.225]),
        normalize=preprocess_config.get('normalize', True),
        convert_to_rgb=preprocess_config.get('convert_to_rgb', True)
    )
    
    return preprocessor


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO)
    
    # 创建测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    preprocessor = ImagePreprocessor(
        input_size=(640, 640),
        normalize=True,
        convert_to_rgb=True
    )
    
    processed = preprocessor.preprocess(test_image)
    print(f"原始图像形状: {test_image.shape}")
    print(f"处理后形状: {processed.shape}")
    print(f"处理后范围: [{processed.min():.2f}, {processed.max():.2f}]")
