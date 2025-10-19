"""
摄像头采集模块
用于实时获取摄像头画面并进行基础处理
"""
import cv2
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class CameraCapture:
    """摄像头采集类"""
    
    def __init__(self, 
                 device_id: int = 0,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 use_dummy: bool = False):
        """
        初始化摄像头
        
        Args:
            device_id: 摄像头设备ID
            width: 帧宽度
            height: 帧高度
            fps: 帧率
            use_dummy: 是否使用虚拟摄像头（用于测试）
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.is_opened = False
        self.frame_count = 0
        self.use_dummy = use_dummy
        
    def initialize(self) -> bool:
        """
        初始化摄像头
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.use_dummy:
                logger.info(f"使用虚拟摄像头 (演示模式): {self.width}x{self.height}@{self.fps}fps")
                self.is_opened = True
                return True
            
            self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                logger.warning(f"无法打开摄像头设备 {self.device_id}，使用虚拟摄像头")
                self.use_dummy = True
                self.is_opened = True
                logger.info(f"已切换到虚拟摄像头: {self.width}x{self.height}@{self.fps}fps")
                return True
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # 设置帧率
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # 设置缓冲区大小，减少延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_opened = True
            logger.info(f"摄像头初始化成功: {self.width}x{self.height}@{self.fps}fps")
            return True
            
        except Exception as e:
            logger.warning(f"摄像头初始化失败: {str(e)}，使用虚拟摄像头")
            self.use_dummy = True
            self.is_opened = True
            return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        捕获单帧图像
        
        Returns:
            np.ndarray: 捕获的帧，失败返回None
        """
        if not self.is_opened:
            logger.error("摄像头未打开")
            return None
        
        try:
            if self.use_dummy:
                # 生成虚拟帧（蓝色背景+随机噪点）
                frame = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
                # 添加一些渐变效果
                for i in range(self.height):
                    frame[i, :] = frame[i, :] * (1.0 - i / self.height * 0.5)
                self.frame_count += 1
                return frame.astype(np.uint8)
            
            ret, frame = self.cap.read()
            
            if not ret:
                logger.warning("无法读取摄像头帧，切换到虚拟摄像头")
                self.use_dummy = True
                return self.capture_frame()
            
            self.frame_count += 1
            return frame
            
        except Exception as e:
            logger.error(f"捕获摄像头帧失败: {str(e)}")
            return None
    
    def get_frame_info(self) -> dict:
        """
        获取摄像头帧信息
        
        Returns:
            dict: 包含宽度、高度、帧率等信息
        """
        if not self.is_opened:
            return {}
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': self.frame_count,
            'codec': int(self.cap.get(cv2.CAP_PROP_FOURCC))
        }
    
    def release(self):
        """释放摄像头资源"""
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            logger.info("摄像头已释放")
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.release()


def create_camera_from_config(config: dict) -> CameraCapture:
    """
    从配置字典创建摄像头对象
    
    Args:
        config: 包含摄像头配置的字典
        
    Returns:
        CameraCapture: 摄像头对象
    """
    camera_config = config.get('camera', {})
    
    camera = CameraCapture(
        device_id=camera_config.get('device_id', 0),
        width=camera_config.get('frame_width', 640),
        height=camera_config.get('frame_height', 480),
        fps=camera_config.get('fps', 30)
    )
    
    return camera


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO)
    
    camera = CameraCapture(device_id=0, width=640, height=480, fps=30)
    
    if camera.initialize():
        frame = camera.capture_frame()
        if frame is not None:
            print(f"成功捕获帧: {frame.shape}")
            info = camera.get_frame_info()
            print(f"摄像头信息: {info}")
        camera.release()
