"""
视障人群AI前方障碍物识别语音提醒系统
主程序入口
"""
import logging
import sys
import os
from pathlib import Path
import signal
import argparse
import numpy as np

# 添加src目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 尝试导入opencv
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

from src.utils.config import (
    setup_logging, load_config, get_config_path,
    ensure_directory_exists
)
from src.camera.camera import create_camera_from_config
from src.preprocess.preprocessor import create_preprocessor_from_config
from src.model.inference import create_model_from_config
from src.detection.detector import create_detector_from_config
from src.voice.tts import create_announcer_from_config


logger = logging.getLogger(__name__)


class ObstacleDetectionSystem:
    """障碍物识别系统主类"""
    
    def __init__(self, config_path: str = None, enable_display: bool = True):
        """
        初始化系统
        
        Args:
            config_path: 配置文件路径
            enable_display: 是否显示摄像头窗口 (需要OpenCV和图形界面)
        """
        # 加载配置
        if config_path is None:
            config_path = get_config_path()
        
        self.config = load_config(config_path)
        self.enable_display = enable_display and OPENCV_AVAILABLE
        
        # 初始化日志
        setup_logging(
            log_dir=self.config.get('system', {}).get('log_dir', './logs'),
            log_level=self.config.get('system', {}).get('log_level', 'INFO')
        )
        
        logger.info("="*60)
        logger.info("视障人群AI前方障碍物识别系统启动")
        if self.enable_display:
            logger.info("运行模式: 窗口显示模式")
        else:
            if enable_display and not OPENCV_AVAILABLE:
                logger.info("运行模式: 后台模式 (OpenCV未安装)")
            else:
                logger.info("运行模式: 后台模式 (--no-display)")
        logger.info("="*60)
        
        # 初始化各个模块
        self.camera = None
        self.preprocessor = None
        self.model = None
        self.detector = None
        self.announcer = None
        
        self.is_running = False
        self.frame_count = 0
        self.window_name = "Obstacle Detection System"
        
    def initialize(self) -> bool:
        """
        初始化所有模块
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            logger.info("初始化摄像头...")
            self.camera = create_camera_from_config(self.config)
            if not self.camera.initialize():
                logger.error("摄像头初始化失败")
                return False
            
            logger.info("初始化图像预处理器...")
            self.preprocessor = create_preprocessor_from_config(self.config)
            
            logger.info("初始化AI模型...")
            self.model = create_model_from_config(self.config)
            if not self.model.initialize():
                logger.error("模型初始化失败")
                return False
            
            logger.info("初始化检测器...")
            self.detector = create_detector_from_config(self.config)
            
            logger.info("初始化语音播报...")
            self.announcer = create_announcer_from_config(self.config)
            if not self.announcer.initialize():
                logger.warning("语音模块初始化失败，继续运行（无语音功能）")
                self.announcer.set_enabled(False)
            
            logger.info("所有模块初始化成功")
            return True
            
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            return False
    
    def process_frame(self):
        """处理单帧图像"""
        try:
            # 1. 捕获摄像头帧
            frame = self.camera.capture_frame()
            if frame is None:
                logger.warning("无法捕获摄像头帧")
                return
            
            # 2. 图像预处理
            preprocessed = self.preprocessor.preprocess(frame)
            
            # 3. 模型推理
            raw_detections = self.model.inference(preprocessed)
            
            # 4. 检测后处理
            detections = self.detector.process_frame(
                raw_detections,
                frame.shape[0],
                frame.shape[1]
            )
            
            # 5. 语音播报
            if detections:
                for det in detections[:3]:  # 只播报距离最近的3个
                    self.announcer.announce_obstacle(
                        obstacle_class=det['class_name'],
                        distance=det['distance'],
                        risk_level=det['risk_level'].name.lower()
                    )
            else:
                # 如果没有检测到障碍物，播报安全提示
                if self.frame_count % 60 == 0:  # 每60帧播报一次
                    self.announcer.announce_safe()
            
            # 6. 窗口显示（如果启用）
            if self.enable_display:
                self._display_frame(frame, detections)
            
            self.frame_count += 1
            
            # 日志记录
            if self.frame_count % 30 == 0:
                logger.info(f"处理帧数: {self.frame_count}, 检测到 {len(detections)} 个障碍物")
            
        except Exception as e:
            logger.error(f"处理帧失败: {str(e)}")
    
    def _display_frame(self, frame: np.ndarray, detections: list):
        """
        显示视频帧和检测结果
        
        Args:
            frame: 原始帧
            detections: 检测结果列表
        """
        try:
            if not OPENCV_AVAILABLE:
                return
            
            # 创建显示帧的副本
            display_frame = frame.copy()
            
            # 绘制检测框
            for det in detections:
                # 从嵌套字典中提取坐标
                bbox = det.get('bbox', {})
                x1 = int(bbox.get('x1', 0))
                y1 = int(bbox.get('y1', 0))
                x2 = int(bbox.get('x2', 0))
                y2 = int(bbox.get('y2', 0))
                
                confidence = det.get('confidence', 0)
                class_name = det.get('class_name', 'unknown')
                distance = det.get('distance', 0)
                risk_level = det.get('risk_level', None)
                
                # 获取风险等级名称
                if hasattr(risk_level, 'name'):
                    risk_name = risk_level.name
                else:
                    risk_name = "UNKNOWN"
                
                # 根据风险级别选择颜色
                if risk_name == "DANGER":
                    color = (0, 0, 255)  # 红色
                    thickness = 3
                elif risk_name == "WARNING":
                    color = (0, 165, 255)  # 橙色
                    thickness = 2
                else:
                    color = (0, 255, 0)  # 绿色
                    thickness = 1
                
                # 绘制检测框
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                
                # 绘制标签
                label = f"{class_name} {confidence:.2f} ({distance:.1f}m) [{risk_name}]"
                cv2.putText(display_frame, label, (x1, max(y1 - 10, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 绘制系统信息
            info_text = f"Frame: {self.frame_count} | Objects: {len(detections)}"
            cv2.putText(display_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示帧
            cv2.imshow(self.window_name, display_frame)
            
            # 处理按键事件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q 或 ESC 退出
                self.is_running = False
                logger.info("用户按下退出键")
            
        except Exception as e:
            logger.debug(f"显示帧失败: {str(e)}")
    
    def run(self):
        """运行系统"""
        if not self.initialize():
            logger.error("系统初始化失败")
            return
        
        self.is_running = True
        logger.info("系统开始运行，按 Ctrl+C 退出")
        
        try:
            while self.is_running:
                self.process_frame()
        
        except KeyboardInterrupt:
            logger.info("收到退出信号")
        except Exception as e:
            logger.error(f"运行时出错: {str(e)}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理系统资源...")
        
        if self.camera:
            self.camera.release()
        
        if self.model:
            self.model.release()
        
        if self.announcer:
            self.announcer.stop()
        
        # 关闭显示窗口
        if self.enable_display and OPENCV_AVAILABLE:
            try:
                cv2.destroyAllWindows()
                logger.info("显示窗口已关闭")
            except Exception as e:
                logger.warning(f"关闭窗口失败: {str(e)}")
        
        logger.info("系统已安全退出")
        logger.info("="*60)


def signal_handler(signum, frame):
    """信号处理器"""
    logger.info("\n收到信号，系统正在退出...")
    sys.exit(0)


def main():
    """主函数"""
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="视障人群AI前方障碍物识别语音提醒系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 正常运行（显示窗口）
  python src/main.py
  
  # 后台运行（不显示窗口）
  python src/main.py --no-display
  
  # 使用自定义配置文件
  python src/main.py --config config/my_config.yaml
  
  # 结合多个选项
  python src/main.py --config config/my_config.yaml --no-display
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='配置文件路径 (默认: config/config.yaml)'
    )
    
    parser.add_argument(
        '--no-display', '-nd',
        action='store_true',
        help='禁用窗口显示，后台运行'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Obstacle Detection System v1.0'
    )
    
    args = parser.parse_args()
    
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 获取配置文件路径
    config_path = args.config if args.config else get_config_path()
    
    # 创建系统实例并运行
    enable_display = not args.no_display
    system = ObstacleDetectionSystem(config_path, enable_display=enable_display)
    system.run()


if __name__ == "__main__":
    main()
