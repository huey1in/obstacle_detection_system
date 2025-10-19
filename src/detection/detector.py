"""
障碍物检测和距离计算模块
用于处理模型输出并进行后处理
"""
import logging
import numpy as np
from typing import List, Tuple, Dict
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """风险等级枚举"""
    SAFE = 0
    WARNING = 1
    DANGER = 2


class ObstacleDetector:
    """障碍物检测和处理类"""
    
    def __init__(self,
                 confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.45,
                 dangerous_distance: float = 2.0,
                 warning_distance: float = 3.5):
        """
        初始化检测器
        
        Args:
            confidence_threshold: 置信度阈值
            nms_threshold: NMS阈值
            dangerous_distance: 危险距离阈值（米）
            warning_distance: 警告距离阈值（米）
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.dangerous_distance = dangerous_distance
        self.warning_distance = warning_distance
        
        # 障碍物类别
        self.class_names = [
            "person",      # 0
            "pole",        # 1
            "tree",        # 2
            "wall",        # 3
            "stairs",      # 4
            "step",        # 5
            "hole",        # 6
            "vehicle",     # 7
            "railing",     # 8
            "other"        # 9
        ]
    
    def filter_detections(self,
                         detections: List[Tuple]) -> List[Dict]:
        """
        过滤和处理原始检测结果
        
        Args:
            detections: 原始检测结果 [(x1, y1, x2, y2, conf, class_id), ...]
            
        Returns:
            List[Dict]: 处理后的检测结果列表
        """
        filtered = []
        
        for det in detections:
            # 检查置信度
            if det[4] < self.confidence_threshold:
                continue
            
            # 构建检测结果字典
            result = {
                'bbox': {
                    'x1': float(det[0]),
                    'y1': float(det[1]),
                    'x2': float(det[2]),
                    'y2': float(det[3])
                },
                'confidence': float(det[4]),
                'class_id': int(det[5]),
                'class_name': self.class_names[int(det[5])] if int(det[5]) < len(self.class_names) else "unknown",
                'distance': None,  # 后续计算
                'risk_level': RiskLevel.SAFE
            }
            
            filtered.append(result)
        
        # 应用NMS
        filtered = self._apply_nms(filtered)
        
        return filtered
    
    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        """
        应用非极大值抑制(NMS)
        
        Args:
            detections: 检测结果列表
            
        Returns:
            List[Dict]: NMS处理后的检测结果
        """
        if len(detections) == 0:
            return []
        
        # 按置信度排序
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        
        while len(detections) > 0:
            # 保留置信度最高的检测框
            current = detections.pop(0)
            keep.append(current)
            
            if len(detections) == 0:
                break
            
            # 计算与其他框的IoU
            remaining = []
            for other in detections:
                iou = self._calculate_iou(current['bbox'], other['bbox'])
                
                # 如果IoU小于阈值，保留该框
                if iou < self.nms_threshold:
                    remaining.append(other)
            
            detections = remaining
        
        return keep
    
    def _calculate_iou(self, box1: Dict, box2: Dict) -> float:
        """
        计算两个检测框的交并比(IoU)
        
        Args:
            box1: 检测框1 {'x1', 'y1', 'x2', 'y2'}
            box2: 检测框2
            
        Returns:
            float: IoU值 (0-1)
        """
        # 计算交集
        x1_inter = max(box1['x1'], box2['x1'])
        y1_inter = max(box1['y1'], box2['y1'])
        x2_inter = min(box1['x2'], box2['x2'])
        y2_inter = min(box1['y2'], box2['y2'])
        
        if x2_inter < x1_inter or y2_inter < y1_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # 计算并集
        box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
        box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
        
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    def estimate_distance(self, 
                         detection: Dict,
                         image_height: int,
                         image_width: int,
                         camera_fov: float = 60.0) -> float:
        """
        估计障碍物距离（基于物体大小和相机参数的简单估计）
        
        Args:
            detection: 检测结果字典
            image_height: 图像高度
            image_width: 图像宽度
            camera_fov: 相机视角（度）
            
        Returns:
            float: 估计距离（米）
        """
        bbox = detection['bbox']
        
        # 计算检测框宽度和高度
        bbox_width = bbox['x2'] - bbox['x1']
        bbox_height = bbox['y2'] - bbox['y1']
        bbox_size = (bbox_width + bbox_height) / 2
        
        # 使用简单的距离估计模型
        # 距离 = 焦距 * 标准物体高度 / 检测框高度
        # 这里使用简化模型：距离与检测框大小成反比
        
        max_distance = 10.0  # 最大检测距离（米）
        min_distance = 0.5   # 最小检测距离（米）
        
        # 标准化检测框大小
        normalized_size = bbox_size / max(image_height, image_width)
        
        # 距离估计（inverse relationship）
        if normalized_size > 0:
            distance = max_distance / (1 + normalized_size * 20)
            distance = max(min_distance, min(max_distance, distance))
        else:
            distance = max_distance
        
        return distance
    
    def assess_risk(self, 
                   detection: Dict) -> RiskLevel:
        """
        评估检测到的障碍物风险等级
        
        Args:
            detection: 检测结果字典
            
        Returns:
            RiskLevel: 风险等级
        """
        distance = detection.get('distance', float('inf'))
        
        if distance <= self.dangerous_distance:
            return RiskLevel.DANGER
        elif distance <= self.warning_distance:
            return RiskLevel.WARNING
        else:
            return RiskLevel.SAFE
    
    def process_frame(self,
                     raw_detections: List[Tuple],
                     image_height: int,
                     image_width: int) -> List[Dict]:
        """
        处理一帧的完整检测流程
        
        Args:
            raw_detections: 原始检测结果
            image_height: 图像高度
            image_width: 图像宽度
            
        Returns:
            List[Dict]: 处理完成的检测结果
        """
        # 过滤和NMS
        detections = self.filter_detections(raw_detections)
        
        # 估计距离
        for detection in detections:
            distance = self.estimate_distance(
                detection,
                image_height,
                image_width
            )
            detection['distance'] = distance
            
            # 评估风险等级
            risk = self.assess_risk(detection)
            detection['risk_level'] = risk
        
        # 按距离排序（最近的在前）
        detections = sorted(detections, key=lambda x: x['distance'])
        
        return detections


def create_detector_from_config(config: dict) -> ObstacleDetector:
    """
    从配置字典创建检测器
    
    Args:
        config: 包含检测器配置的字典
        
    Returns:
        ObstacleDetector: 检测器对象
    """
    detection_config = config.get('detection', {})
    
    detector = ObstacleDetector(
        confidence_threshold=detection_config.get('confidence_threshold', 0.5),
        nms_threshold=detection_config.get('nms_threshold', 0.45),
        dangerous_distance=detection_config.get('dangerous_distance', 2.0),
        warning_distance=detection_config.get('warning_distance', 3.5)
    )
    
    return detector


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO)
    
    detector = ObstacleDetector()
    
    # 模拟原始检测结果
    raw_detections = [
        (100, 100, 300, 400, 0.95, 0),  # 置信度高的检测
        (110, 105, 310, 410, 0.92, 0),  # 与上一个有重叠，IoU高
        (350, 150, 500, 350, 0.87, 1),  # 置信度较低
    ]
    
    # 处理一帧
    results = detector.process_frame(raw_detections, 480, 640)
    
    print(f"处理后检测结果数量: {len(results)}")
    for i, det in enumerate(results):
        print(f"  检测 {i+1}: {det['class_name']} @ {det['distance']:.2f}m (置信度: {det['confidence']:.2f})")
