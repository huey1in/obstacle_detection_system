"""
工具函数模块
包含配置加载、日志设置等常用工具
"""
import logging
import yaml
import os
from typing import Dict, Any
from pathlib import Path


def setup_logging(log_dir: str = "./logs", log_level: str = "INFO") -> None:
    """
    设置日志系统
    
    Args:
        log_dir: 日志目录
        log_level: 日志级别
    """
    # 创建日志目录
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 设置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 文件处理器
    log_file = os.path.join(log_dir, "obstacle_detection.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    # 根记录器配置
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    logging.info(f"日志系统已初始化，日志文件: {log_file}")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Dict: 配置字典
    """
    try:
        if not os.path.exists(config_path):
            logging.error(f"配置文件不存在: {config_path}")
            return {}
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logging.info(f"配置文件加载成功: {config_path}")
        return config if config else {}
        
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        return {}


def save_config(config: Dict, output_path: str) -> bool:
    """
    保存配置到YAML文件
    
    Args:
        config: 配置字典
        output_path: 输出文件路径
        
    Returns:
        bool: 保存是否成功
    """
    try:
        # 创建输出目录
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        logging.info(f"配置文件保存成功: {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"保存配置文件失败: {str(e)}")
        return False


def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
    """
    合并两个配置字典（override覆盖base）
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        
    Returns:
        Dict: 合并后的配置
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def ensure_directory_exists(directory: str) -> bool:
    """
    确保目录存在，不存在则创建
    
    Args:
        directory: 目录路径
        
    Returns:
        bool: 操作是否成功
    """
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建目录失败: {str(e)}")
        return False


def get_project_root() -> str:
    """
    获取项目根目录
    
    Returns:
        str: 项目根目录路径
    """
    current_file = os.path.abspath(__file__)
    # src/utils/config.py -> src/utils -> src -> root
    return os.path.dirname(os.path.dirname(os.path.dirname(current_file)))


def get_model_path(model_name: str = "obstacle_detection_model.mindir") -> str:
    """
    获取模型文件路径
    
    Args:
        model_name: 模型文件名
        
    Returns:
        str: 完整的模型路径
    """
    # 尝试多个可能的模型路径
    possible_paths = [
        # 从项目根目录查找
        os.path.join(get_project_root(), "obstacle_detection_system", "models", model_name),
        # 直接从当前项目文件夹查找
        os.path.join(get_project_root(), "models", model_name),
        # 从项目根目录查找
        os.path.join(os.path.dirname(get_project_root()), "obstacle_detection_system", "models", model_name),
    ]
    
    # 返回第一个存在的路径
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果都不存在，返回默认路径（项目根/models）
    return os.path.join(get_project_root(), "models", model_name)


def get_config_path(config_name: str = "config.yaml") -> str:
    """
    获取配置文件路径
    
    Args:
        config_name: 配置文件名
        
    Returns:
        str: 完整的配置文件路径
    """
    project_root = get_project_root()
    return os.path.join(project_root, "config", config_name)


if __name__ == "__main__":
    # 测试工具函数
    setup_logging(log_level="DEBUG")
    
    test_config = {
        'app': {'name': 'test', 'version': '1.0'},
        'settings': {'debug': True}
    }
    
    test_path = "/tmp/test_config.yaml"
    
    # 测试保存
    if save_config(test_config, test_path):
        # 测试加载
        loaded = load_config(test_path)
        print(f"加载的配置: {loaded}")
