"""
语音播报模块
用于生成和播放语音提示
"""
import logging
import threading
from typing import Optional, Dict
from queue import Queue
from enum import Enum

logger = logging.getLogger(__name__)


class VoiceEngine(Enum):
    """语音引擎类型"""
    PYTTSX3 = "pyttsx3"
    EDGE_TTS = "edge-tts"
    SYSTEM = "system"


class TextToSpeech:
    """文本转语音类"""
    
    def __init__(self,
                 engine: str = "pyttsx3",
                 language: str = "zh-CN",
                 rate: int = 150,
                 volume: float = 1.0):
        """
        初始化TTS引擎
        
        Args:
            engine: TTS引擎类型
            language: 语言代码
            rate: 语速（字/分钟）
            volume: 音量（0-1）
        """
        self.engine = engine
        self.language = language
        self.rate = rate
        self.volume = max(0.0, min(1.0, volume))
        self.tts = None
        self.queue = Queue()
        self.is_playing = False
        self.play_thread = None
        
    def initialize(self) -> bool:
        """
        初始化语音引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            if self.engine == "pyttsx3":
                return self._init_pyttsx3()
            elif self.engine == "edge-tts":
                return self._init_edge_tts()
            else:
                logger.warning(f"不支持的TTS引擎: {self.engine}")
                return self._init_pyttsx3()
                
        except Exception as e:
            logger.error(f"TTS引擎初始化失败: {str(e)}")
            return False
    
    def _init_pyttsx3(self) -> bool:
        """
        初始化pyttsx3引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            import pyttsx3
            
            self.tts = pyttsx3.init()
            
            # 设置语音参数
            self.tts.setProperty('rate', self.rate)
            self.tts.setProperty('volume', self.volume)
            
            # 设置语言（根据系统可用的声音）
            try:
                # 获取可用的声音
                voices = self.tts.getProperty('voices')
                
                # 尝试找到中文声音
                for voice in voices:
                    if 'Chinese' in voice.name or 'zh' in voice.languages[0]:
                        self.tts.setProperty('voice', voice.id)
                        break
            except:
                logger.warning("无法设置中文语音，使用默认声音")
            
            logger.info("pyttsx3引擎初始化成功")
            return True
            
        except ImportError:
            logger.error("pyttsx3未安装，请运行: pip install pyttsx3")
            return False
        except Exception as e:
            logger.error(f"pyttsx3初始化失败: {str(e)}")
            return False
    
    def _init_edge_tts(self) -> bool:
        """
        初始化Edge TTS引擎
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            import edge_tts
            logger.info("Edge TTS引擎初始化成功")
            return True
        except ImportError:
            logger.error("edge-tts未安装，请运行: pip install edge-tts")
            return False
        except Exception as e:
            logger.error(f"Edge TTS初始化失败: {str(e)}")
            return False
    
    def speak(self, text: str, wait: bool = True) -> bool:
        """
        播放文本语音
        
        Args:
            text: 要播放的文本
            wait: 是否等待播放完成
            
        Returns:
            bool: 播放是否成功
        """
        if not text or len(text.strip()) == 0:
            return False
        
        try:
            if self.engine == "pyttsx3":
                return self._speak_pyttsx3(text, wait)
            elif self.engine == "edge-tts":
                return self._speak_edge_tts(text, wait)
            else:
                logger.error(f"未知的TTS引擎: {self.engine}")
                return False
                
        except Exception as e:
            logger.error(f"播放语音失败: {str(e)}")
            return False
    
    def _speak_pyttsx3(self, text: str, wait: bool = True) -> bool:
        """
        使用pyttsx3播放语音
        
        Args:
            text: 文本
            wait: 是否等待
            
        Returns:
            bool: 成功与否
        """
        if self.tts is None:
            return False
        
        try:
            self.tts.say(text)
            
            if wait:
                self.tts.runAndWait()
            else:
                # 在后台线程中播放
                self._run_tts_async()
            
            return True
        except Exception as e:
            logger.error(f"pyttsx3播放失败: {str(e)}")
            return False
    
    def _speak_edge_tts(self, text: str, wait: bool = True) -> bool:
        """
        使用Edge TTS播放语音
        
        Args:
            text: 文本
            wait: 是否等待
            
        Returns:
            bool: 成功与否
        """
        try:
            import edge_tts
            import asyncio
            
            async def async_speak():
                communicate = edge_tts.Communicate(
                    text=text,
                    voice=f"zh-CN-XiaoxiaoNeural",
                    rate=self.rate / 100,
                    volume=self.volume * 100
                )
                await communicate.save("temp_audio.mp3")
                logger.info("Edge TTS生成音频成功")
            
            if wait:
                asyncio.run(async_speak())
            else:
                # 在后台线程中运行
                threading.Thread(target=asyncio.run, args=(async_speak(),)).start()
            
            return True
        except Exception as e:
            logger.error(f"Edge TTS播放失败: {str(e)}")
            return False
    
    def _run_tts_async(self):
        """在后台线程中运行TTS"""
        if self.play_thread is None or not self.play_thread.is_alive():
            self.play_thread = threading.Thread(target=self.tts.runAndWait)
            self.play_thread.start()
    
    def stop(self):
        """停止播放"""
        if self.tts:
            try:
                self.tts.stop()
            except:
                pass


class VoiceAnnouncer:
    """语音播报管理器"""
    
    def __init__(self,
                 engine: str = "pyttsx3",
                 language: str = "zh-CN",
                 rate: int = 150,
                 volume: float = 1.0):
        """
        初始化语音播报管理器
        
        Args:
            engine: TTS引擎
            language: 语言
            rate: 语速
            volume: 音量
        """
        self.tts = TextToSpeech(engine, language, rate, volume)
        self.format_template = "{obstacle_class}距离{distance:.1f}米"
        self.enabled = True
        
    def initialize(self) -> bool:
        """初始化"""
        return self.tts.initialize()
    
    def announce_obstacle(self,
                         obstacle_class: str,
                         distance: float,
                         risk_level: str = "warning") -> bool:
        """
        播报障碍物信息
        
        Args:
            obstacle_class: 障碍物类别
            distance: 距离（米）
            risk_level: 风险等级 ("safe", "warning", "danger")
            
        Returns:
            bool: 播报是否成功
        """
        if not self.enabled:
            return False
        
        # 根据风险等级调整播报内容
        if risk_level == "danger":
            announcement = f"危险！{self.format_template.format(obstacle_class=obstacle_class, distance=distance)}"
        elif risk_level == "warning":
            announcement = f"注意！{self.format_template.format(obstacle_class=obstacle_class, distance=distance)}"
        else:
            announcement = f"{self.format_template.format(obstacle_class=obstacle_class, distance=distance)}"
        
        logger.info(f"播报: {announcement}")
        return self.tts.speak(announcement, wait=False)
    
    def announce_safe(self) -> bool:
        """播报安全提示"""
        if not self.enabled:
            return False
        
        announcement = "前方安全，可以继续前进"
        logger.info(f"播报: {announcement}")
        return self.tts.speak(announcement, wait=False)
    
    def announce_custom(self, text: str) -> bool:
        """
        播报自定义文本
        
        Args:
            text: 播报文本
            
        Returns:
            bool: 播报是否成功
        """
        if not self.enabled or not text:
            return False
        
        logger.info(f"播报: {text}")
        return self.tts.speak(text, wait=False)
    
    def set_enabled(self, enabled: bool):
        """启用/禁用语音播报"""
        self.enabled = enabled
    
    def stop(self):
        """停止播报"""
        self.tts.stop()


def create_announcer_from_config(config: dict) -> VoiceAnnouncer:
    """
    从配置字典创建播报管理器
    
    Args:
        config: 包含语音配置的字典
        
    Returns:
        VoiceAnnouncer: 播报管理器
    """
    voice_config = config.get('voice', {})
    
    announcer = VoiceAnnouncer(
        engine=voice_config.get('tts_engine', 'pyttsx3'),
        language=voice_config.get('language', 'zh-CN'),
        rate=voice_config.get('speaker_rate', 150),
        volume=voice_config.get('volume', 1.0)
    )
    
    return announcer


if __name__ == "__main__":
    # 快速测试
    logging.basicConfig(level=logging.INFO)
    
    announcer = VoiceAnnouncer(
        engine="pyttsx3",
        language="zh-CN",
        rate=150,
        volume=1.0
    )
    
    if announcer.initialize():
        # 测试不同的播报
        announcer.announce_obstacle("行人", 1.5, "danger")
        announcer.announce_obstacle("栏杆", 2.5, "warning")
        announcer.announce_safe()
        announcer.announce_custom("系统启动成功")
    else:
        print("语音引擎初始化失败")
