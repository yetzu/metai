"""
MetAI 统一日志系统（MetLog）
提供统一的日志输出、格式化、彩色控制台输出以及文件持久化能力

功能特性：
- 多级别日志（DEBUG / INFO / WARNING / ERROR / CRITICAL）
- 控制台彩色输出
- 支持文件输出与滚动切分
- 敏感信息自动过滤
- 运行时性能统计
- 计时器上下文管理
- 函数调用装饰器
- 线程安全

"""

import sys
import time
import logging
import logging.handlers
import threading
import re
from datetime import datetime
from typing import Optional, Union, Any, Dict, List
from pathlib import Path
from contextlib import contextmanager
from functools import wraps

class LogLevel:
    """日志级别常量（Log level constants）"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    
    LEVEL_NAMES = {
        DEBUG: "DEBUG",
        INFO: "INFO", 
        WARNING: "WARNING",
        ERROR: "ERROR",
        CRITICAL: "CRITICAL"
    }
    
    LEVEL_COLORS = {
        DEBUG: "\033[36m",      # Cyan
        INFO: "\033[32m",       # Green
        WARNING: "\033[33m",    # Yellow
        ERROR: "\033[31m",      # Red
        CRITICAL: "\033[35m"    # Magenta
    }


class MetLogger:
    """MetAI 统一日志器（MetAI Unified Logger）
    
    提供以下能力：
    - 多级别日志与彩色控制台输出
    - 支持文件输出与自动滚动切分
    - 敏感信息自动过滤
    - 运行时统计与性能监控
    - 线程安全
    - 计时上下文管理器与装饰器
    """
    
    # Sensitive information patterns - using named groups
    SENSITIVE_PATTERNS = [
        (r'(password\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
        (r'(token\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
        (r'(key\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
        (r'(secret\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
        (r'(api_key\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
        (r'(access_token\s*[:=]\s*)([^\s,]+)', r'\1***FILTERED***'),
    ]
    
    def __init__(
        self,
        name: str = "",
        level: int = LogLevel.INFO,
        enable_colors: bool = True,
        enable_file_logging: bool = True,
        log_dir: str = "tmp/logs",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        format_string: Optional[str] = None,
        enable_sensitive_filtering: bool = True
    ):
        self.name = name
        self.level = level
        self.enable_colors = enable_colors and sys.stdout.isatty()
        self.enable_file_logging = enable_file_logging
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_sensitive_filtering = enable_sensitive_filtering
        
        # Statistics tracking
        self.stats = {
            "total_logs": 0,
            "logs_by_level": {level: 0 for level in LogLevel.LEVEL_NAMES.keys()},
            "start_time": time.time(),
            "last_log_time": None,
            "filtered_sensitive_count": 0,
            "file_write_errors": 0,
            "average_log_interval": 0
        }
        
        # Thread lock for thread-safe operations
        self._lock = threading.Lock()
        
        # Performance optimization: timestamp caching
        self._last_timestamp = None
        self._timestamp_cache_duration = 0.001  # Reuse timestamp within 1ms
        
        # Set log format
        if format_string is None:
            self.format_string = "[{timestamp}] [{level}] [{name}] {message}"
        else:
            self.format_string = format_string
            
        # Validate configuration
        self._validate_config()
        
        # Initialize file logging
        if self.enable_file_logging:
            self._setup_file_logging()
    
    def _filter_sensitive_info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> tuple[str, Optional[Dict[str, Any]]]:
        """从消息与扩展参数中过滤敏感信息（password/token/key/secret 等）"""
        if not self.enable_sensitive_filtering:
            return message, extra
        
        # Filter message string
        filtered_message = message
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            filtered_message = re.sub(pattern, replacement, filtered_message, flags=re.IGNORECASE)
        
        # Filter extra parameters
        filtered_extra = None
        if extra:
            filtered_extra = {}
            for key, value in extra.items():
                # Check if key contains sensitive information
                if any(sensitive in key.lower() for sensitive in ['password', 'token', 'key', 'secret', 'api_key', 'access_token']):
                    filtered_extra[key] = "***FILTERED***"
                else:
                    # Check if value contains sensitive information
                    value_str = str(value)
                    filtered_value = value_str
                    for pattern, replacement in self.SENSITIVE_PATTERNS:
                        filtered_value = re.sub(pattern, replacement, filtered_value, flags=re.IGNORECASE)
                    filtered_extra[key] = filtered_value if filtered_value != value_str else value
        
        return filtered_message, filtered_extra
    
    def _validate_config(self):
        """校验配置参数是否合法"""
        if not isinstance(self.level, int) or self.level < 0 or self.level > 4:
            raise ValueError("Log level must be between 0-4")
        
        if self.max_file_size <= 0:
            raise ValueError("Maximum file size must be greater than 0")
        
        if self.backup_count < 0:
            raise ValueError("Backup count cannot be negative")
        
        if not isinstance(self.name, str):
            raise ValueError("Logger name must be a string")
    
    def _setup_file_logging(self):
        """初始化带滚动的文件日志输出"""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create file handler
            log_name = self.name if self.name else "metlog"
            log_file = self.log_dir / f"{log_name}.log"
            
            # Use Python logging module
            logger_name = f"{self.name}_file" if self.name else "metlog_file"
            self.file_logger = logging.getLogger(logger_name)
            self.file_logger.setLevel(logging.DEBUG)
            
            # File handler with rotation
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            
            # File formatter
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            self.file_logger.addHandler(file_handler)
            
        except Exception as e:
            self._log_error(f"Failed to setup file logging: {e}")
            self.enable_file_logging = False
    
    def _log_error(self, message: str):
        """内部错误处理：用于记录日志系统自身的错误"""
        try:
            print(f"[ERROR] MetLog: {message}")
        except Exception:
            # Fallback to basic print if all else fails
            pass
    
    def _get_timestamp(self) -> str:
        """获取时间戳（带毫秒，并做短暂缓存以优化性能）"""
        now = time.time()
        if (self._last_timestamp is None or 
            now - self._last_timestamp[0] > self._timestamp_cache_duration):
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            self._last_timestamp = (now, timestamp_str)
        return self._last_timestamp[1]
    
    def _format_message(
        self, 
        level: int, 
        message: str, 
        extra: Optional[Dict[str, Any]] = None
    ) -> str:
        """格式化日志消息：含时间戳、级别、名称与可选扩展字段"""
        timestamp = self._get_timestamp()
        level_name = LogLevel.LEVEL_NAMES[level]
        
        # Basic formatting
        if self.name:
            formatted = self.format_string.format(
                timestamp=timestamp,
                level=level_name,
                name=self.name,
                message=str(message)
            )
        else:
            # When name is empty, don't show name part
            formatted = f"[{timestamp}] [{level_name}] {str(message)}"
        
        # Add extra information
        if extra:
            extra_str = " | ".join([f"{k}={v}" for k, v in extra.items()])
            formatted += f" | {extra_str}"
        
        return formatted
    
    def _log(self, level: int, message: str, extra: Optional[Dict[str, Any]] = None):
        """核心日志写入流程：过滤敏感信息、格式化、输出控制台与文件"""
        if level < self.level:
            return
        
        with self._lock:
            # Update statistics
            self.stats["total_logs"] += 1
            self.stats["logs_by_level"][level] += 1
            self.stats["last_log_time"] = time.time()
            
            # Filter sensitive information
            filtered_message, filtered_extra = self._filter_sensitive_info(str(message), extra)
            if filtered_message != str(message) or (extra and filtered_extra != extra):
                self.stats["filtered_sensitive_count"] += 1
            
            # Format message
            formatted_message = self._format_message(level, filtered_message, filtered_extra)
            
            # Console output
            if self.enable_colors:
                color = LogLevel.LEVEL_COLORS[level]
                reset_color = "\033[0m"
                print(f"{color}{formatted_message}{reset_color}")
            else:
                print(formatted_message)
            
            # File output
            if self.enable_file_logging:
                try:
                    log_level_name = LogLevel.LEVEL_NAMES[level]
                    if hasattr(self, 'file_logger'):
                        if level == LogLevel.DEBUG:
                            self.file_logger.debug(message)
                        elif level == LogLevel.INFO:
                            self.file_logger.info(message)
                        elif level == LogLevel.WARNING:
                            self.file_logger.warning(message)
                        elif level == LogLevel.ERROR:
                            self.file_logger.error(message)
                        elif level == LogLevel.CRITICAL:
                            self.file_logger.critical(message)
                except Exception as e:
                    self.stats["file_write_errors"] += 1
                    self._log_error(f"Failed to write to log file: {e}")
    
    def debug(self, message: str, **kwargs):
        """输出 DEBUG 级别日志"""
        self._log(LogLevel.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs):
        """输出 INFO 级别日志"""
        self._log(LogLevel.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """输出 WARNING 级别日志"""
        self._log(LogLevel.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """输出 ERROR 级别日志"""
        self._log(LogLevel.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """输出 CRITICAL 级别日志"""
        self._log(LogLevel.CRITICAL, message, kwargs)
    
    def set_level(self, level: int):
        """设置最低输出日志级别"""
        self.level = level
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前日志统计信息（总量/分布/平均间隔等）"""
        with self._lock:
            stats = self.stats.copy()
            stats["uptime"] = time.time() - stats["start_time"]
            
            # Calculate average log interval
            if stats["total_logs"] > 1:
                stats["average_log_interval"] = stats["uptime"] / stats["total_logs"]
            else:
                stats["average_log_interval"] = 0
            
            # Add log level distribution percentages
            total_logs = stats["total_logs"]
            if total_logs > 0:
                stats["level_distribution"] = {
                    LogLevel.LEVEL_NAMES[level]: (count / total_logs) * 100
                    for level, count in stats["logs_by_level"].items()
                }
            else:
                stats["level_distribution"] = {}
            
            return stats
    
    def reset_stats(self):
        """重置日志统计信息"""
        with self._lock:
            self.stats = {
                "total_logs": 0,
                "logs_by_level": {level: 0 for level in LogLevel.LEVEL_NAMES.keys()},
                "start_time": time.time(),
                "last_log_time": None,
                "filtered_sensitive_count": 0,
                "file_write_errors": 0,
                "average_log_interval": 0
            }
    
    @contextmanager
    def timer(self, operation: str):
        """计时上下文：用于度量某个操作的耗时"""
        start_time = time.time()
        self.info(f"Starting: {operation}")
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.info(f"Completed: {operation}", duration=f"{duration:.3f}s")
    
    def log_function_call(self, func_name: str, args: tuple = (), kwargs: Optional[dict] = None):
        """记录函数调用与参数信息（用于调试追踪）"""
        kwargs = kwargs or {}
        args_str = ", ".join([str(arg) for arg in args])
        kwargs_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        params = ", ".join(filter(None, [args_str, kwargs_str]))
        self.debug(f"Function call: {func_name}({params})")


# Global logger instance
_default_logger = MetLogger()

# Convenience functions
def MLOGD(message: str, **kwargs):
    """DEBUG 级别日志"""
    _default_logger.debug(message, **kwargs)

def MLOGI(message: str, **kwargs):
    """INFO 级别日志"""
    _default_logger.info(message, **kwargs)

def MLOGW(message: str, **kwargs):
    """WARNING 级别日志"""
    _default_logger.warning(message, **kwargs)

def MLOGE(message: str, **kwargs):
    """ERROR 级别日志"""
    _default_logger.error(message, **kwargs)

def MLOGC(message: str, **kwargs):
    """CRITICAL 级别日志"""
    _default_logger.critical(message, **kwargs)

def set_level(level: int):
    """设置日志输出级别"""
    _default_logger.set_level(level)

def get_stats() -> Dict[str, Any]:
    """获取日志统计信息"""
    return _default_logger.get_stats()

def reset_stats():
    """重置日志统计信息"""
    _default_logger.reset_stats()

@contextmanager
def timer(operation: str):
    """计时上下文（使用默认全局日志器）"""
    with _default_logger.timer(operation):
        yield


# Decorators
def log_function_calls(level: int = LogLevel.DEBUG):
    """函数调用日志装饰器：按级别记录调用信息"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if level >= _default_logger.level:
                _default_logger.log_function_call(func.__name__, args, kwargs or {})
            return func(*args, **kwargs)
        return wrapper
    return decorator

def log_execution_time(operation_name: Optional[str] = None):
    """耗时日志装饰器：自动测量并记录函数执行时间"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with _default_logger.timer(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Backward compatible MLOG function
def MLOG(message: str = "", level: str = "INFO", **kwargs):
    """
    向后兼容的 MLOG 函数（Backward compatible）
    保持旧调用方式的同时，兼容新的多级别与扩展字段能力
    """
    # If message is empty, output empty line
    if not message:
        print()
        return
    
    # Parse level
    level_map = {
        "DEBUG": LogLevel.DEBUG,
        "INFO": LogLevel.INFO,
        "WARNING": LogLevel.WARNING,
        "ERROR": LogLevel.ERROR,
        "CRITICAL": LogLevel.CRITICAL
    }
    
    # If message contains level identifier, auto-parse
    if message.startswith("[DEBUG]"):
        _default_logger.debug(message[7:].strip(), **kwargs)
    elif message.startswith("[INFO]"):
        _default_logger.info(message[6:].strip(), **kwargs)
    elif message.startswith("[WARNING]"):
        _default_logger.warning(message[9:].strip(), **kwargs)
    elif message.startswith("[ERROR]"):
        _default_logger.error(message[7:].strip(), **kwargs)
    elif message.startswith("[CRITICAL]"):
        _default_logger.critical(message[10:].strip(), **kwargs)
    elif message.startswith("[RESULT]"):
        _default_logger.info(message[8:].strip(), **kwargs)
    elif message.startswith("[STATS]"):
        _default_logger.info(message[7:].strip(), **kwargs)
    elif message.startswith("[EXAMPLE]"):
        _default_logger.info(message[9:].strip(), **kwargs)
    else:
        # Default to specified level
        log_level = level_map.get(level.upper(), LogLevel.INFO)
        _default_logger._log(log_level, message, kwargs)


# 使用示例
if __name__ == "__main__":
    # 基本使用
    MLOG("MetLog 日志系统测试")
    MLOGD("调试信息", user_id=123, action="login")
    MLOGW("警告信息", error_code=404)
    MLOGE("错误信息", exception="FileNotFoundError")
    
    # 敏感信息过滤测试
    MLOGI("用户登录", password="secret123", token="abc123")
    
    # 计时器使用
    with timer("数据处理"):
        import time
        time.sleep(0.1)
    
    # 装饰器使用
    @log_function_calls()
    @log_execution_time("计算函数")
    def calculate_sum(a, b):
        return a + b
    
    result = calculate_sum(1, 2)
    MLOGI(f"计算结果: {result}")
    
    # 统计信息
    stats = get_stats()
    MLOGI("日志统计", **stats)
