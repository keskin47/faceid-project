import logging
import logging.handlers
import os
import sys
from typing import Optional
from faceid.core.utils import ConfigLoader  # ✅ yapılandırma yükleyici

class _UTCFormatter(logging.Formatter):
    """Zaman damgasını UTC formatında göstermek için özel formatter sınıfı."""
    converter = staticmethod(__import__("time").gmtime)


class LoggerManager:
    """
    Uygulama genelinde log (günlük) yönetimini sağlayan merkezi sınıf.

    Özellikler:
      - Tek bir kök logger oluşturur, tüm modüller bu logger'ı paylaşır.
      - Konsola ve dönen (rotating) dosyalara log yazma desteği sunar.
      - Log ayarları YAML dosyasından yüklenebilir.
    """
    _root_logger: Optional[logging.Logger] = None
    _log_file_path: Optional[str] = None

    # -------------------- Yardımcı Fonksiyonlar -------------------- #
    @staticmethod
    def _get_base_path():
        """Uygulamanın taban çalışma dizinini döndürür (PyInstaller uyumlu)."""
        if getattr(sys, "frozen", False):
            return os.path.dirname(sys.executable)
        return os.path.abspath(".")

    @staticmethod
    def _get_log_file_path():
        """Log dosyasının konumunu belirler ve klasörü yoksa oluşturur."""
        if LoggerManager._log_file_path:
            return LoggerManager._log_file_path
        base_path = LoggerManager._get_base_path()
        log_dir = os.path.join(base_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        LoggerManager._log_file_path = os.path.join(log_dir, "system.log")
        return LoggerManager._log_file_path

    # -------------------- Logger Başlatma -------------------- #
    @staticmethod
    def init_logger(
        name: str = "FaceID",
        level: int = None,
        *,
        utc: bool = False,
        max_bytes: int = None,       # YAML’den okunabilir
        backup_count: int = None,    # YAML’den okunabilir
        stream: bool = True
    ) -> logging.Logger:
        """
        Uygulama başlangıcında bir kez çağrılır.
        Diğer modüller logger almak için get_logger(__name__) metodunu kullanır.
        """
        if LoggerManager._root_logger is not None:
            return LoggerManager._root_logger

        # YAML konfigürasyonundan log ayarlarını yükle
        config = ConfigLoader().load().get("logging", {})
        config_level = config.get("level", "INFO").upper()
        config_max_bytes = config.get("max_bytes", 5 * 1024 * 1024)
        config_backup_count = config.get("backup_count", 5)

        # Log seviyesi önceliği: parametre > ortam değişkeni > YAML
        if level is None:
            env_level = os.getenv("FACEID_LOG_LEVEL", config_level)
            level = getattr(logging, env_level, logging.INFO)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False

        # Logger zaten kurulmuşsa tekrar oluşturulmaz
        if logger.handlers:
            LoggerManager._root_logger = logger
            return logger

        # Log formatı
        fmt = "%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(funcName)s - %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        FormatterCls = _UTCFormatter if utc else logging.Formatter
        formatter = FormatterCls(fmt, datefmt=datefmt)

        # Konsol çıktısı (StreamHandler)
        if stream:
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            logger.addHandler(sh)

        # Dosya çıktısı (RotatingFileHandler)
        log_file = LoggerManager._get_log_file_path()
        fh = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes or config_max_bytes,
            backupCount=backup_count or config_backup_count,
            encoding="utf-8"
        )
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        LoggerManager._root_logger = logger
        return logger

    # -------------------- Logger Erişimi -------------------- #
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Belirtilen ad için alt logger nesnesini döndürür."""
        root = LoggerManager._root_logger or LoggerManager.init_logger()
        return root.getChild(name) if root.name else logging.getLogger(name)

    @staticmethod
    def set_level(level: int):
        """Log seviyesini çalışma anında değiştirmeye olanak tanır."""
        if LoggerManager._root_logger:
            LoggerManager._root_logger.setLevel(level)

    # -------------------- Hata Yakalama (Exception Hook) -------------------- #
    @staticmethod
    def attach_exception_hook():
        """Yakalanmamış hataları log dosyasına otomatik kaydeder."""
        def _hook(exc_type, exc, tb):
            logger = LoggerManager.get_logger("uncaught")
            logger.critical("Yakalanmamış hata", exc_info=(exc_type, exc, tb))
            if LoggerManager._orig_hook:
                LoggerManager._orig_hook(exc_type, exc, tb)

        if not hasattr(LoggerManager, "_orig_hook"):
            LoggerManager._orig_hook = sys.excepthook
            sys.excepthook = _hook

    # -------------------- Kısa Log İşlevleri -------------------- #
    @staticmethod
    def info(msg):     LoggerManager.get_logger("app").info(msg)
    @staticmethod
    def warning(msg):  LoggerManager.get_logger("app").warning(msg)
    @staticmethod
    def error(msg):    LoggerManager.get_logger("app").error(msg)
    @staticmethod
    def debug(msg):    LoggerManager.get_logger("app").debug(msg)
    @staticmethod
    def critical(msg): LoggerManager.get_logger("app").critical(msg)
