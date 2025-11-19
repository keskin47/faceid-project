import cv2
import threading
import time
from typing import Optional, Union
from faceid.logging.logger import LoggerManager
from faceid.core.utils import ConfigLoader

Number = Union[int, str]
logger = LoggerManager.get_logger(__name__)


class CameraOpenError(Exception):
    """Kamera açılamadığında fırlatılan özel hata sınıfı."""
    pass


class CameraManager:
    """
    Kamera yönetimini üstlenen sınıf.

    Özellikler:
      - Bir veya birden fazla kamera kaynağı (dahili, harici, IP) ile çalışabilir.
      - Otomatik yeniden bağlanma, çözünürlük ayarlama ve FPS sınırlama desteklenir.
      - Thread tabanlı sürekli kare okuma mekanizması içerir.
    """

    def __init__(
        self,
        source: Optional[Number] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps_limit: Optional[float] = None,
        reconnect_backoff: float = 1.0,
        config_path: str = "configs/default.yaml"
    ):
        """Kamera yöneticisini yapılandırma dosyasına göre başlatır."""
        # YAML konfigürasyonundan değerleri yükle
        config = ConfigLoader(config_path).load().get("camera", {})

        mode = config.get("mode", "auto")
        if source is None:
            if mode == "internal":
                source = config.get("index_internal", 0)
            elif mode == "external":
                source = config.get("index_external", 1)
            elif mode == "custom" and config.get("custom_url"):
                source = config.get("custom_url")
            else:  # otomatik seçim
                source = 0

        self.source = source
        self.width = width or config.get("width", 640)
        self.height = height or config.get("height", 480)
        self.fps_limit = fps_limit or config.get("fps_limit", 30.0)
        self.reconnect_backoff = reconnect_backoff

        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._frame = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(
            f"[CameraManager] Başlatıldı -> source={self.source}, width={self.width}, height={self.height}, fps={self.fps_limit}"
        )

    # --------- Genel Kullanıcı Arayüzü (Public API) --------- #
    def open(self):
        """Kamerayı açar ve okuma thread’ini başlatır."""
        if self._running:
            return
        self._open_capture_or_raise()
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.info(f"[CameraManager] Kamera başlatıldı: {self.source}")

    def close(self):
        """Kamerayı kapatır ve thread'i sonlandırır."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self._release_capture()
        logger.info("[CameraManager] Kamera kapatıldı.")

    def is_opened(self) -> bool:
        """Kamera bağlantısının açık olup olmadığını kontrol eder."""
        return self._cap is not None and self._cap.isOpened()

    def get_latest_frame(self):
        """En son okunan kareyi döndürür (thread-safe)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def snapshot(self, timeout_sec: float = 2.0):
        """Belirtilen süre içinde bir kare yakalamayı dener."""
        t0 = time.time()
        while time.time() - t0 < timeout_sec:
            frame = self.get_latest_frame()
            if frame is not None:
                return True, frame
            time.sleep(0.01)
        return False, None

    def read(self, timeout_sec: float = 0.5):
        """
        Thread aktifse son kareyi döndürür.
        Değilse doğrudan kameradan tek kare okur.
        """
        if self._running:
            frame = self.get_latest_frame()
            return (frame is not None), frame

        if not self.is_opened():
            self._open_capture_or_raise()
        ok, frame = self._cap.read()
        return (ok and frame is not None), frame

    def switch_source(self, new_source: Number):
        """Kamera kaynağını değiştirir ve gerekiyorsa yeniden başlatır."""
        logger.info(f"[CameraManager] Kamera kaynağı değiştiriliyor: {self.source} -> {new_source}")
        self.source = new_source
        was_running = self._running
        self.close()
        if was_running:
            self.open()

    @staticmethod
    def test_open(source: Number, width: int = 640, height: int = 480) -> bool:
        """
        Belirtilen kamera kaynağının çalışıp çalışmadığını test eder.
        İlk kare başarıyla okunabiliyorsa True döner.
        """
        cap = CameraManager._create_capture(source)
        if not cap or not cap.isOpened():
            logger.debug(f"[CameraManager] test_open: kaynak açılamadı -> {source}")
            return False
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        cap.release()
        if not ok:
            logger.debug(f"[CameraManager] test_open: ilk kare okunamadı -> {source}")
        return bool(ok)

    # --------- Dahili Yardımcı Metotlar --------- #
    def _reader_loop(self):
        """
        Arka planda çalışan thread döngüsü.
        Sürekli kare okur, FPS sınırına göre bekleme uygular,
        bağlantı koptuğunda yeniden bağlanmayı dener.
        """
        prev = 0.0
        while self._running:
            if not self.is_opened():
                try:
                    self._open_capture_or_raise()
                    logger.info("[CameraManager] Kamera yeniden bağlandı.")
                except Exception as e:
                    logger.warning(f"[CameraManager] Kamera yeniden bağlanamadı: {e}")
                    time.sleep(self.reconnect_backoff)
                    continue

            ok, frame = self._cap.read()
            if not ok or frame is None:
                logger.debug("[CameraManager] Kare okunamadı, yeniden bağlanma denenecek.")
                self._release_capture()
                time.sleep(self.reconnect_backoff)
                continue

            # Gerekiyorsa çözünürlüğü yeniden boyutlandır
            if isinstance(self.source, int):
                if (self.width and self.height) and (
                    frame.shape[1] != self.width or frame.shape[0] != self.height
                ):
                    frame = cv2.resize(frame, (self.width, self.height))

            with self._lock:
                self._frame = frame

            # FPS sınırına göre bekleme uygula
            if self.fps_limit:
                now = time.time()
                dt = now - prev
                target = 1.0 / self.fps_limit
                if dt < target:
                    time.sleep(target - dt)
                prev = time.time()

    def _open_capture_or_raise(self):
        """Kamerayı açar, açılamazsa hata fırlatır."""
        self._release_capture()
        self._cap = self._create_capture(self.source)
        if not self._cap or not self._cap.isOpened():
            self._release_capture()
            logger.error(f"[CameraManager] Kamera açılamadı: {self.source}")
            raise CameraOpenError(f"Kamera açılamadı: {self.source}")

        # Kamera çözünürlüğünü ayarla
        if isinstance(self.source, int):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            rw = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            rh = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.debug(f"[CameraManager] Açılış çözünürlüğü -> {rw}x{rh}")

    @staticmethod
    def _create_capture(source: Number) -> Optional[cv2.VideoCapture]:
        """Kaynağa uygun şekilde cv2.VideoCapture nesnesi oluşturur."""
        if isinstance(source, int):
            return cv2.VideoCapture(source, cv2.CAP_DSHOW)
        return cv2.VideoCapture(source)

    def _release_capture(self):
        """Kamera bağlantısını güvenli bir şekilde serbest bırakır."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    # --------- Context Manager Desteği --------- #
    def __enter__(self):
        """with ifadesiyle kullanılabilir hale getirir."""
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        """with bloğundan çıkarken kamerayı otomatik kapatır."""
        self.close()
