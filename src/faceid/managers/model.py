import torch
from insightface.app import FaceAnalysis
from faceid.core.analysis import FaceAnalysisWrapper
from faceid.core.utils import ConfigLoader
from faceid.logging.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)


class ModelManager:
    """
    FaceID sisteminde kullanılan modellerin (RetinaFace + ArcFace) yüklenmesini ve yönetimini sağlar.

    Özellikler:
      - FaceAnalysis modeli tek bir örnek (singleton) olarak yüklenir.
      - GPU/CPU ortamına göre otomatik provider seçimi yapılır.
      - Config (YAML) dosyasından model adı, provider ve det_size bilgileri okunur.
    """
    _detector = None  # Tekil model önbelleği

    @classmethod
    def get_detector(cls, det_size=(640, 640)) -> FaceAnalysisWrapper:
        """
        FaceAnalysis modelini yükler ve FaceAnalysisWrapper ile döndürür.

        Parametreler:
          det_size (tuple): Yüz tespiti modelinin giriş çözünürlüğü (varsayılan 640x640)

        Dönüş:
          FaceAnalysisWrapper: Yüklenmiş ve kullanıma hazır model sarmalayıcısı
        """
        if cls._detector is None:
            # YAML yapılandırma dosyasını yükle
            cfg = ConfigLoader().load().get("models", {})
            model_name = cfg.get("arcface_name", "buffalo_l")
            config_providers = cfg.get("providers", None)
            config_det_size = tuple(cfg.get("det_size", det_size))

            # Donanım ortamına göre provider seçimi
            if torch.cuda.is_available():
                default_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                ctx_id = 0
                logger.info("🚀 GPU bulundu, CUDAExecutionProvider kullanılacak.")
            else:
                default_providers = ["CPUExecutionProvider"]
                ctx_id = -1
                logger.info("⚙️ GPU bulunamadı, CPUExecutionProvider kullanılacak.")

            # Config’te provider tanımlıysa onu kullan, aksi halde varsayılanı al
            providers = config_providers or default_providers

            logger.info(f"[ModelManager] Model: {model_name}, Providers: {providers}, det_size={config_det_size}")

            # FaceAnalysis modelini başlat ve hazırla
            app = FaceAnalysis(name=model_name, providers=providers)
            app.prepare(ctx_id=ctx_id, det_size=config_det_size)

            # Tekil model örneğini kaydet
            cls._detector = FaceAnalysisWrapper(app)

        return cls._detector
