# face_analysis_wrapper.py
import cv2
import numpy as np
from typing import Optional, List, Tuple
from faceid.logging.logger import LoggerManager
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

logger = LoggerManager.get_logger(__name__)


class FaceAnalysisWrapper:
    """
    Yüz tespiti ve embedding çıkarımı işlemlerini yöneten yardımcı sınıf.

    Model yükleme işlemi doğrudan bu sınıf içinde değil, ModelManager tarafından yapılır.

    Parametreler:
      - face_analysis_app: InsightFace'in FaceAnalysis nesnesi
      - input_mode:
          "insightface_arcface" → BGR, (img - 127.5) / 127.5
          "rgb_255"             → RGB, img / 255.0
    """

    def __init__(self, face_analysis_app: FaceAnalysis, *, input_mode: str = "insightface_arcface"):
        """Hazır FaceAnalysis nesnesi alarak sınıfı başlatır."""
        if not isinstance(face_analysis_app, FaceAnalysis):
            raise TypeError("face_analysis_app parametresi FaceAnalysis türünde olmalıdır.")
        self.app = face_analysis_app
        self.input_mode = input_mode
        logger.info("FaceAnalysisWrapper: Model başarıyla yüklendi ve hazır.")

    # ------------------- Yüz Tespiti ------------------- #
    def detect_faces(
        self,
        frame,
        return_count: bool = False,
        *,
        min_score: float = 0.5,
        max_faces: Optional[int] = None,
        align: bool = False
    ):
        """Kare üzerinde yüzleri tespit eder ve isteğe bağlı olarak hizalama yapar."""
        if frame is None:
            return ([], 0) if return_count else []

        try:
            faces = self.app.get(frame) or []
        except Exception as e:
            logger.error(f"Yüz tespiti sırasında hata oluştu: {e}")
            return ([], 0) if return_count else []

        # Düşük güven skoruna sahip yüzleri filtrele
        faces = [f for f in faces if getattr(f, "det_score", 1.0) >= min_score]
        faces.sort(key=lambda f: getattr(f, "det_score", 0.0), reverse=True)
        if max_faces is not None:
            faces = faces[:max_faces]

        results = []
        for face in faces:
            if align and getattr(face, "kps", None) is not None:
                # Gerekirse yüz hizalaması yap
                try:
                    aligned_rgb = norm_crop(frame, face.kps)  # 112x112 RGB yüz
                    aligned_bgr = cv2.cvtColor(aligned_rgb, cv2.COLOR_RGB2BGR)
                    results.append((face, aligned_bgr))
                except Exception as e:
                    logger.debug(f"Yüz hizalama başarısız: {e}")
                    results.append((face, None))
            else:
                results.append(face)

        return (results, len(results)) if return_count else results

    @staticmethod
    def draw_annotations(frame, faces, color=(0, 255, 0)):
        """Yüz kutularını kare üzerine çizer."""
        if frame is None:
            return frame
        for f in faces:
            face = f[0] if isinstance(f, (list, tuple)) else f
            if getattr(face, "bbox", None) is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    # ------------------- Embedding İşlemleri ------------------- #
    def get_embedding(self, faces: List) -> np.ndarray:
        """Tek bir yüz için embedding (özellik vektörü) üretir."""
        if not faces:
            raise ValueError("Hiç yüz verilmedi (tekli mod).")
        if len(faces) > 1:
            raise ValueError(f"Tekli modda yalnızca 1 yüz bekleniyor, {len(faces)} yüz verildi.")

        face = faces[0]
        if face.embedding is None or face.embedding.size == 0:
            raise RuntimeError("Model boş embedding döndürdü (tekli).")

        emb = face.embedding.reshape(-1).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(emb))
        if norm == 0.0:
            raise ValueError("Embedding vektörünün normu sıfır. Normalize edilemez.")
        return emb / norm

    def get_embeddings(self, faces: List) -> List[np.ndarray]:
        """Birden fazla yüz için embedding (özellik vektörü) üretir."""
        if not faces:
            raise ValueError("Hiç yüz verilmedi (çoklu mod).")

        embeddings = []
        for face in faces:
            if face.embedding is None or face.embedding.size == 0:
                logger.warning("Boş embedding atlandı.")
                continue
            emb = face.embedding.reshape(-1).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(emb))
            if norm == 0.0:
                logger.warning("Normu sıfır embedding atlandı.")
                continue
            embeddings.append(emb / norm)

        if not embeddings:
            raise RuntimeError("Hiç geçerli embedding üretilemedi (batch).")
        return embeddings
