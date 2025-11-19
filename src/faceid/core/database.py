import numpy as np
import pickle
import hashlib
import os
from typing import Dict, List, Tuple, Optional

from faceid.core.utils import get_resource_path, ConfigLoader  # ✅ gerekli yardımcılar
from faceid.logging.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)

Record = Dict[str, object]  # {"name": str, "embeddings": List[np.ndarray], "centroid": np.ndarray}


def _as_norm_f32(vec: np.ndarray) -> np.ndarray:
    """Verilen vektörü float32 tipine çevirir, bellekte C-contiguous hale getirir ve L2-normalize eder."""
    v = np.asarray(vec, dtype=np.float32)
    if not v.flags.c_contiguous:
        v = np.ascontiguousarray(v)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("Embedding normu sıfır. Normalize edilemez.")
    return v / n


class FaceDatabase:
    """
    Yüz embedding veritabanını yöneten sınıf.

    YAML yapılandırma dosyasından şu değerleri alır:
      - threshold: thresholds.cosine_similarity
      - db_path: paths.database_file
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        max_per_person: Optional[int] = None,
        detailed_margin: float = 0.05,
        db_path: Optional[str] = None,
    ):
        """Veritabanını başlatır ve yapılandırma ayarlarını yükler."""
        config = ConfigLoader().load()
        cfg_threshold = config.get("thresholds", {}).get("cosine_similarity", 0.38)
        cfg_db_path = config.get("paths", {}).get("database_file", "data/app/face_db.pkl")

        # Öncelik sırası: parametre > config > varsayılan değer
        self.threshold = float(threshold or cfg_threshold)
        self.db_path = get_resource_path(db_path or cfg_db_path)
        self.sig_path = os.path.splitext(self.db_path)[0] + ".sig"

        self.db: Dict[str, Record] = {}
        self.max_per_person = max_per_person
        self.detailed_margin = detailed_margin

        logger.info(f"[FaceDB] Başlatıldı | Eşik: {self.threshold:.3f}, Yol: {self.db_path}")

    # -------------------- Kayıt İşlemleri (CRUD) -------------------- #
    def add_person(self, student_id: str, name: str, embedding: np.ndarray):
        """Yeni bir kişi ekler veya mevcut kişinin embedding bilgilerini günceller."""
        emb = _as_norm_f32(embedding)
        rec: Record = self.db.setdefault(student_id, {"name": name, "embeddings": [], "centroid": None})

        # İsim güncellemesi gerekiyorsa değiştir
        if isinstance(rec.get("name"), str) and rec["name"] != name:
            rec["name"] = name

        embs: List[np.ndarray] = rec["embeddings"]  # type: ignore
        embs.append(emb)
        if self.max_per_person is not None and len(embs) > self.max_per_person:
            embs.pop(0)

        arr = np.vstack(embs) if len(embs) > 1 else embs[0][None, :]
        rec["centroid"] = np.mean(arr, axis=0).astype(np.float32)
        rec["centroid"] /= np.linalg.norm(rec["centroid"]) + 1e-12

    def is_student_registered(self, student_id: str) -> bool:
        """Belirtilen öğrencinin sistemde kayıtlı olup olmadığını döndürür."""
        rec = self.db.get(student_id)
        return bool(rec and rec.get("embeddings"))

    def remove_person(self, student_id: str) -> bool:
        """Belirtilen öğrenciyi veritabanından siler."""
        return self.db.pop(student_id, None) is not None

    def list_ids(self) -> List[str]:
        """Kayıtlı tüm öğrenci ID'lerini döndürür."""
        return list(self.db.keys())

    def get_person_name(self, student_id: str) -> Optional[str]:
        """Belirtilen ID'ye ait öğrencinin adını döndürür."""
        rec = self.db.get(student_id)
        return None if rec is None else rec.get("name")  # type: ignore

    # -------------------- Eşik Ayarları -------------------- #
    def set_threshold(self, value: float):
        """Tanıma eşiğini günceller."""
        self.threshold = float(value)

    def set_detailed_margin(self, value: float):
        """Detaylı eşik aralığını günceller."""
        self.detailed_margin = float(value)

    # -------------------- Eşleştirme (Matching) -------------------- #
    def _dot(self, a: np.ndarray, b: np.ndarray) -> float:
        """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
        return float(np.dot(a, b))

    def find_match(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Verilen embedding için en yakın kaydı bulur ve skoruyla birlikte döndürür."""
        if not self.db:
            return "Unknown", 0.0

        q = _as_norm_f32(embedding)
        best_id, best_score = "Unknown", 0.0

        for sid, rec in self.db.items():
            c = rec.get("centroid")
            if c is None:
                embs: List[np.ndarray] = rec.get("embeddings", [])  # type: ignore
                if not embs:
                    continue
                c = np.mean(np.vstack(embs), axis=0)
                c = _as_norm_f32(c)
                rec["centroid"] = c
            s = self._dot(q, c)
            if s > best_score:
                best_score, best_id = s, sid

        # Detaylı kontrol (centroid yerine bireysel embeddinglerle karşılaştırma)
        if best_id != "Unknown" and best_score >= max(self.threshold - self.detailed_margin, 0.0):
            rec = self.db.get(best_id, {})
            embs: List[np.ndarray] = rec.get("embeddings", [])  # type: ignore
            if embs:
                E = np.vstack(embs).T
                detailed = float(np.max(q @ E))
                best_score = max(best_score, detailed)

        if best_score >= self.threshold:
            return best_id, float(best_score)
        return "Unknown", float(best_score)

    # -------------------- Kopya Kontrolü ve Toplu Eşleştirme -------------------- #
    def check_duplicate_for_id(self, student_id: str, embedding: np.ndarray, *, hard_floor: float = 0.8) -> Tuple[bool, Optional[str], float]:
        """Bir embedding'in başka bir öğrenciye ait olup olmadığını kontrol eder."""
        match_id, score = self.find_match(embedding)
        if match_id != "Unknown" and match_id != student_id and score >= max(self.threshold, hard_floor):
            return True, match_id, score
        return False, None, score

    def batch_match(self, embeddings: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Birden fazla embedding için toplu eşleştirme işlemi yapar."""
        return [self.find_match(emb) for emb in embeddings]

    # -------------------- Embedding Güncelleme -------------------- #
    def replace_person_embeddings(self, student_id: str, name: Optional[str], new_embeddings: List[np.ndarray]) -> bool:
        """Belirtilen kişinin embedding'lerini tamamen yeni embedding'lerle değiştirir."""
        if not new_embeddings:
            return False
        embs = [_as_norm_f32(e) for e in new_embeddings]
        rec: Record = self.db.setdefault(student_id, {"name": name or "", "embeddings": [], "centroid": None})
        if name is not None:
            rec["name"] = name
        if self.max_per_person is not None and len(embs) > self.max_per_person:
            embs = embs[-self.max_per_person:]
        rec["embeddings"] = embs
        arr = np.vstack(embs)
        n = float(np.linalg.norm(arr.mean(axis=0)))
        rec["centroid"] = (arr.mean(axis=0) / (n + 1e-12)).astype(np.float32) if n > 0 else embs[0]
        return True

    # -------------------- Kalıcılık (Dosya İşlemleri) -------------------- #
    def _hash_file(self, filepath: str) -> str:
        """Dosyanın SHA256 hash değerini hesaplar."""
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def save(self, path: Optional[str] = None):
        """Veritabanını Pickle formatında kaydeder ve SHA256 imzasını oluşturur."""
        target_path = get_resource_path(path or self.db_path)
        try:
            for sid, rec in self.db.items():
                embs: List[np.ndarray] = rec.get("embeddings", [])  # type: ignore
                rec["embeddings"] = [np.ascontiguousarray(np.asarray(e, dtype=np.float32)) for e in embs]
                if rec.get("centroid") is not None:
                    rec["centroid"] = np.ascontiguousarray(np.asarray(rec["centroid"], dtype=np.float32))

            with open(target_path, "wb") as f:
                pickle.dump(self.db, f)

            signature = self._hash_file(target_path)
            with open(self.sig_path, "w", encoding="utf-8") as sig:
                sig.write(signature)

            logger.info(f"[FaceDB] Kaydedildi: {target_path}")
        except Exception as e:
            logger.error(f"[FaceDB] Kayıt hatası: {e}")

    def load(self, path: Optional[str] = None):
        """Veritabanını yükler ve bütünlüğünü SHA256 imzası ile doğrular."""
        target_path = get_resource_path(path or self.db_path)
        try:
            # İmza kontrolü (dosya bütünlüğü)
            if os.path.exists(self.sig_path):
                current_hash = self._hash_file(target_path) if os.path.exists(target_path) else ""
                with open(self.sig_path, "r", encoding="utf-8") as sig:
                    saved_hash = sig.read().strip()
                if current_hash and current_hash != saved_hash:
                    raise ValueError("Yüz veritabanı dosyası değiştirilmiş olabilir!")

            if not os.path.exists(target_path):
                self.db = {}
                return

            # Pickle verisini yükle
            with open(target_path, "rb") as f:
                self.db = pickle.load(f)

            # Embedding ve centroid verilerini normalize et
            for sid, rec in self.db.items():
                embs: List[np.ndarray] = []
                for e in rec.get("embeddings", []):  # type: ignore
                    embs.append(_as_norm_f32(e))
                rec["embeddings"] = embs
                c = rec.get("centroid")
                rec["centroid"] = _as_norm_f32(c) if c is not None else (
                    _as_norm_f32(np.mean(np.vstack(embs), axis=0)) if embs else None
                )

            logger.info(f"[FaceDB] Yüklendi: {target_path}")
        except Exception as e:
            logger.error(f"[FaceDB] Yükleme hatası: {e}")
            raise
