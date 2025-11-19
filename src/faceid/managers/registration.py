from typing import List, Dict, Optional
import numpy as np
from faceid.logging.logger import LoggerManager
from faceid.managers.excel import ExcelManager
from faceid.core.database import FaceDatabase

logger = LoggerManager.get_logger(__name__)


class RegistrationManager:
    """
    Kayıt (öğrenci-yüz) işlemlerinin iş mantığını yöneten sınıf.

    Sorumluluklar:
      - ExcelManager: Öğrenci ekleme ve güncelleme işlemleri.
      - FaceDatabase: Yüz embedding'lerinin eklenmesi, değiştirilmesi ve çakışma (duplicate) kontrolü.
    """

    def __init__(self, excel: ExcelManager, face_db: FaceDatabase):
        self.excel = excel
        self.face_db = face_db

    def register_student(
        self,
        student_id: str,
        name: str,
        surname: str,
        embeddings: List[np.ndarray],
        *,
        overwrite_excel_name: bool = True,
        check_duplicates: bool = True,
        hard_floor: float = 0.80,
        replace_all_embeddings: bool = True,
        save_path: Optional[str] = None
    ) -> Dict[str, object]:
        """
        Yeni bir öğrenciyi sisteme kaydeder veya mevcut kaydı günceller.

        Adımlar:
          1) Excel'de öğrenci kaydı oluşturulur veya güncellenir.
          2) FaceDatabase üzerinde çakışma (duplicate) kontrolü yapılır.
          3) Embedding verileri eklenir veya değiştirilir.
          4) Değişiklikler dosyaya kaydedilir.

        Dönüş:
          - status: İşlem sonucu ("ok", "duplicate_face", "face_db_error" vb.)
          - excel_updated: Excel dosyasında değişiklik yapılıp yapılmadığı.
          - face_db_updated: FaceDatabase üzerinde işlem yapılıp yapılmadığı.
          - duplicate: Çakışma varsa bilgisi.
          - counts: Kayıt öncesi ve sonrası embedding sayıları.
        """
        if not student_id or not name or not surname or not embeddings:
            return {
                "status": "invalid_data",
                "excel_updated": False,
                "face_db_updated": False,
                "duplicate": None,
                "counts": None
            }

        full_name = f"{name} {surname}".strip()

        # ---- 1) Excel kayıt veya güncelleme ----
        excel_updated = False
        try:
            info = self.excel.get_student_info(student_id)
            if info is None:
                self.excel.add_student(student_id, name, surname, overwrite=False)
                excel_updated = True
            else:
                if overwrite_excel_name and (info.get("ad") != name or info.get("soyad") != surname):
                    self.excel.add_student(student_id, name, surname, overwrite=True)
                    excel_updated = True
        except Exception as e:
            logger.error(f"[RegistrationManager] Excel yazılamadı: {e}")
            return {
                "status": "excel_write_error",
                "excel_updated": False,
                "face_db_updated": False,
                "duplicate": None,
                "counts": None
            }

        # ---- 2) Duplicate (çakışan yüz) kontrolü ----
        if check_duplicates and self.face_db.list_ids():
            for emb in embeddings:
                is_dup, dup_id, score = self.face_db.check_duplicate_for_id(
                    student_id, emb, hard_floor=hard_floor
                )
                if is_dup and dup_id is not None:
                    return {
                        "status": "duplicate_face",
                        "excel_updated": excel_updated,
                        "face_db_updated": False,
                        "duplicate": {"id": dup_id, "score": float(score)},
                        "counts": None
                    }

        # ---- 3) FaceDatabase güncelleme ----
        face_db_updated = False
        counts: Optional[Dict[str, int]] = None

        try:
            before = len(self.face_db.db.get(student_id, {}).get("embeddings", [])) if self.face_db.db else 0

            if not self.face_db.is_student_registered(student_id):
                # Öğrenci veritabanında yoksa yeni kayıt oluştur
                for emb in embeddings:
                    self.face_db.add_person(student_id, full_name, emb)
                face_db_updated = True
            else:
                # Mevcut kayıt varsa tüm embedding’leri değiştir veya yeni ekle
                if replace_all_embeddings:
                    ok = self.face_db.replace_person_embeddings(student_id, full_name, embeddings)
                    face_db_updated = face_db_updated or ok
                else:
                    for emb in embeddings:
                        self.face_db.add_person(student_id, full_name, emb)
                    face_db_updated = True

            after = len(self.face_db.db.get(student_id, {}).get("embeddings", []))
            counts = {"before": int(before), "after": int(after)}

            # Veritabanını kaydet
            if save_path is not None:
                self.face_db.save(save_path)
            else:
                self.face_db.save()

        except Exception as e:
            logger.error(f"[RegistrationManager] FaceDB yazılamadı: {e}")
            return {
                "status": "face_db_error",
                "excel_updated": excel_updated,
                "face_db_updated": False,
                "duplicate": None,
                "counts": counts
            }

        return {
            "status": "ok",
            "excel_updated": excel_updated,
            "face_db_updated": face_db_updated,
            "duplicate": None,
            "counts": counts
        }

    def register_single(
        self,
        student_id: str,
        name: str,
        surname: str,
        embedding: np.ndarray,
        **kwargs
    ) -> Dict[str, object]:
        """
        Tek bir embedding ile öğrenci kaydı yapar.
        register_student() metoduna basitleştirilmiş bir arayüz sağlar.
        """
        return self.register_student(student_id, name, surname, [embedding], **kwargs)
