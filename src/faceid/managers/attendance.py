from typing import List, Tuple, Dict
from faceid.logging.logger import LoggerManager
from faceid.managers.excel import ExcelManager
from faceid.core.database import FaceDatabase

logger = LoggerManager.get_logger(__name__)


class AttendanceManager:
    """
    Yoklama işlemlerinin iş mantığını yöneten sınıf.

    Bileşenler:
      - ExcelManager: Sütun hazırlama, yoklama işaretleme, raporlama.
      - FaceDatabase: Yüz embedding eşleştirmesi (tanıma).
    
    Not:
      - Detector/Embedder bu sınıfa doğrudan bağımlı değildir.
        Embedding’ler senaryolar tarafından üretilip bu sınıfa iletilir.
    """

    def __init__(self, excel: ExcelManager, face_db: FaceDatabase):
        self.excel = excel
        self.face_db = face_db

    # ---------- Tekli Yoklama (ID + yüz doğrulama) ---------- #
    def single_attendance(self, student_id: str, embedding_vec, *, write: bool = True) -> Tuple[bool, str, Dict]:
        """
        Tek bir öğrenci için yoklama işlemini gerçekleştirir.
        Öğrenci ID’si doğrulanır, yüz tanıma yapılır ve Excel’e işlenir.
        """
        info = self.excel.get_student_info(student_id)
        if info is None:
            logger.warning(f"[Attendance] Listede olmayan ID: {student_id}")
            return False, "not_in_list", {}

        marked, val = self.excel.is_marked(student_id)
        if marked:
            logger.info(f"[Attendance] ID zaten işaretli: {student_id} ({val})")
            return True, "already_marked", {"value": val}

        if not self.face_db.is_student_registered(student_id):
            logger.warning(f"[Attendance] Yüz kaydı bulunamadı: {student_id}")
            return False, "no_face_record", {}

        matched_id, score = self.face_db.find_match(embedding_vec)
        details = {"matched_id": matched_id, "score": float(score), "threshold": float(self.face_db.threshold)}

        # Eşleşme analizi
        if matched_id == "Unknown":
            logger.debug(f"[Attendance] Düşük skor, eşleşme yok. ID: {student_id}, Score: {score:.3f}")
            return False, "low_score", details
        if matched_id != student_id:
            logger.warning(f"[Attendance] ID uyuşmazlığı. Beklenen: {student_id}, Bulunan: {matched_id}, Score: {score:.3f}")
            return False, "mismatch", details

        # Yazma işlemi kapalıysa sadece doğrulama yapılır
        if not write:
            logger.info(f"[Attendance] Doğrulama başarılı (işaretlenmedi): {student_id}")
            return True, "ok", details

        # Yoklamayı Excel’e işaretle
        ok = self.excel.mark_present(student_id)
        if ok:
            logger.info(f"[Attendance] Yoklama işaretlendi: {student_id}")
            return True, "ok", details
        else:
            logger.error(f"[Attendance] Yoklama işaretlenemedi: {student_id}")
            return False, "mark_failed", details

    # ---------- Çoklu Yoklama (kadrajdaki tüm yüzler) ---------- #
    def multi_attendance(self, embeddings: List, *, write: bool = True) -> Dict:
        """
        Aynı karede birden fazla yüz bulunduğunda toplu yoklama işlemini gerçekleştirir.
        Her yüz embedding’i için eşleşme yapılır, Excel güncellenir ve rapor döndürülür.
        """
        results = []
        if not embeddings:
            logger.debug("[Attendance] Çoklu yoklama: yüz bulunamadı.")
            return {
                "results": results,
                "report": self.excel.get_report(),
                "total_faces": 0,
                "recognized_count": 0,
                "already_marked_count": 0,
                "marked_present_count": 0,
                "unknown_count": 0
            }

        matches = self.face_db.batch_match(embeddings)  # [(id_or_Unknown, score), ...]

        recognized_ids = set()
        already_marked_ids = set()
        marked_present_ids = set()
        unknown_count = 0

        # Her yüz embedding’i için değerlendirme yapılır
        for i, (mid, sc) in enumerate(matches):
            item = {"idx": i, "id": mid, "score": float(sc)}

            if mid != "Unknown" and sc >= self.face_db.threshold:
                recognized_ids.add(mid)
                already, _ = self.excel.is_marked(mid)
                if already:
                    already_marked_ids.add(mid)
                    logger.debug(f"[Attendance] Yoklaması önceden alınmış: {mid}")
                    item["action"] = "recognized_already_marked"
                else:
                    if write:
                        if self.excel.mark_present(mid):
                            marked_present_ids.add(mid)
                            logger.info(f"[Attendance] Yoklamaya eklendi (listede var): {mid}")
                            item["action"] = "recognized_and_marked"
                        else:
                            logger.error(f"[Attendance] Yoklama işaretlenemedi: {mid}")
                            item["action"] = "mark_failed"
                    else:
                        logger.info(f"[Attendance] Tanındı (işaretlenmedi): {mid}")
                        item["action"] = "recognized"
            else:
                unknown_count += 1
                logger.debug(f"[Attendance] Kayıtlı olmayan kişi (tanınmadı): idx={i}, score={sc:.3f}")
                item["action"] = "unknown"

            results.append(item)

        return {
            "results": results,
            "report": self.excel.get_report(),
            "total_faces": len(embeddings),
            "recognized_count": len(recognized_ids),         # Benzersiz tanınan kişi sayısı
            "already_marked_count": len(already_marked_ids), # Benzersiz önceden işaretli kişi sayısı
            "marked_present_count": len(marked_present_ids), # Benzersiz yeni işaretlenen kişi sayısı
            "unknown_count": unknown_count
        }
