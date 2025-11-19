import time
import cv2
import numpy as np
from typing import Dict, Any, List
from faceid.logging.logger import LoggerManager

logger = LoggerManager.get_logger(__name__)


class AttendanceScenario:
    """
    Yoklama senaryosu sınıfı.

    İşlevler:
      - run_single(student_id): Tek bir öğrenci için yoklama alır.
      - run_multi(): Aynı anda birden fazla öğrenciyi tanıyarak yoklama alır.

    Özellikler:
      - preview_time: (saniye) Kamera önizleme süresi.
      - Önizleme tamamlandığında tek bir kare alınır ve işlem yapılır.
      - ID ve yüz kaydı kontrolleri, kamera açılmadan önce gerçekleştirilir.
    """

    def __init__(
        self,
        camera,
        detector,
        attendance_manager,
        *,
        preview_time: float = 5.0,
        window_name: str = "Live"
    ):
        self.camera = camera
        self.detector = detector
        self.att = attendance_manager
        self.preview_time = float(preview_time)
        self.window_name = window_name

    # ------------------- Genel API ------------------- #
    def run_single(self, student_id: str) -> Dict[str, Any]:
        """Tek bir öğrenci için yoklama işlemini başlatır."""
        # 1) Kamera açılmadan önce gerekli kontroller
        info = self.att.excel.get_student_info(student_id)
        if info is None:
            logger.warning(f"[AttendanceScenario] Listede olmayan ID: {student_id}")
            return {"status": "invalid_id"}

        marked, _ = self.att.excel.is_marked(student_id)
        if marked:
            logger.info(f"[AttendanceScenario] Öğrencinin bugünkü yoklaması zaten alınmış: {student_id}")
            return {"status": "already_marked"}

        if not self.att.face_db.is_student_registered(student_id):
            logger.warning(f"[AttendanceScenario] Öğrencinin yüz kaydı bulunamadı: {student_id}")
            return {"status": "no_face_record"}

        # 2) Kamera oturumu (önizleme + tek kare)
        sess = self._run_camera_session()
        if sess["status"] != "faces":
            return sess

        faces = sess["faces"]

        # Birden fazla yüz algılanması durumunda işlem iptal edilir
        if len(faces) > 1:
            logger.warning(f"[AttendanceScenario] Birden fazla yüz bulundu: {len(faces)}")
            return {"status": "multiple_faces"}

        # Tek bir yüz embedding çıkarımı
        try:
            emb = self.detector.get_embedding([faces[0]])
        except Exception:
            return {"status": "embedding_error"}

        if isinstance(emb, (list, tuple)):
            emb = emb[0]
        emb = np.asarray(emb, dtype=np.float32)

        ok, code, info = self.att.single_attendance(student_id, emb)
        return {"status": "ok", "ok": ok, "code": code, "info": info}

    def run_multi(self) -> Dict[str, Any]:
        """Çoklu yüz algılama ile yoklama işlemini başlatır."""
        # 1) Kamera oturumu (önizleme + tek kare)
        sess = self._run_camera_session()
        if sess["status"] != "faces":
            return sess

        faces = sess["faces"]
        if not faces:
            return {"status": "no_face"}

        # Birden fazla yüz embedding çıkarımı
        try:
            embs = self.detector.get_embeddings(faces)
        except Exception:
            return {"status": "embedding_error_multi"}

        embeddings_list = [np.asarray(e, dtype=np.float32) for e in embs]
        result = self.att.multi_attendance(embeddings_list)
        return {"status": "ok", "result": result}

    # ------------------- Dahili Yardımcı Fonksiyonlar ------------------- #
    def _run_camera_session(self) -> Dict[str, Any]:
        """Kamera önizlemesini başlatır, süre dolduğunda tek kare yakalayıp yüz tespiti yapar."""
        try:
            self.camera.open()
            logger.info("[AttendanceScenario] Kamera oturumu başladı.")
        except Exception as e:
            logger.error(f"[AttendanceScenario] Kamera açılamadı: {e}")
            return {"status": "camera_error"}

        start = time.time()
        try:
            # Önizleme süresi boyunca kamera görüntüsünü göster
            while True:
                ok, frame = self.camera.read()
                if not ok or frame is None:
                    cv2.waitKey(1)
                    if time.time() - start >= self.preview_time:
                        break
                    continue

                elapsed = time.time() - start
                remaining = max(0, self.preview_time - elapsed)
                try:
                    cv2.setWindowTitle(self.window_name, f"{self.window_name} | Önizleme kalan: {remaining:.0f}s")
                except Exception:
                    pass

                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC → iptal
                    return {"status": "timeout"}

                if elapsed >= self.preview_time:
                    break

            # Önizleme tamamlandığında tek kare al ve yüzleri algıla
            ok, final_frame = self.camera.read()
            if not ok or final_frame is None:
                return {"status": "timeout"}

            faces = self.detector.detect_faces(final_frame, align=False)
            vis = self.detector.draw_annotations(final_frame.copy(), faces)
            cv2.imshow(self.window_name, vis)
            cv2.waitKey(500)

            if not faces:
                return {"status": "no_face"}

            return {"status": "faces", "faces": list(faces)}

        finally:
            try:
                self.camera.close()
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            logger.info("[AttendanceScenario] Kamera oturumu sona erdi.")



#------------------------------------------------------------------------------------------------------------#



class RegistrationScenario:
    """
    Öğrenci yüz kayıt senaryosu.

    İşlev:
      - run_registration(student_id, name, surname)

    İşlem adımları:
      1) Kamera açılmadan önce:
         - Excel üzerinde öğrenci kontrolü yapılır (yeni kayıt / güncelleme / iptal)
      2) Kamera açıldıktan sonra:
         - Çoklu yüz tespiti kontrolü
         - Duplicate (tekrar eden yüz) kontrolü
         - 3 farklı poz (ön, sağ, sol) için kare alınır ve embedding çıkarılır
      3) RegistrationManager aracılığıyla kayıt veya güncelleme tamamlanır
    """

    def __init__(
        self,
        camera,
        detector,
        reg_manager,
        face_db,
        excel_manager,
        *,
        preview_time: float = 5.0,
        window_name: str = "Registration"
    ):
        self.camera = camera
        self.detector = detector
        self.reg = reg_manager
        self.face_db = face_db
        self.excel = excel_manager
        self.preview_time = float(preview_time)
        self.window_name = window_name

    def run_registration(self, student_id: str, name: str, surname: str) -> Dict[str, Any]:
        """Yeni öğrenci kaydı veya mevcut kaydın güncellenmesi işlemini gerçekleştirir."""
        # ---------------- Kamera açılmadan önce kontroller ----------------
        info = self.excel.get_student_info(student_id)
        update_mode = False

        if info is None:
            logger.info("[REG] Öğrenci listede yok, yeni kayıt yapılacak.")
            self.excel.add_student(student_id, name, surname, overwrite=False)
        else:
            if self.face_db.is_student_registered(student_id):
                print(f"{student_id} mevcut kayıtlı. Güncellemek ister misiniz? (y/n)")
                ans = input().strip().lower()
                if ans == "y":
                    update_mode = True
                else:
                    return {"status": "cancelled"}
            else:
                logger.info("[REG] Öğrenci listede var ancak yüz kaydı bulunmuyor, yeni kayıt yapılacak.")

        # ---------------- Kamera oturumunun başlatılması ----------------
        try:
            self.camera.open()
            logger.info("[REG] Kamera oturumu başladı.")
        except Exception as e:
            logger.error(f"[REG] Kamera açılamadı: {e}")
            return {"status": "camera_error"}

        try:
            # Kamera ısınma süresi
            frame = None
            for _ in range(5):
                ok, test_frame = self.camera.read()
                if ok and test_frame is not None:
                    frame = test_frame
                    break
                time.sleep(0.1)
            if frame is None:
                return {"status": "camera_no_frame"}

            # İlk karede çoklu yüz kontrolü
            faces = self.detector.detect_faces(frame, align=False)
            if len(faces) > 1:
                return {"status": "multiple_faces"}
            if not faces:
                return {"status": "no_face"}

            # Duplicate (tekrar eden) yüz kontrolü
            try:
                emb_first = self.detector.get_embedding([faces[0]])
            except Exception:
                return {"status": "embedding_error"}

            is_dup, dup_id, score = self.face_db.check_duplicate_for_id(student_id, emb_first)
            if is_dup:
                return {"status": "duplicate_face", "dup_id": dup_id, "score": score}

            # ---------------- Poz çekimleri ----------------
            embeddings: List[np.ndarray] = []
            pose_instructions = [
                ("Kameraya bakın", "front"),
                ("Sağa dönün", "right"),
                ("Sola dönün", "left"),
            ]

            for instruction, pose_name in pose_instructions:
                print(f"{instruction} - {self.preview_time} saniye içinde poz verin...")
                start_time = time.time()

                while time.time() - start_time < self.preview_time:
                    ok, frame = self.camera.read()
                    if not ok or frame is None:
                        continue
                    cv2.putText(frame, f"{instruction} ({int(self.preview_time - (time.time()-start_time))}s)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    cv2.imshow(self.window_name, frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC → iptal
                        return {"status": "cancelled"}

                # Süre sonunda tek kare alınır
                ok, final_frame = self.camera.read()
                if not ok or final_frame is None:
                    return {"status": "camera_no_frame"}

                faces_pose = self.detector.detect_faces(final_frame, align=False)
                if len(faces_pose) != 1:
                    return {"status": "multiple_faces_during_pose", "pose": pose_name}

                try:
                    emb_pose = self.detector.get_embedding([faces_pose[0]])
                except Exception:
                    return {"status": "embedding_error_pose", "pose": pose_name}

                embeddings.append(np.asarray(emb_pose, dtype=np.float32))

            # ---------------- Kayıt işlemi ----------------
            result = self.reg.register_student(
                student_id,
                name,
                surname,
                embeddings,
                overwrite_excel_name=True,
                check_duplicates=False,  # Duplicate kontrolü daha önce yapıldı
                replace_all_embeddings=update_mode
            )
            return {"status": result["status"], "details": result}

        finally:
            try:
                self.camera.close()
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass
            logger.info("[REG] Kamera oturumu sona erdi.")
