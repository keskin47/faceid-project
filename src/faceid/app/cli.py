import sys
import os

# OpenMP kütüphane çakışmalarını önlemek için gerekli ortam değişkeni
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Proje kök dizini ve model dosya yollarının otomatik olarak ayarlanması
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ["INSIGHTFACE_HOME"] = os.path.join(PROJECT_ROOT, "models")

from faceid.logging.logger import LoggerManager
from faceid.managers.model import ModelManager
from faceid.core.analysis import FaceAnalysisWrapper
from faceid.managers.camera import CameraManager

from faceid.managers.excel import ExcelManager
from faceid.core.database import FaceDatabase
from faceid.managers.registration import RegistrationManager
from faceid.managers.attendance import AttendanceManager

from faceid.app.scenarios import AttendanceScenario

logger = LoggerManager.get_logger(__name__)


def build_components():
    """Sistemde kullanılan tüm temel bileşenleri bir kez oluşturur ve geri döndürür."""
    excel = ExcelManager()
    face_db = FaceDatabase()
    try:
        face_db.load()
    except Exception:
        print("⚠️  Yüz veritabanı yüklenemedi. Boş bir veritabanı ile devam ediliyor.")

    reg_manager = RegistrationManager(excel, face_db)
    att_manager = AttendanceManager(excel, face_db)

    # Yüz algılama (RetinaFace) ve embedding (ArcFace) modüllerinin birlikte başlatılması
    app: FaceAnalysisWrapper = ModelManager.get_detector(det_size=(640, 640))

    # Kamera seçimi menüsü
    print("\n🎥 Kamera seçimi:")
    print("0) Dahili sistem kamerası")
    print("1) Harici USB kamera")
    print("2) IP / RTSP kamera (elle adres girilecek)")

    choice = input("Seçiminiz [0/1/2 veya Enter=varsayılan]: ").strip()

    if choice == "1":
        source = 1
    elif choice == "2":
        source = input("🌐 IP kamera adresi (örnek: rtsp://192.168.1.10:554/stream): ").strip()
    else:
        source = 0  # Varsayılan: dahili kamera

    # Kamera bağlantısının önceden test edilmesi
    if not CameraManager.test_open(source):
        print(f"❌ Seçilen kamera (source={source}) açılamadı.")
        print("⚙️ Varsayılan sistem kamerası (0) ile yeniden deneniyor...")
        source = 0
        if not CameraManager.test_open(source):
            raise RuntimeError("📷 Hiçbir kamera açılamadı. Lütfen bağlantıyı kontrol edin.")

    # Kamera nesnesinin oluşturulması
    camera = CameraManager(source=source, width=640, height=480, fps_limit=30.0)
    print(f"✅ Kamera başarıyla başlatılacak -> source={source}")

    return excel, face_db, reg_manager, att_manager, app, camera


def run_attendance_single(camera, app, reg_manager, att_manager):
    """Tek bir öğrenci için yoklama işlemini başlatır."""
    sid = input("🆔 Öğrenci ID: ").strip()
    if not sid:
        print("⛔ Geçersiz ID. Lütfen tekrar deneyin.")
        return

    reg_manager.excel.ensure_today_column()

    scenario = AttendanceScenario(
        camera=camera,
        detector=app, 
        attendance_manager=att_manager,
        preview_time=15.0,
        window_name="Live",
    )
    try:
        res = scenario.run_single(sid)
    except Exception as e:
        logger.error(f"[main] Tekli yoklama hatası: {e}")
        print("⛔ İşlem sırasında beklenmeyen bir hata oluştu.")
        return

    status = res.get("status")
    if status == "invalid_id":
        print("⛔ Öğrenci listede bulunamadı.")
    elif status == "already_marked":
        print("ℹ️ Bu öğrencinin bugünkü yoklaması zaten alınmış.")
    elif status == "no_face_record":
        print("❌ Bu öğrencinin yüz kaydı sistemde bulunamadı.")
    elif status == "multiple_faces":
        print("⚠️ Kadrajda birden fazla yüz tespit edildi. İşlem iptal edildi.")
    elif status == "no_face":
        print("👀 Kadrajda yüz algılanmadı. Lütfen tekrar deneyin.")
    elif status == "timeout":
        print("⏱ Önizleme süresi doldu. Yüz algılanamadı.")
    elif status == "camera_error":
        print("📷 Kamera açılamadı. Lütfen bağlantınızı kontrol edin.")
    elif status == "ok":
        ok, code = res.get("ok"), res.get("code")
        if ok:
            print(f"✅ Yoklama başarıyla alındı. (Kod: {code})")
        else:
            print(f"❌ Yoklama başarısız. (Kod: {code})")
    else:
        print(f"ℹ️ İşlem durumu: {status}")


def run_attendance_multi(camera, app, reg_manager, att_manager):
    """Birden fazla yüzü aynı anda tanıyarak yoklama işlemini yürütür."""
    reg_manager.excel.ensure_today_column()

    scenario = AttendanceScenario(
        camera=camera,
        detector=app,  
        attendance_manager=att_manager,
        preview_time=15.0,
        window_name="Live",
    )
    try:
        res = scenario.run_multi()
    except Exception as e:
        logger.error(f"[main] Çoklu yoklama hatası: {e}")
        print("⛔ İşlem sırasında beklenmeyen bir hata oluştu.")
        return

    status = res.get("status")
    if status == "no_face":
        print("👀 Kadrajda yüz algılanmadı. Lütfen tekrar deneyin.")
    elif status == "timeout":
        print("⏱ Önizleme süresi doldu. Yüz algılanamadı.")
    elif status == "camera_error":
        print("📷 Kamera açılamadı. Lütfen bağlantınızı kontrol edin.")
    elif status == "ok":
        result_data = res.get("result", {})
        report = result_data.get("report")

        if report:
            print(f"📅 Yoklama Tarihi: {report.get('date')}")
            print(f"   ✅ Var: {report.get('present')}")
            print(f"   ❌ Yok: {report.get('absent')}")
            print(f"   ⏳ İşaretlenmemiş: {report.get('unmarked')}")

        print("📊 Özet:")
        print(f"   👥 Toplam algılanan yüz: {result_data.get('total_faces', 0)}")
        print(f"   ✅ Listede var ve işaretlendi: {result_data.get('marked_present_count', 0)}")
        print(f"   ℹ️  Yoklaması önceden alınmış: {result_data.get('already_marked_count', 0)}")
        print(f"   ❌ Kayıtlı olmayan (tanınmadı): {result_data.get('unknown_count', 0)}")
    else:
        print(f"ℹ️ İşlem durumu: {status}")


def run_registration(camera, app, reg_manager, excel, face_db):
    """Yeni öğrenci yüz kaydı oluşturur ve veritabanına ekler."""
    from faceid.app.scenarios import RegistrationScenario

    sid = input("🆔 Öğrenci ID: ").strip()
    name = input("👤 Ad: ").strip()
    surname = input("👤 Soyad: ").strip()
    if not sid or not name or not surname:
        print("⛔ ID, Ad ve Soyad boş olamaz.")
        return

    scenario = RegistrationScenario(
        camera=camera,
        detector=app,  
        reg_manager=reg_manager,
        face_db=face_db,
        excel_manager=excel,
        preview_time=5.0,
        window_name="Registration"
    )

    try:
        res = scenario.run_registration(sid, name, surname)
    except Exception as e:
        logger.error(f"[main] Kayıt senaryosu hatası: {e}")
        print("⛔ İşlem sırasında beklenmeyen bir hata oluştu.")
        return

    status = res.get("status")
    if status == "ok":
        print("✅ Kayıt işlemi başarıyla tamamlandı.")
        print(res.get("details"))
    elif status == "duplicate_face":
        print(f"⚠️ Bu yüz başka bir kayıtla eşleşiyor: {res.get('dup_id')} (Skor: {res.get('score'):.3f})")
    elif status == "multiple_faces":
        print("⚠️ Kadrajda birden fazla yüz tespit edildi. İşlem iptal edildi.")
    elif status == "no_face":
        print("👀 Kadrajda yüz algılanmadı. Lütfen tekrar deneyin.")
    elif status == "cancelled":
        print("ℹ️ İşlem iptal edildi.")
    else:
        print(f"ℹ️ İşlem durumu: {status}")


def main():
    """Uygulamanın ana çalışma döngüsünü yönetir."""
    print("🔧 Bileşenler hazırlanıyor...")
    excel, face_db, reg_manager, att_manager, app, camera = build_components()
    print("✅ Hazır.")

    while True:
        print("\n===== FaceID =====")
        print("1) Tekli Yoklama")
        print("2) Çoklu Yoklama")
        print("3) Kayıt")
        print("q) Çıkış")
        choice = input("Seçiminiz: ").strip().lower()

        if choice == "1":
            run_attendance_single(camera, app, reg_manager, att_manager)
        elif choice == "2":
            run_attendance_multi(camera, app, reg_manager, att_manager)
        elif choice == "3":
            run_registration(camera, app, reg_manager, excel, face_db)
        elif choice == "q":
            print("👋 Görüşmek üzere.")
            break
        else:
            print("⛔ Geçersiz seçim.")

    try:
        face_db.save()
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 İptal edildi.")
        sys.exit(0)
