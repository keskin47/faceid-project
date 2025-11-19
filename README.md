# FaceID – Otomatik Yoklama Sistemi

FaceID, yüz tanıma tabanlı bir otomatik yoklama sistemidir.  
Sistem, InsightFace kütüphanesinin RetinaFace (yüz algılama) ve ArcFace (yüz tanıma) modellerini kullanarak öğrencilerin kimliğini doğrular ve yoklamayı Excel tablosuna otomatik olarak işler.

---

## 1. Genel Bakış

Sistem iki ana moddan oluşur:

| Mod | Açıklama |
|-----|-----------|
| **Kayıt (Registration)** | Öğrencinin yüz görüntüleri alınır, embedding'leri çıkarılır ve veritabanına kaydedilir. |
| **Yoklama (Attendance)** | Kameradan alınan görüntülerdeki yüzler tanınır ve öğrenciler Excel dosyasında "var" olarak işaretlenir. |

Yapı modülerdir. Her görev (kamera, veritabanı, model, Excel yönetimi, kayıt, yoklama) ayrı sınıflar tarafından yürütülür. Bu sayede sistem genişletilebilir ve kolay bakım yapılabilir.

---

## 2. Klasör Yapısı

```
faceid/
│
├── app/
│   ├── cli.py                 # Komut satırı arabirimi
│   ├── scenarios.py           # Kayıt ve yoklama senaryoları
│
├── core/
│   ├── analysis.py            # FaceAnalysisWrapper – tespit ve embedding çıkarma
│   ├── database.py            # FaceDatabase – yüz embedding veritabanı
│   ├── utils.py               # Config ve path yardımcıları
│
├── managers/
│   ├── camera.py              # CameraManager – kamera kontrolü ve akış yönetimi
│   ├── excel.py               # ExcelManager – öğrenci listesi ve yoklama dosyası
│   ├── model.py               # ModelManager – RetinaFace ve ArcFace yükleyici
│   ├── attendance.py          # AttendanceManager – yoklama iş mantığı
│   ├── registration.py        # RegistrationManager – kayıt iş mantığı
│
├── logging/
│   └── logger.py              # LoggerManager – loglama altyapısı
│
├── configs/
│   └── default.yaml           # Varsayılan sistem ayarları
│
├── data/
│   └── app/
│       ├── ogrenci_listesi.xlsx  # Öğrenci listesi dosyası
│       ├── face_db.pkl           # Yüz embedding veritabanı
│       └── backups/              # Excel yedekleri
│
└── run_cli.bat                  # Windows için başlatma betiği
```

---

## 3. Çalışma Prensibi

### 3.1 Kayıt Süreci
1. Kullanıcıdan öğrenci numarası, ad ve soyad bilgisi alınır.  
2. Kamera üzerinden üç farklı poz (ön, sağ, sol) yakalanır.  
3. Her poz için embedding çıkarılır.  
4. Excel dosyasına öğrenci bilgisi eklenir, veritabanına embedding’ler kaydedilir.  
5. Mevcut öğrenci varsa bilgileri güncellenir.

### 3.2 Yoklama Süreci
1. Kamera akışı başlatılır.  
2. Her karedeki yüzler RetinaFace ile tespit edilir.  
3. ArcFace ile embedding çıkarılır.  
4. Veritabanındaki kayıtlarla benzerlik kontrolü yapılır.  
5. Eşik değeri aşılırsa öğrenci Excel’de “var” olarak işaretlenir.  
6. Daha önce işaretlenen öğrenciler tekrar yazılmaz.

---

## 4. Bileşenler

| Sınıf | Görevi |
|-------|---------|
| `CameraManager` | Kamerayı açar, kareleri okur, bağlantı koparsa yeniden bağlanır. |
| `ExcelManager` | Öğrenci listesini, yoklama sütunlarını ve raporları yönetir. |
| `FaceDatabase` | Embedding verilerini saklar, eşleştirme ve çakışma kontrolü yapar. |
| `ModelManager` | RetinaFace ve ArcFace modellerini yükler ve hazırlar. |
| `FaceAnalysisWrapper` | InsightFace modelini Python arayüzüne bağlar. |
| `RegistrationManager` | Öğrenci kayıt işlemlerini yürütür. |
| `AttendanceManager` | Tekli ve çoklu yoklama işlemlerini yürütür. |
| `LoggerManager` | Tüm bileşenlerde tutarlı loglama sağlar. |

---

## 5. Kurulum

### 5.1 Ortam Kurulumu
```bash
conda create -n faceenv38 python=3.8
conda activate faceenv38
pip install -r requirements.txt
```

### 5.2 Model Dosyaları
InsightFace modelleri otomatik olarak yüklenir.  
Gerekirse aşağıdaki ortam değişkeni ayarlanabilir:
```bash
set INSIGHTFACE_HOME=.\models
```

### 5.3 Uygulamanın Başlatılması
Windows için:
```bash
run_cli.bat
```
veya:
```bash
python -m faceid.app.cli
```

---

## 6. Loglama ve Yedekleme

- Tüm sistem olayları `logs/system.log` dosyasına kaydedilir.  
- Her Excel değişikliğinde `data/app/backups/` altında zaman damgalı yedek alınır.  
- Log ayarları (seviye, dosya boyutu, yedek sayısı) `configs/default.yaml` dosyasından yönetilir.

---

## 7. Teknik Özellikler

| Kategori | Teknoloji |
|-----------|------------|
| Derin Öğrenme | InsightFace (RetinaFace, ArcFace) |
| Programlama Dili | Python 3.8 |
| Görüntü İşleme | OpenCV |
| Veri Yönetimi | Pandas, Pickle |
| Konfigürasyon | YAML |
| Loglama | Python logging (RotatingFileHandler) |

---

## 8. CLI Menü Örneği

```
===== FaceID =====
1) Tekli Yoklama
2) Çoklu Yoklama
3) Kayıt
q) Çıkış
```

---

## 9. Özellikler

- Çoklu modda aynı karede birden fazla yüz tespiti yapılabilir.  
- ArcFace tabanlı cosine benzerlik ölçümü kullanılır.  
- Eşik değerleri dinamik olarak yapılandırılabilir.  
- Tanınmayan yüzler “Unknown” olarak raporlanır.  
- Modüler yapı sayesinde GUI veya gömülü sistem entegrasyonu mümkündür.

---

## 10. Lisans ve Yazar

Bu proje akademik araştırma ve eğitim amaçlı olarak geliştirilmiştir.  
Ticari kullanım durumlarında yazarla iletişime geçilmelidir.

**Yazar:** Bünyamin Keskin  
**Üniversite:** Türk-Alman Üniversitesi – Mekatronik Mühendisliği  
**Yıl:** 2025
