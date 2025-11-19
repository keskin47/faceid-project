from pathlib import Path
import sys
import os
import yaml

# -------------------- Yol (Path) Yardımcı Fonksiyonları -------------------- #
def base_path() -> Path:
    """
    Uygulamanın temel çalışma dizinini döndürür.
    - Eğer uygulama PyInstaller ile paketlenmişse (_MEIPASS kullanılır).
    - Aksi halde, proje kök dizinine göre yol hesaplanır.
    """
    try:
        # PyInstaller ile derlenmiş ortam (frozen) için
        return Path(getattr(sys, "_MEIPASS"))
    except AttributeError:
        # Normal geliştirme ortamı
        return Path(__file__).resolve().parent.parent.parent  # Proje taban dizini

def get_resource_path(relative_path: str) -> str:
    """Verilen göreceli yolu proje kök dizinine göre mutlak hale getirir."""
    return str(base_path() / relative_path)

def ensure_dir(relative_dir: str) -> str:
    """Belirtilen göreceli dizini oluşturur (varsa hata vermez) ve yolunu döndürür."""
    p = base_path() / relative_dir
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

def resource_exists(relative_path: str) -> bool:
    """Belirtilen kaynak yolunun mevcut olup olmadığını kontrol eder."""
    return (base_path() / relative_path).exists()


# -------------------- ConfigLoader -------------------- #
class ConfigLoader:
    """
    YAML tabanlı basit bir yapılandırma (konfigürasyon) yükleyici sınıfı.

    Özellikler:
      - Dosya mevcut değilse boş bir sözlük döndürür.
      - Hata durumunda uygulamayı durdurmaz, sadece uyarı verir.
    """

    def __init__(self, path: str = "configs/default.yaml"):
        """Yapılandırma dosyasının yolunu ayarlar."""
        self.path = get_resource_path(path)
        self.config = {}

    def load(self) -> dict:
        """YAML yapılandırma dosyasını yükler ve bir sözlük döndürür."""
        if not os.path.exists(self.path):
            print(f"[ConfigLoader] ⚠️ Konfigürasyon dosyası bulunamadı: {self.path}")
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
            print(f"[ConfigLoader] ✅ Konfigürasyon yüklendi: {self.path}")
        except Exception as e:
            print(f"[ConfigLoader] ❌ YAML okuma hatası: {e}")
            self.config = {}
        return self.config
