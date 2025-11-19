import os
import shutil
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, List, Tuple

from faceid.logging.logger import LoggerManager
from faceid.core.utils import get_resource_path, ConfigLoader  # ✅ ConfigLoader eklendi

logger = LoggerManager.get_logger(__name__)

ID_COL = "öğrenci_no"
NAME_COL = "ad"
SURNAME_COL = "soyad"


class ExcelManager:
    """
    Yoklama verilerini Excel dosyasında yöneten sınıf.

    Özellikler:
      - Öğrenci ekleme, güncelleme, yoklama işaretleme.
      - Günlük sütun oluşturma ve raporlama.
      - Eksik sütunları otomatik tamamlama ve yedekleme.
      - Config dosyasındaki 'paths.excel_file' yolunu kullanır.
    """

    def __init__(self, path: Optional[str] = None):
        """
        Excel dosyasını yönetir.
        Yol belirtilmezse config/default.yaml içindeki
        'paths.excel_file' değeri kullanılır.
        Eğer orada da tanımlı değilse varsayılan olarak
        'data/app/ogrenci_listesi.xlsx' seçilir.
        """
        # Config’ten Excel dosyası yolunu al
        config = ConfigLoader().load()
        cfg_path = config.get("paths", {}).get("excel_file", "data/app/ogrenci_listesi.xlsx")

        # Parametre verilmişse onu, aksi halde config yolunu kullan
        self.path = get_resource_path(path or cfg_path)
        self.required_cols = [ID_COL, NAME_COL, SURNAME_COL]
        self._ensure_file()

    # ------------------ Genel İşlevler ------------------ #
    def ensure_today_column(self, date_str: Optional[str] = None) -> str:
        """Bugünün tarihine ait sütun yoksa oluşturur."""
        dcol = date_str or self._today_str()
        df = self._read()
        if dcol not in df.columns:
            df[dcol] = ""
            self._write(df, backup=True)
            logger.info(f"[ExcelManager] Yeni tarih sütunu eklendi: {dcol}")
        return dcol

    def add_student(self, student_id: str, name: str, surname: str, overwrite: bool = False) -> bool:
        """Yeni öğrenci ekler veya isteğe bağlı olarak mevcut kaydı günceller."""
        df = self._read()
        exists = df[ID_COL].astype(str) == str(student_id)
        if exists.any():
            if not overwrite:
                logger.warning(f"[ExcelManager] ID zaten var: {student_id}")
                return False
            idx = df.index[exists][0]
            df.loc[idx, [NAME_COL, SURNAME_COL]] = [name, surname]
            self._write(df, backup=True)
            logger.info(f"[ExcelManager] Öğrenci bilgileri güncellendi: {student_id}")
            return True

        new_row = {ID_COL: str(student_id), NAME_COL: name, SURNAME_COL: surname}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        self._write(df, backup=True)
        logger.info(f"[ExcelManager] Öğrenci eklendi: {student_id} - {name} {surname}")
        return True

    def find_student_row(self, student_id: str) -> Optional[int]:
        """Verilen öğrenci ID’sine ait satır indeksini döndürür."""
        df = self._read()
        hits = df[ID_COL].astype(str) == str(student_id)
        return int(df.index[hits][0]) if hits.any() else None

    def mark_present(self, student_id: str, date_str: Optional[str] = None, *, allow_override: bool = False) -> bool:
        """Belirtilen öğrenciye bugünkü yoklama sütununda 'var' işaretini koyar."""
        return self._mark(student_id, date_str or self._today_str(), value="var", allow_override=allow_override)

    def mark_absent(self, student_id: str, date_str: Optional[str] = None, *, allow_override: bool = False) -> bool:
        """Belirtilen öğrenciye bugünkü yoklama sütununda 'yok' işaretini koyar."""
        return self._mark(student_id, date_str or self._today_str(), value="yok", allow_override=allow_override)

    def is_marked(self, student_id: str, date_str: Optional[str] = None) -> Tuple[bool, Optional[str]]:
        """Bir öğrencinin belirtilen tarihte işaretlenip işaretlenmediğini döndürür."""
        dcol = date_str or self._today_str()
        df = self._read()
        if dcol not in df.columns:
            return False, None
        row = self.find_student_row(student_id)
        if row is None:
            return False, None

        cell_value = df.at[row, dcol]
        if pd.isna(cell_value) or str(cell_value).strip() == "":
            return False, None

        v = str(cell_value).strip()
        return True, v

    def get_report(self, date_str: Optional[str] = None) -> Dict[str, object]:
        """Belirtilen tarih için yoklama raporu oluşturur."""
        dcol = date_str or self._today_str()
        df = self._read()
        if dcol not in df.columns:
            df[dcol] = ""

        total = len(df)
        vals = df[dcol].astype(str).str.strip().replace("nan", "").fillna("")
        present_mask = vals.eq("var")
        absent_mask = vals.eq("yok")
        empty_mask = vals.eq("")

        report = {
            "date": dcol,
            "total": total,
            "present": int(present_mask.sum()),
            "absent": int(absent_mask.sum()),
            "unmarked": int(empty_mask.sum()),
            "present_ids": df.loc[present_mask, ID_COL].astype(str).tolist(),
            "absent_ids": df.loc[absent_mask, ID_COL].astype(str).tolist(),
            "unmarked_ids": df.loc[empty_mask, ID_COL].astype(str).tolist(),
        }
        return report

    def get_student_info(self, student_id: str) -> Optional[Dict[str, str]]:
        """Belirtilen öğrenci ID’sine ait bilgileri döndürür."""
        df = self._read()
        hit = df[ID_COL].astype(str) == str(student_id)
        if not hit.any():
            return None
        r = df.loc[hit].iloc[0]
        return {ID_COL: str(r[ID_COL]), NAME_COL: str(r[NAME_COL]), SURNAME_COL: str(r[SURNAME_COL])}

    # ------------------ Yardımcı Fonksiyonlar ------------------ #
    def _mark(self, student_id: str, dcol: str, *, value: str, allow_override: bool) -> bool:
        """İçsel yoklama işaretleme işlemini gerçekleştirir."""
        df = self._read()
        if dcol not in df.columns:
            df[dcol] = ""

        row = self.find_student_row(student_id)
        if row is None:
            logger.warning(f"[ExcelManager] ID bulunamadı: {student_id}")
            return False

        current = df.at[row, dcol]
        if (not pd.isna(current) and str(current).strip() != "") and not allow_override:
            logger.debug(f"[ExcelManager] Hücre zaten dolu (override yok): ID={student_id}, {dcol}='{current}'")
            return False

        df.at[row, dcol] = value
        self._write(df, backup=True)
        logger.info(f"[ExcelManager] İşaretleme yapıldı: ID={student_id}, {dcol}='{value}'")
        return True

    def _ensure_file(self):
        """Excel dosyasını ve zorunlu sütunları kontrol eder; eksikse oluşturur."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        if not os.path.exists(self.path):
            logger.warning(f"[ExcelManager] Dosya yok, oluşturuluyor: {self.path}")
            df = pd.DataFrame(columns=self.required_cols)
            self._write(df, backup=False)
        else:
            df = self._read()
            missing = [c for c in self.required_cols if c not in df.columns]
            if missing:
                logger.warning(f"[ExcelManager] Eksik sütunlar eklenecek: {missing}")
                for c in missing:
                    df[c] = "" if c not in [ID_COL] else df.get(c, "")
                other_cols = [c for c in df.columns if c not in self.required_cols]
                df = df[self.required_cols + other_cols]
                self._write(df, backup=True)

    def _read(self) -> pd.DataFrame:
        """Excel dosyasını okur ve bir DataFrame döndürür."""
        try:
            df = pd.read_excel(self.path, engine="openpyxl")
        except FileNotFoundError:
            logger.error(f"[ExcelManager] Dosya bulunamadı: {self.path}")
            raise
        except Exception as e:
            logger.error(f"[ExcelManager] Excel okunamadı: {e}")
            raise

        if ID_COL in df.columns:
            df[ID_COL] = df[ID_COL].astype(str).str.strip()
        for c in [NAME_COL, SURNAME_COL]:
            if c in df.columns:
                df[c] = df[c].astype(str).str.strip()
        return df

    def _write(self, df: pd.DataFrame, *, backup: bool):
        """DataFrame’i Excel dosyasına yazar, gerekirse yedek alır."""
        front = [c for c in self.required_cols if c in df.columns]
        back = [c for c in df.columns if c not in front]
        df = df[front + back]

        if backup and os.path.exists(self.path):
            self._backup_file()

        try:
            df.to_excel(self.path, index=False)
        except Exception as e:
            logger.error(f"[ExcelManager] Excel yazılamadı: {e}")
            raise

    def _backup_file(self):
        """Mevcut Excel dosyasının zaman damgalı bir yedeğini oluşturur."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(self.path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.path))[0]
        backup_path = os.path.join(backup_dir, f"{base}_{ts}.xlsx")
        try:
            shutil.copy2(self.path, backup_path)
            logger.info(f"[ExcelManager] Yedek alındı: {backup_path}")
        except Exception as e:
            logger.warning(f"[ExcelManager] Yedek alınamadı: {e}")

    @staticmethod
    def _today_str() -> str:
        """Bugünün tarihini YYYY-MM-DD formatında döndürür."""
        return date.today().isoformat()
