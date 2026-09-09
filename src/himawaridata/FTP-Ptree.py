import os
import time
from datetime import datetime, timedelta
from ftplib import FTP, error_perm
from typing import Optional

# ==========================================
# 設定情報
# ==========================================
FTP_HOST = os.environ.get("PTREE_FTP_HOST", "ftp.ptree.jaxa.jp")
FTP_USER = os.environ.get("PTREE_FTP_USER")
FTP_PASS = os.environ.get("PTREE_FTP_PASS")

# 保存先ディレクトリ
SAVE_DIR = "/Users/itousoumakoto/workspace/hirahara/SST-Anomaly-Inpainting-hirahara/src/himawaridata/himawari_sst_data"

# ダウンロード期間 (2025年4月1日〜2025年6月30日)
START_DATE = datetime(2025, 4, 1, 0, 0, 0)
END_DATE = datetime(2025, 6, 30, 23, 0, 0)

# バージョン情報 (環境に合わせて v21 などを指定)
VERSION_DIR = "v201_nc4_normal_std"
VER_STR = "v2.1"  # ファイル名内のバージョン表記
FTP_TIMEOUT_SECONDS = 600
RETRY_COUNT = 3
RETRY_DELAY_SECONDS = 10


def _cleanup_partial_file(local_file_path: str) -> None:
    temp_file_path = f"{local_file_path}.part"
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


def download_file(remote_dir: str, remote_filename: str, local_file_path: str) -> None:
    temp_file_path = f"{local_file_path}.part"
    last_error: Optional[Exception] = None

    for attempt in range(1, RETRY_COUNT + 1):
        try:
            _cleanup_partial_file(local_file_path)
            with FTP(FTP_HOST, timeout=FTP_TIMEOUT_SECONDS) as ftp:
                ftp.login(user=FTP_USER, passwd=FTP_PASS)
                ftp.cwd(remote_dir)
                with open(temp_file_path, "wb") as file_handle:
                    ftp.retrbinary(f"RETR {remote_filename}", file_handle.write)
            os.replace(temp_file_path, local_file_path)
            return
        except Exception as exc:
            last_error = exc
            _cleanup_partial_file(local_file_path)
            if attempt < RETRY_COUNT:
                print(
                    f"Retry {attempt}/{RETRY_COUNT} failed for {remote_filename}: {exc}"
                )
                time.sleep(RETRY_DELAY_SECONDS)
                continue
            raise last_error

# ==========================================
# メイン処理
# ==========================================
os.makedirs(SAVE_DIR, exist_ok=True)

if not FTP_USER or not FTP_PASS:
    raise RuntimeError("PTREE_FTP_USER と PTREE_FTP_PASS を環境変数に設定してください。")

print("FTPサーバーに接続中...")
print("認証情報を確認しました。ダウンロードを開始します。")

current_time = START_DATE
while current_time <= END_DATE:
    # パスとファイル名の生成
    year_month = current_time.strftime("%Y%m")
    day = current_time.strftime("%d")
    time_str = current_time.strftime("%Y%m%d%H%M%S")

    # リモート（FTP側）のディレクトリパス
    remote_dir = f"/pub/himawari/L3/SST/{VERSION_DIR}/{year_month}/{day}"

    # 正しい命名規則に沿ったファイル名
    filename = f"{time_str}-JAXA-L3C_GHRSST-SSTskin-H09_AHI-{VER_STR}-v02.0-fv01.0.nc"
    local_file_path = os.path.join(SAVE_DIR, filename)

    # すでにダウンロード済みの場合はスキップ
    if os.path.exists(local_file_path):
        current_time += timedelta(hours=1)
        continue

    try:
        print(f"Downloading: {filename}")
        download_file(remote_dir, filename, local_file_path)
    except error_perm as exc:
        # 該当時間にファイルがない、または権限/パスの問題がある場合は次へ進む
        print(f"Skipped (Not Found or Permission): {filename} -> {exc}")
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(f"{local_file_path}.part"):
            os.remove(f"{local_file_path}.part")
        if "550" in str(exc):
            print("※ファイルが見つかりません。VERSION_DIR または VER_STR を確認してください。")
    except Exception as exc:
        print(f"Skipped (Error): {filename} -> {exc}")
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if os.path.exists(f"{local_file_path}.part"):
            os.remove(f"{local_file_path}.part")

    # 1時間進める
    current_time += timedelta(hours=1)

print("すべてのダウンロード処理が完了しました。")
