import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# =====================
# 1. Config & Logging
# =====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "111204")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "retail_db")
CSV_PATH = os.getenv("CSV_PATH", "online_retail_data.csv")

# =====================
# 2. Kết nối PostgreSQL
# =====================
try:
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )
    logging.info("✅ Kết nối PostgreSQL thành công!")
except Exception as e:
    logging.error(f"❌ Lỗi kết nối PostgreSQL: {e}")
    exit(1)

# =====================
# 3. Đọc dữ liệu CSV
# =====================
if not os.path.exists(CSV_PATH):
    logging.error(f"❌ Không tìm thấy file CSV tại {CSV_PATH}")
    exit(1)

try:
    df = pd.read_csv(CSV_PATH)
    logging.info(f"✅ Đọc file CSV thành công! Số dòng: {len(df)}")
except Exception as e:
    logging.error(f"❌ Lỗi đọc CSV: {e}")
    exit(1)

# =====================
# 4. Tiền xử lý dữ liệu
# =====================
try:
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df = df.dropna(subset=['InvoiceDate', 'Quantity', 'UnitPrice'])
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['Revenue'] = df['Quantity'] * df['UnitPrice']

    daily_revenue = df.groupby(df['InvoiceDate'].dt.date)['Revenue'].sum().rename('sales')
    daily_revenue.index = pd.to_datetime(daily_revenue.index)
    daily_revenue = daily_revenue.asfreq('D', fill_value=0)

    df_revenue_raw = daily_revenue.reset_index()
    df_revenue_raw.columns = ["date", "sales"]

    df_revenue_scaled = df_revenue_raw.copy()
    df_revenue_scaled["sales"] = df_revenue_scaled["sales"] / 10000

    logging.info("✅ Tiền xử lý dữ liệu hoàn tất!")
except Exception as e:
    logging.error(f"❌ Lỗi khi tiền xử lý dữ liệu: {e}")
    exit(1)

# =====================
# 5. Đẩy vào PostgreSQL
# =====================
try:
    df_revenue_raw.to_sql("daily_revenue", engine, if_exists="replace", index=False, method="multi")
    df_revenue_scaled.to_sql("daily_revenue_scaled", engine, if_exists="replace", index=False, method="multi")
    logging.info("✅ Đã đẩy dữ liệu vào PostgreSQL thành công!")
except Exception as e:
    logging.error(f"❌ Lỗi khi đẩy dữ liệu vào PostgreSQL: {e}")
    exit(1)

# =====================
# 6. Kiểm tra dữ liệu
# =====================
try:
    df_check = pd.read_sql("SELECT * FROM daily_revenue_scaled LIMIT 10;", engine)
    logging.info("🔎 10 dòng dữ liệu mẫu (scaled):")
    logging.info(f"\n{df_check}")
except Exception as e:
    logging.error(f"❌ Lỗi khi đọc lại dữ liệu từ PostgreSQL: {e}")

engine.dispose()
