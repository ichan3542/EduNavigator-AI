# 1. 使用輕量級的 Python 3.12 映像檔作為基礎
FROM python:3.12-slim-bookworm

# 2. 設定容器內的工作目錄
WORKDIR /app

# 3. 安裝系統級相依套件 (PyMuPDF 處理 PDF 時需要一些底層庫)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# 4. 複製相依套件清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. 複製專案原始碼到容器中
COPY . .

# 6. 建立必要的資料夾（避免程式執行時找不到路徑）
RUN mkdir -p data/source_pdfs

# 7. 設定環境變數
# 設定 Streamlit 監聽的通訊埠 (Cloud Run 預設為 8080)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    TZ=Asia/Taipei \
    PORT=8080

# 8. 暴露通訊埠
EXPOSE 8080

# 9. 啟動指令：執行 src/app.py
# --server.port 指定埠號，--server.address 必須設定為 0.0.0.0 以接受外部連線
CMD ["streamlit", "run", "src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]