# EduNavigator-AI

EduNavigator-AI 是一個以教育題目檢索為核心的 RAG 專案。系統會先將題目圖片轉成文字，再透過 Gemini embedding、BigQuery 向量檢索與二階段 rerank，找出題庫中最相近的題目，並在 Streamlit 介面中呈現結果。

## 專案特色

- 支援題目圖片上傳與 OCR 檢索
- 題庫資料以「一題一筆」方式切分與儲存
- 使用 BigQuery `VECTOR_SEARCH` 進行 dense retrieval
- 加入 lexical rerank 與 Gemini rerank，改善語義相近但題意不合的誤排
- 支援將教材 PDF 抽題、向量化並寫入 BigQuery

## 系統流程

1. 使用者上傳題目圖片
2. Gemini 2.5 Flash 將圖片 OCR 成文字查詢
3. `gemini-embedding-001` 產生 query embedding
4. BigQuery `VECTOR_SEARCH` 召回候選題目
5. 本地 lexical rerank 先做第一輪重排
6. Gemini rerank 再做第二輪題意排序
7. Streamlit 顯示題目內容、來源頁碼與排序結果

## 技術棧

- Gemini 2.5 Flash：圖片 OCR、LLM rerank
- `gemini-embedding-001`：query / document dense embedding
- BigQuery `VECTOR_SEARCH`：Cosine dense retrieval
- `google-genai`：Gemini API SDK
- `google-cloud-bigquery`：向量資料表與 SQL 檢索
- Streamlit：前端展示介面
- Pillow：圖片讀取與預覽
- python-dotenv：環境變數載入

## 專案結構

```text
.
├─ data/
│  ├─ processed_questions.json
│  ├─ processed_questions_progress.json
│  └─ source_pdfs/
├─ scripts/
│  ├─ extract_questions.py
│  └─ ingest_to_bigquery.py
├─ src/
│  ├─ app.py
│  └─ vector_search.py
├─ requirements.txt
└─ README.md
```

## 環境需求

- Python 3.10 以上
- Gemini API Key
- Google Cloud BigQuery 存取權限

## 安裝方式

在專案根目錄安裝套件：

```bash
python -m pip install -r requirements.txt
```

## 環境變數設定

請在專案根目錄建立 `.env`，並填入下列欄位：

```env
GEMINI_API_KEY=your_gemini_api_key
GCP_PROJECT_ID=your_gcp_project_id
BQ_DATASET_ID=your_bigquery_dataset
BQ_TABLE_ID=your_bigquery_table
LOCATION=asia-east1
```

## 常用指令

啟動 Streamlit 介面：

```bash
streamlit run src/app.py
```

執行向量搜尋 smoke test：

```bash
python src/vector_search.py
```

將題庫寫入 BigQuery：

```bash
python scripts/ingest_to_bigquery.py
```

從 PDF 抽取題目：

```bash
python scripts/extract_questions.py
```

## BigQuery資料設計摘要

- `question_id`：題目的穩定識別碼。
- `question_type`：題型分類，例如選擇題、是非題、填空題。
- `question_no`：題目在頁面上的題號。
- `content`：題目的主要文字內容。
- `display_text`：前端展示用完整題目文字。
- `vector_text`：向量化與檢索用文字。
- `source`：來源檔案名稱。
- `page`：題目所在頁碼。
- `embedding`：由 `gemini-embedding-001` 產生的向量欄位，供 BigQuery `VECTOR_SEARCH` 使用。

