import os
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from google.cloud import bigquery
from google import genai
from google.genai import types

# 1. 載入設定與環境變數
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET_ID")
TABLE_ID = os.getenv("BQ_TABLE_ID")
LOCATION = os.getenv("LOCATION", "asia-east1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPPORTED_CORPUS_SCHEMA_VERSIONS = {"2.0"}
EMBEDDING_MODEL = "gemini-embedding-001"

# 2. 初始化 BigQuery 客戶端與 Gemini AI 設定
bq_client = bigquery.Client(project=PROJECT_ID, location=LOCATION)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def get_required_schema():
    """
    定義目前 ingestion 需要的 BigQuery 欄位。
    保留既有的 content/source/page/embedding 供檢索使用，
    並補上 question_id/question_type/question_no/vector_text 來對齊新版 extractor。
    """
    return [
        bigquery.SchemaField("question_id", "STRING", mode="REQUIRED", description="題目穩定識別碼"),
        bigquery.SchemaField("question_type", "STRING", mode="NULLABLE", description="題型分類"),
        bigquery.SchemaField("question_no", "STRING", mode="NULLABLE", description="頁面上的題號"),
        bigquery.SchemaField("content", "STRING", mode="REQUIRED", description="題目顯示內容"),
        bigquery.SchemaField("display_text", "STRING", mode="NULLABLE", description="前端顯示使用的完整題目文字"),
        bigquery.SchemaField("vector_text", "STRING", mode="NULLABLE", description="向量化使用的文本"),
        bigquery.SchemaField("source", "STRING", mode="NULLABLE", description="來源檔案名稱"),
        bigquery.SchemaField("page", "INTEGER", mode="NULLABLE", description="所在頁碼"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED", description="由模型產生的向量值"),
    ]


def ensure_table_schema():
    """
    確保 BigQuery 中存在目標資料表，並在既有資料表上補齊新版 schema 所需欄位。
    """
    table_full_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    required_schema = get_required_schema()

    try:
        table = bq_client.get_table(table_full_id)
        print(f"[*] 檢查：資料表 {table_full_id} 已存在。")
    except Exception:
        table = bigquery.Table(table_full_id, schema=required_schema)
        bq_client.create_table(table)
        print(f"[+] 成功建立資料表: {table_full_id}")
        return

    existing_field_names = {field.name for field in table.schema}
    missing_fields = [field for field in required_schema if field.name not in existing_field_names]
    if not missing_fields:
        return

    table.schema = list(table.schema) + missing_fields
    bq_client.update_table(table, ["schema"])
    added_field_names = ", ".join(field.name for field in missing_fields)
    print(f"[+] 已補齊資料表欄位: {added_field_names}")


def get_existing_sources():
    """
    查詢 BigQuery，回傳目前資料庫中已經有哪些 PDF 檔案的資料。
    這裡不再拿來做「整份略過」，而是拿來決定哪些來源需要先刪除舊資料再重寫。
    """
    table_full_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    query = f"SELECT DISTINCT source FROM `{table_full_id}`"
    try:
        query_job = bq_client.query(query)
        results = query_job.result()
        return [row.source for row in results]
    except Exception as e:
        print(f"[!] 無法取得現有來源資訊（可能資料表為空）: {e}")
        return []

def get_embedding(text):
    """
    呼叫 Gemini API 將題目文字轉換為向量。
    """
    # task_type "retrieval_document" 適合用於存放進資料庫的文本
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return result.embeddings[0].values

def load_questions_payload(input_file):
    """
    讀取 extractor 輸出的 corpus JSON。
    兼容舊版 flat list 與新版 top-level object，並回傳 schema version 與 question list。
    """
    with open(input_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        return None, payload

    if isinstance(payload, dict):
        questions = payload.get("questions", [])
        if isinstance(questions, list):
            return payload.get("schema_version"), questions

    return None, []


def validate_corpus_schema(schema_version):
    """
    對新版 corpus JSON 做最基本的 schema version 檢查。
    若 extractor 之後再升版，這裡會提早失敗而不是默默吃進不相容資料。
    """
    if schema_version is None:
        print("[*] 偵測到舊版 flat JSON，將以相容模式繼續處理。")
        return

    if schema_version not in SUPPORTED_CORPUS_SCHEMA_VERSIONS:
        raise ValueError(
            f"不支援的 processed_questions schema_version: {schema_version}"
        )


def group_questions_by_source(all_questions):
    """
    依來源檔案分組，讓 ingestion 可以對每份 PDF 執行先刪後寫。
    這樣即使 extractor 重跑後題數變動，也能完整覆蓋 BigQuery 舊資料。
    """
    grouped_questions = defaultdict(list)
    for question in all_questions:
        source_name = question.get("source")
        if not source_name:
            continue
        grouped_questions[source_name].append(question)
    return grouped_questions


def delete_existing_source_rows(source_file_name):
    """
    在寫入同一份 PDF 的新資料前，先移除舊資料。
    這比單純用 source 做略過更符合新版 extractor 的重跑與補抓模式。
    """
    table_full_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    sql = f"DELETE FROM `{table_full_id}` WHERE source = @filename"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("filename", "STRING", source_file_name)
        ]
    )
    bq_client.query(sql, job_config=job_config).result()


def build_display_text(question):
    """
    產生前端顯示用的完整文字。
    選擇題會附上各選項，其餘題型維持原始 content。
    """
    content = (question.get("content") or question.get("question") or "").strip()
    question_type = question.get("type") or question.get("question_type")
    options = question.get("options") or []

    if question_type == "multiple_choice":
        normalized_options = []
        option_labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]

        for index, option in enumerate(options):
            option_text = str(option).strip()
            if not option_text:
                continue

            # 若 extractor 沒保留選項標記，則在 ingestion 時補上穩定的顯示標記。
            if not re.match(r"^\([A-F]\)", option_text, flags=re.IGNORECASE):
                if index < len(option_labels):
                    option_text = f"{option_labels[index]} {option_text}"

            normalized_options.append(option_text)

        if normalized_options:
            return "\n".join([content, *normalized_options])

    return content


def build_row_for_insert(question, vector):
    """
    將單題 question record 轉成 BigQuery row。
    content 保留前端顯示用文字，vector_text 保留實際 embedding 用文字，
    其他欄位則支援後續追蹤題型與去重。
    """
    content = question.get("content") or question.get("question")
    display_text = build_display_text(question)
    vector_text = question.get("vector_text") or content
    question_no = question.get("question_no")
    if question_no is not None:
        question_no = str(question_no)

    return {
        "question_id": question.get("id") or question.get("question_id"),
        "question_type": question.get("type") or question.get("question_type"),
        "question_no": question_no,
        "content": content,
        "display_text": display_text,
        "vector_text": vector_text,
        "source": question.get("source"),
        "page": question.get("page"),
        "embedding": vector,
    }


def build_rows_for_source(source_name, questions):
    """
    針對單一來源檔案產生待寫入的 BigQuery rows。
    若某題缺少必要欄位，直接略過並留下 log，避免整批中斷。
    """
    rows_to_insert = []

    for question in questions:
        content = question.get("content") or question.get("question")
        vector_text = question.get("vector_text") or content

        if not content or not vector_text or not question.get("page"):
            print(f"  - [!] 略過缺少必要欄位的題目: [{source_name}]")
            continue

        print(f"  - 向量化中: [{source_name}] {content[:15]}...")
        try:
            vector = get_embedding(vector_text)
            rows_to_insert.append(build_row_for_insert(question, vector))
        except Exception as e:
            print(f"  - [!] 向量化失敗: {e}")

    return rows_to_insert


def insert_rows(rows_to_insert):
    """
    將已完成向量化的資料寫入 BigQuery。
    寫入錯誤直接回報，方便針對 schema 或資料格式問題追查。
    """
    if not rows_to_insert:
        return

    table_ref = bq_client.dataset(DATASET_ID).table(TABLE_ID)
    errors = bq_client.insert_rows_json(table_ref, rows_to_insert)

    if errors == []:
        print(f"[OK] 成功寫入 {len(rows_to_insert)} 筆資料。")
    else:
        print(f"[!] 寫入 BigQuery 時發生錯誤: {errors}")

def main():
    input_file = "data/processed_questions.json"
    
    if not os.path.exists(input_file):
        print(f"[!] 錯誤：找不到 {input_file}，請先執行 extract_questions.py")
        return

    # A. 確保資料表結構就緒
    ensure_table_schema()

    # B. 取得資料庫中已有的 PDF 檔案清單
    existing_pdfs = get_existing_sources()
    print(f"[*] 目前資料庫中已包含的檔案: {existing_pdfs}")

    # C. 讀取 extractor 產出的 corpus JSON，並驗證 schema version
    schema_version, all_questions = load_questions_payload(input_file)
    validate_corpus_schema(schema_version)

    if not all_questions:
        print("[*] 沒有偵測到可寫入的題目資料。")
        return

    # D. 依來源分組，改用「每個 source 先刪後寫」的同步策略
    questions_by_source = group_questions_by_source(all_questions)
    print(f"[+] 準備同步 {len(questions_by_source)} 份來源檔案，共 {len(all_questions)} 題。")

    for source_name, source_questions in questions_by_source.items():
        print(f"\n[*] 正在同步來源: {source_name} ({len(source_questions)} 題)")

        if source_name in existing_pdfs:
            print(f"  - 先刪除 BigQuery 中既有資料: {source_name}")
            delete_existing_source_rows(source_name)

        rows_to_insert = build_rows_for_source(source_name, source_questions)
        insert_rows(rows_to_insert)

if __name__ == "__main__":
    main()