import os
import re
import json
from io import BytesIO
from dotenv import load_dotenv
from google.cloud import bigquery
from google import genai
from google.genai import types
from PIL import Image

# 這個檔案負責整個檢索核心：
# 1. 將 query_text 轉成 embedding
# 2. 用 BigQuery VECTOR_SEARCH 做第一階段 dense recall
# 3. 用 lexical overlap 修正局部詞面不合但 dense score 偏高的結果
# 4. 用 Gemini 對前段候選做第二階段意圖排序
# 5. 回傳前端可以直接展示的 display_text / source / page / score

# 1. 初始化設定
load_dotenv()
PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET_ID = os.getenv("BQ_DATASET_ID")
TABLE_ID = os.getenv("BQ_TABLE_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "gemini-embedding-001"
VISION_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "gemini-2.5-flash"
RERANK_DENSE_WEIGHT = 0.72
RERANK_LEXICAL_WEIGHT = 0.28
RERANK_MC_QUESTION_TYPE_BONUS = 0.04
RERANK_CANDIDATE_BUFFER = 3
GEMINI_RERANK_MAX_CANDIDATES = 8
GEMINI_RERANK_ENABLED = os.getenv("ENABLE_GEMINI_RERANK", "true").lower() != "false"

# 這裡的 Gemini client 同時負責 embedding、vision 測試與 phase-2 rerank。
# BigQuery client 則負責向量檢索與資料表刪除等操作。
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
bq_client = bigquery.Client(project=PROJECT_ID)


def pil_image_to_part(image):
    """將 PIL Image 轉成 google.genai 可接受的 bytes part。"""
    image_buffer = BytesIO()
    image_format = (image.format or "PNG").upper()
    mime_type = Image.MIME.get(image_format, "image/png")
    image.save(image_buffer, format=image_format)
    return types.Part.from_bytes(data=image_buffer.getvalue(), mime_type=mime_type)

def analyze_image_to_text(image):
    """
    將使用者上傳的題目圖片轉換為文字描述，讓圖片能與資料庫中的文字向量進行比對。
    """
    prompt = "這是一張題目照片，請精確提取圖中的文字內容、數學公式與選項。如果是圖形題，請簡單描述圖形特徵。只需要輸出題目本文。"

    response = gemini_client.models.generate_content(
        model=VISION_MODEL,
        contents=[prompt, pil_image_to_part(image)],
    )
    return response.text

def get_query_embedding(text):
    """
    將搜尋字串轉化為向量。
    注意：搜尋時 task_type 必須使用 'retrieval_query' 以獲得最佳檢索效果。
    """
    result = gemini_client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


def normalize_display_text(content, display_text, vector_text):
    """補齊舊資料缺少的選項標記，確保前端展示與舊資料相容。"""
    normalized_content = (content or "").strip()
    normalized_display_text = (display_text or normalized_content).strip()

    if re.search(r"^\([A-F]\)\s", normalized_display_text, flags=re.MULTILINE):
        return normalized_display_text

    option_matches = re.findall(r"選項(\d+)\s*:\s*(.*?)(?=\s+選項\d+\s*:|$)", vector_text or "")
    if not option_matches:
        return normalized_display_text

    option_labels = ["(A)", "(B)", "(C)", "(D)", "(E)", "(F)"]
    normalized_options = []

    for option_number, option_text in option_matches:
        option_index = int(option_number) - 1
        cleaned_option_text = option_text.strip()
        if not cleaned_option_text:
            continue

        if 0 <= option_index < len(option_labels):
            normalized_options.append(f"{option_labels[option_index]} {cleaned_option_text}")
        else:
            normalized_options.append(cleaned_option_text)

    if normalized_content and normalized_options:
        return "\n".join([normalized_content, *normalized_options])

    return normalized_display_text


def normalize_rerank_text(text):
    """將文字正規化，盡量把不同標點、選項標記轉成可比對的統一形式。"""
    normalized_text = (text or "").lower()
    normalized_text = re.sub(r"\([a-f]\)", " ", normalized_text)
    normalized_text = re.sub(r"選項\s*\d+\s*:", " ", normalized_text)
    normalized_text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", normalized_text)
    return re.sub(r"\s+", " ", normalized_text).strip()


def tokenize_for_rerank(text):
    """同時保留詞片段與中文 bigram，降低中文沒有空白分詞時的漏配問題。"""
    normalized_text = normalize_rerank_text(text)
    if not normalized_text:
        return set()

    tokens = set()
    for chunk in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9_]+", normalized_text):
        if not chunk:
            continue

        tokens.add(chunk)
        if re.fullmatch(r"[\u4e00-\u9fff]+", chunk) and len(chunk) > 1:
            for index in range(len(chunk) - 1):
                tokens.add(chunk[index:index + 2])

    return tokens


def compute_lexical_overlap(query_text, candidate_text):
    """以 query coverage 為主，避免候選只靠主題相近就衝到太前面。"""
    query_tokens = tokenize_for_rerank(query_text)
    candidate_tokens = tokenize_for_rerank(candidate_text)

    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap_count = len(query_tokens & candidate_tokens)
    return overlap_count / len(query_tokens)


def looks_like_multiple_choice(text):
    """檢查 query 是否像選擇題，供 lexical rerank 做非常輕量的 soft bonus。"""
    normalized_text = text or ""
    return bool(
        re.search(r"\([A-D]\)", normalized_text)
        or re.search(r"[A-D][\.、]\s", normalized_text)
        or len(re.findall(r"選項\s*\d+\s*:", normalized_text)) >= 2
    )


def rerank_search_results(query_text, search_results, top_k):
    """phase 1：在 dense retrieval 之後做本地 lexical rerank。"""
    query_is_multiple_choice = looks_like_multiple_choice(query_text)
    reranked_results = []

    for index, result in enumerate(search_results):
        lexical_score = max(
            compute_lexical_overlap(query_text, result.get("vector_text")),
            compute_lexical_overlap(query_text, result.get("display_text") or result.get("content")),
        )
        rerank_score = (
            result["dense_score"] * RERANK_DENSE_WEIGHT
            + lexical_score * RERANK_LEXICAL_WEIGHT
        )

        if query_is_multiple_choice and result.get("question_type") == "multiple_choice":
            rerank_score += RERANK_MC_QUESTION_TYPE_BONUS

        reranked_result = dict(result)
        reranked_result["score"] = round(rerank_score, 4)
        reranked_result["dense_score"] = round(result["dense_score"], 4)
        reranked_result["lexical_score"] = round(lexical_score, 4)
        reranked_result["rerank_reason"] = "dense+lexical"
        reranked_results.append((rerank_score, -index, reranked_result))

    reranked_results.sort(reverse=True)
    return [result for _, _, result in reranked_results[:top_k]]


def build_gemini_rerank_prompt(query_text, search_results):
    """把候選濃縮成短文字，避免 rerank prompt 太長、太吵或成本過高。"""
    candidate_blocks = []
    expected_ranked_ids = list(range(len(search_results)))

    for index, result in enumerate(search_results):
        display_text = (result.get("display_text") or result.get("content") or "").strip()
        preview_text = re.sub(r"\s+", " ", display_text)[:280]
        candidate_blocks.append(
            "\n".join([
                f"candidate_id: {index}",
                f"question_type: {result.get('question_type') or 'unknown'}",
                f"source: {result.get('source') or ''}",
                f"page: {result.get('page') or ''}",
                f"dense_score: {result.get('dense_score', 0):.4f}",
                f"lexical_score: {result.get('lexical_score', 0):.4f}",
                f"text: {preview_text}",
            ])
        )

    joined_candidates = "\n\n".join(candidate_blocks)
    return f"""
你是國中數學題庫的 reranker。你的任務不是找語義大概相關的題目，而是找最符合使用者實際題意、上下文與小題連續性的候選。

排序原則：
1. 優先保留與查詢描述的題幹、情境、問法、小題承接關係最一致的候選。
2. 對於只在局部詞彙、學科主題或數學符號上相似，但實際題意不對的題目，要明確往後排。
3. 如果查詢看起來是同一大題下的第(1)(2)小題，優先保留同一題組或明顯延續該情境的候選。
4. 多選題、填充題、是非題的題型可以作為參考，但不要因題型相同就忽略題意不合。
5. 你必須輸出完整排序，`ranked_ids` 必須包含所有 candidate_id，且每個 id 只能出現一次。
6. 不可以只排前幾名；就算某些候選很差，也要把它們排在較後面。

請只輸出 JSON，格式如下：
{{"ranked_ids":{expected_ranked_ids},"reason":"簡短說明整體排序依據"}}

查詢文字：
{query_text}

候選題目：
{joined_candidates}
""".strip()


def parse_gemini_ranked_ids(response_text, valid_ids):
    """容忍模型輸出不完全乾淨，盡量從回應中救回可用排名。"""
    valid_ids = list(valid_ids)
    valid_id_set = set(valid_ids)
    ranked_ids = []

    try:
        parsed = json.loads(response_text)
        candidate_ids = parsed.get("ranked_ids", []) if isinstance(parsed, dict) else []
        for candidate_id in candidate_ids:
            if isinstance(candidate_id, int) and candidate_id in valid_id_set and candidate_id not in ranked_ids:
                ranked_ids.append(candidate_id)
    except Exception:
        pass

    if not ranked_ids:
        matches = re.findall(r"\d+", response_text or "")
        for match in matches:
            candidate_id = int(match)
            if candidate_id in valid_id_set and candidate_id not in ranked_ids:
                ranked_ids.append(candidate_id)

    if len(ranked_ids) != len(valid_ids):
        return []

    return ranked_ids


def gemini_rerank_search_results(query_text, search_results, top_k, enable_gemini_rerank=None, gemini_rerank_max_candidates=None):
    """phase 2：讓 Gemini 判斷哪個候選最符合題意與題組脈絡。"""
    if enable_gemini_rerank is None:
        enable_gemini_rerank = GEMINI_RERANK_ENABLED

    if gemini_rerank_max_candidates is None:
        gemini_rerank_max_candidates = GEMINI_RERANK_MAX_CANDIDATES

    if not enable_gemini_rerank or len(search_results) <= 1:
        return search_results[:top_k]

    candidate_count = min(
        len(search_results),
        max(top_k + 2, min(gemini_rerank_max_candidates, len(search_results)))
    )
    llm_candidates = search_results[:candidate_count]
    llm_prompt = build_gemini_rerank_prompt(query_text, llm_candidates)

    try:
        # 對候選做 LLM rerank，候選數由 UI 設定與程式內上下限共同決定。
        response = gemini_client.models.generate_content(
            model=RERANK_MODEL,
            contents=llm_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        ranked_ids = parse_gemini_ranked_ids(response.text, range(candidate_count))
        if not ranked_ids:
            print("Gemini rerank 未提供完整排序，改用 lexical rerank")
            return search_results[:top_k]

        ranked_results = []

        for ranked_id in ranked_ids:
            result = dict(llm_candidates[ranked_id])
            result["rerank_reason"] = "gemini+dense+lexical"
            ranked_results.append(result)

        ranked_results.extend(search_results[candidate_count:])
        return ranked_results[:top_k]
    except Exception as error:
        # 如果 rerank call 失敗，直接退回 lexical rerank，避免整段檢索不可用。
        print(f"Gemini rerank 失敗，改用 lexical rerank: {error}")
        return search_results[:top_k]

def search_similar_questions(
    query_text,
    top_k=3,
    candidate_pool_size=None,
    enable_gemini_rerank=None,
    gemini_rerank_max_candidates=None,
):
    """
    對外唯一的搜尋入口。
    流程是：query embedding -> BigQuery dense recall -> lexical rerank -> Gemini rerank。
    """
    # 1. 取得查詢文字的向量
    query_vector = get_query_embedding(query_text)
    
    # 2. 建構 SQL
    # 第一階段先多抓一點候選，不直接把 top_k 當最終答案，這樣 reranker 才有調整空間。
    table_full_id = f"`{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
    if candidate_pool_size is None:
        candidate_pool_size = max(top_k * 2, top_k + RERANK_CANDIDATE_BUFFER)
    else:
        candidate_pool_size = max(int(candidate_pool_size), top_k)
    
    sql = f"""
    SELECT 
        base.content AS content,
        base.question_type AS question_type,
        base.display_text AS display_text,
        base.vector_text AS vector_text,
        base.source AS source,
        base.page AS page,
        distance
    FROM VECTOR_SEARCH(
        TABLE {table_full_id},
        'embedding',
        (SELECT {query_vector} AS query_v),
        top_k => {candidate_pool_size},
        distance_type => 'COSINE'
    )
    """
    
    # 3. 執行查詢並格式化結果
    try:
        query_job = bq_client.query(sql)
        results = query_job.result()
        
        search_results = []
        for row in results:
            dense_score = 1 - row.distance
            search_results.append({
                "content": row.content,
                "question_type": row.question_type,
                "display_text": normalize_display_text(row.content, row.display_text, row.vector_text),
                "vector_text": row.vector_text,
                "source": row.source,
                "page": row.page,
                "dense_score": dense_score,
                "score": round(dense_score, 4) # 初始分數先記錄 dense similarity，後續可能被 rerank 覆寫
            })

        # phase 1 先做 deterministic lexical reorder，再交給 phase 2 Gemini 做更細的題意排序。
        lexically_reranked_results = rerank_search_results(query_text, search_results, len(search_results))
        return gemini_rerank_search_results(
            query_text,
            lexically_reranked_results,
            top_k,
            enable_gemini_rerank=enable_gemini_rerank,
            gemini_rerank_max_candidates=gemini_rerank_max_candidates,
        )
    except Exception as e:
        print(f"搜尋過程中發生錯誤: {e}")
        return []

def delete_file_data(source_file_name):
    """
    當使用者在網頁刪除某份 PDF 時，呼叫此函數移除資料庫對應內容。
    """
    table_full_id = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    sql = f"DELETE FROM `{table_full_id}` WHERE source = @filename"
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("filename", "STRING", source_file_name)
        ]
    )
    bq_client.query(sql, job_config=job_config).result()
    print(f"已從資料庫移除檔案: {source_file_name}")

# 測試邏輯
if __name__ == "__main__":
    # 用 q1.png 做完整圖片檢索測試：先辨識題目文字，再查詢相似題目
    test_image_path = "C:/Users/ichan/Desktop/q1.png"

    if not os.path.exists(test_image_path):
        print(f"找不到測試圖片: {test_image_path}")
        raise SystemExit(1)

    print(f"正在分析圖片: {test_image_path}")
    test_image = Image.open(test_image_path)
    test_text = analyze_image_to_text(test_image)
    test_image.close()

    print("辨識出的查詢文字：")
    print(test_text)
    print("\n正在搜尋相似題目...")

    matches = search_similar_questions(test_text)
    for i, m in enumerate(matches):
        preview_text = m.get('display_text') or m['content']
        print(f"結果 {i+1} [{m['source']} 頁{m['page']}] (分數: {m['score']}):\n{preview_text[:80]}...\n")