import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone

import fitz  # PyMuPDF
from PIL import Image
from dotenv import load_dotenv
from google import genai


# 1. 載入環境設定並初始化 Gemini client
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. 抽取流程與輸出 schema 的全域設定
SCHEMA_VERSION = "2.0"
EXTRACTOR_VERSION = "2026-05-15-v1"
PROMPT_VERSION = "2026-05-15-v1"
MODEL_NAME = "gemini-2.5-flash"
EXTRACTION_SCOPE = ["multiple_choice", "true_false", "fill_in"]
DEFAULT_RETRY_DELAY_SECONDS = 30
MIN_RETRY_DELAY_SECONDS = 5
SUCCESS_DELAY_SECONDS = 8
MAX_RETRIES_PER_PAGE = 5
TAIWAN_TIMEZONE = timezone(timedelta(hours=8))

TRUE_FALSE_OPTION_SETS = {
    frozenset(["o", "x"]),
    frozenset(["○", "×"]),
    frozenset(["正確", "錯誤"]),
    frozenset(["對", "錯"]),
}


# 3. 自訂例外：當 quota 明顯無法在本輪恢復時，中止整批處理
class BatchQuotaExhaustedError(Exception):
    pass


def iso_now():
    """統一產生台灣時區 ISO 時間字串，供 corpus/progress 寫入。"""
    return datetime.now(TAIWAN_TIMEZONE).isoformat(timespec="seconds")


# 4. 基礎 JSON I/O 與 retry 時間解析工具
def load_json_file(path, default):
    """安全讀取 JSON；若檔案不存在或內容損壞則回傳預設值。"""
    if not os.path.exists(path):
        return default

    with open(path, "r", encoding="utf-8") as file_obj:
        try:
            return json.load(file_obj)
        except json.JSONDecodeError:
            return default


def save_json_file(path, data):
    """以 UTF-8 與縮排格式寫回 JSON，便於人工檢查。"""
    with open(path, "w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def get_progress_path(output_json):
    """由 corpus 檔名推導對應的 progress 檔名。"""
    output_dir = os.path.dirname(output_json)
    output_name = os.path.splitext(os.path.basename(output_json))[0]
    return os.path.join(output_dir, f"{output_name}_progress.json")


def parse_retry_delay_seconds(error_message):
    """從 Gemini 429 錯誤訊息中解析建議等待秒數，並加上安全緩衝。"""
    match = re.search(r"Please retry in ([\d.]+)(ms|s)", error_message)
    if not match:
        return DEFAULT_RETRY_DELAY_SECONDS

    value = float(match.group(1))
    unit = match.group(2)
    if unit == "ms":
        value /= 1000

    return max(value + 2, MIN_RETRY_DELAY_SECONDS)


def format_taiwan_retry_time(retry_delay_seconds):
    """把等待秒數轉成台灣時間，方便看何時適合再試。"""
    retry_time = datetime.now(TAIWAN_TIMEZONE) + timedelta(seconds=retry_delay_seconds)
    return retry_time.strftime("%Y-%m-%d %H:%M:%S %Z")


def normalize_response_text(response_text):
    """移除模型偶爾包上的 markdown code fence，留下純 JSON 文字。"""
    return response_text.replace("```json", "").replace("```", "").strip()


# 5. 模型回傳解析：把 Gemini 輸出轉成 page question list
def extract_page_questions(response_text):
    """解析單頁模型輸出；兼容直接 list 與包在 questions 內的物件格式。"""
    text_content = normalize_response_text(response_text)
    if not text_content:
        return []

    payload = json.loads(text_content)
    if isinstance(payload, dict):
        payload = payload.get("questions", [])

    if not isinstance(payload, list):
        return []

    return payload


# 6. Corpus 建立與整理：維護抽取結果的主資料檔
def build_empty_corpus():
    """建立空的 corpus 結構，作為抽取結果的唯一真相來源。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "extraction_scope": list(EXTRACTION_SCOPE),
        "generated_at": None,
        "source_count": 0,
        "question_count": 0,
        "notes": {
            "extractor_version": EXTRACTOR_VERSION,
            "model": MODEL_NAME,
            "prompt_version": PROMPT_VERSION,
        },
        "questions": [],
    }


def finalize_corpus(corpus_state):
    """回填題數、來源數與版本資訊，讓 corpus 每次落盤都保持完整。"""
    questions = corpus_state.get("questions", [])
    source_names = {
        question.get("source")
        for question in questions
        if question.get("source")
    }

    corpus_state["schema_version"] = SCHEMA_VERSION
    corpus_state["extraction_scope"] = list(EXTRACTION_SCOPE)
    corpus_state["generated_at"] = iso_now()
    corpus_state["source_count"] = len(source_names)
    corpus_state["question_count"] = len(questions)

    notes = corpus_state.get("notes")
    if not isinstance(notes, dict):
        notes = {}
    notes["extractor_version"] = EXTRACTOR_VERSION
    notes["model"] = MODEL_NAME
    notes["prompt_version"] = PROMPT_VERSION
    corpus_state["notes"] = notes
    return corpus_state


# 7. 題目正規化工具：統一空白、題型、marker 與向量文本
def normalize_whitespace(text):
    """壓縮多餘空白，避免同題因排版差異造成重複。"""
    return re.sub(r"\s+", " ", text or "").strip()


def is_true_false_options(options):
    """判斷選項是否屬於常見的是非題集合。"""
    normalized_options = [
        normalize_whitespace(str(option)).lower()
        for option in options
        if normalize_whitespace(str(option))
    ]
    if len(normalized_options) != 2:
        return False

    return frozenset(normalized_options) in TRUE_FALSE_OPTION_SETS


def detect_markers(question_text):
    """偵測填空題與是非題常見訊號，供題型判斷與後續除錯使用。"""
    text = normalize_whitespace(question_text)
    markers = []

    if re.search(r"答[:：]\s*[_＿]{2,}", text):
        markers.append("answer_colon")
    if re.search(r"[_＿]{2,}", text):
        markers.append("underline")
    if re.search(r"[=＝]\s*[_＿]{2,}$", text):
        markers.append("tail_equals")
    if re.search(r"為\s*[_＿]{2,}$", text):
        markers.append("tail_wei")
    if re.search(r"(?:=>|＝＞)\s*[_＿]{2,}$", text):
        markers.append("tail_arrow")
    if re.search(r"^[（(]\s*[）)]", text):
        markers.append("paren_blank")

    return markers


def normalize_question_type(type_value, question_text, options):
    """將模型回傳或規則推斷的題型收斂成固定三類。"""
    normalized_type = normalize_whitespace(str(type_value or "")).lower().replace("-", "_")
    type_aliases = {
        "multiple_choice": "multiple_choice",
        "multiple choice": "multiple_choice",
        "mcq": "multiple_choice",
        "choice": "multiple_choice",
        "選擇題": "multiple_choice",
        "true_false": "true_false",
        "true false": "true_false",
        "tf": "true_false",
        "是非題": "true_false",
        "fill_in": "fill_in",
        "fill in": "fill_in",
        "fill_blank": "fill_in",
        "fill_in_the_blank": "fill_in",
        "填空題": "fill_in",
    }
    if normalized_type in type_aliases:
        return type_aliases[normalized_type]

    if is_true_false_options(options):
        return "true_false"

    if detect_markers(question_text):
        return "fill_in"

    if options:
        return "multiple_choice"

    return "fill_in"


def normalize_markers(raw_markers, question_text, question_type, options):
    """合併模型提供的 marker 與本地規則推得的 marker。"""
    markers = []
    if isinstance(raw_markers, list):
        markers.extend(
            normalize_whitespace(str(marker)).lower().replace(" ", "_")
            for marker in raw_markers
            if normalize_whitespace(str(marker))
        )

    markers.extend(detect_markers(question_text))

    if question_type == "true_false" or is_true_false_options(options):
        markers.append("ox_marker")

    return sorted(set(marker for marker in markers if marker))


def build_vector_text(question_type, content, options):
    """建立向量檢索用文字；選擇題會把選項一併拼入。"""
    if question_type != "multiple_choice" or not options:
        return content

    option_text = " ".join(
        f"選項{index}: {option}"
        for index, option in enumerate(options, start=1)
    )
    return f"{content} {option_text}".strip()


def build_question_id(source_name, page_number, question_type, content):
    """以來源、頁碼、題型、題面生成穩定 id，供去重與追蹤使用。"""
    seed = "|".join([
        normalize_whitespace(source_name),
        str(page_number),
        question_type,
        normalize_whitespace(content),
    ])
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def normalize_question_record(question, source_name, page_number, default_question_no):
    """將單題原始資料整理成統一 question record。"""
    if not isinstance(question, dict):
        return None

    raw_question_text = question.get("raw_text") or question.get("question") or question.get("content")
    content = normalize_whitespace(question.get("content") or question.get("question") or "")
    if not content:
        return None

    raw_options = question.get("options")
    if not isinstance(raw_options, list):
        raw_options = []
    options = [
        normalize_whitespace(str(option))
        for option in raw_options
        if normalize_whitespace(str(option))
    ]

    question_type = normalize_question_type(question.get("type"), content, options)
    normalized_options = options if question_type == "multiple_choice" else []
    markers = normalize_markers(question.get("markers"), content, question_type, options)
    question_no = question.get("question_no")
    if question_no in (None, ""):
        question_no = default_question_no

    return {
        "id": build_question_id(source_name, page_number, question_type, content),
        "source": source_name,
        "page": page_number,
        "question_no": question_no,
        "type": question_type,
        "content": content,
        "options": normalized_options,
        "vector_text": build_vector_text(question_type, content, normalized_options),
        "raw_text": normalize_whitespace(raw_question_text or content),
        "extraction_method": question.get("extraction_method") or "gemini",
        "markers": markers,
    }


def load_corpus_file(path):
    """讀取既有 corpus；若遇舊版 flat list 也一併轉成新版結構。"""
    raw_payload = load_json_file(path, None)
    corpus_state = build_empty_corpus()

    if isinstance(raw_payload, list):
        for index, question in enumerate(raw_payload, start=1):
            normalized_question = normalize_question_record(
                question,
                question.get("source", ""),
                question.get("page", 0),
                question.get("question_no") or index,
            )
            if normalized_question:
                corpus_state["questions"].append(normalized_question)
        return finalize_corpus(corpus_state)

    if isinstance(raw_payload, dict):
        questions = raw_payload.get("questions", [])
        if isinstance(questions, list):
            normalized_questions = []
            for index, question in enumerate(questions, start=1):
                normalized_question = normalize_question_record(
                    question,
                    question.get("source", ""),
                    question.get("page", 0),
                    question.get("question_no") or index,
                )
                if normalized_question:
                    normalized_questions.append(normalized_question)
            corpus_state.update(raw_payload)
            corpus_state["questions"] = normalized_questions

    return finalize_corpus(corpus_state)


def save_corpus_file(path, corpus_state):
    """寫回 corpus 前先補齊統計與版本欄位。"""
    save_json_file(path, finalize_corpus(corpus_state))


# 8. Progress 建立與整理：維護續跑、狀態與錯誤資訊
def build_empty_progress():
    """建立空的 progress 結構，專門追蹤執行狀態。"""
    return {
        "schema_version": SCHEMA_VERSION,
        "extractor_version": EXTRACTOR_VERSION,
        "extraction_scope": list(EXTRACTION_SCOPE),
        "started_at": iso_now(),
        "updated_at": iso_now(),
        "sources": {},
    }


def normalize_progress_state(progress_state):
    """把舊的或不完整的 progress 資料整理成固定結構。"""
    if not isinstance(progress_state, dict):
        progress_state = {}

    normalized_state = build_empty_progress()
    normalized_state["started_at"] = progress_state.get("started_at") or normalized_state["started_at"]

    sources = progress_state.get("sources", {})
    if not isinstance(sources, dict):
        sources = {}

    normalized_sources = {}
    for source_name, source_progress in sources.items():
        if not isinstance(source_progress, dict):
            continue

        pages = source_progress.get("pages", {})
        if not isinstance(pages, dict):
            pages = {}

        normalized_pages = {}
        for page_number, page_state in pages.items():
            if not isinstance(page_state, dict):
                page_state = {}
            normalized_pages[str(page_number)] = {
                "status": page_state.get("status") or "pending",
                "question_count": page_state.get("question_count", 0),
                "question_ids": list(page_state.get("question_ids", [])),
                "last_attempt_at": page_state.get("last_attempt_at"),
                "last_error": page_state.get("last_error"),
                "methods_used": list(page_state.get("methods_used", [])),
            }

        normalized_sources[source_name] = {
            "total_pages": source_progress.get("total_pages", 0),
            "pages": normalized_pages,
        }

    normalized_state["sources"] = normalized_sources
    normalized_state["updated_at"] = iso_now()
    return normalized_state


def load_progress_state(path):
    """讀取並正規化 progress 檔案。"""
    return normalize_progress_state(load_json_file(path, {}))


def save_progress_state(path, progress_state):
    """寫回 progress 前補齊版本資訊與最後更新時間。"""
    progress_state["schema_version"] = SCHEMA_VERSION
    progress_state["extractor_version"] = EXTRACTOR_VERSION
    progress_state["extraction_scope"] = list(EXTRACTION_SCOPE)
    progress_state["updated_at"] = iso_now()
    save_json_file(path, progress_state)


def ensure_source_progress(progress_state, source_name, total_pages):
    """確保某一 PDF 在 progress 中有對應的 source 節點。"""
    sources = progress_state.setdefault("sources", {})
    source_progress = sources.get(source_name)
    if not isinstance(source_progress, dict):
        source_progress = {"total_pages": total_pages, "pages": {}}
        sources[source_name] = source_progress

    source_progress["total_pages"] = total_pages
    source_progress.setdefault("pages", {})
    return source_progress


def get_page_progress(progress_state, source_name, page_number, total_pages):
    """取得單頁 progress；若不存在就建立預設狀態。"""
    source_progress = ensure_source_progress(progress_state, source_name, total_pages)
    pages = source_progress["pages"]
    page_key = str(page_number)
    if page_key not in pages:
        pages[page_key] = {
            "status": "pending",
            "question_count": 0,
            "question_ids": [],
            "last_attempt_at": None,
            "last_error": None,
            "methods_used": [],
        }
    return pages[page_key]


def mark_page_completed(progress_state, source_name, page_number, total_pages, question_ids, methods_used):
    """將單頁標記為完成，並記錄題數、題目 id 與使用方法。"""
    page_progress = get_page_progress(progress_state, source_name, page_number, total_pages)
    page_progress["status"] = "completed"
    page_progress["question_count"] = len(question_ids)
    page_progress["question_ids"] = list(question_ids)
    page_progress["last_attempt_at"] = iso_now()
    page_progress["last_error"] = None
    page_progress["methods_used"] = sorted(set(methods_used))


def mark_page_failed(progress_state, source_name, page_number, total_pages, error_message):
    """將單頁標記為失敗，保留最後錯誤訊息供後續續跑判讀。"""
    page_progress = get_page_progress(progress_state, source_name, page_number, total_pages)
    page_progress["status"] = "failed"
    page_progress["last_attempt_at"] = iso_now()
    page_progress["last_error"] = error_message


# 9. 頁級資料整理：單頁結果會先正規化，再整頁替換進 corpus
def replace_page_questions(all_questions, source_name, page_number, replacement_questions):
    """以整頁為單位替換 corpus 內容，避免同頁殘留舊版本題目。"""
    retained_questions = [
        question
        for question in all_questions
        if not (
            question.get("source") == source_name
            and question.get("page") == page_number
        )
    ]
    retained_questions.extend(replacement_questions)
    retained_questions.sort(
        key=lambda item: (
            item.get("source", ""),
            item.get("page", 0),
            str(item.get("question_no", "")),
        )
    )
    return retained_questions


def prepare_page_questions(page_questions, source_name, page_number):
    """整理單頁題目並依 question id 去除同頁重複資料。"""
    prepared_questions = []
    seen_question_ids = set()

    for index, question in enumerate(page_questions, start=1):
        normalized_question = normalize_question_record(
            question,
            source_name,
            page_number,
            index,
        )
        if not normalized_question:
            continue

        question_id = normalized_question["id"]
        if question_id in seen_question_ids:
            continue

        prepared_questions.append(normalized_question)
        seen_question_ids.add(question_id)

    return prepared_questions


# 10. 主抽取流程：逐 PDF、逐頁送模型、處理 retry，並同步寫入 corpus/progress
def process_single_pdf(pdf_path, output_json, request_stats):
    """處理單一 PDF 的完整抽取流程。"""
    if not os.path.exists(pdf_path):
        print(f"錯誤：找不到檔案 {pdf_path}")
        return

    doc = fitz.open(pdf_path)
    source_name = os.path.basename(pdf_path)
    progress_path = get_progress_path(output_json)
    corpus_state = load_corpus_file(output_json)
    progress_state = load_progress_state(progress_path)
    all_questions = corpus_state["questions"]

    print(f"開始處理: {pdf_path} (共 {len(doc)} 頁)")

    for page_num in range(len(doc)):
        page_number = page_num + 1
        page_progress = get_page_progress(progress_state, source_name, page_number, len(doc))
        if page_progress.get("status") == "completed":
            print(f"  - 跳過第 {page_number}/{len(doc)} 頁：已在先前執行完成")
            continue

        print(f"  - 正在解析第 {page_number}/{len(doc)} 頁...")

        # 先把當前頁渲染成圖片，供 Gemini 視覺模型辨識
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_path = f"temp_page_{page_num}.png"
        pix.save(img_path)
        img = Image.open(img_path)

        page_completed = False
        last_retry_delay = None

        try:
            for attempt in range(1, MAX_RETRIES_PER_PAGE + 1):
                try:
                    # 記錄 request 次數，方便對照 quota 消耗與錯誤時機
                    next_request_number = request_stats["total_requests"] + 1
                    print(f"    [*] 準備送出第 {next_request_number} 次 request")
                    request_stats["total_requests"] = next_request_number

                    # 核心 prompt：要求模型一次回傳三種題型的結構化 JSON
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=[
                            (
                                "請從這一頁提取所有題目，涵蓋選擇題、是非題、填空題。"
                                "只回傳合法 JSON 陣列，不要加 markdown 或說明文字。"
                                "每個元素格式必須是 "
                                '[{"question_no": "1", "type": "multiple_choice", "question": "題目", '
                                '"options": ["選項1", "選項2"], "raw_text": "原始題面", '
                                '"markers": ["underline"]}]'
                                "。"
                                "type 只能是 multiple_choice、true_false、fill_in。"
                                "選擇題保留 options；是非題與填空題的 options 回傳 []。"
                                "填空題請保留題目中的底線、答：、等號等空格標記。"
                                "若本頁沒有題目，回傳 []。"
                            ),
                            img,
                        ],
                    )

                    print(f"    [*] 已送出 request 總數: {request_stats['total_requests']}")

                    # 解析模型輸出後，正規化成統一的 question record
                    page_questions = extract_page_questions(response.text)
                    prepared_page_questions = prepare_page_questions(
                        page_questions,
                        source_name,
                        page_number,
                    )

                    all_questions = replace_page_questions(
                        all_questions,
                        source_name,
                        page_number,
                        prepared_page_questions,
                    )
                    corpus_state["questions"] = all_questions

                    # 把本頁題目 id 寫入 progress，後續可用於續跑與核對
                    question_ids = [question["id"] for question in prepared_page_questions]
                    methods_used = {
                        question.get("extraction_method", "gemini")
                        for question in prepared_page_questions
                    }
                    if not methods_used:
                        methods_used = {"gemini"}

                    mark_page_completed(
                        progress_state,
                        source_name,
                        page_number,
                        len(doc),
                        question_ids,
                        methods_used,
                    )
                    save_corpus_file(output_json, corpus_state)
                    save_progress_state(progress_path, progress_state)

                    print(f"    ✅ 成功提取 {len(prepared_page_questions)} 個題目")
                    page_completed = True
                    time.sleep(SUCCESS_DELAY_SECONDS)
                    break

                except json.JSONDecodeError:
                    # 若模型回傳不是合法 JSON，短暫等待後重試同一頁
                    print(f"    ⚠️ JSON 解析失敗 (嘗試 {attempt}/{MAX_RETRIES_PER_PAGE})")
                    if attempt == MAX_RETRIES_PER_PAGE:
                        mark_page_failed(
                            progress_state,
                            source_name,
                            page_number,
                            len(doc),
                            "JSON 解析失敗",
                        )
                        print("    ⚠️ 已達 JSON 重試上限，保留下次續跑")
                    else:
                        time.sleep(MIN_RETRY_DELAY_SECONDS)

                except Exception as error:
                    # 其餘錯誤統一在這裡處理，429 會依 retry delay 判斷是否繼續重試
                    error_message = str(error)
                    print(f"    ❌ 錯誤 (頁 {page_number}，嘗試 {attempt}/{MAX_RETRIES_PER_PAGE}): {error_message}")
                    print(f"    [*] 已送出 request 總數: {request_stats['total_requests']}")

                    if "429" in error_message and attempt < MAX_RETRIES_PER_PAGE:
                        # 若 retry delay 沒有改善，視為本輪 quota 已耗盡，直接中止整批
                        retry_delay = parse_retry_delay_seconds(error_message)
                        if last_retry_delay is not None and retry_delay >= last_retry_delay:
                            raise BatchQuotaExhaustedError(
                                "疑似 project/model free tier 已用盡，請稍後或隔日再試"
                            )

                        last_retry_delay = retry_delay
                        retry_time_taiwan = format_taiwan_retry_time(retry_delay)
                        print(f"    建議下次重試時間（台灣時間）: {retry_time_taiwan}")
                        print(f"    等待 {retry_delay:.1f} 秒後重試同一頁...")
                        time.sleep(retry_delay)
                        continue

                    mark_page_failed(
                        progress_state,
                        source_name,
                        page_number,
                        len(doc),
                        error_message,
                    )

                    if "429" in error_message:
                        raise BatchQuotaExhaustedError(
                            "疑似 project/model free tier 已用盡，請稍後或隔日再試"
                        )
                    break
        finally:
            # 無論成功或失敗，都清除暫存圖片以免累積垃圾檔
            img.close()
            if os.path.exists(img_path):
                os.remove(img_path)

        if not page_completed:
            # 單頁失敗也先把目前 corpus/progress 狀態落盤，方便下次續跑
            corpus_state["questions"] = all_questions
            save_corpus_file(output_json, corpus_state)
            save_progress_state(progress_path, progress_state)

    # 單一 PDF 結束後再次完整落盤，確保最後狀態與累計題數一致
    corpus_state["questions"] = all_questions
    save_corpus_file(output_json, corpus_state)
    save_progress_state(progress_path, progress_state)
    doc.close()

    print(f"PDF 處理完成！目前累計題目總數: {len(all_questions)}")


if __name__ == "__main__":
    # 11. 批次入口：逐份 PDF 執行抽取，若 quota 用盡則中止本輪並保留進度
    PDF_DIR = "data/source_pdfs"
    OUTPUT_FILE = "data/processed_questions.json"
    os.makedirs("data", exist_ok=True)
    request_stats = {"total_requests": 0}

    pdf_files = [file_name for file_name in os.listdir(PDF_DIR) if file_name.lower().endswith(".pdf")]
    try:
        for pdf_file in pdf_files:
            process_single_pdf(os.path.join(PDF_DIR, pdf_file), OUTPUT_FILE, request_stats)
            time.sleep(5)
    except BatchQuotaExhaustedError as error:
        print(f"[!] {error}")