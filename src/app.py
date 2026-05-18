import streamlit as st
import os
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from google import genai
from google.genai import types
from vector_search import search_similar_questions

# 這個檔案負責 UI 與互動流程：
# 使用者上傳圖片後，先用 Gemini 做 OCR，再把辨識出的 query_text 交給檢索模組。
# 檢索模組內部會完成 dense retrieval、lexical rerank 與 Gemini rerank，
# 這裡只負責收集輸入與展示最終結果。

# 1. 初始化與頁面設定
load_dotenv()
st.set_page_config(page_title="EduNavigator-AI", layout="centered")

# 這裡的 Gemini client 只負責「圖片 -> 文字」這段 OCR/vision 任務；
# 真正的向量檢索與 rerank 在 vector_search.py 中進行。
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
RAG_STACK = [
    "Gemini 2.5 Flash：圖片 OCR 與 LLM rerank",
    "gemini-embedding-001：query / document dense embedding",
    "BigQuery VECTOR_SEARCH：Cosine dense retrieval",
    "題目級切分：一題一筆 record",
    "display_text / vector_text：展示文本與向量文本分離",
    "Lexical rerank：字詞重疊重排",
    "Gemini rerank：候選題意重排，失敗時退回 lexical",
    "google-genai SDK：Gemini API 呼叫",
    "google-cloud-bigquery：向量資料表與 SQL 檢索",
    "Streamlit：Web UI",
    "Pillow：圖片讀取與預覽",
    "python-dotenv：環境變數載入",
]
PREVIEW_MAX_WIDTH = 520
PREVIEW_TALL_HEIGHT_THRESHOLD = 1200
PREVIEW_TALL_ASPECT_RATIO = 1.35
DEFAULT_TOP_K = 3
DEFAULT_CANDIDATE_POOL_SIZE = 6
DEFAULT_ENABLE_GEMINI_RERANK = True
DEFAULT_GEMINI_RERANK_CANDIDATES = 5

# 標題與簡介
st.title("🍎 EduNavigator-AI")
st.subheader("數位教學題目檢索系統")
st.markdown("請上傳一張包含題目的圖片，我將為您從題庫中檢索出最相似的題目。")

if "top_k" not in st.session_state:
    st.session_state["top_k"] = DEFAULT_TOP_K
if "candidate_pool_size" not in st.session_state:
    st.session_state["candidate_pool_size"] = DEFAULT_CANDIDATE_POOL_SIZE
if "enable_gemini_rerank" not in st.session_state:
    st.session_state["enable_gemini_rerank"] = DEFAULT_ENABLE_GEMINI_RERANK
if "gemini_rerank_candidate_count" not in st.session_state:
    st.session_state["gemini_rerank_candidate_count"] = DEFAULT_GEMINI_RERANK_CANDIDATES

# --- 側邊欄：設定與參數 ---
with st.sidebar:
    st.header("系統設定")

    if st.button("重設為推薦值", use_container_width=True):
        st.session_state["top_k"] = DEFAULT_TOP_K
        st.session_state["candidate_pool_size"] = DEFAULT_CANDIDATE_POOL_SIZE
        st.session_state["enable_gemini_rerank"] = DEFAULT_ENABLE_GEMINI_RERANK
        st.session_state["gemini_rerank_candidate_count"] = DEFAULT_GEMINI_RERANK_CANDIDATES

    top_k = st.slider("檢索題目數量", min_value=1, max_value=5, key="top_k")

    st.session_state["candidate_pool_size"] = max(st.session_state["candidate_pool_size"], top_k)
    candidate_pool_size = st.slider(
        "第一階段候選池大小",
        min_value=top_k,
        max_value=12,
        key="candidate_pool_size",
    )

    enable_gemini_rerank = st.checkbox("啟用 Gemini rerank", key="enable_gemini_rerank")

    st.session_state["gemini_rerank_candidate_count"] = min(
        max(st.session_state["gemini_rerank_candidate_count"], top_k),
        candidate_pool_size,
    )

    if candidate_pool_size == top_k:
        gemini_rerank_candidate_count = candidate_pool_size
        st.session_state["gemini_rerank_candidate_count"] = gemini_rerank_candidate_count
        st.caption(f"Gemini rerank 候選數固定為 {gemini_rerank_candidate_count}，因為候選池大小已等於 top_k。")
    else:
        gemini_rerank_candidate_count = st.slider(
            "Gemini rerank 候選數",
            min_value=top_k,
            max_value=candidate_pool_size,
            key="gemini_rerank_candidate_count",
            disabled=not enable_gemini_rerank,
        )

    st.info("目前流程: Gemini OCR -> BigQuery Dense Retrieval -> Lexical Rerank -> Gemini Rerank")
    st.caption(
        f"實驗設定：top_k={top_k}、candidate_pool={candidate_pool_size}、"
        f"Gemini rerank={'on' if enable_gemini_rerank else 'off'}、"
        f"Gemini candidates={gemini_rerank_candidate_count}"
    )
    st.markdown("**RAG 技術棧**")
    for item in RAG_STACK:
        st.write(f"- {item}")

# --- 主畫面：圖片上傳 ---
uploaded_file = st.file_uploader("選擇題目圖片...", type=["jpg", "jpeg", "png"])


def pil_image_to_part(image):
    """將 PIL Image 轉成 google.genai 可接受的 bytes part。"""
    image_buffer = BytesIO()
    image_format = (image.format or "PNG").upper()
    mime_type = Image.MIME.get(image_format, "image/png")
    image.save(image_buffer, format=image_format)
    return types.Part.from_bytes(data=image_buffer.getvalue(), mime_type=mime_type)


def render_uploaded_image_preview(image, show_above_button):
    """以可伸縮區塊顯示上傳圖片，避免高圖片把結果區整個往下推。"""
    width, height = image.size
    expander_label = "已上傳圖片"
    caption_text = f"圖片尺寸：{width} x {height}"

    with st.expander(expander_label, expanded=show_above_button):
        st.caption(caption_text)
        st.image(image, caption="已上傳圖片", width=min(width, PREVIEW_MAX_WIDTH))


def format_rerank_reason(result):
    """將內部排序標記轉成較容易展示給使用者的說明文字。"""
    rerank_reason = result.get("rerank_reason") or "dense_only"
    reason_map = {
        "gemini+dense+lexical": "Gemini + dense + lexical：先由 BigQuery dense retrieval 召回，再經 lexical 與 Gemini 兩段重排。",
        "dense+lexical": "Dense + lexical：先由 BigQuery dense retrieval 召回，再經本地 lexical 規則重排。",
        "dense_only": "Dense only：目前僅使用向量相似度排序。",
    }
    return reason_map.get(rerank_reason, rerank_reason)


def format_shared_rerank_reason(results):
    """整理本次檢索使用的排序流程，避免每張卡片重複顯示相同說明。"""
    rerank_reasons = []
    for result in results:
        rerank_reason = result.get("rerank_reason") or "dense_only"
        if rerank_reason not in rerank_reasons:
            rerank_reasons.append(rerank_reason)

    if not rerank_reasons:
        return "本次檢索沒有可用的排序說明。"

    if len(rerank_reasons) == 1:
        return f"本次檢索流程：{format_rerank_reason({'rerank_reason': rerank_reasons[0]})}"

    joined_reasons = "\n".join(
        f"- {format_rerank_reason({'rerank_reason': rerank_reason})}"
        for rerank_reason in rerank_reasons
    )
    return f"本次檢索中出現多種排序流程：\n{joined_reasons}"


if uploaded_file is not None:
    # 先依圖片尺寸決定預覽區塊位置，讓長圖不要壓縮主要操作區。
    image = Image.open(uploaded_file)
    image_width, image_height = image.size
    is_large_preview = (
        image_height >= PREVIEW_TALL_HEIGHT_THRESHOLD
        or image_height / max(image_width, 1) >= PREVIEW_TALL_ASPECT_RATIO
    )

    if not is_large_preview:
        render_uploaded_image_preview(image, show_above_button=True)
    
    if st.button("🔍 開始檢索"):
        with st.spinner("Gemini 正在辨識題目內容..."):
            # 2. OCR：把圖片先轉成 query_text，後續所有 retrieval / rerank 都以此文字為核心。
            prompt = "這是一張考卷或教材的截圖。請精確地辨識並提取圖片中的題目文字內容（含選項），不需要額外的解釋。"
            try:
                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[prompt, pil_image_to_part(image)],
                )
                query_text = response.text
                
                st.success("辨識成功！")
                with st.expander("查看辨識出的文字內容"):
                    st.write(query_text)
                
                # 3. 檢索：交給 vector_search.py 執行 dense retrieval + phase 1 lexical rerank + phase 2 Gemini rerank。
                st.divider()
                st.subheader(f"📍 為您找到的前 {top_k} 個相似題目：")
                
                results = search_similar_questions(
                    query_text,
                    top_k=top_k,
                    candidate_pool_size=candidate_pool_size,
                    enable_gemini_rerank=enable_gemini_rerank,
                    gemini_rerank_max_candidates=gemini_rerank_candidate_count,
                )
                
                if not results:
                    st.warning("在資料庫中找不到足夠相似的題目。")
                else:
                    with st.expander("檢索說明"):
                        st.write(format_shared_rerank_reason(results))

                    # 4. 呈現結果：這裡只顯示最終排序，不重複暴露底層 rerank 細節，避免 UI 太噪音。
                    for i, res in enumerate(results):
                        with st.container():
                            # 使用 Markdown 語法美化輸出
                            st.markdown(f"### 第 {i+1} 名 (相似度: {res['score']})")
                            st.info(f"📄 來源檔案: {res['source']} | 頁碼: 第 {res['page']} 頁")
                            st.write(res.get('display_text') or res['content'])
                            st.divider()
                            
            except Exception as e:
                st.error(f"發生錯誤: {str(e)}")

    if is_large_preview:
        render_uploaded_image_preview(image, show_above_button=False)

# 底部頁腳
st.caption("=====RAG 應用專案 for AI 數位教學=====")