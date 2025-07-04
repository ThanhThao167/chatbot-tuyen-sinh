import os
import streamlit as st
import pandas as pd
import uuid
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from functools import lru_cache

# Load API key từ .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# UI cấu hình
st.set_page_config(page_title="Chatbot Tư Vấn Tuyển Sinh lớp 10 trường THPT Marie Curie", page_icon="🧠")
st.title("🧠 Chatbot Tư Vấn Tuyển Sinh lớp 10 trường THPT Marie Curie")

# Khởi tạo session
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load Q&A từ CSV
@lru_cache(maxsize=1)
def get_cached_qa_pairs():
    try:
        df = pd.read_csv("MC_chatbot.csv", encoding="utf-8")
        return list(zip(df['cauhoi'], df['cautraloi']))
    except Exception as e:
        st.error(f"❌ Lỗi tải dữ liệu: {e}")
        return []

@lru_cache(maxsize=1)
def get_vectorizer_and_matrix():
    qa_pairs = get_cached_qa_pairs()
    questions = [q for q, _ in qa_pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

def find_best_match(query, qa_pairs):
    vectorizer, tfidf_matrix = get_vectorizer_and_matrix()
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    idx = sims.argmax()
    score = sims[idx]
    return qa_pairs[idx][1], score

def save_chat_to_csv(session_id: str, role: str, content: str):
    history_file = "chat_history.csv"
    row = {
        "timestamp": datetime.now().isoformat(),
        "session_id": session_id,
        "role": role,
        "content": content
    }
    df = pd.DataFrame([row])
    if os.path.exists(history_file):
        df.to_csv(history_file, mode='a', index=False, header=False, encoding='utf-8')
    else:
        df.to_csv(history_file, mode='w', index=False, header=True, encoding='utf-8')

# Nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_chat_to_csv(st.session_state.session_id, "user", user_input)

    qa_pairs = get_cached_qa_pairs()
    if qa_pairs:
        answer, score = find_best_match(user_input, qa_pairs)
        if score >= 0.5:
            reply = answer
        else:
            try:
                context = st.session_state.messages[-3:] if len(st.session_state.messages) > 3 else st.session_state.messages
                completion = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": msg["role"], "content": msg["content"]} for msg in context
                    ]
                )
                reply = completion.choices[0].message.content
            except Exception as e:
                reply = f"❌ Lỗi khi gọi GPT: {e}"
    else:
        reply = "❌ Không thể tải dữ liệu từ file CSV."

    st.session_state.messages.append({"role": "assistant", "content": reply})
    save_chat_to_csv(st.session_state.session_id, "assistant", reply)

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

st.divider()

# Hiển thị bảng lịch sử
st.subheader("📂 Lịch sử trò chuyện (ghi nhận từ chat_history.csv)")
try:
    df = pd.read_csv("chat_history.csv")
    df_filtered = df[df["session_id"] == st.session_state.session_id]
    st.dataframe(df_filtered[["timestamp", "role", "content"]], use_container_width=True)
except FileNotFoundError:
    st.info("Chưa có dữ liệu lịch sử được lưu.")
