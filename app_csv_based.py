import os
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache
from openai import OpenAI
from datetime import datetime

# Load .env
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class ChatPayload(BaseModel):
    session_id: str
    messages: List[dict]

# Load Q&A từ CSV
@lru_cache(maxsize=1)
def get_cached_qa_pairs():
    try:
        df = pd.read_csv("MC_chatbot.csv", encoding="utf-8")
        return list(zip(df['cauhoi'], df['cautraloi']))
    except Exception as e:
        logger.error(f"❌ CSV load error: {e}")
        return []

@lru_cache(maxsize=1)
def get_vectorizer_and_matrix():
    qa_pairs = get_cached_qa_pairs()
    questions = [q for q, _ in qa_pairs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(questions)
    return vectorizer, tfidf_matrix

def find_best_match(query, qa_pairs, threshold=0.5):
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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(payload: ChatPayload, request: Request):
    body = await request.body()
    logger.info(f"📦 Payload: {body.decode('utf-8')}")
    session_id = payload.session_id
    user_message = payload.messages[-1]['content']
    qa_pairs = get_cached_qa_pairs()

    if not qa_pairs:
        return {"response": "Không thể tải dữ liệu từ CSV."}

    answer, score = find_best_match(user_message, qa_pairs)
    logger.info(f"🔍 TF-IDF score: {score:.2f} | {user_message}")

    save_chat_to_csv(session_id, "user", user_message)

    if score >= 0.5:
        save_chat_to_csv(session_id, "assistant", answer)
        return {"response": answer, "source": "knowledge_base", "similarity": round(score, 2)}

    try:
        context = payload.messages[-3:] if len(payload.messages) > 3 else payload.messages
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context
        )
        reply = completion.choices[0].message.content
        save_chat_to_csv(session_id, "assistant", reply)
        return {"response": reply, "source": "gpt", "similarity": round(score, 2)}
    except Exception as e:
        logger.error(f"❌ GPT error: {e}")
        return {"response": f"❌ Lỗi khi gọi GPT: {e}", "source": "error"}
