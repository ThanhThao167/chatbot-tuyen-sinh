# 🤖 Chatbot Tư Vấn Tuyển Sinh

Giải đáp các thắc mắc thường gặp của học sinh và phụ huynh trong công tác tuyển sinh lớp 10 tại trường THPT Marie Curie.

## 📦 Cấu trúc dự án

- `app_csv_based.py`: API FastAPI để xử lý câu hỏi (dùng TF-IDF và GPT)
- `MC_chatbot.csv`: Dữ liệu câu hỏi - trả lời
- `chat_history.csv`: File lịch sử hội thoại được ghi lại
- `streamlit_chat.py`: Giao diện chatbot Streamlit
- `requirements.txt`: Các thư viện cần thiết để chạy app trên Streamlit Cloud

## 🚀 Cách triển khai trên [Streamlit Cloud](https://streamlit.io/cloud)

1. Fork hoặc push repo này lên GitHub
2. Truy cập [https://streamlit.io/cloud](https://streamlit.io/cloud) → Connect repo
3. Chọn file chính: `streamlit_chat.py`
4. Thêm biến môi trường:  
   - `OPENAI_API_KEY`: 🔑 API key từ [https://platform.openai.com/account/api-keys](https://platform.openai.com/account/api-keys)

## 🧪 Demo Local

```bash
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy FastAPI
uvicorn app_csv_based:app --reload

# Chạy giao diện
streamlit run streamlit_chat.py
```