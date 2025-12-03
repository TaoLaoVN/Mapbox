__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
import time
import os

# --- 1. SETUP CLASS CALLBACK ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = "SYSTEM_LOG_INIT: Waiting for agents...\n"

    def _append_text(self, text):
        self.text += text
        self.container.code(self.text, language="bash")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._append_text(f"\n[🧠 THINKING]: Agent đang phân tích...\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._append_text(f"\n[🛠️ TOOL]: {serialized.get('name')} > {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        display_out = output[:200] + "..." if len(output) > 200 else output
        self._append_text(f"   > Result: {display_out}\n")

    def on_agent_action(self, action, **kwargs):
        self._append_text(f"\n[⚡ ACTION]: {action.tool}\n")
        
    def on_chain_end(self, outputs, **kwargs):
        self._append_text(f"\n[✅ DONE]: Hoàn thành bước xử lý.\n")

# --- 2. STREAMLIT APP UI ---
st.set_page_config(page_title="AI Arena", layout="wide")
st.title("🤖 Autonomous AI Agents Arena")

# Input Key và Chủ đề
with st.sidebar:
    st.header("Cấu hình")
    # Nếu bạn có key trong st.secrets thì dùng, không thì hiện ô nhập
    google_key = st.text_input("Google Gemini Key:", type="password")
    topic = st.text_input("Chủ đề:", "Tương lai của AI")
    start_btn = st.button("🚀 Chạy ngay")

# --- 3. MAIN LOGIC ---
if start_btn and google_key:
    st.subheader("🖥️ Terminal Output")
    terminal_placeholder = st.empty()
    
    # Khởi tạo Callback
    st_callback = StreamlitCallbackHandler(terminal_placeholder)

    # 1. Khởi tạo LLM (Google Gemini)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_key,
        temperature=0.7,
        callbacks=[st_callback]
    )

    # 2. Khởi tạo Tool (DuckDuckGo) - Đã sửa lỗi thụt đầu dòng ở đây
    search_tool = DuckDuckGoSearchRun()

    # 3. Định nghĩa Agents
    researcher = Agent(
        role='Researcher',
        goal=f'Tìm kiếm thông tin về {topic}',
        backstory='Chuyên gia tìm kiếm thông tin.',
        tools=[search_tool],
        llm=llm,
        verbose=True,
        callbacks=[st_callback]
    )

    writer = Agent(
        role='Writer',
        goal=f'Viết bài ngắn về {topic}',
        backstory='Nhà văn viết nội dung tóm tắt súc tích.',
        llm=llm,
        verbose=True,
        callbacks=[st_callback]
    )

    # 4. Định nghĩa Tasks
    task1 = Task(
        description=f"Tìm kiếm thông tin quan trọng nhất về: {topic}",
        expected_output="Gạch đầu dòng các ý chính.",
        agent=researcher
    )

    task2 = Task(
        description="Tổng hợp thông tin trên thành một đoạn văn ngắn.",
        expected_output="Đoạn văn khoảng 100 từ.",
        agent=writer
    )

    # 5. Chạy Crew
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential
    )

    with st.spinner('Đang chạy...'):
        try:
            result = crew.kickoff()
            st.success("Hoàn thành!")
            st.markdown("### 📝 Kết quả:")
            st.write(result)
        except Exception as e:
            st.error(f"Lỗi: {e}")

elif start_btn and not google_key:
    st.error("Vui lòng nhập Google API Key!")
