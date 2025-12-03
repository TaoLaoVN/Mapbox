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

# --- 1. CALLBACK STREAMLIT ---
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
        display_out = str(output)
        if len(display_out) > 200:
            display_out = display_out[:200] + "..."
        self._append_text(f"   > Result: {display_out}\n")

    def on_agent_action(self, action, **kwargs):
        self._append_text(f"\n[⚡ ACTION]: {action.tool}\n")
        
    def on_chain_end(self, outputs, **kwargs):
        self._append_text(f"\n[✅ DONE]: Hoàn thành bước xử lý.\n")


# ------------------------------
# 2. STREAMLIT UI
# ------------------------------
st.set_page_config(page_title="AI Arena", layout="wide")
st.title("🤖 Autonomous AI Agents Arena")

with st.sidebar:
    st.header("Cấu hình")
    google_key = st.text_input("Google Gemini Key:", type="password")
    topic = st.text_input("Chủ đề:", "Tương lai của AI")
    start_btn = st.button("🚀 Chạy ngay")


# ------------------------------
# 3. MAIN LOGIC
# ------------------------------
if start_btn and google_key:
    st.subheader("🖥️ Terminal Output")
    terminal_placeholder = st.empty()
    
    # Callback
    st_callback = StreamlitCallbackHandler(terminal_placeholder)

    # Google LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_key,
        temperature=0.7
    )

    # Search Tool
    search_tool = DuckDuckGoSearchRun()

    # Agents
    researcher = Agent(
        role='Researcher',
        goal=f'Tìm thông tin mới nhất về {topic}',
        backstory='Chuyên gia điều tra, tìm kiếm thông tin.',
        tools=[search_tool],
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        role='Writer',
        goal=f'Viết bài báo súc tích về {topic}',
        backstory='Nhà văn chuyên tổng hợp thông tin.',
        allow_delegation=False,
        verbose=True,
    )

    # Tasks
    task1 = Task(
        description=f"Tìm kiếm và trích xuất các thông tin quan trọng nhất về: {topic}.",
        expected_output="Danh sách bullet gọn gàng.",
        agent=researcher
    )

    task2 = Task(
        description="Dùng kết quả của task trước để viết 1 đoạn văn 100 từ.",
        expected_output="Đoạn văn hoàn chỉnh.",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        llm=llm,
        callbacks=[st_callback]
    )

    with st.spinner('Đang chạy...'):
        try:
            result = crew.kickoff()
            st.success("Hoàn thành!")
            st.markdown("### 📝 Kết quả:")
            
            if hasattr(result, "output"):
                st.write(result.output)
            else:
                st.write(str(result))

        except Exception as e:
            st.error(f"Lỗi: {e}")


elif start_btn and not google_key:
    st.error("Vui lòng nhập Google API Key!")
