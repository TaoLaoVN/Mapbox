__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
from duckduckgo_search import DDGS
from langchain_core.callbacks import BaseCallbackHandler
import os
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- CALLBACK STREAMLIT LOG ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = "SYSTEM_LOG_INIT: Waiting for agents...\n"

    def _append_text(self, text):
        self.text += text
        self.container.code(self.text, language="bash")

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._append_text("\n[🧠 THINKING]: Agent đang phân tích...\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        self._append_text(f"\n[🛠️ TOOL]: {serialized.get('name')} > {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        display = str(output)
        if len(display) > 200:
            display = display[:200] + "..."
        self._append_text(f"   > Result: {display}\n")

    def on_agent_action(self, action, **kwargs):
        self._append_text(f"\n[⚡ ACTION]: {action.tool}\n")

    def on_chain_end(self, outputs, **kwargs):
        self._append_text("\n[✅ DONE]: Hoàn thành bước xử lý.\n")



# --- TOOL SEARCH DUCKDUCKGO CHUẨN CREWAI ---
@tool("duckduckgo_search")
def duckduckgo_search(query: str):
    """Tìm kiếm nhanh bằng DuckDuckGo và trả về text."""
    with DDGS() as ddg:
        results = ddg.text(query, max_results=5)
        out = "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        return out



# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Multi-Agent Arena", layout="wide")
st.title("🤖 Autonomous AI Agents Arena")

with st.sidebar:
    st.header("Cấu hình")
    google_key = st.text_input("Google Gemini API Key:", type="password")
    topic = st.text_input("Chủ đề:", "Tương lai của AI")
    start_btn = st.button("🚀 Chạy ngay")



# ==================================================================
# ===================== MAIN LOGIC =================================
# ==================================================================

if start_btn and google_key:

    # Terminal log output
    st.subheader("🖥️ Terminal Output")
    terminal_placeholder = st.empty()

    callbacks = [StreamlitCallbackHandler(terminal_placeholder)]

    # --- LLM CREWAI (không dùng LangChain) ---
    llm = LLM(
        model="gemini/gemini-1.5-flash",
        api_key=google_key,
        temperature=0.7,
        callbacks=callbacks
    )

    # --- AGENTS ---
    researcher = Agent(
        role="Researcher",
        goal=f"Tìm các dữ liệu quan trọng nhất về chủ đề: {topic}",
        backstory="Chuyên gia nghiên cứu, phân tích thông tin.",
        tools=[duckduckgo_search],
        allow_delegation=False
    )

    writer = Agent(
        role="Writer",
        goal=f"Viết bài súc tích dựa trên thông tin về {topic}",
        backstory="Nhà văn chuyên tóm tắt dữ liệu.",
        allow_delegation=False
    )

    # --- TASKS ---
    task1 = Task(
        description=f"Tìm kiếm 5 thông tin quan trọng nhất về: {topic}.",
        expected_output="Danh sách bullet rõ ràng.",
        agent=researcher
    )

    task2 = Task(
        description="Dựa vào dữ liệu task1, viết đoạn văn khoảng 100 từ.",
        expected_output="Đoạn văn hoàn chỉnh.",
        agent=writer
    )

    # --- CREW ---
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential,
        llm=llm,
        callbacks=callbacks
    )

    with st.spinner("🔄 Đang chạy tác tử..."):
        try:
            result = crew.kickoff()
            st.success("🎉 Hoàn thành!")

            st.markdown("### 📝 Kết quả cuối cùng:")
            st.write(result.output if hasattr(result, "output") else str(result))

        except Exception as e:
            st.error(f"Lỗi xảy ra: {e}")

elif start_btn and not google_key:
    st.error("⚠️ Vui lòng nhập Google API Key!")

