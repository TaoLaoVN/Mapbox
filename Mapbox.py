import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
import time
import os

# --- 1. SETUP CLASS CALLBACK (PHẦN QUAN TRỌNG NHẤT) ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = "SYSTEM_LOG_INIT: Waiting for agents...\n"
        self.last_update = time.time()

    def _append_text(self, text):
        """Hàm phụ trợ để thêm text và update UI"""
        self.text += text
        # Update UI: Dùng st.code để tạo cảm giác Terminal hacker
        self.container.code(self.text, language="bash")

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Kích hoạt khi LLM bắt đầu suy nghĩ"""
        self._append_text(f"\n[🧠 THINKING]: Agent đang phân tích yêu cầu...\n")

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Kích hoạt khi Agent gọi Tool"""
        self._append_text(f"\n[🛠️ TOOL USE]: Đang sử dụng công cụ: {serialized.get('name')}")
        self._append_text(f"\n   > Params: {input_str}\n")

    def on_tool_end(self, output, **kwargs):
        """Kích hoạt khi Tool trả về kết quả"""
        # Cắt ngắn nếu output quá dài để đỡ rối mắt
        display_out = output[:200] + "..." if len(output) > 200 else output
        self._append_text(f"   > Result: {display_out}\n")

    def on_agent_action(self, action, **kwargs):
        """Kích hoạt khi Agent quyết định hành động"""
        self._append_text(f"\n[⚡ ACTION]: {action.tool} -> {action.tool_input}\n")
        
    def on_chain_end(self, outputs, **kwargs):
        """Kết thúc chuỗi xử lý"""
        self._append_text(f"\n[✅ FINISHED]: Hoàn thành tác vụ.\n")

# --- 2. STREAMLIT APP UI ---
st.set_page_config(page_title="Agent Terminal", layout="wide", page_icon="🤖")

st.markdown("""
<style>
    /* CSS hack để terminal trông ngầu hơn (nền đen, chữ xanh) */
    .stCodeBlock {
        border: 1px solid #00ff41;
        box-shadow: 0 0 10px #00ff41;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ Autonomous Agent: Live Terminal")
st.caption("Quan sát suy nghĩ của AI theo thời gian thực (Real-time Logs)")

# Input từ người dùng
col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("Nhiệm vụ cho Agent:", "Tìm hiểu giá Bitcoin hôm nay và phân tích xu hướng.")
with col2:
    start_btn = st.button("🚀 EXECUTE", type="primary", use_container_width=True)

# --- 3. CORE LOGIC ---
if start_btn:
    # A. Tạo khu vực hiển thị Log (Terminal)
    st.subheader("🖥️ Terminal Output")
    terminal_placeholder = st.empty() # Khung chứa nội dung sẽ thay đổi liên tục
    
    # Khởi tạo Callback Handler và truyền cái khung placeholder vào đó
    st_callback = StreamlitCallbackHandler(terminal_placeholder)

# 2. Khởi tạo LLM Google Gemini
# model="gemini-1.5-flash" là bản nhanh, rẻ (miễn phí mức cơ bản) và thông minh
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    verbose=True,
    temperature=0.5,
    google_api_key="AIzaSyDoQeCKhWno2H1aq9N5h2WFhPvkvw9jKxU", # Hoặc lấy từ st.secrets
    callbacks=[st_callback] # <--- QUAN TRỌNG: Vẫn giữ callback để hiện Terminal UI
)
    
    search_tool = DuckDuckGoSearchRun()

    # C. Định nghĩa Agent
    # Lưu ý quan trọng: Phải truyền callbacks vào cả Agent để bắt sự kiện Tool
    researcher = Agent(
        role='Market Analyst',
        goal='Tìm kiếm dữ liệu chính xác',
        backstory='Bạn là chuyên gia phân tích số liệu.',
        tools=[search_tool],
        llm=llm,
        verbose=True, # Bắt buộc True để sinh log
        callbacks=[st_callback] # <--- GẮN CALLBACK VÀO AGENT
    )

    writer = Agent(
        role='Content Writer',
        goal='Viết báo cáo ngắn gọn',
        backstory='Bạn viết báo cáo súc tích, dễ hiểu.',
        llm=llm,
        verbose=True,
        callbacks=[st_callback] # <--- GẮN CALLBACK VÀO AGENT
    )

    # D. Định nghĩa Task
    task1 = Task(
        description=f"Nghiên cứu về: {user_input}",
        expected_output="Danh sách các thông tin chính tìm được.",
        agent=researcher
    )
    
    task2 = Task(
        description="Tổng hợp thông tin từ researcher thành một đoạn văn ngắn.",
        expected_output="Đoạn văn tổng hợp 3 câu.",
        agent=writer
    )

    crew = Crew(
        agents=[researcher, writer],
        tasks=[task1, task2],
        process=Process.sequential
    )

    # E. Chạy (Kickoff)
    with st.spinner('Agents are working...'):
        try:
            final_result = crew.kickoff()
            
            # Hiển thị kết quả cuối cùng ra ngoài Terminal
            st.success("Mission Complete!")
            st.markdown("### 📝 Final Report")
            st.write(final_result)
            
        except Exception as e:
            st.error(f"Error: {e}")


