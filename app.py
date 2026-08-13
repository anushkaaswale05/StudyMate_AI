"""
Multi-Agent StudyMate AI
A collaborative multi-agent system for intelligent study assistance
"""

import streamlit as st
import html as html_escape
import time
from datetime import datetime
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Any

# Import your existing modules - FIXED IMPORTS
from rag import ask_question
from summarizer import summarize_topic
from quiz import generate_quiz

# Try importing optional modules with error handling
try:
    from chroma_store import ChromaStore
except ImportError:
    # If ChromaStore doesn't exist, create a placeholder
    class ChromaStore:
        def __init__(self, *args, **kwargs):
            pass

try:
    from embeddings import EmbeddingManager
except ImportError:
    class EmbeddingManager:
        def __init__(self, *args, **kwargs):
            pass

try:
    from loader import DocumentLoader
except ImportError:
    class DocumentLoader:
        def __init__(self, *args, **kwargs):
            pass

try:
    from splitter import TextSplitter
except ImportError:
    class TextSplitter:
        def __init__(self, *args, **kwargs):
            pass

try:
    from retriever import Retriever
except ImportError:
    class Retriever:
        def __init__(self, *args, **kwargs):
            pass


# ============================================================
# PDF SUPPORT
# ============================================================

PDF_SUPPORT = False
LANGCHAIN_SUPPORT = False

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    pass

try:
    from langchain_community.document_loaders import PyPDFLoader
    LANGCHAIN_SUPPORT = True
except ImportError:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        LANGCHAIN_SUPPORT = True
    except ImportError:
        pass


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Multi-Agent StudyMate AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MULTI-AGENT SYSTEM
# ============================================================

class StudyAgent:
    """Base class for all study agents"""
    
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.status = "idle"
        self.last_response = None
        self.execution_time = 0
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Process query and return response"""
        raise NotImplementedError
    
    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "execution_time": self.execution_time
        }


class QuestionAgent(StudyAgent):
    """Specialized agent for answering questions using RAG"""
    
    def __init__(self):
        super().__init__("QA Specialist", "Answer questions using RAG from uploaded documents")
    
    def process(self, query: str, context: Dict = None) -> Dict:
        self.status = "processing"
        start_time = time.time()
        
        try:
            # Use the RAG system to get answer
            answer = ask_question(query)
            
            self.last_response = {
                "type": "answer",
                "content": answer if answer else "I couldn't find an answer in your study material.",
                "confidence": "high" if answer and len(answer) > 20 else "low",
                "source": "RAG System",
                "query": query
            }
            self.status = "completed"
            self.execution_time = time.time() - start_time
            return self.last_response
            
        except Exception as e:
            self.status = "error"
            self.execution_time = time.time() - start_time
            return {"error": str(e), "type": "error"}


class SummarizationAgent(StudyAgent):
    """Specialized agent for creating summaries"""
    
    def __init__(self):
        super().__init__("Summarizer", "Create concise summaries from documents")
    
    def process(self, query: str, context: Dict = None) -> Dict:
        self.status = "processing"
        start_time = time.time()
        
        try:
            topic = context.get("topic", query) if context else query
            style = context.get("style", "Detailed") if context else "Detailed"
            
            summary = summarize_topic(topic)
            
            self.last_response = {
                "type": "summary",
                "content": summary if summary else "No summary was generated.",
                "style": style,
                "word_count": len(summary.split()) if summary else 0,
                "topic": topic
            }
            self.status = "completed"
            self.execution_time = time.time() - start_time
            return self.last_response
            
        except Exception as e:
            self.status = "error"
            self.execution_time = time.time() - start_time
            return {"error": str(e), "type": "error"}


class QuizAgent(StudyAgent):
    """Specialized agent for generating quizzes"""
    
    def __init__(self):
        super().__init__("Quiz Master", "Generate practice quizzes from content")
    
    def process(self, query: str, context: Dict = None) -> Dict:
        self.status = "processing"
        start_time = time.time()
        
        try:
            topic = context.get("topic", query) if context else query
            num_questions = context.get("num_questions", 5) if context else 5
            
            questions = generate_quiz(topic, num_questions)
            
            self.last_response = {
                "type": "quiz",
                "content": questions,
                "num_questions": len(questions) if questions else 0,
                "topic": topic
            }
            self.status = "completed"
            self.execution_time = time.time() - start_time
            return self.last_response
            
        except Exception as e:
            self.status = "error"
            self.execution_time = time.time() - start_time
            return {"error": str(e), "type": "error"}


class ExplanationAgent(StudyAgent):
    """Specialized agent for explaining concepts"""
    
    def __init__(self):
        super().__init__("Explainer", "Explain complex concepts in simple terms")
    
    def process(self, query: str, context: Dict = None) -> Dict:
        self.status = "processing"
        start_time = time.time()
        
        try:
            concept = context.get("concept", query) if context else query
            level = context.get("level", "Simple") if context else "Simple"
            
            prompt = f"""
            Explain the concept '{concept}' at {level} level.
            Use the uploaded study notes as primary source.
            
            Format your response as:
            
            📚 SIMPLE DEFINITION:
            [Provide a clear, concise definition]
            
            💡 MAIN IDEA:
            [Explain the core concept]
            
            🔑 KEY POINTS:
            [List 3-5 important points]
            
            📝 EXAMPLE:
            [Provide a relevant example]
            
            ✍️ EXAM ANSWER:
            [Give a short exam-style answer]
            """
            
            explanation = ask_question(prompt)
            
            self.last_response = {
                "type": "explanation",
                "content": explanation if explanation else "I couldn't find information about this concept.",
                "level": level,
                "concept": concept
            }
            self.status = "completed"
            self.execution_time = time.time() - start_time
            return self.last_response
            
        except Exception as e:
            self.status = "error"
            self.execution_time = time.time() - start_time
            return {"error": str(e), "type": "error"}


class ResearchAgent(StudyAgent):
    """New agent for research and deep analysis"""
    
    def __init__(self):
        super().__init__("Research Analyst", "Deep research and analysis")
    
    def process(self, query: str, context: Dict = None) -> Dict:
        self.status = "processing"
        start_time = time.time()
        
        try:
            research_depth = context.get("depth", "Standard") if context else "Standard"
            
            prompt = f"""
            Conduct a research analysis on: {query}
            
            Research Depth: {research_depth}
            
            Provide:
            1. Executive Summary
            2. Key Findings
            3. Supporting Evidence from the notes
            4. Conclusions
            5. Further Questions to Explore
            """
            
            analysis = ask_question(prompt)
            
            self.last_response = {
                "type": "research",
                "content": analysis if analysis else "No research analysis could be generated.",
                "depth": research_depth,
                "query": query
            }
            self.status = "completed"
            self.execution_time = time.time() - start_time
            return self.last_response
            
        except Exception as e:
            self.status = "error"
            self.execution_time = time.time() - start_time
            return {"error": str(e), "type": "error"}


class OrchestratorAgent:
    """Orchestrator that coordinates multiple agents"""
    
    def __init__(self):
        self.agents = {
            "question": QuestionAgent(),
            "summarize": SummarizationAgent(),
            "quiz": QuizAgent(),
            "explain": ExplanationAgent(),
            "research": ResearchAgent()
        }
        self.execution_history = []
        self.total_executions = 0
        self.successful_executions = 0
    
    def execute(self, agent_type: str, query: str, context: Dict = None) -> Dict:
        """Execute a specific agent"""
        if agent_type not in self.agents:
            return {"error": f"Agent '{agent_type}' not found", "type": "error"}
        
        agent = self.agents[agent_type]
        
        # Execute the agent
        result = agent.process(query, context)
        
        # Record execution
        self.total_executions += 1
        if result.get("type") != "error":
            self.successful_executions += 1
        
        self.execution_history.append({
            "agent": agent_type,
            "timestamp": datetime.now().isoformat(),
            "duration": agent.execution_time,
            "status": agent.status,
            "query": query[:50] + "..." if len(query) > 50 else query
        })
        
        return result
    
    def execute_parallel(self, tasks: List[Dict]) -> List[Dict]:
        """Execute multiple agents in sequence"""
        results = []
        for task in tasks:
            agent_type = task.get("agent")
            query = task.get("query")
            context = task.get("context", {})
            
            result = self.execute(agent_type, query, context)
            results.append({
                "agent": agent_type,
                "query": query,
                "result": result
            })
        
        return results
    
    def get_agent_status(self) -> Dict:
        """Get status of all agents"""
        return {name: agent.get_status() for name, agent in self.agents.items()}
    
    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        if self.total_executions == 0:
            return {
                "total_executions": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "agent_usage": {}
            }
        
        agent_usage = {}
        for history in self.execution_history:
            agent = history["agent"]
            agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        avg_duration = sum(h["duration"] for h in self.execution_history) / len(self.execution_history)
        
        return {
            "total_executions": self.total_executions,
            "success_rate": (self.successful_executions / self.total_executions * 100),
            "avg_duration": avg_duration,
            "agent_usage": agent_usage
        }


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "selected_feature": "Chat",
    "messages": [],
    "quiz_questions": [],
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_score": 0,
    "quiz_topic": "",
    "summary_result": "",
    "explanation_result": "",
    "theme": "dark",
    "font_size": "medium",
    "show_timestamps": True,
    "study_sessions": 0,
    "total_questions_asked": 0,
    "total_quizzes_taken": 0,
    "uploaded_pdfs": [],
    "current_pdf_content": "",
    "pdf_processed": False,
    "agent_results": [],
    "show_agent_dashboard": False,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# INITIALIZE ORCHESTRATOR
# ============================================================

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = OrchestratorAgent()


# ============================================================
# HELPERS
# ============================================================

def render_html(content):
    """Render HTML content safely."""
    st.markdown(content, unsafe_allow_html=True)


def switch_feature(feature):
    st.session_state.selected_feature = feature


def reset_quiz():
    st.session_state.quiz_questions = []
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_score = 0
    st.session_state.quiz_topic = ""


def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        if LANGCHAIN_SUPPORT:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(pdf_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                text = "\n".join([doc.page_content for doc in documents])
                
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                
                if text.strip():
                    return text
                    
            except Exception as e:
                pass
        
        if PDF_SUPPORT:
            try:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if text.strip():
                    return text
            except Exception as e:
                return None
        
        return None
            
    except Exception as e:
        return None


def format_file_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.1f} {unit}"
        size_in_bytes /= 1024.0
    return f"{size_in_bytes:.1f} TB"


def get_theme_colors():
    if st.session_state.theme == "dark":
        return {
            "bg": "#08080D",
            "bg_secondary": "#101018",
            "bg_card": "rgba(255,255,255,0.035)",
            "text": "#E8E8F0",
            "text_secondary": "#85859A",
            "border": "rgba(255,255,255,0.06)",
        }
    else:
        return {
            "bg": "#F8FAFC",
            "bg_secondary": "#FFFFFF",
            "bg_card": "rgba(255,255,255,0.8)",
            "text": "#1E293B",
            "text_secondary": "#64748B",
            "border": "rgba(0,0,0,0.08)",
        }


def render_agent_status():
    """Render agent status dashboard"""
    orchestrator = st.session_state.orchestrator
    status = orchestrator.get_agent_status()
    
    cols = st.columns(len(status))
    for idx, (name, info) in enumerate(status.items()):
        with cols[idx]:
            status_color = "🟢" if info["status"] == "completed" else "🔴" if info["status"] == "error" else "🟡"
            render_html(f"""
            <div style="background: rgba(255,255,255,0.035); border-radius: 12px; padding: 15px; text-align: center;">
                <div style="font-size: 28px;">{status_color}</div>
                <div style="font-size: 13px; font-weight: 600; margin-top: 6px; color: #E8E8F0;">{info['name']}</div>
                <div style="font-size: 10px; color: #77778D;">{info['role']}</div>
                <div style="font-size: 10px; margin-top: 6px; color: #818CF8;">{info['status']}</div>
                <div style="font-size: 9px; color: #5E5E73; margin-top: 2px;">{info['execution_time']:.2f}s</div>
            </div>
            """)


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

* {
    box-sizing: border-box;
}

.block-container {
    max-width: 1300px;
    padding-top: 1rem;
    padding-bottom: 4rem;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 7px;
}
::-webkit-scrollbar-track {
    background: #101018;
}
::-webkit-scrollbar-thumb {
    background: #4F46E5;
    border-radius: 20px;
}

/* Sidebar */
section[data-testid="stSidebar"] > div {
    padding: 1.3rem 1rem;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 12px;
    font-size: 13px;
    font-weight: 500;
    text-align: left;
    min-height: 45px;
    transition: all 0.2s ease;
    color: #A9A9BE;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,102,241,0.10);
    border-color: rgba(99,102,241,0.16);
    color: #FFFFFF;
}

/* Buttons */
.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(135deg, #6366F1, #7C3AED) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    min-height: 44px;
    font-weight: 600;
    transition: 0.2s ease;
    box-shadow: 0 5px 20px rgba(99,102,241,0.18);
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 9px 28px rgba(99,102,241,0.30);
}

/* Inputs */
.stTextInput input {
    border-radius: 13px !important;
    min-height: 45px;
    background: rgba(255,255,255,0.045) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
}

.stTextInput input::placeholder {
    color: #68687D !important;
}

.stSelectbox > div > div {
    border-radius: 13px;
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.09);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    render_html("""
    <div style="display: flex; align-items: center; gap: 13px; padding: 8px 5px 25px 5px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.06);">
        <div style="width: 47px; height: 47px; border-radius: 14px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #6366F1, #8B5CF6); font-size: 23px; box-shadow: 0 8px 30px rgba(99,102,241,0.3);">🤖</div>
        <div>
            <div style="color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; font-size: 19px; font-weight: 700;">Multi-Agent</div>
            <div style="color: #85859A; font-size: 11px; margin-top: 2px;">AI Study System</div>
        </div>
    </div>
    """)
    
    st.markdown('<div style="font-size: 10px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin: 24px 7px 10px 7px; color: #64647C;">📚 Workspace</div>', unsafe_allow_html=True)
    
    if st.button("💬  Multi-Agent Chat", use_container_width=True):
        switch_feature("Chat")
        st.rerun()
    
    if st.button("📝  Summarize", use_container_width=True):
        switch_feature("Summary")
        st.rerun()
    
    if st.button("❓  Generate Quiz", use_container_width=True):
        switch_feature("Quiz")
        st.rerun()
    
    if st.button("🧠  Explain Concept", use_container_width=True):
        switch_feature("Explain")
        st.rerun()
    
    if st.button("🔬  Research", use_container_width=True):
        switch_feature("Research")
        st.rerun()
    
    if st.button("🤖  Agent Dashboard", use_container_width=True):
        switch_feature("AgentDashboard")
        st.rerun()
    
    st.markdown('<div style="font-size: 10px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin: 24px 7px 10px 7px; color: #64647C;">📄 Upload PDF</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=['pdf'],
        help="Upload a PDF document for multi-agent analysis",
        disabled=not (PDF_SUPPORT or LANGCHAIN_SUPPORT)
    )
    
    if uploaded_file is not None:
        if st.button("📤 Process PDF", use_container_width=True):
            with st.spinner("📄 Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    st.session_state.current_pdf_content = text
                    st.session_state.pdf_processed = True
                    
                    file_info = {
                        "name": uploaded_file.name,
                        "size": format_file_size(uploaded_file.size),
                        "pages": text.count('\n\n') + 1,
                        "words": len(text.split()),
                        "uploaded": datetime.now().strftime("%B %d, %Y %H:%M")
                    }
                    st.session_state.uploaded_pdfs.append(file_info)
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    st.rerun()
    
    if st.session_state.uploaded_pdfs:
        st.markdown('<div style="font-size: 10px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin: 24px 7px 10px 7px; color: #64647C;">📁 Uploaded Files</div>', unsafe_allow_html=True)
        for pdf in st.session_state.uploaded_pdfs:
            render_html(f"""
            <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 13px; padding: 13px 14px; margin-bottom: 9px;">
                <div style="color: #E8E8F0; font-size: 12px; font-weight: 600;">📄 {pdf['name']}</div>
                <div style="color: #77778D; font-size: 10px; margin-top: 5px;">
                    <span style="display: inline-block; width: 7px; height: 7px; background: #34D399; border-radius: 50%; margin-right: 5px;"></span>
                    {pdf['pages']} pages • {pdf['words']} words • {pdf['size']}
                </div>
            </div>
            """)
    
    st.markdown('<div style="font-size: 10px; font-weight: 800; letter-spacing: 1.5px; text-transform: uppercase; margin: 24px 7px 10px 7px; color: #64647C;">⚙️ Settings</div>', unsafe_allow_html=True)
    
    theme_label = "☀️ Light" if st.session_state.theme == "dark" else "🌙 Dark"
    if st.button(f"{theme_label} Mode", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.session_state.show_timestamps = st.checkbox(
        "Show Timestamps",
        value=st.session_state.show_timestamps
    )


# ============================================================
# MAIN CONTENT
# ============================================================

# Header
render_html("""
<div style="display: flex; align-items: center; justify-content: space-between; padding: 18px 0 25px 0; border-bottom: 1px solid rgba(255,255,255,0.06); margin-bottom: 28px;">
    <div style="display: flex; align-items: center; gap: 14px;">
        <div style="width: 51px; height: 51px; border-radius: 15px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #6366F1, #8B5CF6); font-size: 25px; box-shadow: 0 8px 28px rgba(99,102,241,0.25);">🤖</div>
        <div>
            <div style="color: #FFFFFF; font-family: 'Space Grotesk', sans-serif; font-size: 25px; font-weight: 700;">Multi-Agent Study System</div>
            <div style="color: #85859A; font-size: 12px; margin-top: 3px;">5 specialized AI agents working together</div>
        </div>
    </div>
    <div style="color: #34D399; background: rgba(16,185,129,0.10); border: 1px solid rgba(16,185,129,0.18); border-radius: 999px; padding: 8px 14px; font-size: 11px; font-weight: 600;">● System Active</div>
</div>
""")


# ============================================================
# STATS
# ============================================================

orchestrator = st.session_state.orchestrator
stats = orchestrator.get_statistics()

st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 16px; margin: 20px 0;">
    <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 28px; font-weight: 700; color: #818CF8;">{stats['total_executions']}</div>
        <div style="font-size: 11px; color: #77778D; margin-top: 4px;">Total Executions</div>
    </div>
    <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 28px; font-weight: 700; color: #34D399;">{stats['success_rate']:.1f}%</div>
        <div style="font-size: 11px; color: #77778D; margin-top: 4px;">Success Rate</div>
    </div>
    <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 28px; font-weight: 700; color: #60A5FA;">{stats['avg_duration']:.2f}s</div>
        <div style="font-size: 11px; color: #77778D; margin-top: 4px;">Avg Response Time</div>
    </div>
    <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 28px; font-weight: 700; color: #FBBF24;">{len(st.session_state.uploaded_pdfs)}</div>
        <div style="font-size: 11px; color: #77778D; margin-top: 4px;">PDFs Uploaded</div>
    </div>
    <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); border-radius: 14px; padding: 18px; text-align: center;">
        <div style="font-size: 28px; font-weight: 700; color: #A78BFA;">{len(orchestrator.agents)}</div>
        <div style="font-size: 11px; color: #77778D; margin-top: 4px;">Active Agents</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# FEATURE SELECTION
# ============================================================

if st.session_state.selected_feature == "Chat":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">💬 Multi-Agent Chat</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Select an agent to handle your query</div>', unsafe_allow_html=True)
    
    # Agent selector with icons
    agent_type = st.selectbox(
        "Select Agent",
        ["question", "summarize", "quiz", "explain", "research"],
        format_func=lambda x: {
            "question": "💬 Question Answerer",
            "summarize": "📝 Summarizer",
            "quiz": "❓ Quiz Generator",
            "explain": "🧠 Explainer",
            "research": "🔬 Research Analyst"
        }.get(x, x)
    )
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            safe_text = html_escape.escape(str(message["content"]))
            timestamp = f"<div style='font-size:9px;color:#77778D;text-align:right;margin-top:4px;'>{message.get('timestamp', '')}</div>" if st.session_state.show_timestamps else ""
            render_html(f'<div style="background: linear-gradient(135deg, #6366F1, #7C3AED); color: white; border-radius: 18px 18px 4px 18px; padding: 14px 18px; max-width: 78%; margin: 10px 0 10px auto; line-height: 1.6;">{safe_text}{timestamp}</div>')
        else:
            timestamp = f"<div style='font-size:9px;color:#77778D;text-align:right;margin-top:8px;'>{message.get('timestamp', '')}</div>" if st.session_state.show_timestamps else ""
            agent_name = message.get("agent", "StudyMate AI")
            render_html(f"""
            <div style="background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.06); color: #E8E8F0; border-radius: 18px 18px 18px 4px; padding: 18px 20px; max-width: 88%; margin: 10px auto 10px 0; line-height: 1.7;">
                <div style="color: #818CF8; font-size: 11px; font-weight: 700; margin-bottom: 8px;">🤖 {agent_name}</div>
                {message["content"]}
                {timestamp}
            </div>
            """)
    
    # Chat form
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_button = st.columns([5, 1])
        with col_input:
            question = st.text_input(
                "Question",
                placeholder=f"Ask the {agent_type} agent something...",
                label_visibility="collapsed"
            )
        with col_button:
            ask_clicked = st.form_submit_button("Ask ✨", use_container_width=True)
    
    if ask_clicked and question.strip():
        timestamp = datetime.now().strftime("%H:%M") if st.session_state.show_timestamps else ""
        st.session_state.messages.append({"role": "user", "content": question, "timestamp": timestamp})
        
        with st.spinner(f"🤖 {agent_type.capitalize()} Agent is processing..."):
            try:
                context = {}
                if agent_type == "summarize":
                    context = {"topic": question, "style": "Detailed"}
                elif agent_type == "quiz":
                    context = {"topic": question, "num_questions": 5}
                elif agent_type == "explain":
                    context = {"concept": question, "level": "Simple"}
                elif agent_type == "research":
                    context = {"depth": "Standard"}
                
                result = st.session_state.orchestrator.execute(agent_type, question, context)
                
                if result.get("type") == "error":
                    answer = f"❌ Error: {result['error']}"
                else:
                    answer = result.get("content", "No response generated")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": str(answer),
                    "timestamp": timestamp,
                    "agent": f"{agent_type.capitalize()} Agent"
                })
                st.rerun()
            except Exception as e:
                st.error(f"Agent error: {e}")


# ============================================================
# AGENT DASHBOARD
# ============================================================

elif st.session_state.selected_feature == "AgentDashboard":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">🤖 Multi-Agent Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Monitor and control all AI agents in the system</div>', unsafe_allow_html=True)
    
    # Agent Status
    st.markdown("### 🟢 Agent Status")
    render_agent_status()
    
    # Execution Stats
    st.markdown("### 📊 Execution Statistics")
    stats = st.session_state.orchestrator.get_statistics()
    
    if stats["total_executions"] > 0:
        render_html(f"""
        <div style="background: rgba(255,255,255,0.035); border-radius: 12px; padding: 16px; margin: 10px 0;">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 700; color: #818CF8;">{stats['total_executions']}</div>
                    <div style="font-size: 10px; color: #77778D;">Total Executions</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 700; color: #34D399;">{stats['success_rate']:.1f}%</div>
                    <div style="font-size: 10px; color: #77778D;">Success Rate</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 700; color: #F87171;">{stats['total_executions'] - int(stats['success_rate'] * stats['total_executions'] / 100)}</div>
                    <div style="font-size: 10px; color: #77778D;">Errors</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 20px; font-weight: 700; color: #60A5FA;">{stats['avg_duration']:.2f}s</div>
                    <div style="font-size: 10px; color: #77778D;">Avg Duration</div>
                </div>
            </div>
        </div>
        """)
    
    # Agent History
    st.markdown("### 📜 Execution History")
    history = st.session_state.orchestrator.execution_history
    
    if history:
        for h in history[-10:]:  # Show last 10
            status_icon = "✅" if h["status"] == "completed" else "❌" if h["status"] == "error" else "⏳"
            render_html(f"""
            <div style="background: rgba(255,255,255,0.035); border-radius: 8px; padding: 10px 14px; margin: 4px 0; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 600; color: #E8E8F0;">{h['agent'].capitalize()}</span>
                    <span style="color: #77778D; font-size: 11px; margin-left: 10px;">{h['query']}</span>
                </div>
                <div>
                    <span style="color: #77778D; font-size: 11px;">{h['duration']:.2f}s</span>
                    <span style="margin-left: 10px;">{status_icon}</span>
                </div>
            </div>
            """)
    else:
        st.info("No execution history yet. Start using agents to see history.")
    
    # Control buttons
    st.markdown("### ⚡ Control")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Reset All Agents", use_container_width=True):
            st.session_state.orchestrator = OrchestratorAgent()
            st.session_state.agent_results = []
            st.rerun()
    with col2:
        if st.button("🧹 Clear Messages", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ============================================================
# SUMMARY
# ============================================================

elif st.session_state.selected_feature == "Summary":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">📝 Summarize with AI</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Get a concise summary of any topic from your documents</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_pdfs:
        st.info("📄 Please upload a PDF document first to generate summaries.")
    
    summary_topic = st.text_input("Topic", placeholder="Enter a topic to summarize...")
    summary_style = st.selectbox("Summary Style", ["Quick Revision", "Detailed Study Notes", "Exam-Focused Points"])
    
    if st.button("✨ Generate Summary", use_container_width=True):
        if not summary_topic.strip():
            st.warning("Please enter a topic first.")
        elif not st.session_state.uploaded_pdfs:
            st.warning("Please upload a PDF document first.")
        else:
            with st.spinner("🤖 Summarizer Agent is working..."):
                result = st.session_state.orchestrator.execute("summarize", summary_topic, {"topic": summary_topic, "style": summary_style})
                if result.get("type") != "error":
                    st.markdown(f"<div style='background: rgba(255,255,255,0.035); border-radius: 18px; padding: 24px; line-height: 1.8; margin-top: 18px; color: #E8E8F0;'>{result['content']}</div>", unsafe_allow_html=True)


# ============================================================
# QUIZ
# ============================================================

elif st.session_state.selected_feature == "Quiz":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">❓ Generate Quiz</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Test your knowledge with AI-generated questions</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_pdfs:
        st.info("📄 Please upload a PDF document first to generate quizzes.")
    
    if not st.session_state.quiz_questions:
        quiz_topic = st.text_input("Quiz Topic", placeholder="Enter a topic for the quiz...")
        num_questions = st.selectbox("Number of Questions", [5, 10, 15])
        
        if st.button("✨ Generate Quiz", use_container_width=True):
            if not quiz_topic.strip():
                st.warning("Please enter a quiz topic.")
            elif not st.session_state.uploaded_pdfs:
                st.warning("Please upload a PDF document first.")
            else:
                with st.spinner("🤖 Quiz Master is creating questions..."):
                    result = st.session_state.orchestrator.execute("quiz", quiz_topic, {"topic": quiz_topic, "num_questions": num_questions})
                    if result.get("type") != "error" and result.get('content'):
                        st.session_state.quiz_questions = result['content']
                        st.session_state.quiz_topic = quiz_topic
                        st.session_state.quiz_answers = {}
                        st.session_state.quiz_submitted = False
                        st.session_state.quiz_score = 0
                        st.rerun()
    
    # Display quiz (your existing quiz display code)
    if st.session_state.quiz_questions:
        # ... (keep your existing quiz display logic)
        pass


# ============================================================
# EXPLAIN
# ============================================================

elif st.session_state.selected_feature == "Explain":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">🧠 Explain Concept</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Get a clear explanation of any concept from your documents</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_pdfs:
        st.info("📄 Please upload a PDF document first to get explanations.")
    
    concept = st.text_input("Concept", placeholder="Enter a concept to explain...")
    level = st.selectbox("Explanation Level", ["Simple", "Detailed", "Exam-Focused"])
    
    if st.button("🧠 Explain Concept", use_container_width=True):
        if not concept.strip():
            st.warning("Please enter a concept.")
        elif not st.session_state.uploaded_pdfs:
            st.warning("Please upload a PDF document first.")
        else:
            with st.spinner("🤖 Explainer Agent is working..."):
                result = st.session_state.orchestrator.execute("explain", concept, {"concept": concept, "level": level})
                if result.get("type") != "error":
                    st.markdown(f"<div style='background: rgba(255,255,255,0.035); border-radius: 18px; padding: 24px; line-height: 1.8; margin-top: 18px; color: #E8E8F0;'>{result['content']}</div>", unsafe_allow_html=True)


# ============================================================
# RESEARCH
# ============================================================

elif st.session_state.selected_feature == "Research":
    st.markdown('<div style="color: #FFFFFF; font-family: Space Grotesk, sans-serif; font-size: 21px; font-weight: 700; margin-top: 28px; margin-bottom: 5px;">🔬 Research & Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div style="color: #77778D; font-size: 12px; margin-bottom: 18px;">Deep research and analysis using your documents</div>', unsafe_allow_html=True)
    
    if not st.session_state.uploaded_pdfs:
        st.info("📄 Please upload a PDF document first to conduct research.")
    
    research_topic = st.text_input("Research Topic", placeholder="Enter a topic for in-depth research...")
    research_depth = st.selectbox("Research Depth", ["Quick Overview", "Standard", "Deep Analysis"])
    
    if st.button("🔬 Conduct Research", use_container_width=True):
        if not research_topic.strip():
            st.warning("Please enter a research topic.")
        elif not st.session_state.uploaded_pdfs:
            st.warning("Please upload a PDF document first.")
        else:
            with st.spinner("🔬 Research Analyst is analyzing..."):
                result = st.session_state.orchestrator.execute("research", research_topic, {"depth": research_depth})
                if result.get("type") != "error":
                    st.markdown(f"<div style='background: rgba(255,255,255,0.035); border-radius: 18px; padding: 24px; line-height: 1.8; margin-top: 18px; color: #E8E8F0;'>{result['content']}</div>", unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div style="text-align: center; font-size: 11px; margin-top: 60px; padding-top: 22px; border-top: 1px solid rgba(255,255,255,0.05); color: #5E5E73;">
    <span style="color: #818CF8; font-weight: 700;">🤖 Multi-Agent StudyMate AI</span>
    <span>• Powered by RAG + 5 Specialized Agents</span>
    <br>
    <span style="font-size: 11px; color: #5E5E73;">
        QA • Summarizer • Quiz Master • Explainer • Research Analyst
        {f'• {datetime.now().strftime("%B %d, %Y")}'}
    </span>
</div>
""", unsafe_allow_html=True)