import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import streamlit as st
from datetime import datetime
from langchain_core.tools import StructuredTool
import time
from module2 import agent,State,secure_input

# Load environment variables FIRST
load_dotenv()
try :
    api_key = os.getenv('GROQ_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')  
except Exception as e:
    print(f"Error loading environment variables: {str(e)}")  
# AGENT CORE LOGIC
max_queries = 15


def create_agent_graph() -> function:
    return agent
# STREAMLIT UI

# Page configuration - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Headers */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0
if 'total_queries' not in st.session_state:
    st.session_state.total_queries = 0
if 'agent' not in st.session_state:
    with st.spinner("🔄 Initializing AI agent..."):
        st.session_state.agent = create_agent_graph()

# Sidebar
with st.sidebar:
    st.markdown("# 🤖 AI Research Assistant")
    st.markdown("---")
    
    st.markdown("### 📊 Statistics")
    col1,col2 = st.columns(2)
    with col1:
        st.metric("Total Queries", st.session_state.total_queries, 
                 delta=None if st.session_state.total_queries == 0 else "+1")
    with col2 :
        remaining = max_queries - st.session_state.total_queries
        st.metric("Remaining", max(0, remaining))
    if st.session_state.total_queries >= max_queries * 0.8:
        st.warning(f"⚠️ Approaching query limit ({remaining} remaining)")
    elif st.session_state.total_queries >= max_queries:
        st.error("❌ **Query limit reached**")
        st.info("Please refresh the page to start a new session or contact support for extended access.")
        st.stop()
    
    
    
    st.markdown("---")
    
    st.markdown("### Available Tools")
    tools_info = {
        "🖥️ Anything_about_python": "All about Python 🐍🖥️",
        "📚 Academic_web_recents_requests": "Scholar and academics researches,Encyclopedia knowledge and definitions",
        "📄 Document_summarizer": "For document summarization"
    }
    
    for tool, description in tools_info.items():
        with st.expander(tool):
            st.write(description)
    
    st.markdown("---")
    
    st.markdown("### ℹ️ How to Use")
    st.info("""
    **Ask me anything!**
    
    - 📰 Latest news,General knowledge and events
    - 🔬 Scientific research
    - 🌍  Python Definitions and concepts
    - 📖  Documents summarization
    
    I'll automatically select the best tool for your query!
    """)
    
    st.markdown("---")
    
    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count += 1
        st.rerun()
    
    st.markdown("---")
    
    # API Status
    st.markdown("### 🔐 API Status")
    groq_status = "✅" if os.getenv("GROQ_API_KEY") else "❌"
    tavily_status = "✅" if os.getenv("TAVILY_API_KEY") else "❌"
    
    st.markdown(f"""
    - GROQ: {groq_status}
    - Tavily: {tavily_status}
    """)

# Main content
st.markdown("# 🤖 ALPHA AI")
st.markdown("### *Your intelligent companion for research,learning and information*")

# Welcome message
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3> Welcome to Alpha AI!</h3>
        <p>I'm here to help you find information, learn Python and summarize documents using multiple powerful tools:</p>
        <ul>
            <li><strong>🌐 Anything_about_python</strong> - Using RAG for efficiency response and code about Python</li>
            <li><strong>📚 Academic_web_recents_requests</strong> - For recents news,academic researche papers,encyclopedic knowledge</li>
            <li><strong>📄 Document_summarizer</strong> - For documents summarization</li>
        </ul>
        <p><strong>Just type your question below to get started! 🚀</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
if prompt := st.chat_input("Ask me anything... 💬"):
    if prompt.len() < 5 :
        st.error("Input too short..")
        
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Researching your query..."):
            try:
                # Create initial state
                initial_state = State(
                    messages=[{"role": "user", "content": prompt}]
                )
                start_time = time.time()
                
                # Invoke agent
                result = st.session_state.agent.invoke(initial_state)
                response_time = time.time() - start_time
                if  response_time > 100 :
                    st.warning("Your query took a long time  maybe something went wrong , Please retry or refresh the page")
                
                response = result['messages'][-1].content
                
               
                # Display response
                st.markdown(response)
                
                # Check if tools were used
                tools_used = []
                for msg in result['messages']:
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        if hasattr(msg, 'name'):
                            tools_used.append(msg.name)
                
                # Display tools used
                if tools_used:
                    st.markdown("---")
                    st.caption(f"🛠️ Tools used: {', '.join(set(tools_used))}")
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🤖 Powered by**")
    st.markdown("LangGraph & Groq")
with col2:
    st.markdown("**📅 Session Started**")
    st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M"))
with col3:
    st.markdown("**💡 Status**")
    st.markdown("🟢 Ready to assist")

# Example queries (collapsible)
with st.expander("💡 Example Queries to Try"):
    st.markdown("""
    **News & Current Events:**
    - What are the latest developments in AI?
    - Recent news about climate change
    
    **Python questions:**
    - What is metaclasses in Python ?
    - How to use decorators in Python ?
    - Explain the GIL in Python
    
    **Document summarization:**
    - Summarize the key points of the attached research paper.
    - Provide a concise summary of the following article.
    """)