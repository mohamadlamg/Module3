import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import streamlit as st
from datetime import datetime
from langchain_core.tools import StructuredTool
import time
from module2 import agent, State, secure_input

# Load environment variables
load_dotenv()
try:
    api_key = os.getenv('GROQ_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')  
except Exception as e:
    print(f"Error loading environment variables: {str(e)}")

# Configuration
MAX_QUERIES = 15

def create_agent_graph():
    """Initialize and return the configured agent"""
    return agent

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Alpha AI - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2a5298;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #2a5298 0%, #1e3c72 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 28px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 24px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #2a5298;
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(30, 60, 114, 0.95);
    }
    
    /* Sidebar text */
    .sidebar .markdown-text-container {
        color: white;
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
    with st.spinner("Initializing AI agent..."):
        st.session_state.agent = create_agent_graph()

# Sidebar
with st.sidebar:
    st.markdown("# Alpha AI")
    st.markdown("### Intelligent Research Assistant")
    st.markdown("---")
    
    # Usage statistics
    st.markdown("### Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        remaining = MAX_QUERIES - st.session_state.total_queries
        st.metric("Remaining", max(0, remaining))
    
    # Limit management
    if st.session_state.total_queries >= MAX_QUERIES:
        st.error("**Query limit reached**")
        st.info("Please refresh the page to start a new session.")
        st.stop()
    elif st.session_state.total_queries >= MAX_QUERIES * 0.8:
        st.warning(f"Warning: {remaining} queries remaining")
    
    st.markdown("---")
    
    # Available tools
    st.markdown("### Available Tools")
    
    with st.expander("Python Expert", expanded=False):
        st.markdown("""
        **Specialized in:**
        - Python concepts and definitions
        - Code examples
        - Best practices
        - Debugging and optimization
        """)
    
    with st.expander("Academic Research", expanded=False):
        st.markdown("""
        **Access to:**
        - Research articles
        - Scientific publications
        - Encyclopedic knowledge
        - Recent news
        """)
    
    with st.expander("Document Analyzer", expanded=False):
        st.markdown("""
        **Capabilities:**
        - Document summaries
        - Key information extraction
        - Content analysis
        """)
    
    st.markdown("---")
    
    # Usage guide
    st.markdown("### Usage Guide")
    st.info("""
    **Areas of expertise:**
    
    • Recent news and events
    • Scientific and academic research
    • Python programming
    • Document analysis
    
    Simply ask your question, the agent will automatically select the best tools.
    """)
    
    st.markdown("---")
    
    # Reset button
    if st.button("New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count += 1
        st.rerun()
    
    st.markdown("---")
    
    # API status
    st.markdown("### Service Status")
    groq_status = "✓ Active" if os.getenv("GROQ_API_KEY") else "✗ Inactive"
    tavily_status = "✓ Active" if os.getenv("TAVILY_API_KEY") else "✗ Inactive"
    
    st.markdown(f"""
    **GROQ:** {groq_status}  
    **Tavily:** {tavily_status}
    """)

# Main content
st.markdown("# Alpha AI")
st.markdown("### Intelligent assistant for research, learning and analysis")

# Welcome message
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3>Welcome to Alpha AI</h3>
        <p>Your intelligent assistant capable of:</p>
        <ul>
            <li><strong>Python Research</strong> - Concepts, code examples and best practices via RAG</li>
            <li><strong>Academic Research</strong> - Access to scientific articles, news and encyclopedic knowledge</li>
            <li><strong>Document Analysis</strong> - Summaries and key information extraction</li>
        </ul>
        <p><strong>Start by asking your question below.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input area
if prompt := st.chat_input("Ask your question..."):
    # Input validation
    if len(prompt.strip()) < 5:
        st.error("Your question is too short. Please provide more details.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query..."):
            try:
                # Create initial state
                initial_state = State(
                    messages=[{"role": "user", "content": prompt}]
                )
                start_time = time.time()
                
                # Invoke agent
                result = st.session_state.agent.invoke(initial_state)
                response_time = time.time() - start_time
                
                # Warning if response time is too long
                if response_time > 100:
                    st.warning("The query took an unusually long time. If you encounter issues, please refresh the page.")
                
                response = result['messages'][-1].content
                
                # Display response
                st.markdown(response)
                
                # Detect tools used
                tools_used = []
                for msg in result['messages']:
                    if hasattr(msg, 'type') and msg.type == 'tool':
                        if hasattr(msg, 'name'):
                            tools_used.append(msg.name)
                
                # Display tools used
                if tools_used:
                    st.markdown("---")
                    unique_tools = list(set(tools_used))
                    tools_display = ", ".join(unique_tools)
                    st.caption(f"Tools used: {tools_display}")
                    st.caption(f"Response time: {response_time:.2f}s")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Powered by**")
    st.markdown("LangGraph & Groq")
with col2:
    st.markdown("**Session Started**")
    st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M"))
with col3:
    st.markdown("**Status**")
    st.markdown("🟢 Operational")

# Example queries
with st.expander("Example Queries"):
    st.markdown("""
    **News & Current Events:**
    - What are the latest developments in artificial intelligence?
    - Recent news about climate change
    
    **Python Questions:**
    - What are metaclasses in Python?
    - How to use decorators in Python?
    - Explain the GIL (Global Interpreter Lock) in Python
    
    **Document Analysis:**
    - Summarize the key points of this research document
    - Provide a concise summary of the following article
    
    **Academic Research:**
    - What are the latest research findings on machine learning?
    - Explain the concept of convolutional neural networks
    """)