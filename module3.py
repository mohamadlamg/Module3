import os
from dotenv import load_dotenv
from langgraph.graph.message import add_messages
import streamlit as st
from datetime import datetime
from langchain_core.tools import StructuredTool
import time

# Import conditionnel pour éviter l'exécution du main de module2
import sys
if __name__ == '__main__':
    from module2 import agent_assistant_graph, State
    agent = agent_assistant_graph()  # Créer l'agent directement ici
else:
    from module2 import agent_assistant_graph, State
    agent = agent_assistant_graph()

# Load environment variables
load_dotenv()
try:
    api_key = os.getenv('GROQ_API_KEY')
    tavily_api_key = os.getenv('TAVILY_API_KEY')  
except Exception as e:
    print(f"Error loading environment variables: {str(e)}")

# Configuration
MAX_QUERIES = 15

# Configuration de la page - DOIT ÊTRE LA PREMIÈRE COMMANDE STREAMLIT
st.set_page_config(
    page_title="Alpha AI - Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS Styles
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
    
    /* Tool usage badge */
    .tool-badge {
        display: inline-block;
        background: #2a5298;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85em;
        margin: 2px;
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
if 'uploaded_file_content' not in st.session_state:
    st.session_state.uploaded_file_content = None
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'quick_action' not in st.session_state:
    st.session_state.quick_action = None

# *** CORRECTION PRINCIPALE : Initialisation de l'agent ***
if 'agent' not in st.session_state:
    try:
        # L'agent est déjà créé lors de l'import
        st.session_state.agent = agent
        st.session_state.agent_loaded = True
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement de l'agent : {str(e)}")
        st.session_state.agent_loaded = False
        st.stop()

# Sidebar
with st.sidebar:
    st.markdown("# 🤖 Alpha AI")
    st.markdown("### Intelligent Research Assistant")
    st.markdown("---")
    
    # Agent status
    if st.session_state.agent_loaded:
        st.success("✅ Agent opérationnel")
    else:
        st.error("❌ Agent non chargé")
    
    st.markdown("---")
    
    # Usage statistics
    st.markdown("### 📊 Session Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Queries", st.session_state.total_queries)
    with col2:
        remaining = MAX_QUERIES - st.session_state.total_queries
        st.metric("Remaining", max(0, remaining))
    
    # Progress bar
    progress = min(st.session_state.total_queries / MAX_QUERIES, 1.0)
    st.progress(progress)
    
    # Limit management
    if st.session_state.total_queries >= MAX_QUERIES:
        st.error("**⚠️ Query limit reached**")
        st.info("Please refresh the page to start a new session.")
        st.stop()
    elif st.session_state.total_queries >= MAX_QUERIES * 0.8:
        st.warning(f"⚠️ Warning: {remaining} queries remaining")
    
    st.markdown("---")
    
    # Available tools
    st.markdown("### 🛠️ Available Tools")
    
    with st.expander("🐍 Python Expert", expanded=False):
        st.markdown("""
        **Specialized in:**
        - Python concepts and definitions
        - Code examples
        - Best practices
        - Debugging and optimization
        """)
    
    with st.expander("🔍 Academic Research", expanded=False):
        st.markdown("""
        **Access to:**
        - Research articles
        - Scientific publications
        - Encyclopedic knowledge
        - Recent news
        """)
    
    with st.expander("📄 Document Analyzer", expanded=False):
        st.markdown("""
        **Capabilities:**
        - Document summaries
        - Key information extraction
        - Content analysis
        """)
    
    st.markdown("---")
    
    # File Upload Section
    st.markdown("### 📎 Upload Document")
    uploaded_file = st.file_uploader(
        "Upload a file to analyze",
        type=['txt'],
        help="Supported formats: TXT"
    )
    
    if uploaded_file is not None:
        # Read file content
        try:
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'txt' or file_extension == 'md':
                content = uploaded_file.read().decode('utf-8')
            
            st.session_state.uploaded_file_content = content
            st.session_state.uploaded_file_name = uploaded_file.name
            
            st.success(f"✅ File loaded: {uploaded_file.name}")
            st.info(f"📊 Size: {len(content)} characters")
            
            # Preview
            with st.expander("👁️ Preview "):
                st.text(content[:500] + "..." if len(content) > 500 else content)
            
            # Quick action buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📝 Summarize", use_container_width=True):
                    st.session_state.quick_action = f"Summarize this document:\n\n{content}"
                    st.rerun()
            with col2:
                if st.button("🔍 Analyze", use_container_width=True):
                    st.session_state.quick_action = f"Analyze the key points of this document:\n\n{content}"
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("💡 Make sure you have the required libraries: `pip install PyPDF2 python-docx`")
    
    elif st.session_state.uploaded_file_name:
        st.info(f"📄 Current file: {st.session_state.uploaded_file_name}")
        if st.button("🗑️ Clear file", use_container_width=True):
            st.session_state.uploaded_file_content = None
            st.session_state.uploaded_file_name = None
            st.rerun()
    
    st.markdown("---")
    
    # Usage guide
    st.markdown("### 💡 Usage Guide")
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
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_count += 1
        st.rerun()
    
    st.markdown("---")
    
    # API status
    st.markdown("### 🔌 Service Status")
    groq_status = "✅ Active" if os.getenv("GROQ_API_KEY") else "❌ Inactive"
    tavily_status = "✅ Active" if os.getenv("TAVILY_API_KEY") else "❌ Inactive"
    Rag_tool = "✅RAG"
    
    st.markdown(f"""
    **GROQ:** {groq_status}  
    **Tavily:** {tavily_status}
    **RAG:**{Rag_tool}
    """)

# Main content
st.markdown("# 🤖 Alpha AI")
st.markdown("### Intelligent assistant for research, learning and analysis")

# Welcome message
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div class="info-box">
        <h3>👋 Welcome to Alpha AI</h3>
        <p>Your intelligent assistant capable of:</p>
        <ul>
            <li><strong>🐍 Python Research</strong> - Concepts, code examples and best practices via RAG</li>
            <li><strong>🔍 Academic Research</strong> - Access to scientific articles, news and encyclopedic knowledge</li>
            <li><strong>📄 Document Analysis</strong> - Summaries and key information extraction</li>
        </ul>
        <p><strong>Start by asking your question below.</strong></p>
    </div>
    """, unsafe_allow_html=True)

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display tools if available
        if "tools_used" in message:
            st.caption(f"🛠️ Tools: {message['tools_used']}")

# Handle quick actions from file upload buttons
if st.session_state.quick_action:
    prompt = st.session_state.quick_action
    st.session_state.quick_action = None  # Reset
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": "📄 [Document Analysis Request]"})
    st.session_state.total_queries += 1
    
    with st.chat_message("user"):
        st.markdown("📄 **Document Analysis Request**")
    
    # Generate response (same logic as below but triggered by button)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🔄 Analyzing document..."):
            try:
                initial_state = State(
                    messages=[{"role": "user", "content": prompt}]
                )
                
                start_time = time.time()
                result = st.session_state.agent.invoke(initial_state)
                response_time = time.time() - start_time
                
                if response_time > 100:
                    st.warning("⚠️ The query took an unusually long time.")
                
                response = result['messages'][-1].content
                message_placeholder.markdown(response)
                
                # Detect tools used
                tools_used = []
                for msg in result['messages']:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if 'name' in tool_call:
                                tools_used.append(tool_call['name'])
                    elif hasattr(msg, 'name'):
                        tools_used.append(msg.name)
                
                if tools_used:
                    st.markdown("---")
                    unique_tools = list(set(tools_used))
                    tools_html = " ".join([f'<span class="tool-badge">{tool}</span>' for tool in unique_tools])
                    st.markdown(f"🛠️ **Tools used:** {tools_html}", unsafe_allow_html=True)
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    tools_display = ", ".join(unique_tools)
                else:
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    tools_display = None
                
                message_data = {
                    "role": "assistant", 
                    "content": response
                }
                if tools_display:
                    message_data["tools_used"] = tools_display
                    
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                
                with st.expander("🔍 Error details (for debugging)"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())
    
    st.rerun()

# Input area
if prompt := st.chat_input("💬 Ask your question..."):
    # Input validation
    if len(prompt.strip()) < 5:
        st.error("⚠️ Your question is too short. Please provide more details.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.total_queries += 1
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check if user wants to analyze uploaded file
    if st.session_state.uploaded_file_content and any(keyword in prompt.lower() for keyword in ['document', 'file', 'uploaded', 'résumé', 'resume', 'analyze', 'summary']):
        # Append file content to prompt
        full_prompt = f"{prompt}\n\n--- DOCUMENT CONTENT ---\n{st.session_state.uploaded_file_content}"
        st.info(f"📎 Using uploaded file: {st.session_state.uploaded_file_name}")
    else:
        full_prompt = prompt
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🔄 Analyzing your query..."):
            try:
                # Create initial state
                initial_state = State(
                    messages=[{"role": "user", "content": full_prompt}]
                )
                
                start_time = time.time()
                
                # *** INVOCATION DE L'AGENT ***
                result = st.session_state.agent.invoke(initial_state)
                
                response_time = time.time() - start_time
                
                # Warning if response time is too long
                if response_time > 100:
                    st.warning("⚠️ The query took an unusually long time. If you encounter issues, please refresh the page.")
                
                # Extract response
                response = result['messages'][-1].content
                
                # Display response
                message_placeholder.markdown(response)
                
                # Detect tools used
                tools_used = []
                for msg in result['messages']:
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            if 'name' in tool_call:
                                tools_used.append(tool_call['name'])
                    elif hasattr(msg, 'name'):
                        tools_used.append(msg.name)
                
                # Display tools used and metrics
                if tools_used:
                    st.markdown("---")
                    unique_tools = list(set(tools_used))
                    tools_html = " ".join([f'<span class="tool-badge">{tool}</span>' for tool in unique_tools])
                    st.markdown(f"🛠️ **Tools used:** {tools_html}", unsafe_allow_html=True)
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    tools_display = ", ".join(unique_tools)
                else:
                    st.caption(f"⏱️ Response time: {response_time:.2f}s")
                    tools_display = None
                
                # Add to history
                message_data = {
                    "role": "assistant", 
                    "content": response
                }
                if tools_display:
                    message_data["tools_used"] = tools_display
                    
                st.session_state.messages.append(message_data)
                
            except Exception as e:
                error_message = f"❌ An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                
                # Display more detailed error for debugging
                with st.expander("🔍 Error details (for debugging)"):
                    st.code(str(e))
                    import traceback
                    st.code(traceback.format_exc())

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**⚡ Powered by**")
    st.markdown("LangGraph & Groq")
with col2:
    st.markdown("**🕐 Session Started**")
    st.markdown(datetime.now().strftime("%Y-%m-%d %H:%M"))
with col3:
    st.markdown("**📊 Status**")
    st.markdown("🟢 Operational" if st.session_state.agent_loaded else "🔴 Agent Error")

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    **📰 News & Current Events:**
    - What are the latest developments in artificial intelligence?
    - Recent news about climate change
    
    **🐍 Python Questions:**
    - What are metaclasses in Python?
    - How to use decorators in Python?
    - Explain the GIL (Global Interpreter Lock) in Python
    
    **📄 Document Analysis:**
    - Upload a document and click "Summarize" or "Analyze"
    - Or ask: "Summarize the uploaded document"
    - "What are the key points in this document?"
    
    **🎓 Academic Research:**
    - What are the latest research findings on machine learning?
    - Explain the concept of convolutional neural networks
    """)