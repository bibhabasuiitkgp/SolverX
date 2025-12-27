"""
SolverX Frontend - Streamlit UI for the JEE problem solver
"""

import streamlit as st
import requests
import base64

# Backend API URL
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="SolverX - JEE Problem Solver",
    page_icon="🧮",
    layout="wide"
)

# MathJax configuration - inject into page
st.markdown("""
<script>
MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  },
  svg: {
    fontCache: 'global'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js">
</script>
""", unsafe_allow_html=True)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .solution-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    .solution-container h2, .solution-container h3 {
        color: #333;
        margin-top: 1.5rem;
    }
    .solution-container p {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧮 SolverX</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered JEE Mains & Advanced Problem Solver</p>', unsafe_allow_html=True)

# Input method selection
input_method = st.radio(
    "Choose input method:",
    ["📝 Text Problem", "🖼️ Image Problem"],
    horizontal=True
)

# Initialize session state
if "solution" not in st.session_state:
    st.session_state.solution = None

# Input section
if input_method == "📝 Text Problem":
    problem_text = st.text_area(
        "Enter your JEE problem:",
        height=150,
        placeholder="Type or paste your JEE Mains/Advanced problem here..."
    )
    
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        if problem_text.strip():
            with st.spinner("🔍 Analyzing and solving the problem..."):
                try:
                    response = requests.post(
                        f"{API_URL}/solve/text",
                        json={"problem_text": problem_text}
                    )
                    if response.status_code == 200:
                        st.session_state.solution = response.json()
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the backend. Make sure the server is running!")
        else:
            st.warning("⚠️ Please enter a problem first!")

else:  # Image Problem
    uploaded_file = st.file_uploader(
        "Upload an image of the problem:",
        type=["png", "jpg", "jpeg", "webp"]
    )
    
    additional_text = st.text_input(
        "Additional context (optional):",
        placeholder="Any additional information about the problem..."
    )
    
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Problem", use_container_width=True)
    
    if st.button("🚀 Solve Problem", type="primary", use_container_width=True):
        if uploaded_file:
            with st.spinner("🔍 Analyzing image and solving the problem..."):
                try:
                    files = {"image": uploaded_file.getvalue()}
                    data = {"additional_text": additional_text} if additional_text else {}
                    
                    response = requests.post(
                        f"{API_URL}/solve/image",
                        files={"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                        data=data
                    )
                    if response.status_code == 200:
                        st.session_state.solution = response.json()
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to the backend. Make sure the server is running!")
        else:
            st.warning("⚠️ Please upload an image first!")


def render_latex_markdown(content: str):
    """Render markdown with LaTeX using HTML component with MathJax."""
    import markdown
    
    # Convert markdown to HTML (but preserve LaTeX delimiters)
    # We'll wrap the content in a div and let MathJax handle the math
    html_content = f"""
    <div class="solution-container">
        {content}
    </div>
    <script>
        if (typeof MathJax !== 'undefined') {{
            MathJax.typesetPromise();
        }}
    </script>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# Display solution
if st.session_state.solution:
    st.divider()
    st.subheader("📋 Solution")
    
    # Tabs for formatted and raw solution
    tab1, tab2 = st.tabs(["✨ Formatted Solution", "📄 Raw Solution"])
    
    with tab1:
        # Use the custom renderer for LaTeX
        render_latex_markdown(st.session_state.solution["formatted_solution"])
    
    with tab2:
        st.code(st.session_state.solution["raw_solution"], language=None)
    
    # Clear button
    if st.button("🗑️ Clear Solution"):
        st.session_state.solution = None
        st.rerun()

# Footer
st.divider()
st.markdown(
    "<p style='text-align: center; color: #888;'>Powered by Gemini 2.5 Pro Preview | Built with ❤️ for JEE Aspirants</p>",
    unsafe_allow_html=True
)
