"""
Code Reviewer - Advanced Agentic Application
Main Streamlit application for intelligent code review with context awareness
"""

import streamlit as st
import os
from pathlib import Path
import json
from datetime import datetime

from code_analyzer import CodeAnalyzer
from context_manager import ContextManager
from utils import load_files, save_review_history, load_review_history

# Page configuration
st.set_page_config(
    page_title="Code Reviewer - Agentic AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 16px;
    }
    .review-section {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .code-block {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 15px;
        border-radius: 5px;
        font-family: 'Courier New', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'context_manager' not in st.session_state:
    st.session_state.context_manager = ContextManager()
if 'code_analyzer' not in st.session_state:
    st.session_state.code_analyzer = CodeAnalyzer()
if 'review_history' not in st.session_state:
    st.session_state.review_history = []
if 'indexed_files' not in st.session_state:
    st.session_state.indexed_files = []

def main():
    st.title("🔍 Code Reviewer - Agentic AI")
    st.markdown("### Advanced multi-file code review with codebase context awareness")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Ollama settings
        st.subheader("Ollama Settings")
        ollama_host = st.text_input("Ollama Host", value="http://localhost:11434")
        model_name = st.text_input("Model Name", value="codellama")

        # Update analyzer settings
        st.session_state.code_analyzer.update_settings(ollama_host, model_name)

        st.divider()

        # Review settings
        st.subheader("Review Settings")
        review_depth = st.select_slider(
            "Review Depth",
            options=["Quick", "Standard", "Deep", "Comprehensive"],
            value="Standard"
        )

        focus_areas = st.multiselect(
            "Focus Areas",
            ["Code Quality", "Security", "Performance", "Best Practices",
             "Documentation", "Testing", "Architecture"],
            default=["Code Quality", "Security", "Best Practices"]
        )

        max_context_files = st.slider(
            "Max Context Files",
            min_value=1,
            max_value=20,
            value=5,
            help="Maximum number of related files to include as context"
        )

        st.divider()

        # Statistics
        st.subheader("📊 Statistics")
        st.metric("Indexed Files", len(st.session_state.indexed_files))
        st.metric("Reviews Completed", len(st.session_state.review_history))

    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Code Review", "📚 Context Manager", "📜 History", "ℹ️ About"])

    with tab1:
        code_review_tab(review_depth, focus_areas, max_context_files)

    with tab2:
        context_manager_tab()

    with tab3:
        history_tab()

    with tab4:
        about_tab()

def code_review_tab(review_depth, focus_areas, max_context_files):
    st.header("Code Review")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Upload Files for Review")
        uploaded_files = st.file_uploader(
            "Choose files to review",
            accept_multiple_files=True,
            type=['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cpp', 'c', 'h', 'go', 'rs', 'rb', 'php']
        )

    with col2:
        st.subheader("Options")
        use_context = st.checkbox("Use Codebase Context", value=True,
                                   help="Include related files from indexed codebase")
        auto_suggest_fixes = st.checkbox("Auto-suggest Fixes", value=True)
        include_examples = st.checkbox("Include Code Examples", value=False)

    if uploaded_files:
        st.divider()

        # Display uploaded files
        with st.expander("📁 Uploaded Files", expanded=True):
            for idx, file in enumerate(uploaded_files):
                st.text(f"{idx + 1}. {file.name} ({file.size} bytes)")

        # Review button
        if st.button("🚀 Start Review", type="primary", use_container_width=True):
            with st.spinner("Analyzing code... This may take a moment..."):
                # Process files
                files_data = []
                for file in uploaded_files:
                    content = file.read().decode('utf-8')
                    files_data.append({
                        'name': file.name,
                        'content': content,
                        'language': Path(file.name).suffix[1:]
                    })

                # Get relevant context if enabled
                context_files = []
                if use_context and len(st.session_state.indexed_files) > 0:
                    with st.status("Retrieving relevant context..."):
                        # Combine all file contents for context search
                        query_text = "\n".join([f['content'] for f in files_data])
                        context_files = st.session_state.context_manager.get_relevant_context(
                            query_text,
                            top_k=max_context_files
                        )
                        st.write(f"Found {len(context_files)} relevant context files")

                # Perform analysis
                with st.status("Performing code analysis..."):
                    review_config = {
                        'depth': review_depth,
                        'focus_areas': focus_areas,
                        'auto_suggest_fixes': auto_suggest_fixes,
                        'include_examples': include_examples
                    }

                    results = st.session_state.code_analyzer.analyze(
                        files_data,
                        context_files,
                        review_config
                    )

                # Display results
                display_review_results(results, files_data)

                # Save to history
                st.session_state.review_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'files': [f['name'] for f in files_data],
                    'results': results,
                    'config': review_config
                })

def display_review_results(results, files_data):
    st.success("✅ Code review completed!")

    # Overall summary
    st.subheader("📊 Review Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Issues Found", results.get('total_issues', 0))
    with col2:
        st.metric("Critical", results.get('critical_count', 0))
    with col3:
        st.metric("Warnings", results.get('warning_count', 0))
    with col4:
        st.metric("Suggestions", results.get('suggestion_count', 0))

    # Overall feedback
    if 'overall_feedback' in results:
        with st.container():
            st.markdown("### 💡 Overall Feedback")
            st.markdown(results['overall_feedback'])

    st.divider()

    # Per-file reviews
    st.subheader("📄 Detailed File Reviews")

    for file_result in results.get('file_reviews', []):
        with st.expander(f"📄 {file_result['filename']}", expanded=True):
            # File summary
            if 'summary' in file_result:
                st.markdown("**Summary:**")
                st.info(file_result['summary'])

            # Issues
            if file_result.get('issues'):
                st.markdown("**Issues Found:**")
                for issue in file_result['issues']:
                    severity = issue.get('severity', 'info')
                    icon = {'critical': '🔴', 'warning': '🟡', 'suggestion': '🔵', 'info': 'ℹ️'}.get(severity, 'ℹ️')

                    st.markdown(f"{icon} **{issue.get('type', 'General')}** (Line {issue.get('line', 'N/A')})")
                    st.markdown(f"> {issue.get('message', '')}")

                    if issue.get('suggestion'):
                        st.markdown(f"**Suggested Fix:** {issue['suggestion']}")

                    if issue.get('code_example'):
                        st.code(issue['code_example'], language=file_result.get('language', 'python'))

            # Positive aspects
            if file_result.get('positive_aspects'):
                st.markdown("**✨ Positive Aspects:**")
                for aspect in file_result['positive_aspects']:
                    st.markdown(f"- {aspect}")

def context_manager_tab():
    st.header("Codebase Context Manager")
    st.markdown("Index your codebase to provide intelligent context for code reviews")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Index Codebase")

        # Directory input
        codebase_path = st.text_input(
            "Codebase Directory Path",
            placeholder="/path/to/your/project",
            help="Enter the path to your project directory"
        )

        # File patterns
        file_patterns = st.text_input(
            "File Patterns (comma-separated)",
            value="*.py,*.js,*.ts,*.java,*.go",
            help="Specify file patterns to include"
        )

        # Exclusion patterns
        exclude_patterns = st.text_input(
            "Exclude Patterns (comma-separated)",
            value="*test*,*node_modules*,*venv*,*.min.js",
            help="Specify patterns to exclude"
        )

        if st.button("🔍 Index Codebase", type="primary"):
            if codebase_path and os.path.isdir(codebase_path):
                with st.spinner("Indexing codebase... This may take a while for large projects..."):
                    patterns = [p.strip() for p in file_patterns.split(',')]
                    excludes = [p.strip() for p in exclude_patterns.split(',')]

                    indexed_count = st.session_state.context_manager.index_codebase(
                        codebase_path,
                        patterns,
                        excludes
                    )

                    st.session_state.indexed_files = st.session_state.context_manager.get_indexed_files()
                    st.success(f"✅ Successfully indexed {indexed_count} files!")
            else:
                st.error("❌ Invalid directory path")

    with col2:
        st.subheader("Quick Actions")

        if st.button("🗑️ Clear Index"):
            st.session_state.context_manager.clear_index()
            st.session_state.indexed_files = []
            st.success("Index cleared!")

        if st.button("💾 Export Index"):
            index_data = st.session_state.context_manager.export_index()
            st.download_button(
                "Download Index",
                data=json.dumps(index_data, indent=2),
                file_name="codebase_index.json",
                mime="application/json"
            )

    # Display indexed files
    if st.session_state.indexed_files:
        st.divider()
        st.subheader("📚 Indexed Files")

        search_query = st.text_input("🔍 Search indexed files", placeholder="Search by filename or content...")

        display_files = st.session_state.indexed_files
        if search_query:
            display_files = [f for f in display_files if search_query.lower() in f.lower()]

        # Paginated display
        items_per_page = 20
        total_pages = (len(display_files) + items_per_page - 1) // items_per_page

        if total_pages > 0:
            page = st.slider("Page", 1, total_pages, 1)
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page

            for idx, file_path in enumerate(display_files[start_idx:end_idx], start=start_idx + 1):
                st.text(f"{idx}. {file_path}")

def history_tab():
    st.header("Review History")

    if not st.session_state.review_history:
        st.info("No reviews yet. Complete a code review to see history here.")
        return

    # Display reviews in reverse chronological order
    for idx, review in enumerate(reversed(st.session_state.review_history)):
        with st.expander(
            f"Review #{len(st.session_state.review_history) - idx} - "
            f"{review['timestamp']} ({len(review['files'])} files)",
            expanded=(idx == 0)
        ):
            st.markdown(f"**Files Reviewed:** {', '.join(review['files'])}")
            st.markdown(f"**Review Depth:** {review['config']['depth']}")
            st.markdown(f"**Focus Areas:** {', '.join(review['config']['focus_areas'])}")

            results = review['results']

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Issues", results.get('total_issues', 0))
            with col2:
                st.metric("Critical", results.get('critical_count', 0))
            with col3:
                st.metric("Warnings", results.get('warning_count', 0))

            if st.button(f"View Full Report", key=f"view_{idx}"):
                st.json(results)

def about_tab():
    st.header("About Code Reviewer")

    st.markdown("""
    ### 🎯 Features

    **Code Reviewer** is an advanced agentic application for intelligent code analysis powered by Ollama LLMs.

    #### Key Capabilities:

    - **Multi-File Analysis**: Review multiple files simultaneously with cross-file insights
    - **Context-Aware Reviews**: Index your codebase to provide relevant context for better analysis
    - **Semantic Search**: Uses vector embeddings to find related code automatically
    - **Customizable Depth**: Choose from Quick to Comprehensive review levels
    - **Focus Areas**: Target specific aspects like security, performance, or best practices
    - **Auto-Suggest Fixes**: Get actionable suggestions with code examples
    - **Review History**: Track all your reviews with full reports

    #### Technology Stack:

    - **Streamlit**: Interactive web interface
    - **Ollama**: Local LLM inference
    - **Sentence Transformers**: Semantic embeddings
    - **ChromaDB**: Vector database for context retrieval

    #### How It Works:

    1. **Index Your Codebase**: Point to your project directory to build a semantic index
    2. **Upload Files**: Select files you want to review
    3. **Configure Review**: Choose depth, focus areas, and other options
    4. **Get Insights**: Receive detailed analysis with context from your codebase

    ---

    ### 🚀 Getting Started

    1. Ensure Ollama is running: `ollama serve`
    2. Pull a code model: `ollama pull codellama`
    3. Index your codebase in the "Context Manager" tab
    4. Upload files and start reviewing!

    ---

    ### 📝 Tips

    - Index your codebase first for the best results
    - Use "Deep" or "Comprehensive" review for critical code
    - Enable "Auto-suggest Fixes" to get improvement recommendations
    - Check the history tab to track improvements over time

    """)

if __name__ == "__main__":
    main()
