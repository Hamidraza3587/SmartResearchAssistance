import re
import streamlit as st
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import backend components
from backend.processor import DocumentProcessor
from backend.embeddings import EmbeddingModel
from backend.summarizer import Summarizer
from backend.qa_engine import QAEngine
from backend.challenge_engine import ChallengeEngine
from backend.llm import LLMClient
from backend.chunker import TextChunker

# Custom CSS for better UI
def load_css():
    st.markdown("""
    <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        
        /* Sidebar */
        .css-1d391kg, .css-1d391kg > div:first-child {
            background-color: #f8f9fa !important;
            border-right: 1px solid #dee2e6;
        }
        
        /* Buttons */
        .stButton > button {
            border-radius: 8px;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        
        /* Text areas */
        .stTextArea textarea {
            min-height: 100px;
            border-radius: 8px;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.5rem 1rem;
            margin: 0;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #f0f2f6;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #4e73df;
        }
        
        /* Chat messages */
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        
        .stChatMessage .assistant-message {
            background-color: #f0f7ff;
            border-left: 4px solid #4e97ff;
        }
        
        .stChatMessage .user-message {
            background-color: #f8f9fa;
            border-left: 4px solid #6c757d;
        }
        
        /* Code blocks */
        pre {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            overflow-x: auto;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }
        
        th, td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            .stTabs [data-baseweb="tab"] {
                padding: 0.5rem;
                font-size: 0.9rem;
            }
        }
    </style>
    """, unsafe_allow_html=True)

load_css()

# Initialize session state
def initialize_session_state():
    """Initialize or reset the session state."""
    if 'document_processed' not in st.session_state:
        st.session_state.document_processed = False
        st.session_state.processing_steps = []
        st.session_state.document_name = None
        st.session_state.chunks = None
        st.session_state.embedding_model = None
        st.session_state.qa_engine = None
        st.session_state.summarizer = None
        st.session_state.challenge_engine = None
        st.session_state.summary = None
        st.session_state.generated_questions = None
        st.session_state.processing_complete = False
        st.session_state.processing_progress = 0
        st.session_state.processing_status = ""
        st.session_state.processing_errors = []
        st.session_state.qa_history = []

# Initialize components
def initialize_components():
    """Initialize backend components."""
    try:
        if 'embedding_model' not in st.session_state or st.session_state.embedding_model is None:
            with st.spinner("Initializing AI models..."):
                st.session_state.embedding_model = EmbeddingModel()
                st.session_state.summarizer = Summarizer()
                st.session_state.qa_engine = QAEngine(
                    embedding_model=st.session_state.embedding_model
                )
                st.session_state.challenge_engine = ChallengeEngine()
    except Exception as e:
        error_msg = f"Failed to initialize AI components: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        st.stop()

def update_processing_status(step: str, progress: int, error: bool = False, error_msg: str = None):
    """Update the processing status and progress."""
    if error:
        st.session_state.processing_errors.append(f"{step}: {error_msg}")
        st.session_state.processing_steps.append(f"❌ {step} - Failed: {error_msg}")
    else:
        st.session_state.processing_steps.append(f"✅ {step}")
    
    st.session_state.processing_status = step
    st.session_state.processing_progress = progress
    st.experimental_rerun()

# Initialize session state
initialize_session_state()

def process_document(file_path: str):
    """Process the uploaded document."""
    try:
        update_processing_status("Initializing document processor...", 10)
        processor = DocumentProcessor()
        
        # Process document
        update_processing_status("Extracting text from document...", 20)
        chunks = processor.process_document(file_path)
        st.session_state.chunks = chunks
        
        # Create embeddings
        update_processing_status("Creating document embeddings...", 40)
        st.session_state.embedding_model.create_embeddings(chunks)
        
        # Initialize QA engine with embeddings
        update_processing_status("Setting up question answering system...", 60)
        st.session_state.qa_engine = QAEngine(
            embedding_model=st.session_state.embedding_model
        )
        
        # Generate initial summary
        update_processing_status("Generating document summary...", 80)
        full_text = "\n\n".join([chunk[0] for chunk in chunks])
        st.session_state.summary = st.session_state.summarizer.summarize(full_text)
        
        # Mark as complete
        update_processing_status("Document processing complete!", 100)
        st.session_state.document_processed = True
        st.session_state.processing_complete = True
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Document processing error: {error_msg}", exc_info=True)
        update_processing_status("Document processing", 0, True, error_msg)
        st.error(f"Error processing document: {error_msg}")
        st.stop()

# Initialize components
@st.cache_resource
def load_llm_client():
    """Initialize and cache the LLM client."""
    return LLMClient()

@st.cache_resource
def load_embedding_model():
    """Initialize and cache the embedding model."""
    return EmbeddingModel()

@st.cache_resource
def load_qa_engine(_llm_client, _embedding_model):
    """Initialize and cache the QA engine."""
    return QAEngine(embedding_model=_embedding_model, llm_client=_llm_client)

@st.cache_resource
def load_challenge_engine(_llm_client):
    """Initialize and cache the challenge engine."""
    return ChallengeEngine(llm_client=_llm_client)

@st.cache_resource
def load_document_processor():
    """Initialize and cache the document processor."""
    return DocumentProcessor()

# Initialize components
llm_client = load_llm_client()
embedding_model = load_embedding_model()
qa_engine = load_qa_engine(llm_client, embedding_model)
challenge_engine = load_challenge_engine(llm_client)
document_processor = load_document_processor()

def process_uploaded_file(uploaded_file) -> Optional[str]:
    """
    Process the uploaded file and return its text content.
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        The extracted text content if successful, None otherwise
    """
    try:
        # Get the file extension
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Process the file
            if file_ext == '.pdf':
                # For PDFs, use the document processor's PDF extraction
                text = document_processor.extract_text_from_pdf(tmp_file_path)
            elif file_ext == '.txt':
                # For text files, read directly
                with open(tmp_file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Clean the text
            text = document_processor._clean_text(text)
            
            # Process the document with the extracted text
            process_document(text)
            
            return text
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(tmp_file_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary file: {e}")
                
    except Exception as e:
        error_msg = f"Error processing file: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None

def process_document(content):
    """
    Fast document processing that focuses on quick text extraction and simple summarization.
    This version skips complex processing for maximum speed.
    """
    try:
        # Clear previous state
        doc_name = st.session_state.get('document_name', 'document.txt')
        st.session_state.update({
            'progress': 0,
            'document_processed': False,
            'document_text': '',
            'summary': '',
            'questions': [],
            'chunks': [],
            'document_name': doc_name
        })
        
        # Extract text (fast reading)
        with st.spinner("Reading document..."):
            if isinstance(content, str) and os.path.exists(content):
                # Read from file with error handling for encoding
                try:
                    with open(content, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # Fallback to binary read if UTF-8 fails
                    with open(content, 'rb') as f:
                        text = f.read().decode('latin-1')
            else:
                # Use content directly
                text = str(content)
            
            if not text.strip():
                raise ValueError("The document is empty or could not be read.")
            
            # Store raw text
            st.session_state.document_text = text
            st.session_state.progress = 50
        
        # Generate quick extractive summary (first few sentences)
        with st.spinner("Generating quick summary..."):
            # Simple sentence splitting (faster than regex)
            sentences = []
            current = ""
            for char in text:
                current += char
                if char in '.!?':
                    sentences.append(current.strip())
                    current = ""
                    if len(sentences) >= 5:  # Limit to 5 sentences
                        break
            
            summary = ' '.join(sentences)
            st.session_state.summary = summary if summary else "No summary could be generated."
            st.session_state.progress = 100
        
        # Set basic questions
        st.session_state.questions = [
            "What is this document about?",
            "What are the main points?",
            "Can you summarize the key information?"
        ]
        
        # Simple document chunks for Q&A (if needed)
        st.session_state.chunks = [text[i:i+1000] for i in range(0, min(3000, len(text)), 1000)]
        
        st.session_state.document_processed = True
        st.success("Document processed successfully!")
        
    except Exception as e:
        error_msg = f"Error processing document: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        st.session_state.document_processed = False
        st.session_state.progress = 0

def show_upload_prompt():
    """Show upload prompt if no document is processed."""
    st.title("Smart Research Assistant")
    st.markdown("""
    Welcome to the Smart Research Assistant! This tool helps you:
    - 📄 Upload and process research documents (PDF/TXT)
    - 📝 Generate concise summaries of your documents
    - ❓ Ask questions about the content
    - 🧠 Test your knowledge with challenge questions
    
    Get started by uploading a document using the sidebar.
    """)
    
    # Add some visual elements
    cols = st.columns(3)
    features = [
        ("📄", "Upload", "Upload your research document in PDF or TXT format."),
        ("🤖", "Analyze", "The AI will process and understand your document."),
        ("💡", "Interact", "Ask questions or test your knowledge with challenges.")
    ]
    
    for i, (icon, title, desc) in enumerate(features):
        with cols[i]:
            st.markdown(f"<h3 style='text-align: center;'>{icon} {title}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center;'>{desc}</p>", unsafe_allow_html=True)
    
    # Add example section
    st.markdown("---")
    st.subheader("How It Works")
    st.video("https://www.youtube.com/watch?v=example_video_id")  # Replace with actual demo video

def show_processing_screen():
    """Show processing screen while document is being processed."""
    st.title("Processing Your Document")
    
    # Show progress bar
    progress_bar = st.progress(st.session_state.processing_progress / 100)
    st.caption(st.session_state.processing_status)
    
    # Show processing steps
    st.subheader("Processing Steps")
    for step in st.session_state.processing_steps:
        st.write(step)
    
    # Show errors if any
    if st.session_state.processing_errors:
        st.error("### Errors occurred during processing:")
        for error in st.session_state.processing_errors:
            st.error(f"- {error}")
    
    # Add a cancel button
    if st.button("❌ Cancel Processing"):
        st.session_state.processing_complete = True
        st.experimental_rerun()

def main():
    """Main application function."""
    # Load custom CSS
    load_css()
    
    # Initialize components
    initialize_components()
    
    # Main title
    st.title("📚 Smart Research Assistant")
    
    # Sidebar for document upload
    with st.sidebar:
        st.header("📂 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF or TXT file",
            type=["pdf", "txt"],
            help="Supported formats: PDF, TXT"
        )
        
        if uploaded_file is not None and not st.session_state.get('processing', False):
            try:
                st.session_state.processing = True
                st.session_state.processing_status = "Processing document..."
                st.session_state.document_processed = False
                
                try:
                    # Set document name
                    st.session_state.document_name = uploaded_file.name
                    
                    # Process the file directly (bypassing process_uploaded_file for speed)
                    with st.spinner("Processing document..."):
                        # Read file content directly
                        try:
                            if uploaded_file.name.lower().endswith('.pdf'):
                                try:
                                    # For PDFs, use PyPDF2 if available
                                    import PyPDF2
                                    try:
                                        pdf_reader = PyPDF2.PdfReader(uploaded_file)
                                        text = ""
                                        for page in pdf_reader.pages[:10]:  # Limit to first 10 pages for speed
                                            page_text = page.extract_text()
                                            if page_text:  # Only add if text was extracted
                                                text += page_text + "\n"
                                            if len(text) > 50000:  # Limit to ~50KB of text
                                                break
                                        if not text.strip():
                                            raise ValueError("No text could be extracted from the PDF")
                                    except Exception as e:
                                        st.warning("Could not extract text using PyPDF2. Trying fallback method...")
                                        # Fallback: read as binary and try to extract text
                                        text = uploaded_file.getvalue().decode('latin-1', errors='replace')
                                        text = ' '.join(text.split())  # Clean up whitespace
                                except ImportError:
                                    st.warning("PyPDF2 not available. Using basic text extraction.")
                                    # Fallback: read as binary and try to extract text
                                    text = uploaded_file.getvalue().decode('latin-1', errors='replace')
                                    text = ' '.join(text.split())  # Clean up whitespace
                            else:
                                # For text files, read directly
                                text = uploaded_file.getvalue().decode('utf-8', errors='replace')
                            
                            # Process the extracted text
                            process_document(text)
                            st.session_state.document_processed = True
                            st.session_state.processing = False
                            st.rerun()
                            
                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")
                            logger.error(f"File read error: {str(e)}")
                            st.session_state.processing = False
                            st.session_state.document_processed = False
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Upload error: {str(e)}")
                    st.session_state.processing = False
                    st.session_state.document_processed = False
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
                    logger.error(f"Document processing error: {str(e)}", exc_info=True)
                    st.session_state.processing = False
                    st.session_state.document_processed = False
                    return
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
                logger.error(f"Unexpected error: {str(e)}", exc_info=True)
                st.session_state.processing = False
                st.session_state.document_processed = False
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        **Smart Research Assistant** helps you:
        - 📄 Process and understand documents
        - 📊 Generate concise summaries
        - ❓ Answer your questions
        - 🧠 Test your knowledge
        
        Powered by Hugging Face Transformers and Streamlit.
        """)
    
    # Show processing screen if processing
    if st.session_state.get('processing', False):
        show_processing_screen()
    # Show main interface if document is processed
    elif st.session_state.get('document_processed', False):
        st.sidebar.success("✅ Document processed successfully!")
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["📝 Summary", "❓ Q&A", "🎯 Challenge Me"])
        
        with tab1:
            show_summary_tab()
            
        with tab2:
            show_qa_tab()
            
        with tab3:
            show_challenge_tab()
    
    # Show upload prompt if no document is processed
    else:
        st.markdown("### Get started with your research")
        
        # Three columns for features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 1. Upload")
            st.markdown("Upload your research paper, article, or any text document in PDF or TXT format.")
            
        with col2:
            st.markdown("#### 2. Analyze")
            st.markdown("Let our AI analyze the content and extract key information.")
            
        with col3:
            st.markdown("#### 3. Interact")
            st.markdown("Ask questions, get summaries, or test your understanding with challenges.")
        
        st.markdown("---")
        st.info("👈 Use the sidebar to upload your document and get started!")

def show_summary_tab():
    """Display the summary tab content."""
    st.header("📝 Document Summary")
    
    if not st.session_state.get('document_processed', False):
        st.warning("Please process a document first to see the summary.")
        return
        
    if not st.session_state.get('summary'):
        st.warning("No summary was generated for this document.")
        return
    
    # Display document info
    doc_name = st.session_state.get('document_name', 'document')
    st.caption(f"Document: {doc_name}")
    
    # Word count and processing time
    try:
        word_count = len(st.session_state.summary.split())
        st.caption(f"Summary: {word_count} words")
        
        # Display the summary with nice formatting
        with st.container():
            st.markdown("### Key Points")
            st.markdown(st.session_state.summary)
        
        # Add export options
        st.download_button(
            label="📥 Download Summary",
            data=st.session_state.summary,
            file_name=f"{os.path.splitext(doc_name)[0]}_summary.md",
            mime="text/markdown"
        )
    except Exception as e:
        st.error(f"Error displaying summary: {str(e)}")
        logger.error(f"Error in show_summary_tab: {str(e)}", exc_info=True)

def show_qa_tab():
    """Display the Q&A tab content."""
    st.header("❓ Ask a Question")
    
    if not st.session_state.get('chunks'):
        st.warning("Please process a document first to enable question answering.")
        return
    
    # Initialize chat history if it doesn't exist
    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []
    
    # Display chat history
    for i, (question, answer, sources) in enumerate(st.session_state.qa_history):
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user question to chat history
        st.session_state.qa_history.append((prompt, "", []))
        
        # Generate response
        with st.spinner("Generating answer..."):
            try:
                answer, sources = st.session_state.qa_engine.answer_question(prompt)
                st.session_state.qa_history[-1] = (prompt, answer, sources)
                st.experimental_rerun()
            except Exception as e:
                error_msg = f"Error generating answer: {str(e)}"
                st.error(error_msg)
                logger.error(error_msg, exc_info=True)

def show_challenge_tab():
    """Display the challenge tab content."""
    st.header("🧠 Challenge Me")
    
    if not st.session_state.get('chunks'):
        st.warning("Please process a document first to enable challenge questions.")
        return
    
    # Initialize challenge state
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = []
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = {}
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = {}
    
    # Generate questions if none exist
    if not st.session_state.challenge_questions:
        if st.button("🎯 Generate Challenge Questions"):
            with st.spinner("Generating challenge questions..."):
                try:
                    st.session_state.challenge_questions = st.session_state.challenge_engine.generate_questions(
                        st.session_state.chunks,
                        num_questions=3
                    )
                    st.experimental_rerun()
                except Exception as e:
                    error_msg = f"Error generating questions: {str(e)}"
                    st.error(error_msg)
                    logger.error(error_msg, exc_info=True)
        
        st.info("Click the button above to generate challenge questions about the document.")
        return
    
    # Show questions and answer inputs
    st.subheader("Test Your Knowledge")
    st.markdown("Answer the following questions based on the document. After submitting your answers, you'll receive feedback and a score.")
    
    # Tab interface for each question
    tabs = st.tabs([f"Q{i+1}" for i in range(len(st.session_state.challenge_questions))])
    
    for i, (tab, question) in enumerate(zip(tabs, st.session_state.challenge_questions)):
        with tab:
            st.markdown(f"**Question {i+1}:** {question}")
            
            # Text area for user's answer
            answer_key = f"answer_{i}"
            user_answer = st.text_area(
                "Your answer:",
                value=st.session_state.user_answers.get(answer_key, ""),
                key=answer_key,
                height=150
            )
            
            # Store the answer
            st.session_state.user_answers[answer_key] = user_answer
            
            # Submit button for each question
            if st.button(f"Submit Answer {i+1}", key=f"submit_{i}"):
                if not user_answer.strip():
                    st.warning("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating your answer..."):
                        try:
                            evaluation = st.session_state.challenge_engine.evaluate_answer(
                                question=question,
                                user_answer=user_answer,
                                context_chunks=st.session_state.chunks
                            )
                            st.session_state.evaluations[i] = evaluation
                            st.experimental_rerun()
                        except Exception as e:
                            error_msg = f"Error evaluating answer: {str(e)}"
                            st.error(error_msg)
                            logger.error(error_msg, exc_info=True)
            
            # Show evaluation if available
            if i in st.session_state.evaluations:
                evaluation = st.session_state.evaluations[i]
                st.markdown("---")
                st.markdown("### Evaluation")
                
                # Score visualization
                score = evaluation.get('score', 0)
                st.metric("Score", f"{score}/10")
                
                # Feedback
                if 'feedback' in evaluation:
                    with st.expander("📝 Feedback"):
                        st.markdown(evaluation['feedback'])
                
                # Suggested improvements
                if 'suggested_improvements' in evaluation and evaluation['suggested_improvements']:
                    with st.expander("💡 Suggested Improvements"):
                        for improvement in evaluation['suggested_improvements']:
                            st.markdown(f"- {improvement}")
                
                # Model's ideal answer
                if 'model_answer' in evaluation and evaluation['model_answer']:
                    with st.expander("📚 Model's Suggested Answer"):
                        st.markdown(evaluation['model_answer'])
    
    # Reset button
    if st.button("🔄 Start New Challenge"):
        st.session_state.challenge_questions = []
        st.session_state.user_answers = {}
        st.session_state.evaluations = {}
        st.experimental_rerun()

if __name__ == "__main__":
    main()
