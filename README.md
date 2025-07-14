# Smart Research Assistant

An AI-powered research assistant that helps you summarize and interact with your research documents.

## Features

- **Document Upload**: Upload PDF or TXT documents
- **Automatic Summarization**: Get concise summaries of your documents
- **Ask Anything**: Ask questions about the document content
- **Challenge Me**: Test your understanding with generated questions

## Setup

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Running the Application

1. Navigate to the project directory
2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
3. Open your browser and go to `http://localhost:8501`

## Usage

1. **Upload Document**: Click "Upload Document" and select a PDF or TXT file
2. **View Summary**: See an automatic summary of the document
3. **Ask Questions**: Use the "Ask Anything" tab to ask questions about the document
4. **Challenge Yourself**: Use the "Challenge Me" tab to test your understanding

## Project Structure

- `app.py`: Main Streamlit application
- `backend/`: Contains all the backend logic
  - `processor.py`: Handles document processing
  - `summarizer.py`: Generates document summaries
  - `qa_engine.py`: Handles question answering
  - `challenge_engine.py`: Generates and evaluates questions
  - `embeddings.py`: Manages document embeddings
  - `llm.py`: Wraps LLM functionality
- `utils/`: Utility functions
  - `prompts.py`: Contains prompt templates
