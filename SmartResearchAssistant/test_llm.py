import os
import sys
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def test_llm_connection():
    """Test the LLM connection and model loading."""
    # Check if Hugging Face token is set
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token or hf_token == "your_huggingface_token_here":
        logger.error("HUGGINGFACEHUB_API_TOKEN not set in .env file")
        return False
    
    logger.info("Hugging Face token found in environment variables")
    
    # Try to import and initialize the LLM client
    try:
        from backend.llm import LLMClient
        
        # Initialize with a smaller model for testing
        logger.info("Initializing LLM client with a smaller model for testing...")
        client = LLMClient(
            model_name="gpt2",  # Using a small model for testing
            use_4bit=False  # Disable 4-bit for testing
        )
        
        # Test a simple generation
        logger.info("Testing model generation...")
        test_prompt = "Hello, how are you?"
        response = client.generate(test_prompt, max_tokens=20)
        
        logger.info(f"Test prompt: {test_prompt}")
        logger.info(f"Model response: {response}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing LLM: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    print("Testing LLM connection and model loading...")
    success = test_llm_connection()
    if success:
        print("\n✅ LLM test completed successfully!")
    else:
        print("\n❌ LLM test failed. Check the logs above for details.")
